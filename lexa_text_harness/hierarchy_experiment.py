#!/usr/bin/env python3
"""Frozen-codec hierarchical retrieval pilot. Existing 5 uM cases are development data."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from agent import ChatCompletionsClient, TransportError
from annotate_cards import atomic_write_json
from experiment import metrics, selected_cases, training_deltas
from tools import sha256


ROOT = Path(__file__).resolve().parent
SOURCES = ['hierarchy_experiment.py', 'hierarchy_data.py', 'hierarchy_agent.py', 'agent.py',
           'experiment.py', 'tools.py', 'annotate_cards.py']
ANNOTATION_PROMPT = '''Describe the molecular scope of frozen LEXA axes in plain language.
Each axis can include several functions. Its states are categorical signed expression
patterns, not ordered activity levels. Use the supplied training-derived mask/effect
genes and existing state labels as evidence, without claiming causal regulation or
formal GO enrichment. Be cautious: these are A549 pilot features, not universal axes.
Return JSON only: {"axes":[{"axis_id":0,"name":"short scope name",
"summary":"one concise sentence","functions":["tentative function"],
"evidence_genes":["ENSG00000123456.7"],"uncertainty":"specific limitation"}]}.
Cover every requested axis exactly once. At most 3 functions and 4 evidence genes per
axis. Use unknown if evidence is insufficient. Name<=100 characters, summary<=400,
uncertainty<=300. Do not assign an up/down direction to the entire axis.'''
ANNOTATION_PROMPT += '''\nIMPORTANT: evidence_genes must copy exact versioned Ensembl IDs from that
axis's allowed_gene_ids list (the gene_id fields), never gene symbols such as
MKI67 or CDKN1A. Symbols are display names and are NOT valid evidence_genes.
Use an empty evidence_genes list if none of the allowed IDs supports a claim.'''


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def save(path, value):
    atomic_write_json(Path(path), value)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def axes_list(cards):
    if isinstance(cards, list):
        return cards
    axes = cards.get('axes', cards.get('axis_cards'))
    if isinstance(axes, dict):
        return [axes[k] for k in sorted(axes, key=lambda k: int(str(k).replace('A', ''))) ]
    if not isinstance(axes, list):
        raise ValueError('Unknown axis card structure')
    return axes


def extract_genes(obj):
    found = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ('gene', 'gene_id') and isinstance(v, str):
                found.add(v)
            elif k == 'evidence_genes' and isinstance(v, list):
                found.update(g for g in v if isinstance(g, str))
            else:
                found.update(extract_genes(v))
    elif isinstance(obj, list):
        for item in obj:
            found.update(extract_genes(item))
    return found


def parse_annotations(content, batch):
    content = content.strip()
    if content.startswith('```json') and content.endswith('```'):
        content = content[7:-3].strip()
    obj = json.loads(content)
    expected = {int(c['axis_id']): c for c in batch}
    if set(obj) != {'axes'} or not isinstance(obj['axes'], list) or len(obj['axes']) != len(expected):
        raise ValueError('Return exactly one annotation for every requested axis')
    output = {}
    fields = {'axis_id', 'name', 'summary', 'functions', 'evidence_genes', 'uncertainty'}
    for row in obj['axes']:
        if not isinstance(row, dict) or set(row) != fields:
            raise ValueError('Each axis must contain exactly the six declared fields')
        axis = row['axis_id']
        if type(axis) is not int or axis not in expected or axis in output:
            raise ValueError('Missing, duplicate or unexpected axis ID')
        for field, maximum in [('name', 100), ('summary', 400), ('uncertainty', 300)]:
            if not isinstance(row[field], str) or not row[field].strip() or len(row[field]) > maximum:
                raise ValueError(f'{field} must be a nonempty string within its limit')
        if not isinstance(row['functions'], list) or len(row['functions']) > 3 or any(
                not isinstance(x, str) or not x.strip() or len(x) > 160 for x in row['functions']):
            raise ValueError('functions must contain at most three short strings')
        genes = row['evidence_genes']
        if not isinstance(genes, list) or len(genes) > 4 or any(not isinstance(g, str) for g in genes) or len(set(genes)) != len(genes) or not set(genes) <= extract_genes(expected[axis]):
            raise ValueError(f'Axis {axis}: evidence_genes must cite at most four unique displayed gene IDs; copy exact gene_id strings from allowed_gene_ids, never gene symbols')
        output[axis] = row
    return output


def annotate_axes(cards, client, output, max_retries=2):
    output = Path(output)
    fingerprint = {'cards': digest(cards), 'prompt': digest(ANNOTATION_PROMPT),
                   'model': client.model, 'endpoint': client.url, 'max_retries': max_retries,
                   'source': sha256(__file__)}
    destination = output / 'axis_annotations.json'
    if destination.exists():
        old = read(destination)
        if old['fingerprint'] != fingerprint:
            raise ValueError('Axis annotation inputs changed; choose a new output')
        return old
    rows = axes_list(cards)
    all_annotations = {}
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        # The full state list is available; keep prompt bounded by abbreviating each label.
        compact = []
        for axis in batch:
            item = {k: v for k, v in axis.items() if k != 'states'}
            item['states'] = [{k: v for k, v in s.items() if k in
                               ('state_id', 'name', 'summary', 'label', 'language_annotation',
                                'evidence_genes')} for s in axis.get('states', [])]
            item['allowed_gene_ids'] = sorted(extract_genes(item))
            compact.append(item)
        batch_dir = output / 'annotation_requests' / f'axes_{start:02d}_{start+len(batch)-1:02d}'
        success_path = batch_dir / 'completed.json'
        if success_path.exists():
            completed = read(success_path)
            if completed['fingerprint'] != fingerprint:
                raise ValueError('Cached annotation fingerprint differs')
            all_annotations.update(parse_annotations(json.dumps({'axes': completed['axes']}), compact))
            continue
        messages = [{'role': 'system', 'content': ANNOTATION_PROMPT},
                    {'role': 'user', 'content': json.dumps({'axes': compact}, ensure_ascii=False)}]
        success = None
        for attempt in range(max_retries + 1):
            path = batch_dir / f'attempt_{attempt:02d}'
            request_path=path.with_suffix('.request.json')
            response_path=path.with_suffix('.response.json')
            status_path=path.with_suffix('.status.json')
            request={'messages':messages,'fingerprint':fingerprint}
            if request_path.exists():
                if read(request_path)!=request:
                    raise ValueError('Cached annotation request changed')
                status=read(status_path) if status_path.exists() else {'transport':'unknown_interrupted_attempt'}
                response=read(response_path) if response_path.exists() else None
                if response_path.exists() and status.get('response_sha256') and status['response_sha256']!=sha256(response_path):
                    raise ValueError('Cached annotation response changed')
            else:
                save(request_path, request)
                try:
                    response = client(messages=messages, tools=[])
                    save(response_path, response)
                    status = {'transport': 'success'}
                except TransportError as exc:
                    response = None
                    status = {'transport': 'failed', 'error': str(exc)}
            if response_path.exists():
                status['response_sha256']=sha256(response_path)
            try:
                if response is None:
                    raise ValueError('Transport failed; retry the same request')
                content = response['choices'][0]['message']['content']
                success = parse_annotations(content, compact)
                status['validated'] = True
                save(path.with_suffix('.status.json'), status)
                break
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                error = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError) else 'Invalid JSON response structure'
                status.update(validated=False, validation_error=error)
                save(path.with_suffix('.status.json'), status)
                messages = messages[:2] + [{'role': 'user', 'content': 'Previous output failed validation: ' + error + '. Return the complete corrected JSON.'}]
        if success is None:
            raise RuntimeError('Axis annotation budget exhausted; attempts preserved')
        all_annotations.update(success)
        save(success_path, {'fingerprint': fingerprint, 'axes': list(success.values())})
        print(json.dumps({'stage': 'axis_annotation', 'completed_axes': len(all_annotations)}), flush=True)
    result = {'status': 'complete', 'created': now(), 'fingerprint': fingerprint,
              'annotation_method': 'DeepSeek scope descriptions grounded in supplied training-derived genes and existing state labels; no formal GO enrichment or biological review',
              'axes': [all_annotations[i] for i in range(32)]}
    save(destination, result)
    return result


def source_hashes():
    return {name: sha256(ROOT / name) for name in SOURCES}


def predict(work, output, client, workers=2):
    from hierarchy_data import load_case_bundle
    from hierarchy_agent import run_case, evaluate_selection
    work, output = Path(work), Path(output)
    manifest = read(output / 'inputs/manifest.json')
    cards = read(manifest['cards_path'])
    protocol = {'version': 'hierarchical_occupancy_v3', 'created': now(),
                'evaluation_role': 'previously unblinded 5uM development experiment',
                'case_ids': [c['case_id'] for c in manifest['cases']],
                'methods': ['expression_nearest', 'expression_interpolate', 'code_nearest',
                            'code_interpolate', 'hierarchical_local', 'hierarchical_joint'],
                'candidate_ids': ['keep', 'nearest', 'interpolate', 'lower', 'upper', 'half', 'onehalf'],
                'local_batches': 4, 'axes_per_batch': 8, 'joint_successful_rounds': 1,
                'repair_retries_per_stage': 2, 'model': client.model, 'endpoint': client.url,
                'workers': workers, 'source_hashes': source_hashes(),
                'input_manifest_sha256': sha256(output / 'inputs/manifest.json'),
                'limitations': ['Joint uses one extra successful call; not budget matched.',
                    'Candidate search is restricted to training-derived occupancy profiles/scales, not arbitrary 512 probabilities.',
                    'Axis functionality is provisional language annotation; no GO enrichment.',
                    'Independent axis marginals determine additive mean, not a unique joint cell population.']}
    plan_path = output / 'protocol.json'
    if plan_path.exists():
        previous = read(plan_path)
        if {k:v for k,v in previous.items() if k!='created'} != {k:v for k,v in protocol.items() if k!='created'}:
            raise ValueError('Prediction protocol changed; use a new output')
    else:
        save(plan_path, protocol)
    annotations = annotate_axes(cards, client, output)
    annotations = {a['axis_id']: a for a in annotations['axes']}
    for axis in axes_list(cards):
        axis['language_annotation'] = annotations[axis['axis_id']]
    merged = output / 'axis_cards_annotated.json'
    save(merged, cards)

    def run_one(case):
        bundle = load_case_bundle(work, case['npz'], case['context'])
        folder = output / 'predictions' / case['slug']
        folder.mkdir(parents=True, exist_ok=True)
        for candidate, method in [('nearest', 'code_nearest'), ('interpolate', 'code_interpolate')]:
            selections = [{'axis_id': a, 'candidate_id': candidate} for a in range(32)]
            evaluation = evaluate_selection(bundle, selections)
            path = folder / (method + '.npz')
            np.savez_compressed(path, q=evaluation['q'], p0=bundle['p0'],
                                delta_mean=evaluation['delta_mean'], genes=bundle['genes'])
        result = run_case(bundle, cards, client, folder, max_retries=2)
        print(json.dumps({'stage': 'prediction', 'case_id': case['case_id'], 'status': 'complete'}), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, case) for case in manifest['cases']]
        for future in as_completed(futures):
            future.result()
    # Seal every required result before the score command opens any treated outcomes.
    files = {}
    for case in manifest['cases']:
        folder = output / 'predictions' / case['slug']
        for name in ['code_nearest.npz', 'code_interpolate.npz', 'local.npz', 'joint.npz', 'local.json', 'joint.json']:
            path = folder / name
            files[str(path.relative_to(output))] = sha256(path)
    for path in (output / 'inputs').rglob('*'):
        if path.is_file():
            files[str(path.relative_to(output))] = sha256(path)
    files['axis_annotations.json'] = sha256(output / 'axis_annotations.json')
    files['axis_cards_annotated.json'] = sha256(merged)
    files['protocol.json'] = sha256(plan_path)
    seal = {'status': 'all_predictions_complete', 'sealed': now(), 'n_cases': len(manifest['cases']),
            'source_hashes': source_hashes(), 'files': files,
            'score_exists_at_seal': (output / 'scores_summary.csv').exists()}
    if (output / 'prediction_seal.json').exists():
        prior = read(output / 'prediction_seal.json')
        if prior['files'] != files or prior['source_hashes'] != seal['source_hashes']:
            raise ValueError('Previously sealed results changed')
    else:
        save(output / 'prediction_seal.json', seal)
    return seal


def verify_seal(output):
    output = Path(output)
    seal = read(output / 'prediction_seal.json')
    if seal['status'] != 'all_predictions_complete':
        raise ValueError('Predictions incomplete')
    for name, expected in seal['files'].items():
        if sha256(output / name) != expected:
            raise ValueError('Sealed file changed: ' + name)
    if seal['source_hashes'] != source_hashes():
        raise ValueError('Source changed since prediction seal')
    return seal


def score(work, output):
    work, output = Path(work), Path(output)
    seal = verify_seal(output)
    manifest = read(output / 'inputs/manifest.json')
    cases = sorted(selected_cases(work / 'prepared', 0), key=lambda c:c['case_id'])
    if [c['case_id'] for c in cases] != [c['case_id'] for c in manifest['cases']]:
        raise ValueError('Cases differ from planned full16 population')
    # Outcomes are accessed only below, after all candidate choices have been sealed.
    with np.load(work / 'prepared/evaluation.npz', allow_pickle=False) as z:
        data = {k:z[k] for k in z.files}
    genes = data['genes'].astype(str)
    with np.load(work / 'codec_changes/case_arrays.npz', allow_pickle=False) as z:
        true_occupancy = {str(k): (z['pcontrol'][i], z['ptreated'][i]) for i,k in enumerate(z['case_ids'])}
    # Verify previously encoded observed occupancies belong to the same frozen model.
    codec_manifest = read(work / 'codec_changes/manifest.json')
    code_path=(work/'codec_changes/case_arrays.npz').resolve()
    expected_code_hash=codec_manifest['outputs_sha256'].get('case_arrays.npz',codec_manifest['outputs_sha256'].get(str(code_path)))
    if expected_code_hash!=sha256(code_path):
        raise ValueError('Observed code array checksum differs from original diagnostic')
    for name in ['lexa/best.pt','lexa/exact_code_effects.npz','prepared/evaluation.npz']:
        path=(work/name).resolve()
        if codec_manifest['inputs_sha256'].get(str(path)) != sha256(path):
            raise ValueError('Observed code diagnostic source differs: '+name)
    rows, occupancy_rows, all_q = [], [], {}
    old_manifest = read(work / 'experiment_v2_feedback/inputs/manifest.json')
    old_cases = {c['case_id']:c for c in old_manifest['cases']}
    old_responses = training_deltas(work / 'prepared')
    for case, detail in zip(manifest['cases'], cases):
        case_id = case['case_id']
        treatment = data['x_test'][data['test_case_ids'].astype(str) == case_id]
        control = data['control_x'][np.isin(data['control_cell_ids'].astype(str), detail['control_cell_ids'])]
        if len(treatment)!=detail['n_treated'] or len(control)!=detail['n_control'] or min(len(treatment),len(control))<20:
            raise ValueError('Observed sample membership/count differs from ready case')
        observed = treatment.mean(axis=0) - control.mean(axis=0)
        with np.load(case['npz'], allow_pickle=False) as z:
            inputs = {k:z[k] for k in z.files}
        if not np.array_equal(inputs['genes'].astype(str),genes):
            raise ValueError('Expression retrieval gene order differs')
        p0, ptrue = true_occupancy[case_id]
        if not np.allclose(p0, inputs['p0'], atol=1e-7):
            raise ValueError('Observed code control differs from prediction input')
        predictions = {'no_edit': np.zeros_like(observed),
                       'expression_nearest': inputs['nearest_observed_delta'],
                       'expression_interpolate': inputs['interpolate_observed_delta'],
                       'old_training_mean': np.stack([r[2] for r in old_responses]).mean(axis=0)}
        available = [(dose, delta) for drug,dose,delta in old_responses if drug==case['drug']]
        predictions['old_nearest_training_dose'] = min(available, key=lambda r: abs(np.log1p(r[0])-np.log1p(case['dose_uM'])))[1]
        folder = output / 'predictions' / case['slug']
        for filename, method in [('code_nearest', 'code_nearest'), ('code_interpolate', 'code_interpolate'),
                                  ('local', 'hierarchical_local'), ('joint', 'hierarchical_joint')]:
            with np.load(folder / (filename+'.npz'), allow_pickle=False) as z:
                if not np.array_equal(z['genes'].astype(str), genes):
                    raise ValueError('Prediction gene order differs')
                predictions[method] = z['delta_mean'].copy()
                q = z['q'].copy()
            if q.shape != (32,16) or not np.isfinite(q).all() or q.min() < -1e-9 or not np.allclose(q.sum(1),1):
                raise ValueError('Invalid predicted occupancy')
            all_q[(case_id, method)] = q
            occupancy_rows.append({'case_id': case_id, 'drug':case['drug'], 'method':method,
                  'occupancy_tv_error_mean': float((.5*np.abs(q-ptrue).sum(1)).mean()),
                  'dominant_code_accuracy': float((q.argmax(1)==ptrue.argmax(1)).mean()),
                  'predicted_changed_axes': int((q.argmax(1)!=p0.argmax(1)).sum()),
                  'observed_changed_axes': int((ptrue.argmax(1)!=p0.argmax(1)).sum()),
                  'predicted_mean_axis_tv': float((.5*np.abs(q-p0).sum(1)).mean()),
                  'observed_mean_axis_tv': float((.5*np.abs(ptrue-p0).sum(1)).mean())})
        for old_method in ['one_shot', 'agent']:
            old = read(work / 'experiment_v2_feedback' / old_method / old_cases[case_id]['slug'] / 'agent_result.json')
            path = old['evaluation']['artifact']
            if sha256(path) != old['evaluation']['artifact_sha256']:
                raise ValueError('Old prediction artifact changed')
            with np.load(path, allow_pickle=False) as z:
                assert np.array_equal(z['genes'].astype(str), genes)
                predictions['old_'+old_method] = z['delta_mean'].copy()
        for method, prediction in predictions.items():
            if prediction.shape!=observed.shape or not np.isfinite(prediction).all():
                raise ValueError('Invalid expression prediction: '+method)
            row = {'case_id':case_id, 'drug':case['drug'], 'dose_uM':case['dose_uM'], 'method':method,
                   'n_treated':len(treatment), 'n_control':len(control), **metrics(prediction, observed)}
            row['response_l2_ratio'] = float(np.linalg.norm(prediction)/np.linalg.norm(observed))
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(output/'scores_by_well.csv', index=False)
    columns = ['delta_rmse','delta_pearson','top100_delta_rmse','top100_direction_accuracy',
               'top100_abs_delta_overlap','response_l2_ratio']
    condition = table.groupby(['method','drug','dose_uM'],as_index=False)[columns].mean()
    summary = condition.groupby('method',as_index=False)[columns].mean()
    condition.to_csv(output/'scores_by_condition.csv',index=False)
    summary.to_csv(output/'scores_summary.csv',index=False)
    occupancy = pd.DataFrame(occupancy_rows)
    occupancy.to_csv(output/'occupancy_by_well.csv',index=False)
    occupancy.groupby(['method','drug'],as_index=False).mean(numeric_only=True).to_csv(output/'occupancy_by_condition.csv',index=False)
    changes = []
    for drug in sorted({c['drug'] for c in cases}):
        ids = [c['case_id'] for c in cases if c['drug']==drug]
        p0 = np.mean([true_occupancy[c][0] for c in ids],axis=0)
        values = {'observed':np.mean([true_occupancy[c][1] for c in ids],axis=0)}
        for method in ['code_nearest','code_interpolate','hierarchical_local','hierarchical_joint']:
            values[method] = np.mean([all_q[(c,method)] for c in ids],axis=0)
        for method,q in values.items():
            tv = .5*np.abs(q-p0).sum(1)
            changes.append({'drug':drug,'method':method,'dominant_changed_axes':int((q.argmax(1)!=p0.argmax(1)).sum()),
                            'tv_gt_0_2_axes':int((tv>.2).sum()),'mean_tv':float(tv.mean())})
    pd.DataFrame(changes).to_csv(output/'axis_changes_by_condition.csv',index=False)
    save(output/'evaluation_scope.json', {'status':'complete','scored':now(), 'role':'development;5uM previously unblinded',
         'n_cases':len(cases),'n_treated':int(table.drop_duplicates('case_id').n_treated.sum()),'n_genes':len(genes),
         'aggregation':'equal wells within drug, then equal four drugs; per-well metrics averaged',
         'prediction_seal_sha256':sha256(output/'prediction_seal.json'),
         'evaluation_sha256':sha256(work/'prepared/evaluation.npz'),
         'observed_codes_sha256':sha256(work/'codec_changes/case_arrays.npz'),
         'limitations':['No new independent test; no significance claim.','Fixed24h A549 only.',
                        'Top100 is largest observed mean shifts, not statistical DEGs.',
                        'Old v2 baseline used nonempty train wells; new retrieval requires >=20 per treatment/control.',
                        'Code marginals do not specify individual cell transitions.']})
    return summary


def report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output)
    table=pd.read_csv(output/'scores_summary.csv').set_index('method')
    order=['no_edit','old_agent','old_nearest_training_dose','expression_nearest','expression_interpolate',
           'code_nearest','code_interpolate','hierarchical_local','hierarchical_joint']
    labels=['No edit','Old text agent','Old nearest expression','Matched expression nearest',
            'Matched expression interpolate','Code nearest','Code interpolate','Hierarchical local','Hierarchical joint']
    fig,axes=plt.subplots(1,3,figsize=(15,5.8))
    colors=['#bbc2c8','#9e90ba','#909ca8','#7aa7ae','#427f89','#e7ae63','#d58a35','#5277aa','#295487']
    for ax,metric,title in zip(axes,['delta_rmse','delta_pearson','response_l2_ratio'],
                             ['Delta RMSE (lower better)','Delta Pearson (higher better)','Predicted / observed response norm']):
        values=table.loc[order,metric].to_numpy()
        ax.barh(range(len(order)),np.nan_to_num(values),color=colors)
        ax.set_yticks(range(len(order)),labels if ax==axes[0] else ['']*len(order))
        ax.invert_yaxis();ax.set_title(title,fontsize=10);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        for i,v in enumerate(values):
            ax.text((v if np.isfinite(v) else 0)+.007,i,f'{v:.3f}' if np.isfinite(v) else 'undefined',va='center',fontsize=8)
        ax.set_xlim(0,max(np.nanmax(values)*1.25,.1))
    fig.suptitle('Hierarchical LEXA pilot: 16 wells, 4 drugs, 5 uM / 24 h\nDevelopment data; equal-well metrics, then equal-drug average',fontsize=12)
    fig.tight_layout();fig.savefig(output/'metrics_comparison.png',dpi=170);plt.close(fig)
    scope=read(output/'evaluation_scope.json')
    changes=pd.read_csv(output/'axis_changes_by_condition.csv')
    request_status=[]
    usage=[]
    for path in output.rglob('*response.json'):
        if 'requests' not in str(path):
            continue
        obj=read(path)
        usage.append({'file':str(path.relative_to(output)),**obj.get('usage',{})})
    if usage:
        pd.DataFrame(usage).to_csv(output/'api_usage.csv',index=False)
    html_text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>LEXA分层预测v3</title>
<style>body{font-family:system-ui;max-width:1250px;margin:40px auto;padding:0 20px;line-height:1.7;color:#223}table{border-collapse:collapse;font-size:14px}td,th{padding:7px 10px;border:1px solid #ccd}img{max-width:100%}code{background:#eee}</style>
<h1>LEXA分层占用预测 v3</h1><p>已揭盲5µM开发验证：A549，4种药物，固定24小时，16个匹配孔、786个处理细胞、2000个训练选择基因。逐孔计算指标，药内孔等权，再四药等权。</p>
<p>冻结原codec，训练响应检索→轴功能与状态文字卡→32轴候选选择→一次全局联合修正。候选仅限训练剂量响应及其比例缩放；轴功能为DeepSeek暂定解释，未做正式GO富集。模型未微调。</p>
<img src="metrics_comparison.png"><h2>表达变化预测</h2>'''
    html_text+=table.to_html(float_format=lambda v:f'{v:.6f}',border=0)
    html_text+='<h2>各药物群体主导code更换轴数</h2><p>先对4孔占用等权平均再统计；不代表同一细胞的前后转移或显著性。</p>'
    html_text+=changes.pivot(index='drug',columns='method',values='dominant_changed_axes').to_html(border=0)
    html_text+='''<h2>解释边界</h2><p>新表达与code检索对照使用相同>=20细胞训练孔库；旧nearest用非空训练孔，单独展示。local与joint共享局部预测，但joint多一次模型调用，不能将差异单独归因为反馈机制。代码层容量扩大、检索加入和分层解释共同变化，不能把相对旧版的提升全部归因于语言。真实测试已被查看，当前结果不能作为新泛化证据。</p>
<p>轨迹、所有API尝试、输入来源、预测封存、逐孔/逐药物指标均在同目录保存。code边际可精确计算加法解码器的群体均值，不唯一确定真实细胞联合分布。</p></html>'''
    (output/'RESULTS.html').write_text(html_text)
    text='LEXA分层占用预测v3：已揭盲开发验证\n'+json.dumps(scope,ensure_ascii=False,indent=2)+'\n\n'+table.to_string()+'\n\n主导code变化轴数（条件等权占用）\n'+changes.to_string(index=False)+'\n'
    (output/'RESULTS.txt').write_text(text)
    return {'report':str(output/'RESULTS.html'),'api_responses':len(usage)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','predict','score','report'])
    parser.add_argument('--work',type=Path,default=ROOT/'runs/pilot_seen_dose')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--api-key-file',type=Path)
    parser.add_argument('--model',default=os.environ.get('DEEPSEEK_MODEL','deepseek-flash'))
    parser.add_argument('--base-url',default=os.environ.get('DEEPSEEK_BASE_URL','https://api.deepseek.com'))
    parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();args.work=args.work.resolve()
    output=(args.output or args.work/'experiment_v3_hierarchy').resolve()
    if args.stage=='prepare':
        from hierarchy_data import prepare_inputs
        result=prepare_inputs(args.work,output/'inputs')
        print(json.dumps({'prepared_cases':len(result['cases']),'output':str(output)},ensure_ascii=False))
    elif args.stage=='predict':
        if not 1<=args.workers<=4: parser.error('workers must be 1 to 4')
        key=args.api_key_file.read_text().strip() if args.api_key_file else os.environ.get('DEEPSEEK_API_KEY','')
        client=ChatCompletionsClient(args.model,key,args.base_url)
        result=predict(args.work,output,client,args.workers)
        print(json.dumps({'sealed':result['sealed'],'n_cases':result['n_cases']}))
    elif args.stage=='score':
        print(score(args.work,output).to_string(index=False))
    else:
        print(json.dumps(report(output),ensure_ascii=False))


if __name__=='__main__':
    main()
