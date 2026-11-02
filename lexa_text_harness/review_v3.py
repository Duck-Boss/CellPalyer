"""Independent train-only input/numerical audit for the hierarchy development run.

Does not import hierarchy_data/hierarchy_agent or read held-out treatment expression.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def occupancy(codes):
    return np.stack([np.bincount(codes[:, a], minlength=16) / len(codes)
                     for a in range(32)])


def simplex_bisection(values):
    """Independent threshold bisection, rather than sorting projection."""
    values = np.asarray(values, dtype=np.float64)
    lower, upper = float(values.min() - 1), float(values.max())
    for _ in range(80):
        middle = (lower + upper) / 2
        if np.maximum(values - middle, 0).sum() > 1:
            lower = middle
        else:
            upper = middle
    return np.maximum(values - (lower + upper) / 2, 0)


def project(values):
    return np.stack([simplex_bisection(row) for row in values])


def audit(work, experiment, require_predictions=False):
    work, experiment = Path(work), Path(experiment)
    inputs = experiment / 'inputs'
    manifest_path = inputs / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for artifact, digest in manifest['output_sha256'].items():
        assert sha(artifact) == digest, artifact
    case_entries = manifest['cases']
    cases = {c['case_id']: c for c in json.loads((work / 'prepared/cases.json').read_text())['cases']
             if c['status'] == 'ready'}
    assert len(case_entries) == len(cases) == 16
    assert {c['case_id'] for c in case_entries} == set(cases)
    with np.load(work / 'lexa/exact_code_effects.npz', allow_pickle=False) as z:
        effects = z['contributions'].astype(np.float64)
        genes = z['genes'].astype(str)
    with np.load(work / 'lexa/train_codes.npz', allow_pickle=False) as z:
        codes, ids = z['codes'], z['cell_ids'].astype(str)
        assert np.array_equal(z['genes'].astype(str), genes)
    metadata = pd.read_csv(work / 'prepared/metadata.csv').set_index('cell_id').loc[ids]
    assert metadata.index.is_unique and metadata['split'].eq('train').all()
    assert not metadata['dose_uM'].eq(5).any()
    with np.load(work / 'prepared/prepared.npz', allow_pickle=False) as z:
        assert np.array_equal(z['genes'].astype(str), genes)
        train_ids = z['train_cell_ids'].astype(str)
        indices = pd.Index(train_ids).get_indexer(ids)
        assert (indices >= 0).all()
        expression = z['x_train'][indices].astype(np.float64)
    with np.load(work / 'lexa/baseline.npz', allow_pickle=False) as z:
        control_ids, control_codes = z['cell_ids'].astype(str), z['codes']
        assert np.array_equal(z['genes'].astype(str), genes)
    doses = {}
    eligible = []
    excluded = []
    drug_values = metadata['drug'].to_numpy()
    dose_values = metadata['dose_uM'].to_numpy()
    plate_values = metadata['plate'].to_numpy()
    replicate_values = metadata['replicate'].to_numpy()
    for (drug, dose, plate, replicate), _ in metadata[metadata.dose_uM > 0].groupby(
            ['drug', 'dose_uM', 'plate', 'replicate']):
        shared = (drug_values == drug) & (plate_values == plate) & (replicate_values == replicate)
        treated = shared & (dose_values == dose)
        baseline = shared & (dose_values == 0)
        item = dict(drug=drug, dose_uM=float(dose), plate=plate, replicate=int(replicate),
                    n_treated=int(treated.sum()), n_control=int(baseline.sum()))
        if min(treated.sum(), baseline.sum()) < 20:
            excluded.append(item)
            continue
        dp = occupancy(codes[treated]) - occupancy(codes[baseline])
        dx = expression[treated].mean(0) - expression[baseline].mean(0)
        doses.setdefault((drug, float(dose)), []).append((dp, dx))
        eligible.append(item)
    reference = {key: (np.mean([v[0] for v in rows], axis=0),
                       np.mean([v[1] for v in rows], axis=0)) for key, rows in doses.items()}
    assert len(eligible) == 61 and len(excluded) == 30
    max_error = {}

    def check(name, actual, expected, atol=1e-6):
        actual, expected = np.asarray(actual), np.asarray(expected)
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        error = float(np.max(np.abs(actual - expected)))
        max_error[name] = max(error, max_error.get(name, 0))
        assert np.isfinite(actual).all() and error <= atol, (name, error, atol)

    checked_inputs = []
    expected_by_slug = {}
    for entry in case_entries:
        case = cases[entry['case_id']]
        positions = pd.Index(control_ids).get_indexer(case['control_cell_ids'])
        assert (positions >= 0).all()
        p0 = occupancy(control_codes[positions])
        available = sorted(d for drug, d in reference if drug == case['drug'])
        target = float(case['dose_uM'])
        nearest = min(available, key=lambda d: abs(np.log(d) - np.log(target)))
        lower = max((d for d in available if d <= target), default=available[0])
        upper = min((d for d in available if d >= target), default=available[-1])
        weight = ((np.log(target) - np.log(lower)) / (np.log(upper) - np.log(lower))
                  if lower != upper else 0.)
        weight = float(np.clip(weight, 0, 1))
        lo_dp, lo_dx = reference[(case['drug'], lower)]
        hi_dp, hi_dx = reference[(case['drug'], upper)]
        near_dp, near_dx = reference[(case['drug'], nearest)]
        inter_dp = lo_dp * (1-weight) + hi_dp * weight
        inter_dx = lo_dx * (1-weight) + hi_dx * weight
        displacements = dict(keep=np.zeros_like(p0), nearest=near_dp,
                             interpolate=inter_dp, lower=lo_dp, upper=hi_dp,
                             half=.5*inter_dp, onehalf=1.5*inter_dp)
        npz_path = Path(entry['npz'])
        if not npz_path.is_absolute():
            npz_path = inputs / npz_path
        with np.load(npz_path, allow_pickle=False) as z:
            names = z['candidate_names'].astype(str).tolist()
            assert set(names) == set(displacements)
            check('p0', z['p0'], p0)
            expected_q = np.stack([project(p0+displacements[name]) for name in names], axis=1)
            check('candidate_q', z['candidate_q'], expected_q)
            expected_delta = np.einsum('aks,asg->akg', expected_q-p0[:, None, :], effects)
            check('candidate_delta_genes', z['candidate_delta_genes'], expected_delta, 5e-6)
            check('nearest_observed_delta', z['nearest_observed_delta'], near_dx, 2e-6)
            check('interpolate_observed_delta', z['interpolate_observed_delta'], inter_dx, 2e-6)
            projection_l1 = np.stack([np.abs(expected_q[:, i] - (p0+displacements[name])).sum(1)
                                      for i, name in enumerate(names)], axis=1)
            if 'projection_l1' in z:
                check('projection_l1', z['projection_l1'], projection_l1, 2e-6)
            assert np.array_equal(z['genes'].astype(str), genes)
        expected_by_slug[entry['slug']] = (p0, expected_q, names)
        checked_inputs.append(dict(case_id=case['case_id'], slug=entry['slug'],
                                   npz_sha256=sha(npz_path), lower=lower, upper=upper,
                                   nearest=nearest, upper_weight=weight,
                                   lower_n_wells=len(doses[(case['drug'], lower)]),
                                   upper_n_wells=len(doses[(case['drug'], upper)])))

    predictions = []
    baseline_predictions = []
    for path in sorted((experiment / 'predictions').rglob('*.npz')):
        if path.name not in ['local.npz', 'joint.npz', 'code_nearest.npz', 'code_interpolate.npz']:
            continue
        slug = next((part for part in path.parts if part in expected_by_slug), None)
        assert slug is not None, path
        p0, candidate_q, names = expected_by_slug[slug]
        with np.load(path, allow_pickle=False) as z:
            q = z['q'].astype(np.float64)
            check('prediction_p0', z['p0'], p0)
            check('prediction_q_sum', q.sum(1), np.ones(32))
            assert q.shape == (32, 16) and np.isfinite(q).all() and q.min() >= -1e-10
            check('prediction_delta', z['delta_mean'], np.einsum('as,asg->g', q-p0, effects), 5e-6)
            assert np.array_equal(z['genes'].astype(str), genes)
            selected = []
            for a in range(32):
                distances = np.max(np.abs(candidate_q[a]-q[a]), axis=1)
                assert distances.min() < 1e-6, (path, a)
                selected.append(names[int(np.argmin(distances))])
        if path.stem.startswith('code_'):
            candidate_name = path.stem.removeprefix('code_')
            check('code_baseline_selection', q, candidate_q[:, names.index(candidate_name)])
            baseline_predictions.append(dict(path=str(path.resolve()), sha256=sha(path),
                                              method=path.stem, slug=slug))
            continue
        document = json.loads(path.with_suffix('.json').read_text())
        assert document['arrays_sha256'] == sha(path)
        selections = document['selections']
        assert len(selections) == 32
        assert {row['axis_id'] for row in selections} == set(range(32))
        choices = {row['axis_id']: row['candidate_id'] for row in selections}
        assert all(type(row['axis_id']) is int for row in selections)
        assert set(choices.values()).issubset(names)
        declared_q = np.stack([candidate_q[a, names.index(choices[a])] for a in range(32)])
        check('prediction_matches_declared_choices', q, declared_q)
        contract = json.loads((path.parent/'run_contract.json').read_text())
        identity_hash = hashlib.sha256(json.dumps(contract['identity'], ensure_ascii=False,
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        assert identity_hash == contract['fingerprint'] == document['fingerprint']
        for source, digest in contract['identity']['source_sha256'].items():
            assert sha(Path(__file__).parent/source) == digest, source
        stage_names = [f'local_{i:02d}' for i in range(4)] if path.stem == 'local' else ['joint_00']
        raw_choices = []
        for stage_name in stage_names:
            stage_dir = path.parent/'requests'/stage_name
            completed = [status for status in sorted(stage_dir.glob('*.status.json'))
                         if json.loads(status.read_text())['status'] == 'completed']
            assert len(completed) == 1, stage_dir
            status = json.loads(completed[0].read_text())
            response_path = completed[0].with_name(completed[0].name.replace('.status.json', '.response.json'))
            assert sha(response_path) == status['response_sha256']
            content = json.loads(response_path.read_text())['choices'][0]['message']['content'].strip()
            if content.startswith('```'):
                content = '\n'.join(content.splitlines()[1:-1])
            raw_choices.extend(json.loads(content)['selections'])
        assert sorted(raw_choices, key=lambda row: row['axis_id']) == sorted(selections, key=lambda row: row['axis_id'])
        predictions.append(dict(path=str(path.resolve()), sha256=sha(path), method=path.stem,
                                slug=slug, declared_candidate_choices=choices,
                                numeric_candidate_choices=selected))
    if require_predictions:
        assert len(predictions) == 32, len(predictions)
        assert len({(p['slug'], p['method']) for p in predictions}) == 32
        assert len(baseline_predictions) == 32, len(baseline_predictions)
    code_manifest = json.loads((work/'codec_changes/manifest.json').read_text())
    observed_code_sha = sha(work/'codec_changes/case_arrays.npz')
    assert observed_code_sha == code_manifest['outputs_sha256']['case_arrays.npz']
    report = dict(status='passed', created_asia_shanghai=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  no_heldout_treatment_expression_read=True,
                  independent_algorithms='Train-only regrouping; Euclidean simplex threshold bisection; full-gene additive recomputation.',
                  n_train_cells=len(ids), n_eligible_training_wells=len(eligible),
                  n_excluded_training_wells=len(excluded), n_target_cases=len(case_entries),
                  n_predictions_checked=len(predictions), max_absolute_errors=max_error,
                  n_code_baselines_checked=len(baseline_predictions),
                  input_cases=checked_inputs, predictions=predictions,
                  baseline_predictions=baseline_predictions,
                  observed_code_artifact_hash_verified_without_reading_arrays=observed_code_sha,
                  limitations=['Does not infer biological independence or causal effects.',
                               'Does not establish that API content lacked all possible semantic leakage.',
                               'Uses no held-out target expression and does not compute prediction accuracy.'],
                  implementation_sha256=sha(__file__), manifest_sha256=sha(manifest_path))
    out = experiment / ('independent_prediction_audit.json' if require_predictions else 'independent_input_audit.json')
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    assert json.loads(out.read_text())['status'] == 'passed'
    return report


def audit_scores(work, experiment):
    """Only run after the experiment has sealed predictions and produced scores."""
    work, experiment = Path(work), Path(experiment)
    seal_path = experiment/'prediction_seal.json'
    seal = json.loads(seal_path.read_text())
    assert seal['status'] == 'all_predictions_complete' and not seal['score_exists_at_seal']
    assert (experiment/'scores_summary.csv').exists()
    for name, digest in seal['files'].items():
        assert sha(experiment/name) == digest, name
    manifest = json.loads((experiment/'inputs/manifest.json').read_text())
    cases = {c['case_id']: c for c in json.loads((work/'prepared/cases.json').read_text())['cases']
             if c['status'] == 'ready'}
    with np.load(work/'prepared/evaluation.npz', allow_pickle=False) as z:
        x_test = z['x_test'].astype(np.float64)
        test_cases = z['test_case_ids'].astype(str)
        x_control = z['control_x'].astype(np.float64)
        control_ids = z['control_cell_ids'].astype(str)
        genes = z['genes'].astype(str)
    with np.load(work/'codec_changes/case_arrays.npz', allow_pickle=False) as z:
        true_codes = {case_id: (z['pcontrol'][i], z['ptreated'][i])
                      for i, case_id in enumerate(z['case_ids'].astype(str))}
    # Recompute old nonempty-well expression baselines independently as well.
    with np.load(work/'prepared/prepared.npz', allow_pickle=False) as z:
        x_train = z['x_train'].astype(np.float64)
        train_ids = z['train_cell_ids'].astype(str)
    metadata = pd.read_csv(work/'prepared/metadata.csv').set_index('cell_id').loc[train_ids]
    drug_values, dose_values = metadata.drug.to_numpy(), metadata.dose_uM.to_numpy()
    plate_values, replicate_values = metadata.plate.to_numpy(), metadata.replicate.to_numpy()
    old_wells = {}
    for (drug, dose, plate, replicate), _ in metadata[metadata.dose_uM > 0].groupby(
            ['drug', 'dose_uM', 'plate', 'replicate']):
        shared = (drug_values == drug) & (plate_values == plate) & (replicate_values == replicate)
        treatment, control = shared & (dose_values == dose), shared & (dose_values == 0)
        if treatment.any() and control.any():
            old_wells.setdefault((drug, float(dose)), []).append(x_train[treatment].mean(0)-x_train[control].mean(0))
    old_responses = {key: np.mean(values, axis=0) for key, values in old_wells.items()}
    old_global_mean = np.mean(list(old_responses.values()), axis=0)
    old_cases = {c['case_id']: c for c in json.loads((work/'experiment_v2_feedback/inputs/manifest.json').read_text())['cases']}
    rows, occupancy_rows, q_lookup = [], [], {}
    for entry in manifest['cases']:
        case_id = entry['case_id']
        case = cases[case_id]
        observed = x_test[test_cases == case_id].mean(0) - x_control[np.isin(control_ids,case['control_cell_ids'])].mean(0)
        n_treated = int((test_cases == case_id).sum())
        n_control = int(np.isin(control_ids,case['control_cell_ids']).sum())
        with np.load(entry['npz'], allow_pickle=False) as z:
            predictions = {'no_edit': np.zeros_like(observed),
                           'expression_nearest': z['nearest_observed_delta'],
                           'expression_interpolate': z['interpolate_observed_delta']}
        predictions['old_training_mean'] = old_global_mean
        available = sorted(dose for drug,dose in old_responses if drug == case['drug'])
        nearest = min(available, key=lambda dose: abs(np.log1p(dose)-np.log1p(case['dose_uM'])))
        predictions['old_nearest_training_dose'] = old_responses[(case['drug'], nearest)]
        p0, ptrue = true_codes[case_id]
        for short, method in [('code_nearest','code_nearest'),('code_interpolate','code_interpolate'),
                              ('local','hierarchical_local'),('joint','hierarchical_joint')]:
            with np.load(experiment/'predictions'/entry['slug']/(short+'.npz'), allow_pickle=False) as z:
                assert np.array_equal(z['genes'].astype(str),genes)
                predictions[method], q = z['delta_mean'].copy(), z['q'].copy()
            q_lookup[(case_id, method)] = q
            occupancy_rows.append(dict(case_id=case_id, method=method, drug=case['drug'],
                occupancy_tv_error_mean=float(np.abs(q-ptrue).sum(1).mean()/2),
                dominant_code_accuracy=float(np.mean(q.argmax(1)==ptrue.argmax(1))),
                predicted_changed_axes=int(np.sum(q.argmax(1)!=p0.argmax(1))),
                observed_changed_axes=int(np.sum(ptrue.argmax(1)!=p0.argmax(1))),
                predicted_mean_axis_tv=float(np.abs(q-p0).sum(1).mean()/2),
                observed_mean_axis_tv=float(np.abs(ptrue-p0).sum(1).mean()/2)))
        for method in ['one_shot','agent']:
            result=json.loads((work/'experiment_v2_feedback'/method/old_cases[case_id]['slug']/'agent_result.json').read_text())
            assert sha(result['evaluation']['artifact'])==result['evaluation']['artifact_sha256']
            with np.load(result['evaluation']['artifact'],allow_pickle=False) as z:
                predictions['old_'+method] = z['delta_mean'].copy()
        top = np.argsort(-np.abs(observed))[:100]
        active = np.abs(observed[top]) > 1e-8
        centered_truth = observed-observed.mean()
        for method, predicted in predictions.items():
            centered_pred = predicted-predicted.mean()
            denominator = np.linalg.norm(centered_truth)*np.linalg.norm(centered_pred)
            r = float(centered_truth@centered_pred/denominator) if denominator > 1e-12 else np.nan
            pred_top = set(np.argsort(-np.abs(predicted))[:100]) if np.any(np.abs(predicted)>1e-8) else set()
            rows.append(dict(case_id=case_id, drug=case['drug'],dose_uM=case['dose_uM'],method=method,
                n_treated=n_treated,n_control=n_control,
                delta_rmse=float(np.linalg.norm(predicted-observed)/np.sqrt(len(genes))),
                delta_pearson=r,
                top100_delta_rmse=float(np.linalg.norm(predicted[top]-observed[top])/np.sqrt(len(top))),
                top100_direction_accuracy=float(np.mean(np.sign(predicted[top][active])==np.sign(observed[top][active]))),
                top100_abs_delta_overlap=len(pred_top.intersection(top))/len(top),
                response_l2_ratio=float(np.linalg.norm(predicted)/np.linalg.norm(observed))))
    expected=pd.DataFrame(rows)
    actual=pd.read_csv(experiment/'scores_by_well.csv')
    keys=['case_id','method']
    assert set(map(tuple,expected[keys].to_numpy()))==set(map(tuple,actual[keys].to_numpy()))
    metrics=['delta_rmse','delta_pearson','top100_delta_rmse','top100_direction_accuracy',
             'top100_abs_delta_overlap','response_l2_ratio']
    comparisons={}
    def compare(name,a,b,index,columns,atol=1e-6):
        a,b=a.set_index(index).sort_index(),b.set_index(index).sort_index()
        assert a.index.equals(b.index)
        av,bv=a[columns].to_numpy(float),b[columns].to_numpy(float)
        assert np.array_equal(np.isnan(av),np.isnan(bv)),name
        error=float(np.nanmax(np.abs(av-bv)))
        comparisons[name]=error
        assert np.allclose(av,bv,rtol=0,atol=atol,equal_nan=True),(name,error)
    compare('scores_by_well',actual,expected,keys,metrics+['n_treated','n_control'])
    expected_conditions=expected.groupby(['method','drug','dose_uM'],as_index=False)[metrics].mean()
    compare('scores_by_condition',pd.read_csv(experiment/'scores_by_condition.csv'),expected_conditions,
            ['method','drug','dose_uM'],metrics)
    expected_summary=expected_conditions.groupby('method',as_index=False)[metrics].mean()
    compare('scores_summary',pd.read_csv(experiment/'scores_summary.csv'),expected_summary,['method'],metrics)
    expected_occupancy=pd.DataFrame(occupancy_rows)
    compare('occupancy_by_well',pd.read_csv(experiment/'occupancy_by_well.csv'),expected_occupancy,keys,
            [c for c in expected_occupancy if c not in ['case_id','method','drug']])
    changes=[]
    for drug in sorted({c['drug'] for c in cases.values()}):
        ids=[case_id for case_id,c in cases.items() if c['drug']==drug]
        p0=np.mean([true_codes[c][0] for c in ids],axis=0)
        distributions={'observed':np.mean([true_codes[c][1] for c in ids],axis=0)}
        for method in ['code_nearest','code_interpolate','hierarchical_local','hierarchical_joint']:
            distributions[method]=np.mean([q_lookup[(c,method)] for c in ids],axis=0)
        for method,q in distributions.items():
            tv=np.abs(q-p0).sum(1)/2
            changes.append(dict(drug=drug,method=method,dominant_changed_axes=int(np.sum(q.argmax(1)!=p0.argmax(1))),
                                tv_gt_0_2_axes=int(np.sum(tv>.2)),mean_tv=float(tv.mean())))
    compare('axis_changes_by_condition',pd.read_csv(experiment/'axis_changes_by_condition.csv'),pd.DataFrame(changes),
            ['drug','method'],['dominant_changed_axes','tv_gt_0_2_axes','mean_tv'])
    report=dict(status='passed',created_asia_shanghai=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                heldout_expression_read_only_after_prediction_seal_and_existing_scores=True,
                independent_float64_observed_mean_computation=True,n_well_method_rows=len(expected),
                n_cases=len(cases),n_treated=int(expected.drop_duplicates('case_id').n_treated.sum()),
                max_absolute_errors=comparisons,
                independently_recomputed_summary=expected_summary.replace({np.nan:None}).to_dict('records'),
                prediction_seal_sha256=sha(seal_path),implementation_sha256=sha(__file__),
                limitations=['5uM is development data already unblinded before this version.',
                             'Point estimates do not establish independent biological replication or causality.'])
    (experiment/'independent_score_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report


def audit_execution(work, experiment):
    from collections import Counter
    work, experiment = Path(work), Path(experiment)
    manifest=json.loads((experiment/'inputs/manifest.json').read_text())
    selections={'local':Counter(),'joint':Counter()}
    cases=[]
    for entry in manifest['cases']:
        folder=experiment/'predictions'/entry['slug']
        documents={name:json.loads((folder/(name+'.json')).read_text()) for name in ['local','joint']}
        choices={name:{row['axis_id']:row['candidate_id'] for row in doc['selections']}
                 for name,doc in documents.items()}
        for name in choices:
            selections[name].update(choices[name].values())
        arrays={}
        for name in ['local','joint','code_interpolate']:
            with np.load(folder/(name+'.npz'),allow_pickle=False) as z:
                arrays[name]={'q':z['q'].copy(),'delta':z['delta_mean'].copy()}
        qdiff=arrays['joint']['q']-arrays['local']['q']
        row=dict(case_id=entry['case_id'],drug=entry['drug'],
                 local_joint_candidate_id_changed_axes=sum(choices['local'][a]!=choices['joint'][a] for a in range(32)),
                 local_joint_actual_q_changed_axes=int((np.max(np.abs(qdiff),axis=1)>1e-10).sum()),
                 local_joint_q_total_variation_sum=float(np.abs(qdiff).sum()/2),
                 local_joint_gene_delta_l2=float(np.linalg.norm(arrays['joint']['delta']-arrays['local']['delta'])))
        for name in ['local','joint']:
            d=arrays[name]['q']-arrays['code_interpolate']['q']
            row[name+'_non_interpolate_candidate_id_axes']=sum(v!='interpolate' for v in choices[name].values())
            row[name+'_actual_q_changed_axes_from_interpolate']=int((np.max(np.abs(d),axis=1)>1e-10).sum())
            row[name+'_gene_delta_l2_from_interpolate']=float(np.linalg.norm(arrays[name]['delta']-arrays['code_interpolate']['delta']))
        cases.append(row)

    def requests_for(folder):
        return list((folder/'annotation_requests').rglob('*.request.json'))+list((folder/'predictions').glob('*/requests/*/*.request.json'))
    def count_requests(folder):
        request_rows=[]
        for request_path in sorted(requests_for(folder)):
            response_path=request_path.with_name(request_path.name.replace('.request.json','.response.json'))
            status_path=request_path.with_name(request_path.name.replace('.request.json','.status.json'))
            status=json.loads(status_path.read_text()) if status_path.exists() else {}
            response=json.loads(response_path.read_text()) if response_path.exists() else {}
            group=('annotation' if 'annotation_requests' in request_path.parts else
                   'joint' if request_path.parent.name.startswith('joint') else 'local')
            success=status.get('validated',status.get('status')=='completed')
            request_rows.append(dict(path=str(request_path.relative_to(folder)),group=group,
                                     response_saved=response_path.exists(),validated=bool(success),
                                     prompt_tokens=response.get('usage',{}).get('prompt_tokens',0),
                                     completion_tokens=response.get('usage',{}).get('completion_tokens',0),
                                     total_tokens=response.get('usage',{}).get('total_tokens',0),
                                     error=status.get('error',status.get('validation_error'))))
        table=pd.DataFrame(request_rows)
        summary=[]
        if not table.empty:
            for group,t in table.groupby('group'):
                summary.append(dict(group=group,requests=len(t),responses=int(t.response_saved.sum()),
                                    validated=int(t.validated.sum()),failed_or_unknown=int((~t.validated).sum()),
                                    prompt_tokens=int(t.prompt_tokens.sum()),completion_tokens=int(t.completion_tokens.sum()),
                                    total_tokens=int(t.total_tokens.sum())))
        return dict(requests=len(request_rows),responses=sum(row['response_saved'] for row in request_rows),
                    groups=summary,request_rows=request_rows)
    api=count_requests(experiment)
    prior_path=work/'experiment_v3_hierarchy'
    previous=count_requests(prior_path) if prior_path != experiment and prior_path.exists() else None
    case_table=pd.DataFrame(cases)
    case_table.to_csv(experiment/'independent_execution_by_case.csv',index=False)
    report=dict(status='complete',created_asia_shanghai=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                cases=cases,candidate_selection_counts={k:dict(v) for k,v in selections.items()},
                cases_with_joint_candidate_id_changes=int(case_table.local_joint_candidate_id_changed_axes.gt(0).sum()),
                cases_with_joint_actual_q_changes=int(case_table.local_joint_actual_q_changed_axes.gt(0).sum()),
                cases_with_joint_actual_gene_delta_changes=int(case_table.local_joint_gene_delta_l2.gt(1e-10).sum()),
                cases_local_actual_q_differs_from_interpolate=int(case_table.local_actual_q_changed_axes_from_interpolate.gt(0).sum()),
                cases_joint_actual_q_differs_from_interpolate=int(case_table.joint_actual_q_changed_axes_from_interpolate.gt(0).sum()),
                current_api=api,previous_incomplete_annotation_run=previous,
                limitations=['Joint consumes additional API work; local/joint comparison is not budget matched.',
                             'Duplicate candidates mean changed IDs do not necessarily change numeric predictions.',
                             'Token totals count saved response usage; not a billing reconciliation.',
                             'Only annotation_requests and predictions directories count as real requests; offline mock smoke excluded.'])
    (experiment/'independent_execution_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--require-predictions', action='store_true')
    parser.add_argument('--scores', action='store_true')
    args=parser.parse_args()
    if args.scores:
        r=audit_scores(args.work,args.experiment)
        print(json.dumps({k:r[k] for k in ['status','n_cases','n_well_method_rows','max_absolute_errors']},indent=2))
    else:
        r=audit(args.work,args.experiment,args.require_predictions)
        if args.require_predictions:
            execution=audit_execution(args.work,args.experiment)
            print(json.dumps({k:execution[k] for k in ['cases_with_joint_candidate_id_changes',
                'cases_with_joint_actual_q_changes','cases_with_joint_actual_gene_delta_changes']},indent=2))
        print(json.dumps({k:r[k] for k in ['status','n_target_cases','n_predictions_checked','max_absolute_errors']},indent=2))


if __name__=='__main__':
    main()
