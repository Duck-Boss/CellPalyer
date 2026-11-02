#!/usr/bin/env python3
"""Pure mechanism/semantic factorial pilot: prepare, predict+seal, then score.

The 5uM cases have been inspected in earlier work and are development cases.
Prediction dependencies never include treated outcomes or response retrieval.
"""
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
import shutil

import numpy as np
import pandas as pd

from agent import ChatCompletionsClient
from pure_mechanism_data import prepare_inputs, load_case_bundle, literal_keyword_decisions
from pure_mechanism_agent import run_case, evaluate_decisions

ROOT=Path(__file__).resolve().parent
ARMS=['full','no_mechanism','no_semantics','neither']
ALPHAS=[.1,.3,.6]
SOURCES=['pure_mechanism_experiment.py','pure_mechanism_data.py','pure_mechanism_agent.py',
         'mechanism_cards_v4.json','agent.py','tools.py','annotate_cards.py']

def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)
def source_hashes():return {name:sha(ROOT/name) for name in SOURCES}

def freeze_protocol(output,client,workers):
    manifest=read(output/'inputs/manifest.json')
    protocol={'version':'pure-mechanism-factorial-v4','created':now(),
        'evaluation_role':'Previously unblinded 5uM development experiment; not a fresh test',
        'case_ids':[c['case_id'] for c in manifest['cases']],
        'arms':ARMS,'primary_alpha':.3,'sensitivity_alphas':[.1,.6],
        'alpha_policy':'Prespecified interpolation intensity, not confidence or a calibrated dose response; no outcome-based selection',
        'local_batches':4,'axes_per_batch':8,'mechanism_calls':1,'joint_calls':1,
        'nominal_calls_per_case_arm':6,'nominal_total_calls':len(manifest['cases'])*len(ARMS)*6,
        'max_repair_retries_per_stage':2,'model':client.model,'endpoint':client.url,'workers':workers,
        'condition_response_retrieval':False,'target_expression_visible':False,
        'source_hashes':source_hashes(),'input_manifest_sha256':sha(output/'inputs/manifest.json'),
        'numerical_baselines':['no_edit','literal_keyword'],
        'metrics':{'primary':'Well-wise RMSE of predicted vs true mean expression changes, averaged within drug then across drugs',
                   'secondary':['Pearson','cosine','top100 sign agreement','response L2 ratio','occupancy TV error'],
                   'units':'log1p(CP10K) over 2000 train-selected genes',
                   'top100':'largest absolute observed shifts; not statistical DEGs'},
        'limitations':['Frozen codec and labels were learned from training data that include other drug doses; pure here means no condition-specific response access at inference.',
          'Pretrained LLM knowledge is not restricted to the supplied citations; absence of memorized response knowledge cannot be established.',
          'Semantic hiding changes information content and prompt length, not number of allowed calls.',
          'No semantics also removes gene-grounded hypothesis feedback, to avoid recreating the identity map.',
          'Local vs joint differs by one successful call; not a matched-budget causal feedback comparison.',
          'Marginal occupancies specify an additive population mean, not a unique joint single-cell distribution.',
          'Only four drugs, one dose/time/cell line; author replicates are not verified independent biological replicates.']}
    path=output/'protocol.json'
    if path.exists():
        old=read(path)
        if {k:v for k,v in old.items() if k!='created'}!={k:v for k,v in protocol.items() if k!='created'}:
            raise ValueError('Frozen protocol changed; use a new output')
        protocol=old
    else:save(path,protocol)
    snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name,digest in protocol['source_hashes'].items():
        path=snapshot/name
        if path.exists():
            if sha(path)!=digest:raise ValueError('Source snapshot changed: '+name)
        else:shutil.copyfile(ROOT/name,path)
    return protocol

def baseline(bundle,folder):
    folder.mkdir(parents=True,exist_ok=True)
    decisions=literal_keyword_decisions(bundle)
    values=[evaluate_decisions(bundle,decisions,alpha=a,arm='full') for a in ALPHAS]
    np.savez_compressed(folder/'literal_keyword.npz',p0=bundle['p0'],q=values[1]['q'],
        delta_mean=values[1]['delta_mean'],genes=bundle['genes'],alphas=ALPHAS,
        q_by_alpha=np.stack([v['q'] for v in values]),delta_by_alpha=np.stack([v['delta_mean'] for v in values]))
    save(folder/'literal_keyword.json',{'decisions':decisions,'main_alpha':.3,
        'description':'Prespecified conservative normalized literal program-term and signed wording match; no LLM or response data'})

def predict(work,output,client,workers):
    protocol=freeze_protocol(output,client,workers)
    manifest=read(output/'inputs/manifest.json')
    save(output/'RUN_STATUS.json',{'status':'running','started':now(),'nominal_requests':protocol['nominal_total_calls']})
    for case in manifest['cases']:
        bundle=load_case_bundle(work,case['npz'],case['context'])
        baseline(bundle,output/'baselines'/case['slug'])
    def one(case,arm):
        bundle=load_case_bundle(work,case['npz'],case['context'])
        folder=output/'predictions'/arm/case['slug']
        result=run_case(bundle,bundle['cards'],client,folder,arm=arm,max_retries=2)
        print(json.dumps({'time':now(),'stage':'case_arm_complete','case_id':case['case_id'],'arm':arm}),flush=True)
        return result
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures=[pool.submit(one,c,a) for c in manifest['cases'] for a in ARMS]
            for future in as_completed(futures):
                try:future.result()
                except Exception:
                    for pending in futures:pending.cancel()
                    raise
        if source_hashes()!=protocol['source_hashes']:raise ValueError('Source changed during prediction')
        for c in manifest['cases']:
            for a in ARMS:
                folder=output/'predictions'/a/c['slug']
                for name in ['local.npz','joint.npz','local.json','joint.json','COMPLETE.json']:
                    if not (folder/name).is_file():raise ValueError('Missing required output: '+str(folder/name))
        files={str(p.relative_to(output)):sha(p) for top in ['inputs','predictions','baselines','source_snapshot']
            for p in (output/top).rglob('*') if p.is_file()}
        files['protocol.json']=sha(output/'protocol.json')
        seal={'status':'all_predictions_complete','sealed':now(),'n_cases':len(manifest['cases']),
            'n_case_arms':len(manifest['cases'])*len(ARMS),'source_hashes':source_hashes(),'files':files,
            'score_exists_at_seal':(output/'scores_summary.csv').exists()}
        if (output/'prediction_seal.json').exists():
            old=read(output/'prediction_seal.json')
            if old['files']!=files:raise ValueError('Previously sealed predictions changed')
            seal=old
        else:save(output/'prediction_seal.json',seal)
        save(output/'RUN_STATUS.json',{'status':'predictions_sealed','time':now(),'n_case_arms':seal['n_case_arms']})
        return seal
    except Exception as exc:
        save(output/'RUN_STATUS.json',{'status':'failed','time':now(),'error_type':type(exc).__name__,
             'message':'Prediction incomplete; inspect retained per-stage attempts. No scores produced by this command.'})
        raise

def verify_seal(output):
    seal=read(output/'prediction_seal.json')
    if seal['status']!='all_predictions_complete' or seal['score_exists_at_seal']:raise ValueError('Invalid pre-score seal')
    for name,digest in seal['files'].items():
        if sha(output/name)!=digest:raise ValueError('Sealed file changed: '+name)
    if source_hashes()!=seal['source_hashes']:raise ValueError('Source changed since seal')
    return seal

def metrics(pred,true):
    top=np.argsort(-np.abs(true))[:100]
    nonzero=np.linalg.norm(pred)>1e-12
    centered=np.std(pred)>1e-12 and np.std(true)>1e-12
    return {'delta_rmse':float(np.sqrt(np.mean((pred-true)**2))),
        'delta_pearson':float(np.corrcoef(pred,true)[0,1]) if centered else np.nan,
        'delta_cosine':float(np.dot(pred,true)/(np.linalg.norm(pred)*np.linalg.norm(true))) if nonzero else np.nan,
        'top100_delta_rmse':float(np.sqrt(np.mean((pred[top]-true[top])**2))),
        'top100_direction_accuracy':float(np.mean(np.sign(pred[top])==np.sign(true[top]))),
        'response_l2_ratio':float(np.linalg.norm(pred)/np.linalg.norm(true))}

def score(work,output):
    seal=verify_seal(output);manifest=read(output/'inputs/manifest.json')
    # Target access starts ONLY after the complete four-arm prediction seal.
    details={c['case_id']:c for c in read(work/'prepared/cases.json')['cases'] if c['status']=='ready'}
    if set(details)!={c['case_id'] for c in manifest['cases']}:raise ValueError('Case population changed')
    evaluation=work/'prepared/evaluation.npz';code_path=work/'codec_changes/case_arrays.npz'
    diagnostic=read(work/'codec_changes/manifest.json')
    for name in ['lexa/best.pt','lexa/exact_code_effects.npz','prepared/evaluation.npz']:
        path=(work/name).resolve()
        if diagnostic['inputs_sha256'].get(str(path))!=sha(path):raise ValueError('Diagnostic provenance mismatch: '+name)
    code_hash=diagnostic['outputs_sha256'].get('case_arrays.npz',diagnostic['outputs_sha256'].get(str(code_path.resolve())))
    if code_hash!=sha(code_path):raise ValueError('Observed occupancy file changed')
    with np.load(evaluation,allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with np.load(code_path,allow_pickle=False) as z:
        truth={str(c):(z['pcontrol'][i],z['ptreated'][i],z['decoded_delta'][i]) for i,c in enumerate(z['case_ids'])}
    genes=data['genes'].astype(str);rows=[];occ=[];decisions=[];raw=[]
    for case in manifest['cases']:
        cid=case['case_id'];detail=details[cid]
        tx=data['x_test'][data['test_case_ids'].astype(str)==cid].astype(float)
        ctrl=data['control_x'][np.isin(data['control_cell_ids'].astype(str),detail['control_cell_ids'])].astype(float)
        if (len(tx),len(ctrl))!=(detail['n_treated'],detail['n_control']):raise ValueError('Cell count mismatch')
        true=tx.mean(0)-ctrl.mean(0);p0,pt,codec_delta=truth[cid]
        with np.load(case['npz'],allow_pickle=False) as z:
            if not np.array_equal(genes,z['genes'].astype(str)) or not np.allclose(p0,z['p0']):raise ValueError('Vocabulary/baseline mismatch')
            decoded_ctrl=z['control_decoded_mean'].copy()
        preds=[('no_edit',np.zeros_like(true),p0),('answer_codec_diagnostic',codec_delta,pt)]
        for stem,folder in [('literal_keyword',output/'baselines'/case['slug'])]+[(a+'_'+s,output/'predictions'/a/case['slug']) for a in ARMS for s in ['local','joint']]:
            filename='literal_keyword' if stem=='literal_keyword' else stem.rsplit('_',1)[1]
            with np.load(folder/(filename+'.npz'),allow_pickle=False) as z:
                if not np.array_equal(genes,z['genes'].astype(str)) or not np.allclose(z['alphas'],ALPHAS):raise ValueError('Prediction metadata mismatch')
                for i,alpha in enumerate(ALPHAS):preds.append((stem+f'_a{alpha:.1f}',z['delta_by_alpha'][i].copy(),z['q_by_alpha'][i].copy()))
            d=read(folder/(filename+'.json'))['decisions']
            decisions.append({'case_id':cid,'drug':case['drug'],'method':stem,
                **{k+'_axes':sum(x['decision']==k for x in d) for k in ['shift','keep','unknown']}})
        for method,pred,q in preds:
            if not np.isfinite(pred).all() or q.min() < -1e-8 or not np.allclose(q.sum(1),1):raise ValueError('Invalid prediction')
            base={'case_id':cid,'drug':case['drug'],'method':method,'n_treated':len(tx),'n_control':len(ctrl)}
            rows.append({**base,**metrics(pred,true)})
            occ.append({**base,'occupancy_tv_error_mean':float((.5*np.abs(q-pt).sum(1)).mean()),
                'dominant_code_accuracy':float(np.mean(q.argmax(1)==pt.argmax(1))),
                'predicted_changed_axes':int(np.sum(q.argmax(1)!=p0.argmax(1))),
                'observed_changed_axes':int(np.sum(pt.argmax(1)!=p0.argmax(1))),
                'predicted_mean_axis_tv':float((.5*np.abs(q-p0).sum(1)).mean())})
            raw.append({**base,'raw_decoded_absolute_rmse':float(np.sqrt(np.mean((decoded_ctrl+pred-tx.mean(0))**2))),
                'definition':'decoded control mean + predicted delta vs true treated mean; different anchor from delta RMSE'})
    table=pd.DataFrame(rows);columns=list(metrics(np.ones(2),np.array([1.,2.])).keys())
    table.to_csv(output/'scores_by_well.csv',index=False)
    condition=table.groupby(['method','drug'],as_index=False)[columns].mean()
    condition.to_csv(output/'scores_by_condition.csv',index=False)
    summary=condition.groupby('method',as_index=False)[columns].mean()
    summary.to_csv(output/'scores_summary.csv',index=False)
    pd.DataFrame(occ).to_csv(output/'occupancy_by_well.csv',index=False)
    pd.DataFrame(decisions).to_csv(output/'decisions_by_well.csv',index=False)
    pd.DataFrame(raw).to_csv(output/'raw_decoded_absolute_by_well.csv',index=False)
    counts={k:sum(d['n_treated'] for d in details.values() if d['drug']==k) for k in sorted({d['drug'] for d in details.values()})}
    save(output/'score_manifest.json',{'status':'complete','time':now(),'prediction_seal_sha256':sha(output/'prediction_seal.json'),
        'evaluation_sha256':sha(evaluation),'observed_codes_sha256':sha(code_path),'n_wells':len(details),
        'n_treated':sum(counts.values()),'n_unique_controls':len(set(x for d in details.values() for x in d['control_cell_ids'])),
        'treated_by_drug':counts,'n_genes':len(genes),'aggregation':'equal wells within drug, equal drugs',
        'development_only':True,'all_sensitivity_results_retained':True,
        'outputs_sha256':{p.name:sha(p) for p in output.glob('*.csv')}})
    return summary

def report(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=pd.read_csv(output/'scores_summary.csv').set_index('method')
    main=['no_edit','literal_keyword_a0.3']+[a+'_joint_a0.3' for a in ARMS]+['answer_codec_diagnostic']
    frame=summary.loc[main]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    labels=['No edit','Literal rule','Full','No mechanism','No semantics','Neither','Answer codec\n(diagnostic)']
    for ax,column,title in zip(axes,['delta_rmse','delta_pearson','response_l2_ratio'],['Delta RMSE (lower)','Delta Pearson (higher)','Response norm / truth']):
        ax.barh(labels,frame[column],color=['#888888','#a2a2a2','#277da1','#90be6d','#f9c74f','#f9844a','#c8c8c8'])
        ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.2)
    fig.suptitle('Pure mechanism/code text pilot — alpha 0.3; 16 development wells / 4 drugs')
    fig.tight_layout();fig.savefig(output/'metrics_comparison.png',dpi=150);plt.close(fig)
    protocol=read(output/'protocol.json');samples=read(output/'score_manifest.json')
    text='LEXA pure mechanism v4\n\n'+frame.to_string(float_format=lambda x:f'{x:.6f}')+'\n\n'
    text+='16 previously unblinded development wells; 786 treated cells and 951 unique controls, four drugs at 5uM/24h in A549. No condition-response retrieval at prediction. Alpha .3 prespecified; .1/.6 sensitivity only.\n'
    text+='Answer codec uses actual treated answers and is a diagnostic, not a predictor or strict lower bound. Literal rule is intentionally limited exact text matching, not a comprehensive pathway predictor.\n'
    text+='The fixed dictionary learned from training data containing other doses. This is a pure mechanism INPUT experiment, not a demonstration of an entirely perturbation-naive representation.\n'
    text+='Four unique drug conditions do not support broad generalization claims; 786 cells are not 786 independent perturbation experiments.\n'
    (output/'RESULTS.txt').write_text(text)
    page='<!doctype html><meta charset="utf-8"><title>LEXA pure mechanism v4</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}pre{white-space:pre-wrap}img{max-width:100%}</style>'
    page+='<h1>LEXA 纯机制文字预测实验 v4</h1><p>主参数 alpha=0.3；同一批 16 个已揭盲开发孔，四药各四孔。预测不读取药物响应检索或真实处理表达。</p>'
    page+='<img src="metrics_comparison.png">'+frame.to_html(float_format=lambda x:f'{x:.6f}')
    page+='<h2>全部预设强度与 local/joint 结果</h2>'+summary.to_html(float_format=lambda x:f'{x:.6f}')
    page+='<h2>定义与限制</h2><pre>'+html.escape(text)+'\n'+html.escape(json.dumps(protocol['limitations'],ensure_ascii=False,indent=2))+'</pre>'
    page+='<p><a href="scores_by_condition.csv">逐药指标</a> · <a href="scores_by_well.csv">逐孔指标</a> · <a href="protocol.json">冻结协议</a> · <a href="prediction_seal.json">预测封存</a></p>'
    (output/'RESULTS.html').write_text(page)
    return {'report':str(output/'RESULTS.html'),'n_wells':samples['n_wells']}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','predict','score','report'])
    p.add_argument('--work',type=Path,default=ROOT/'runs/pilot_seen_dose');p.add_argument('--output',type=Path)
    p.add_argument('--mechanisms',type=Path,default=ROOT/'mechanism_cards_v4.json')
    p.add_argument('--api-key-file',type=Path);p.add_argument('--model',default='deepseek-flash')
    p.add_argument('--base-url',default='https://api.deepseek.com');p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();work=args.work.resolve();out=(args.output or work/'experiment_v4_pure_mechanism').resolve()
    if args.action=='prepare':
        m=prepare_inputs(work,out/'inputs',args.mechanisms);print(json.dumps({'cases':len(m['cases']),'output':str(out)}))
    elif args.action=='predict':
        key=args.api_key_file.read_text().strip() if args.api_key_file else os.environ.get('DEEPSEEK_API_KEY','')
        client=ChatCompletionsClient(args.model,key,args.base_url)
        result=predict(work,out,client,args.workers);print(json.dumps({'sealed':result['sealed'],'n_case_arms':result['n_case_arms']}))
    elif args.action=='score':print(score(work,out).to_string(index=False))
    else:print(json.dumps(report(out)))

if __name__=='__main__':main()
