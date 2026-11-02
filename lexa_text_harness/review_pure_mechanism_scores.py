#!/usr/bin/env python3
"""Independent post-seal score audit. Never imported by the prediction runner.

Target files are opened only after verifying a complete prediction seal and a
completed score manifest. All metrics are recomputed here without importing the
prediction or scoring implementation. --self-test reads no experiment files.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ARMS=('full','no_mechanism','no_semantics','neither')
ALPHAS=(0.1,0.3,0.6)
METRICS=('delta_rmse','delta_pearson','delta_cosine','top100_delta_rmse',
         'top100_direction_accuracy','response_l2_ratio')
OCCUPANCY=('occupancy_tv_error_mean','dominant_code_accuracy','predicted_changed_axes',
           'observed_changed_axes','predicted_mean_axis_tv')
CSV_NAMES=('scores_by_well.csv','scores_by_condition.csv','scores_summary.csv',
           'occupancy_by_well.csv','decisions_by_well.csv','raw_decoded_absolute_by_well.csv')


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition,message):
    if not condition:raise AssertionError(message)


def close(actual,expected,label,atol=2e-10,rtol=2e-9):
    np.testing.assert_allclose(actual,expected,atol=atol,rtol=rtol,equal_nan=True,err_msg=label)


def fresh_metrics(predicted,observed):
    p=np.asarray(predicted,dtype=np.float64);y=np.asarray(observed,dtype=np.float64)
    require(p.shape==y.shape and p.ndim==1,'Metric vectors must align')
    require(np.isfinite(p).all() and np.isfinite(y).all(),'Metric vectors must be finite')
    residual=p-y;pn=float(np.sqrt(np.dot(p,p)));yn=float(np.sqrt(np.dot(y,y)))
    require(yn>1e-12,'Observed response is numerically zero: cosine/ratio undefined')
    pc=p-p.mean();yc=y-y.mean();pcn=np.sqrt(np.dot(pc,pc));ycn=np.sqrt(np.dot(yc,yc))
    top=np.argsort(-np.abs(y))[:100]
    return dict(zip(METRICS,[
        float(np.sqrt(np.dot(residual,residual)/len(residual))),
        float(np.dot(pc,yc)/(pcn*ycn)) if np.sqrt(np.mean(pc**2))>1e-12 and np.sqrt(np.mean(yc**2))>1e-12 else np.nan,
        float(np.dot(p,y)/(pn*yn)) if pn>1e-12 else np.nan,
        float(np.sqrt(np.dot(residual[top],residual[top])/len(top))),
        float(np.count_nonzero(np.sign(p[top])==np.sign(y[top]))/len(top)),
        pn/yn]))


def from_decisions(p0,decisions,alpha):
    """Reconstruct every categorical occupancy independently of runner helpers."""
    p0=np.asarray(p0,dtype=float);q=p0.copy()
    require(len(decisions)==p0.shape[0],'Incomplete decision axes')
    seen=set()
    for decision in decisions:
        axis=decision['axis_id'];kind=decision['decision']
        require(type(axis)is int and 0<=axis<len(p0) and axis not in seen,'Invalid/duplicate decision axis')
        seen.add(axis);require(kind in ('shift','keep','unknown'),'Invalid decision action')
        targets=decision['targets']
        if kind!='shift':
            require(targets==[],'Nonshift decision has target states');continue
        require(1<=len(targets)<=3,'Shift target cardinality')
        weights=np.zeros(p0.shape[1]);seen_states=set()
        for state,weight in targets:
            require(type(state)is int and 0<=state<p0.shape[1] and state not in seen_states,'Invalid target state')
            require(np.isfinite(weight) and weight>0,'Invalid state weight')
            weights[state]=weight;seen_states.add(state)
        weights/=weights.sum()
        q[axis]=p0[axis]*(1-alpha)+weights*alpha
    require(np.isfinite(q).all() and q.min()>=0,'Invalid reconstructed probability')
    close(q.sum(1),np.ones(len(q)),'Reconstructed probability mass')
    return q


def _table_check(path,expected,keys,values):
    actual=pd.read_csv(path)
    require(not actual.duplicated(keys).any(),f'Duplicate keys: {path.name}')
    require(len(actual)==len(expected),f'Wrong row count: {path.name}')
    a=actual.set_index(keys).sort_index();e=pd.DataFrame(expected).set_index(keys).sort_index()
    require(a.index.equals(e.index),f'Population differs: {path.name}')
    for col in values:close(a[col].to_numpy(),e[col].to_numpy(),path.name+': '+col)
    return {c:float(np.nanmax(np.abs(a[c].to_numpy()-e[c].to_numpy())))
            if np.isfinite(a[c].to_numpy()).any() else 0. for c in values}


def audit(work,output):
    work=Path(work).resolve();output=Path(output).resolve()
    # Hard gate before reading any outcome, observed treated codes, or score CSV.
    require((output/'prediction_seal.json').is_file(),'Prediction seal is absent; targets remain unopened')
    require((output/'score_manifest.json').is_file(),'Scoring is incomplete; targets remain unopened')
    seal=read(output/'prediction_seal.json');score_manifest=read(output/'score_manifest.json')
    require(seal['status']=='all_predictions_complete' and seal['score_exists_at_seal'] is False,'Invalid prediction seal')
    require(score_manifest['status']=='complete','Incomplete score manifest')
    require(score_manifest['prediction_seal_sha256']==sha(output/'prediction_seal.json'),'Score manifest references a different seal')
    require(seal['n_cases']==16 and seal['n_case_arms']==64,'Unexpected case/arm design')
    for name,digest in seal['files'].items():
        path=(output/name).resolve();require(path.is_relative_to(output),'Sealed path escapes output')
        require(sha(path)==digest,'Sealed artifact hash changed: '+name)
    actual_sealed={str(p.relative_to(output)) for folder in ('inputs','predictions','baselines','source_snapshot')
                   for p in (output/folder).rglob('*') if p.is_file()}|{'protocol.json'}
    require(actual_sealed==set(seal['files']),'Sealed artifact inventory differs')
    source_root=Path(__file__).resolve().parent
    for name,digest in seal['source_hashes'].items():
        require(sha(source_root/name)==digest,'Current source changed: '+name)
        require(sha(output/'source_snapshot'/name)==digest,'Snapshot source changed: '+name)
    protocol=read(output/'protocol.json');manifest=read(output/'inputs/manifest.json')
    require(protocol['source_hashes']==seal['source_hashes'],'Protocol/source seal mismatch')
    require(protocol['input_manifest_sha256']==sha(output/'inputs/manifest.json'),'Protocol input manifest mismatch')
    require(tuple(protocol['arms'])==ARMS and protocol['primary_alpha']==.3,'Protocol arm/alpha mismatch')
    require(set(protocol['sensitivity_alphas'])=={.1,.6},'Sensitivity alpha mismatch')
    for name in CSV_NAMES:
        require(score_manifest['outputs_sha256'].get(name)==sha(output/name),'Scored CSV hash mismatch: '+name)

    # Outcome access is now permitted: the complete four-arm predictions are sealed.
    evaluation_path=work/'prepared/evaluation.npz';code_path=work/'codec_changes/case_arrays.npz'
    require(score_manifest['evaluation_sha256']==sha(evaluation_path),'Evaluation source hash mismatch')
    require(score_manifest['observed_codes_sha256']==sha(code_path),'Observed occupancy hash mismatch')
    diagnostic=read(work/'codec_changes/manifest.json')
    for name in ('lexa/best.pt','lexa/exact_code_effects.npz','prepared/evaluation.npz'):
        path=(work/name).resolve()
        require(diagnostic['inputs_sha256'].get(str(path))==sha(path),'Codec diagnostic source hash mismatch: '+name)
    require(diagnostic['outputs_sha256'].get('case_arrays.npz',diagnostic['outputs_sha256'].get(str(code_path)))==sha(code_path),
            'Codec diagnostic array hash mismatch')
    with np.load(evaluation_path,allow_pickle=False) as z:ev={k:z[k].copy() for k in z.files}
    with np.load(code_path,allow_pickle=False) as z:observed_codes={k:z[k].copy() for k in z.files}
    with np.load(work/'lexa/exact_code_effects.npz',allow_pickle=False) as z:
        effects=z['contributions'].astype(float);genes=z['genes'].astype(str)
    require(np.array_equal(genes,ev['genes'].astype(str)),'Evaluation/effects genes differ')
    require(effects.shape==(32,16,2000),'Expected frozen 32 x 16 x 2000 dictionary')
    case_details={c['case_id']:c for c in read(work/'prepared/cases.json')['cases'] if c['status']=='ready' and float(c['dose_uM'])==5.}
    require(len(manifest['cases'])==16 and len(case_details)==16,'Expected sixteen wells')
    require(set(case_details)=={c['case_id'] for c in manifest['cases']}==set(protocol['case_ids']),'Case selection changed')
    observed_lookup={str(cid):i for i,cid in enumerate(observed_codes['case_ids'])}
    require(len(observed_lookup)==len(observed_codes['case_ids']),'Duplicate observed-code cases')
    eval_control_ids=ev['control_cell_ids'].astype(str);eval_test_cases=ev['test_case_ids'].astype(str)
    require(len(set(eval_control_ids))==len(eval_control_ids),'Duplicate evaluation control IDs')
    all_controls=set();all_treated=set();drug_counts={};well_rows=[];occ_rows=[];raw_rows=[];decision_rows=[]
    expected_methods={'no_edit','answer_codec_diagnostic'}|{s+f'_a{a:.1f}' for s in
        ['literal_keyword']+[arm+'_'+stage for arm in ARMS for stage in ('local','joint')] for a in ALPHAS}
    decoder_residual=0.;prediction_residual=0.;control_residual=0.
    for case in manifest['cases']:
        cid=case['case_id'];detail=case_details[cid];ix=observed_lookup[cid]
        test_mask=eval_test_cases==cid;control_mask=np.isin(eval_control_ids,detail['control_cell_ids'])
        treated=ev['x_test'][test_mask].astype(float);control=ev['control_x'][control_mask].astype(float)
        require(len(treated)==detail['n_treated'] and len(control)==detail['n_control'],'Case sample count mismatch: '+cid)
        require(len(control)==case['n_control'],'Manifest control count mismatch')
        require(set(eval_control_ids[control_mask])==set(detail['control_cell_ids']),'Case control identity mismatch')
        treated_ids=set(ev['test_cell_ids'][test_mask].astype(str))
        require(len(treated_ids)==len(treated) and not treated_ids&all_treated,'Duplicate treated assignment')
        if 'treated_cell_ids' in detail:require(treated_ids==set(detail['treated_cell_ids']),'Treated cell identity mismatch')
        all_treated.update(treated_ids);all_controls.update(detail['control_cell_ids'])
        drug_counts[case['drug']]=drug_counts.get(case['drug'],0)+len(treated)
        target=treated.mean(0)-control.mean(0)
        close(observed_codes['observed_delta'][ix],target,'Cached observed expression shift')
        p0=observed_codes['pcontrol'][ix].astype(float);pt=observed_codes['ptreated'][ix].astype(float)
        for matrix in (p0,pt):
            require(np.isfinite(matrix).all() and matrix.min()>=0,'Invalid observed occupancy')
            close(matrix.sum(1),np.ones(32),'Observed occupancy mass')
        with np.load(case['npz'],allow_pickle=False) as z:
            close(z['p0'],p0,'Conditioning/diagnostic p0')
            close(z['control_mean'],control.mean(0),'Conditioning observed control mean')
            control_decoded=z['control_decoded_mean'].astype(float)
            baseline_codes=z['control_codes'].astype(int)
            require(np.array_equal(genes,z['genes'].astype(str)),'Case gene order mismatch')
        p0_recomputed=np.array([(baseline_codes[:,a,None]==np.arange(16)).mean(0) for a in range(32)])
        close(p0,p0_recomputed,'Independent control code occupancy')
        analytic_control=sum(p0[a]@effects[a] for a in range(32))
        control_residual=max(control_residual,float(np.max(np.abs(analytic_control-control_decoded))))
        close(analytic_control,control_decoded,'Control additive decoder consistency',atol=2e-6,rtol=1e-5)
        answer_delta=observed_codes['decoded_delta'][ix].astype(float)
        analytic_answer=sum((pt[a]-p0[a])@effects[a] for a in range(32))
        decoder_residual=max(decoder_residual,float(np.max(np.abs(answer_delta-analytic_answer))))
        close(answer_delta,analytic_answer,'Answer codec additive consistency',atol=2e-6,rtol=1e-5)
        per_case=[('no_edit',np.zeros_like(target),p0),('answer_codec_diagnostic',answer_delta,pt)]
        stems=[('literal_keyword',output/'baselines'/case['slug'],'literal_keyword')]+[
            (arm+'_'+stage,output/'predictions'/arm/case['slug'],stage) for arm in ARMS for stage in ('local','joint')]
        for stem,folder,filename in stems:
            decisions=read(folder/(filename+'.json'))['decisions']
            decision_rows.append({'case_id':cid,'drug':case['drug'],'method':stem,
                **{s+'_axes':sum(d['decision']==s for d in decisions) for s in ('shift','keep','unknown')}})
            with np.load(folder/(filename+'.npz'),allow_pickle=False) as z:
                require(np.array_equal(z['genes'].astype(str),genes),'Prediction gene order mismatch')
                close(z['alphas'],ALPHAS,'Saved alpha order');close(z['p0'],p0,'Prediction baseline occupancy')
                require(z['q_by_alpha'].shape==(3,32,16) and z['delta_by_alpha'].shape==(3,2000),'Prediction array dimensions')
                for j,alpha in enumerate(ALPHAS):
                    q=from_decisions(p0,decisions,alpha)
                    delta=sum((q[a]-p0[a])@effects[a] for a in range(32))
                    close(z['q_by_alpha'][j],q,'Decision-derived q: '+stem)
                    close(z['delta_by_alpha'][j],delta,'Decision-derived expression delta: '+stem)
                    prediction_residual=max(prediction_residual,float(np.max(np.abs(z['delta_by_alpha'][j]-delta))))
                    per_case.append((stem+f'_a{alpha:.1f}',delta,q))
                close(z['q'],z['q_by_alpha'][1],'Main alpha q');close(z['delta_mean'],z['delta_by_alpha'][1],'Main alpha delta')
        require(len(per_case)==29 and {m for m,_,_ in per_case}==expected_methods,'Method population mismatch')
        for method,delta,q in per_case:
            base={'case_id':cid,'drug':case['drug'],'method':method,'n_treated':len(treated),'n_control':len(control)}
            well_rows.append({**base,**fresh_metrics(delta,target)})
            occ_rows.append({**base,'occupancy_tv_error_mean':float(np.abs(q-pt).sum()/64),
                'dominant_code_accuracy':float(np.count_nonzero(q.argmax(1)==pt.argmax(1))/32),
                'predicted_changed_axes':int(np.count_nonzero(q.argmax(1)!=p0.argmax(1))),
                'observed_changed_axes':int(np.count_nonzero(pt.argmax(1)!=p0.argmax(1))),
                'predicted_mean_axis_tv':float(np.abs(q-p0).sum()/64)})
            residual=control_decoded+delta-treated.mean(0)
            raw_rows.append({**base,'raw_decoded_absolute_rmse':float(np.sqrt(np.dot(residual,residual)/2000))})
    condition_rows=[];summary_rows=[];metric_support=[]
    for method in sorted(expected_methods):
        conditions=[]
        for drug in sorted(drug_counts):
            wells=[r for r in well_rows if r['method']==method and r['drug']==drug]
            require(len(wells)==4,'Expected four wells per drug')
            means={k:float(np.mean([r[k] for r in wells if np.isfinite(r[k])]))
                   if any(np.isfinite(r[k]) for r in wells) else np.nan for k in METRICS}
            conditions.append(means);condition_rows.append({'method':method,'drug':drug,**means})
        means={k:float(np.mean([r[k] for r in conditions if np.isfinite(r[k])]))
               if any(np.isfinite(r[k]) for r in conditions) else np.nan for k in METRICS}
        summary_rows.append({'method':method,**means})
        for metric in ('delta_pearson','delta_cosine'):
            defined={drug:sum(np.isfinite(r[metric]) for r in well_rows if r['method']==method and r['drug']==drug)
                     for drug in sorted(drug_counts)}
            metric_support.append({'method':method,'metric':metric,
                'defined_wells':int(sum(defined.values())),'defined_drugs':int(sum(n>0 for n in defined.values())),
                'defined_wells_by_drug':{drug:int(n) for drug,n in defined.items()},
                'aggregation_denominator':'Finite wells within each drug, then drugs with a defined mean'})
    discrepancies={}
    discrepancies['well']=_table_check(output/'scores_by_well.csv',well_rows,['case_id','drug','method'],[*METRICS,'n_treated','n_control'])
    discrepancies['condition']=_table_check(output/'scores_by_condition.csv',condition_rows,['method','drug'],METRICS)
    discrepancies['summary']=_table_check(output/'scores_summary.csv',summary_rows,['method'],METRICS)
    discrepancies['occupancy']=_table_check(output/'occupancy_by_well.csv',occ_rows,['case_id','drug','method'],[*OCCUPANCY,'n_treated','n_control'])
    discrepancies['raw_absolute']=_table_check(output/'raw_decoded_absolute_by_well.csv',raw_rows,['case_id','drug','method'],['raw_decoded_absolute_rmse','n_treated','n_control'])
    discrepancies['decisions']=_table_check(output/'decisions_by_well.csv',decision_rows,['case_id','drug','method'],['shift_axes','keep_axes','unknown_axes'])
    require(score_manifest['n_wells']==16 and score_manifest['n_genes']==2000,'Score manifest dimension mismatch')
    require(score_manifest['n_treated']==len(all_treated)==sum(drug_counts.values()),'Treated total mismatch')
    require(score_manifest['n_unique_controls']==len(all_controls),'Unique control count mismatch')
    require(score_manifest['treated_by_drug']==drug_counts,'Per-drug treated count mismatch')
    require(score_manifest['aggregation']=='equal wells within drug, equal drugs','Aggregation declaration mismatch')
    require(score_manifest['development_only'] is True and score_manifest['all_sensitivity_results_retained'] is True,'Evaluation limitation flags missing')
    return {'status':'pass','created_asia_shanghai':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'reviewer_script_sha256':sha(__file__),'prediction_seal_sha256':sha(output/'prediction_seal.json'),
        'score_manifest_sha256':sha(output/'score_manifest.json'),'sealed_artifact_count':len(seal['files']),
        'n_wells':16,'n_methods':29,'n_metric_rows':len(well_rows),'n_condition_rows':len(condition_rows),
        'n_decision_rows':len(decision_rows),'n_treated':len(all_treated),'n_unique_controls':len(all_controls),
        'treated_by_drug':drug_counts,'n_genes':len(genes),'metrics_max_absolute_discrepancy':discrepancies,
        'correlation_metric_support':metric_support,
        'independently_recomputed_main_results':[{k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()}
            for row in summary_rows if row['method'] in {'no_edit','answer_codec_diagnostic','literal_keyword_a0.3'}|
            {arm+'_joint_a0.3' for arm in ARMS}],
        'prediction_decoder_max_residual':prediction_residual,'answer_codec_additive_max_residual':decoder_residual,
        'control_codec_additive_max_residual':control_residual,
        'definition_notes':['Independent metric/aggregation and decision-to-occupancy implementations; no imports from scoring/agent code.',
            'Target access starts only after all predictions and source/input artifacts pass sealed-hash validation.',
            'Answer codec diagnostic uses saved float32 decoder means, checked against exact additive code effects within 2e-6/1e-5 tolerance.',
            'No-edit Pearson and cosine are undefined and must remain NaN, rather than assigned an artificial score.',
            'Correlation aggregation skips undefined values: compare support counts before comparing method means. Literal keyword has four defined Nutlin wells and no defined wells for the other three drugs.',
            'Raw absolute RMSE anchors at decoded control; delta RMSE anchors at observed control.',
            'Top100 denotes observed largest absolute mean shifts, not statistical differential-expression calls.',
            'This verifies numerical correctness; the sixteen wells remain previously inspected development data.']}


def self_test():
    y=np.array([1.,-2.,3.,-4.]);identity=fresh_metrics(y,y);zero=fresh_metrics(np.zeros(4),y)
    close(identity['delta_rmse'],0.,'Perfect prediction RMSE');close(identity['delta_pearson'],1.,'Perfect Pearson')
    close(identity['delta_cosine'],1.,'Perfect cosine');close(identity['response_l2_ratio'],1.,'Perfect norm ratio')
    close(zero['delta_rmse'],np.sqrt(7.5),'Zero prediction RMSE')
    require(np.isnan(zero['delta_pearson']) and np.isnan(zero['delta_cosine']),'Zero prediction correlation must be undefined')
    close(zero['top100_direction_accuracy'],0.,'Zero prediction direction')
    p=np.array([[.6,.4],[.25,.75]])
    decisions=[{'axis_id':0,'decision':'shift','targets':[[1,2.]]},{'axis_id':1,'decision':'unknown','targets':[]}]
    close(from_decisions(p,decisions,.3),[[.42,.58],[.25,.75]],'Independent categorical edit')
    try:from_decisions(p,[decisions[0],decisions[0]],.3)
    except AssertionError:pass
    else:raise AssertionError('Duplicate axis guard failed')
    return {'status':'self-test passed','target_files_read':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=Path(__file__).resolve().parent/'runs/pilot_seen_dose')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:result=self_test()
    else:
        if args.output is None:parser.error('--output is required for an actual post-score audit')
        result=audit(args.work,args.output)
        path=args.output/'review_scores.json'
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
