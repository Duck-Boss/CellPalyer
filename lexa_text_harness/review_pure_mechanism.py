#!/usr/bin/env python3
"""Independent input/prediction audit for the pure-mechanism pilot.

Does not import the new data or agent implementation. Treatment outcomes are
not opened by the input or prediction audits.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd


ARMS = ('full', 'no_mechanism', 'no_semantics', 'neither')
ALPHAS = np.asarray([.1, .3, .6])


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def occupancy(codes):
    return np.stack([np.bincount(codes[:,a],minlength=16)/len(codes) for a in range(32)])


def expected_q(p0, decisions, alpha):
    assert len(decisions)==32
    assert {r['axis_id'] for r in decisions}==set(range(32))
    q=p0.copy()
    for row in decisions:
        a=row['axis_id'];assert type(a) is int
        assert row['decision'] in ('shift','keep','unknown')
        targets=row['targets']
        if row['decision']!='shift':
            assert targets==[]
            continue
        assert 1<=len(targets)<=3 and len({p[0] for p in targets})==len(targets)
        assert all(type(s) is int and 0<=s<16 and isinstance(v,(float,int)) and not isinstance(v,bool)
                   and np.isfinite(v) and v>0 for s,v in targets)
        weights=np.zeros(16)
        for s,v in targets:weights[s]=v
        weights/=weights.sum()
        q[a]=(1-alpha)*p0[a]+alpha*weights
    return q


def content_json(response):
    value=response['choices'][0]['message']['content'].strip()
    if value.startswith('```'):
        value='\n'.join(value.splitlines()[1:-1])
    return json.loads(value)


def successful_response(stage):
    statuses=[p for p in sorted(stage.glob('*.status.json')) if read(p).get('status')=='completed']
    assert len(statuses)==1,(stage,len(statuses))
    path=statuses[0].with_name(statuses[0].name.replace('.status.json','.response.json'))
    status=read(statuses[0]);assert status['response_sha256']==sha(path)
    return content_json(read(path))


def decoded_payload(request_path):
    messages=read(request_path)['messages']
    assert messages[0]['role']=='system' and messages[1]['role']=='user'
    return json.loads(messages[1]['content'])


def json_digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def numeric_gene_rows_are_anonymous(obj, path=''):
    """Check structured code evidence, leaving mechanism hypotheses separate."""
    if isinstance(obj,list):
        for i,v in enumerate(obj):numeric_gene_rows_are_anonymous(v,path+f'[{i}]')
    elif isinstance(obj,dict):
        for key,value in obj.items():
            if key in ('feature','gene','gene_id','symbol') and isinstance(value,str):
                assert re.fullmatch(r'G\d{4,}',value),(path,key,value)
            elif key=='scope':
                assert value=='anonymous expression axis',(path,key,value)
            elif key in ('name','summary','language_annotation','functions','evidence_genes'):
                raise AssertionError(('Semantic field in hidden code payload',path,key))
            else:numeric_gene_rows_are_anonymous(value,path+'/'+key)


def feature_rows(vector, info, semantics, count):
    scaled=vector/np.maximum(info['std'],.05)
    def rows(indices):
        output=[]
        for i in indices:
            row={'feature':str(info['genes'][i] if semantics else info['anonymous'][i]),
                 'delta':round(float(vector[i]),5),'scaled_delta':round(float(scaled[i]),5)}
            if semantics:row['symbol']=str(info['symbols'][i])
            output.append(row)
        return output
    up=np.flatnonzero(scaled>1e-8);down=np.flatnonzero(scaled < -1e-8)
    return {'up':rows(up[np.argsort(-scaled[up],kind='stable')][:count]),
            'down':rows(down[np.argsort(scaled[down],kind='stable')][:count])}


def audit_payload(payload,arm,info,plan=None,local_decisions=None):
    """Reconstruct supplied control/code evidence without importing its producer."""
    mechanism=arm in ('full','no_semantics');semantics=arm in ('full','no_mechanism')
    context=payload['context'];case=info['case'];source=info['context']
    base={'case','cell_context','dose_uM','exposure_hours','context_ref','n_control_cells','measurement'}
    expected_keys=base|({'mechanism_card','mechanism_ref'} if mechanism else {'mechanism_status'})
    expected_keys|={'control_expressed_features'}
    if not semantics:expected_keys|={'control_feature_semantics'}
    assert set(context)==expected_keys,(arm,set(context)^expected_keys)
    assert context['case']==info['entry']['opaque_id'] and context['context_ref']=='C'
    assert context['dose_uM']==case['dose_uM'] and context['exposure_hours']==case['exposure_time_hours']
    assert context['n_control_cells']==len(case['control_cell_ids'])
    assert context['cell_context']==source['request']['cell_context']
    if mechanism:
        assert context['mechanism_card']==source['mechanism_card'] and context['mechanism_ref']=='M'
    if semantics:
        mean=info['control_mean'];indices=np.argsort(-mean,kind='stable')[:12]
        expected=[{'gene':str(info['genes'][i]),'symbol':str(info['symbols'][i]),'mean':round(float(mean[i]),4)} for i in indices]
        assert context['control_expressed_features']==expected
    else:
        assert context['control_feature_semantics']=='withheld; no mapping from anonymous features to genes is available'
        mean=info['control_mean'];indices=np.argsort(-mean,kind='stable')[:12]
        expected=[{'feature':str(info['anonymous'][i]),'mean':round(float(mean[i]),4)} for i in indices]
        assert context['control_expressed_features']==expected
        numeric_gene_rows_are_anonymous(context['control_expressed_features'])
    stage=payload['stage']
    if stage=='mechanism_hypotheses':
        assert set(payload)=={'stage','prompt_version','context','instruction','available_evidence_refs'}
        refs={'C'}
        if mechanism:refs|={'M'}|{s['id'] for s in source['mechanism_card'].get('sources',[])}
        assert set(payload['available_evidence_refs'])==refs
        if plan is not None:
            assert all(set(h['evidence_refs'])<=refs for h in plan['hypotheses'])
        return
    assert payload['frozen_hypotheses']==plan
    assert payload['main_alpha']==.3
    if stage=='local_state_preferences':
        assert set(payload)=={'stage','context','frozen_hypotheses','requested_axes','main_alpha','response_field','axis_overview','axes'}
        assert payload['response_field']=='decisions'
        assert len(payload['requested_axes'])==8
        assert {r['axis_id'] for r in payload['axis_overview']}==set(range(32))
    else:
        assert stage=='joint_consistency_review'
        assert set(payload)=={'stage','context','frozen_hypotheses','response_field','requested_axes','instruction','main_alpha','local_decisions','decoder_feedback','axes'}
        assert payload['response_field']=='updates' and payload['requested_axes']==list(range(32))
        assert payload['local_decisions']==local_decisions
        q=expected_q(info['p0'],local_decisions,.3)
        axis_delta=((q-info['p0'])[:,:,None]*info['effects']).sum(1);delta=axis_delta.sum(0)
        feedback=payload['decoder_feedback']
        assert feedback['top_gene_changes']==feature_rows(delta,info,semantics,10)
        assert np.allclose(feedback['axis_tv'],.5*np.abs(q-info['p0']).sum(1),atol=1e-12)
        if not semantics:
            assert isinstance(feedback['hypothesis_gene_checks'],str)
            numeric_gene_rows_are_anonymous(feedback['top_gene_changes'])
            numeric_gene_rows_are_anonymous(feedback['cross_axis_cancellation'])
    assert [r['axis_id'] for r in payload['axes']]==payload['requested_axes']
    for axis in payload['axes']:
        a=axis['axis_id'];base=info['p0'][a]@info['effects'][a]
        assert axis['control_occupancy']==np.round(info['p0'][a],5).tolist()
        assert [r['state_id'] for r in axis['states']]==list(range(16))
        if not semantics:numeric_gene_rows_are_anonymous(axis)
        else:
            card=info['cards'][a]
            assert axis['scope']==card['scope']
            assert axis['language_annotation']==card['language_annotation']
        for row in axis['states']:
            s=row['state_id'];compact=stage=='joint_consistency_review'
            assert row['evidence_ref']==f'A{a}/C{s}'
            assert row['train_occupancy']==round(float(info['train_occupancy'][a,s]),5)
            expected=feature_rows(info['effects'][a,s]-base,info,semantics,1 if compact else 2)
            assert row['effect_from_control']==expected,(arm,a,s,'effect_from_control')
            if not compact:
                expected=feature_rows(info['effects'][a,s]-info['effects'][a].mean(0),info,semantics,1)
                assert row['effect_from_axis_reference']==expected,(arm,a,s,'effect_from_axis_reference')
            if semantics:
                assert row['name']==info['cards'][a]['states'][s]['name']
                if not compact:assert row['summary']==info['cards'][a]['states'][s]['summary']


def audit(work, experiment, require_predictions=False):
    work=Path(work).resolve();experiment=Path(experiment).resolve()
    manifest=read(experiment/'inputs/manifest.json')
    allowed={'effects','baseline','state_labels','cases','axis_annotations','mechanisms'}
    assert set(manifest['provenance']['source_paths'])==allowed
    for key,path in manifest['provenance']['source_paths'].items():
        assert sha(path)==manifest['provenance']['source_sha256'][key],key
    assert manifest['condition_response_retrieval'] is False and manifest['heldout_expression_accessed'] is False
    for name,digest in manifest.get('output_sha256',{}).items():
        p=Path(name);p=p if p.is_absolute() else experiment/'inputs'/p
        assert sha(p)==digest,p
    cases={c['case_id']:c for c in read(work/'prepared/cases.json')['cases'] if c['status']=='ready'}
    entries=manifest['cases'];assert len(entries)==16 and {c['case_id'] for c in entries}==set(cases)
    assert len({c['opaque_id'] for c in entries})==16
    assert all(not any(word in c['opaque_id'].lower() for word in ('bms','dex','nutlin','saha','plate','replicate')) for c in entries)
    with np.load(work/'lexa/exact_code_effects.npz',allow_pickle=False) as z:
        effects=z['contributions'].astype(float);genes=z['genes'].astype(str)
        symbols=z['gene_symbols'].astype(str);std=z['train_gene_std'].astype(float)
        train_occupancy=z['train_occupancy'].astype(float)
    cards=read(manifest['cards_path'])['axes']
    assert [c['axis_id'] for c in cards]==list(range(32))
    anonymous=np.asarray([f'G{i:04d}' for i in np.random.default_rng(20261027).permutation(len(genes))])
    with np.load(work/'lexa/baseline.npz',allow_pickle=False) as z:
        base_ids=z['cell_ids'].astype(str);base_codes=z['codes'];base_x=z['observed_x'].astype(float)
        base_decoded=z['reconstructed_x'].astype(float)
        assert np.array_equal(z['genes'].astype(str),genes)
    assert len(set(base_ids))==len(base_ids)
    meta=pd.read_csv(work/'prepared/metadata.csv').set_index('cell_id')
    expected_by_id={};max_error={}
    def check(name,actual,expected,tol=1e-8):
        actual=np.asarray(actual);expected=np.asarray(expected)
        assert actual.shape==expected.shape,(name,actual.shape,expected.shape)
        error=float(np.max(np.abs(actual-expected)))
        max_error[name]=max(error,max_error.get(name,0))
        assert np.isfinite(actual).all() and error<=tol,(name,error)
    for entry in entries:
        case=cases[entry['case_id']]
        ix=pd.Index(base_ids).get_indexer(case['control_cell_ids']);assert (ix>=0).all()
        cm=meta.loc[base_ids[ix]]
        assert cm.dose_uM.eq(0).all() and cm.drug.eq(case['drug']).all()
        assert cm.plate.eq(case['plate']).all() and cm.replicate.eq(case['replicate']).all()
        assert not set(case['control_cell_ids']) & set(case['treated_cell_ids'])
        p0=occupancy(base_codes[ix])
        with np.load(entry['npz'],allow_pickle=False) as z:
            check('p0',z['p0'],p0)
            assert np.array_equal(z['genes'].astype(str),genes)
            assert np.array_equal(z['anonymous_gene_ids'].astype(str),anonymous)
            assert not any(any(bad in key for bad in ('candidate','reference_observed','treated','retriev')) for key in z.files)
            if 'control_mean' in z:check('control_mean',z['control_mean'],base_x[ix].mean(0))
            if 'control_decoded_mean' in z:check('control_decoded_mean',z['control_decoded_mean'],base_decoded[ix].mean(0))
        context=read(entry['context'])
        text=json.dumps(context)
        for bad in ['reference_observed_delta','training_library','candidate_q','delta_p','treated_cell_ids']:
            assert bad not in text,(entry['case_id'],bad)
        expected_by_id[entry['opaque_id']]={'p0':p0,'case':case,'entry':entry,'context':context,
            'effects':effects,'genes':genes,'symbols':symbols,'std':std,'train_occupancy':train_occupancy,
            'anonymous':anonymous,'cards':cards,'control_mean':base_x[ix].mean(0)}
    predictions=[];payload_checks=[];request_count=0;response_count=0;usage={'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}
    baseline_checks=[]
    for entry in entries:
        folder=experiment/'baselines'/entry['slug']
        if not folder.exists():continue
        declared=read(folder/'literal_keyword.json')['decisions'];p0=expected_by_id[entry['opaque_id']]['p0']
        with np.load(folder/'literal_keyword.npz',allow_pickle=False) as z:
            assert np.array_equal(z['genes'].astype(str),genes)
            check('baseline_p0',z['p0'],p0)
            for k,alpha in enumerate(ALPHAS):
                q=expected_q(p0,declared,float(alpha));delta=((q-p0)[:,:,None]*effects).sum((0,1))
                check('baseline_q',z['q_by_alpha'][k],q)
                check('baseline_delta',z['delta_by_alpha'][k],delta,1e-7)
            check('baseline_main_q',z['q'],z['q_by_alpha'][1])
            check('baseline_main_delta',z['delta_mean'],z['delta_by_alpha'][1],1e-7)
        baseline_checks.append({'opaque_id':entry['opaque_id'],'arrays_sha256':sha(folder/'literal_keyword.npz')})
    for complete_path in sorted((experiment/'predictions').rglob('COMPLETE.json')):
        folder=complete_path.parent;complete=read(complete_path)
        contract=read(folder/'run_contract.json')
        identity=contract['identity']
        assert json_digest(identity)==contract['fingerprint']==complete['fingerprint']
        opaque=identity.get('opaque_id',complete.get('opaque_id'))
        arm=identity.get('arm',complete.get('arm'))
        assert arm in ARMS and opaque in expected_by_id,(folder,arm,opaque)
        p0=expected_by_id[opaque]['p0']
        assert complete['case_id']==identity['case_id']==expected_by_id[opaque]['case']['case_id']
        assert complete['local_sha256']==sha(folder/'local.json')
        assert complete['joint_sha256']==sha(folder/'joint.json')
        assert complete['hypotheses_sha256']==sha(folder/'hypotheses.json')
        plan=successful_response(folder/'requests/plan')
        assert plan==read(folder/'hypotheses.json')
        for name,digest in identity.get('source_sha256',{}).items():
            assert sha(Path(__file__).parent/name)==digest,name
        locals_raw=[]
        stages=list((folder/'requests').iterdir())
        assert sorted(p.name for p in stages)==['joint','local_00','local_01','local_02','local_03','plan']
        assert complete['api_attempts']==len(list((folder/'requests').rglob('*.request.json')))
        for stage in sorted((folder/'requests').glob('local_*')):
            response=successful_response(stage)
            locals_raw.extend(response['decisions'])
        if not locals_raw:
            for stage in sorted((folder/'requests').glob('axes_*')):
                locals_raw.extend(successful_response(stage)['decisions'])
        assert len(locals_raw)==32,(folder,len(locals_raw))
        joint_dirs=list((folder/'requests').glob('joint*'));assert len(joint_dirs)==1
        updates=successful_response(joint_dirs[0])['updates']
        joint_by_axis={d['axis_id']:d for d in locals_raw}
        assert len({d['axis_id'] for d in updates})==len(updates)
        joint_by_axis.update({d['axis_id']:d for d in updates})
        for phase,declared in [('local',locals_raw),('joint',list(joint_by_axis.values()))]:
            doc=read(folder/(phase+'.json'));artifact=folder/(phase+'.npz')
            assert doc['case_id']==identity['case_id'] and doc['arm']==arm and doc['opaque_id']==opaque
            assert doc['fingerprint']==complete['fingerprint'] and doc['hypotheses_sha256']==json_digest(plan)
            assert doc['arrays_sha256']==sha(artifact)
            assert sorted(doc['decisions'],key=lambda d:d['axis_id'])==sorted(declared,key=lambda d:d['axis_id'])
            with np.load(artifact,allow_pickle=False) as z:
                assert np.array_equal(z['genes'].astype(str),genes)
                check('prediction_p0',z['p0'],p0)
                check('alphas',z['alphas'],ALPHAS)
                for k,alpha in enumerate(ALPHAS):
                    q=expected_q(p0,declared,float(alpha))
                    delta=((q-p0)[:,:,None]*effects).sum((0,1))
                    check('q_by_alpha',z['q_by_alpha'][k],q)
                    check('delta_by_alpha',z['delta_by_alpha'][k],delta,1e-7)
                check('main_q',z['q'],z['q_by_alpha'][1])
                check('main_delta',z['delta_mean'],z['delta_by_alpha'][1],1e-7)
            predictions.append({'opaque_id':opaque,'case_id':expected_by_id[opaque]['case']['case_id'],
                                'arm':arm,'phase':phase,'artifact':str(artifact),'sha256':sha(artifact)})
        for request_path in sorted((folder/'requests').rglob('*.request.json')):
            request_count+=1;payload=decoded_payload(request_path);request=read(request_path)
            audit_payload(payload,arm,expected_by_id[opaque],plan,locals_raw)
            prompt=identity['plan_prompt'] if payload['stage']=='mechanism_hypotheses' else identity['decision_prompt']
            assert request['messages'][0]=={'role':'system','content':prompt}
            assert request['tools']==[] and 1<=request['attempt']<=3
            sf=json_digest({'run':complete['fingerprint'],'payload':payload,'prompt':prompt})
            assert request['stage_fingerprint']==sf==read(request_path.parent/'contract.json')['fingerprint']
            for message in request['messages'][2:]:
                if message['role']=='user':
                    assert message['content'].startswith('Format validation failed: ')
                    assert message['content'].endswith('Correct the complete JSON. No treatment outcome has been scored.')
                else:
                    assert message['role']=='assistant'
                    earlier=list(request_path.parent.glob('*.response.json'))
                    possible=[read(p)['choices'][0]['message'].get('content') for p in earlier]
                    assert message['content'] in possible or message['content']=='Invalid completion'
            # Detailed stage-specific checks are deliberately independent of runner helpers.
            text=json.dumps(payload,ensure_ascii=False)
            for forbidden in ('reference_observed_delta','training_library','retrieval_context','candidate_q'):
                assert forbidden not in text,(request_path,forbidden)
            assert 'drug=' not in text and 'plate=' not in text and 'treated_cell_ids' not in text
            if arm in ('no_mechanism','neither'):
                # Direct external identity fields must be absent. Model-generated hypotheses
                # are audited separately and are not treated as externally supplied knowledge.
                direct={k:v for k,v in payload.items() if k not in ('frozen_hypotheses','hypotheses','hypothesis','plan','local_proposal','local_decisions','decoder_feedback','feedback')}
                dt=json.dumps(direct,ensure_ascii=False).lower()
                for drug in ('bms-345541','dexamethasone','nutlin','vorinostat','saha','ethanol','dmso','pubmed.ncbi','doi.org'):
                    assert drug not in dt,(request_path,drug)
            if arm in ('no_semantics','neither'):
                for key in ('axes','axis_cards','axis_overview','control_background','control_summary'):
                    if key in payload:numeric_gene_rows_are_anonymous(payload[key],key)
                for key in ('decoder_feedback','local_decoder_feedback','feedback'):
                    if key in payload:
                        feedback=payload[key]
                        assert 'plan_gene_checks' not in feedback
                        for sub in ('top_gene_changes','cross_axis_cancellation','largest_cross_axis_cancellation','gene_rows'):
                            if sub in feedback:numeric_gene_rows_are_anonymous(feedback[sub],key+'/'+sub)
            payload_checks.append({'file':str(request_path.relative_to(experiment)),'arm':arm,'sha256':sha(request_path)})
        for path in sorted((folder/'requests').rglob('*.response.json')):
            response_count+=1
            counts=read(path).get('usage',{})
            for key in usage:usage[key]+=counts.get(key,0)
    if require_predictions:
        assert len(predictions)==128,len(predictions)
        assert len({(x['opaque_id'],x['arm'],x['phase']) for x in predictions})==128
        assert len(baseline_checks)==16
        assert len(list((experiment/'predictions').rglob('*.request.json')))==request_count
        seal=read(experiment/'prediction_seal.json')
        assert seal['status']=='all_predictions_complete' and seal['score_exists_at_seal'] is False
        assert seal['n_cases']==16 and seal['n_case_arms']==64
        for path,expected in seal['files'].items():assert sha(experiment/path)==expected,path
        for name,expected in seal['source_hashes'].items():assert sha(Path(__file__).parent/name)==expected,name
        protocol=read(experiment/'protocol.json')
        assert protocol['primary_alpha']==.3 and protocol['sensitivity_alphas']==[.1,.6]
        assert protocol['nominal_total_calls']==384 and protocol['condition_response_retrieval'] is False
        assert protocol['target_expression_visible'] is False and protocol['source_hashes']==seal['source_hashes']
    report={'status':'passed','created_asia_shanghai':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'prediction_targets_read':False,'n_cases':len(entries),'n_unique_control_cells':len(set(
                v for c in cases.values() for v in c['control_cell_ids'])),
            'max_numerical_errors':max_error,'n_phase_predictions_checked':len(predictions),
            'predictions':predictions,'numeric_baselines_checked':baseline_checks,'actual_request_payloads_checked':payload_checks,
            'api_requests':request_count,'api_responses':response_count,'usage':usage,
            'prediction_seal_verified':require_predictions,
            'limitations':['Known development population; not an independent new test.',
                'Payload identity checks examine supplied data; spontaneous model hypotheses are not proof of external leakage.',
                'Model knowledge may include drug mechanisms from pretraining.']}
    destination=experiment/('independent_prediction_audit.json' if require_predictions else 'independent_input_audit.json')
    save(destination,report)
    print(json.dumps({k:report[k] for k in ['status','n_cases','n_phase_predictions_checked','max_numerical_errors','api_requests','api_responses']},ensure_ascii=False,indent=2))
    return report


def score_audit(work,experiment):
    """Only callable after all predictions are sealed and the producer scored."""
    work=Path(work).resolve();experiment=Path(experiment).resolve()
    seal=read(experiment/'prediction_seal.json');score_meta=read(experiment/'score_manifest.json')
    assert seal['status']=='all_predictions_complete' and seal['score_exists_at_seal'] is False
    assert score_meta['status']=='complete' and score_meta['prediction_seal_sha256']==sha(experiment/'prediction_seal.json')
    for path,expected in seal['files'].items():assert sha(experiment/path)==expected,path
    for name,expected in score_meta['outputs_sha256'].items():assert sha(experiment/name)==expected,name
    assert sha(work/'prepared/evaluation.npz')==score_meta['evaluation_sha256']
    assert sha(work/'codec_changes/case_arrays.npz')==score_meta['observed_codes_sha256']
    entries=read(experiment/'inputs/manifest.json')['cases']
    details={c['case_id']:c for c in read(work/'prepared/cases.json')['cases'] if c['status']=='ready'}
    with np.load(work/'prepared/evaluation.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with np.load(work/'codec_changes/case_arrays.npz',allow_pickle=False) as z:
        observed={str(cid):{'p0':z['pcontrol'][i],'p1':z['ptreated'][i],'delta':z['decoded_delta'][i]} for i,cid in enumerate(z['case_ids'])}
    with np.load(work/'lexa/exact_code_effects.npz',allow_pickle=False) as z:effects=z['contributions'].astype(float)
    expected=[];occ=[];raw=[];error={}
    def metric(pred,true):
        pc=pred-pred.mean();tc=true-true.mean();pn=np.linalg.norm(pred);tn=np.linalg.norm(true)
        top=np.argsort(-np.abs(true))[:100]
        return {'delta_rmse':float(np.linalg.norm(pred-true)/np.sqrt(len(true))),
            'delta_pearson':float(pc@tc/(np.linalg.norm(pc)*np.linalg.norm(tc))) if np.std(pred)>1e-12 and np.std(true)>1e-12 else np.nan,
            'delta_cosine':float(pred@true/(pn*tn)) if pn>1e-12 else np.nan,
            'top100_delta_rmse':float(np.linalg.norm((pred-true)[top])/10),
            'top100_direction_accuracy':float(np.count_nonzero(np.sign(pred[top])==np.sign(true[top]))/100),
            'response_l2_ratio':float(pn/tn)}
    for entry in entries:
        cid=entry['case_id'];detail=details[cid];truth=observed[cid]
        selected=data['test_case_ids'].astype(str)==cid
        treatment=data['x_test'][selected].astype(float)
        indices=pd.Index(data['control_cell_ids'].astype(str)).get_indexer(detail['control_cell_ids'])
        assert (indices>=0).all() and len(set(indices))==len(indices)
        control=data['control_x'][indices].astype(float)
        assert (len(treatment),len(control))==(detail['n_treated'],detail['n_control'])
        true=treatment.mean(0)-control.mean(0)
        with np.load(entry['npz'],allow_pickle=False) as z:anchor=z['control_decoded_mean'].copy()
        p0=truth['p0'];p1=truth['p1']
        diagnostic=((p1-p0)[:,:,None]*effects).sum((0,1))
        error['answer_codec_delta']=max(error.get('answer_codec_delta',0.),float(np.max(np.abs(diagnostic-truth['delta']))))
        assert np.allclose(diagnostic,truth['delta'],atol=2e-6)
        outputs=[('no_edit',np.zeros_like(true),p0),('answer_codec_diagnostic',truth['delta'],p1)]
        for method,folder,filename in [('literal_keyword',experiment/'baselines'/entry['slug'],'literal_keyword')]+[
                (arm+'_'+phase,experiment/'predictions'/arm/entry['slug'],phase) for arm in ARMS for phase in ['local','joint']]:
            with np.load(folder/(filename+'.npz'),allow_pickle=False) as z:
                assert np.array_equal(z['genes'].astype(str),data['genes'].astype(str))
                for k,alpha in enumerate(ALPHAS):outputs.append((method+f'_a{alpha:.1f}',z['delta_by_alpha'][k],z['q_by_alpha'][k]))
        for method,pred,q in outputs:
            base={'case_id':cid,'drug':entry['drug'],'method':method,'n_treated':len(treatment),'n_control':len(control)}
            expected.append({**base,**metric(pred,true)})
            occ.append({**base,'occupancy_tv_error_mean':float(np.abs(q-p1).sum()/64),
                'dominant_code_accuracy':float(np.count_nonzero(q.argmax(1)==p1.argmax(1))/32),
                'predicted_changed_axes':int(np.count_nonzero(q.argmax(1)!=p0.argmax(1))),
                'observed_changed_axes':int(np.count_nonzero(p1.argmax(1)!=p0.argmax(1))),
                'predicted_mean_axis_tv':float(np.abs(q-p0).sum()/64)})
            raw.append({**base,'raw_decoded_absolute_rmse':float(np.linalg.norm(anchor+pred-treatment.mean(0))/np.sqrt(len(true)))})
    def compare(filename,frame,keys,columns):
        declared=pd.read_csv(experiment/filename).set_index(keys).sort_index()
        target=frame.set_index(keys).sort_index()
        assert declared.index.equals(target.index),(filename,len(declared),len(target))
        a=declared[columns].to_numpy(float);b=target[columns].to_numpy(float)
        assert np.array_equal(np.isnan(a),np.isnan(b)),filename
        finite=np.isfinite(a)&np.isfinite(b);delta=float(np.max(np.abs(a[finite]-b[finite])))
        error[filename]=delta;assert delta<1e-8,(filename,delta)
    frame=pd.DataFrame(expected);metrics=list(metric(np.ones(2000),np.arange(2000)).keys())
    compare('scores_by_well.csv',frame,['method','case_id'],metrics)
    condition=frame.groupby(['method','drug'],as_index=False)[metrics].mean()
    compare('scores_by_condition.csv',condition,['method','drug'],metrics)
    summary=condition.groupby('method',as_index=False)[metrics].mean()
    compare('scores_summary.csv',summary,['method'],metrics)
    compare('occupancy_by_well.csv',pd.DataFrame(occ),['method','case_id'],[
        'occupancy_tv_error_mean','dominant_code_accuracy','predicted_changed_axes','observed_changed_axes','predicted_mean_axis_tv'])
    compare('raw_decoded_absolute_by_well.csv',pd.DataFrame(raw),['method','case_id'],['raw_decoded_absolute_rmse'])
    report={'status':'passed','created_asia_shanghai':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'target_access_after_seal':True,'n_wells':len(entries),'n_methods':len(summary),'n_well_method_rows':len(frame),
        'max_absolute_errors':error,'prediction_seal_sha256':sha(experiment/'prediction_seal.json'),
        'primary_aggregation':'Equal wells within each drug, then equal four drugs',
        'target_definition':'Observed treated mean minus matched observed control mean, log1p(CP10K)',
        'main_alpha':.3,'sensitivity_alphas':[.1,.6]}
    save(experiment/'independent_score_audit.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def audit_live_payloads(work,experiment):
    """Audit requests already written, including cases that are still running."""
    work=Path(work).resolve();experiment=Path(experiment).resolve()
    manifest=read(experiment/'inputs/manifest.json');entries={e['opaque_id']:e for e in manifest['cases']}
    cases={c['case_id']:c for c in read(work/'prepared/cases.json')['cases'] if c['status']=='ready'}
    cards=read(manifest['cards_path'])['axes']
    with np.load(work/'lexa/exact_code_effects.npz',allow_pickle=False) as z:
        fixed={'effects':z['contributions'].astype(float),'genes':z['genes'].astype(str),
            'symbols':z['gene_symbols'].astype(str),'std':z['train_gene_std'].astype(float),
            'train_occupancy':z['train_occupancy'].astype(float)}
    checked=[];statuses=[]
    for contract_path in sorted((experiment/'predictions').rglob('run_contract.json')):
        folder=contract_path.parent;contract=read(contract_path);arm=contract['identity']['arm']
        entry=next(e for e in entries.values() if e['case_id']==contract['identity']['case_id'])
        with np.load(entry['npz'],allow_pickle=False) as z:
            info={**fixed,'p0':z['p0'],'anonymous':z['anonymous_gene_ids'].astype(str),'control_mean':z['control_mean'],
                'entry':entry,'case':cases[entry['case_id']],'context':read(entry['context']),'cards':cards}
        plan=read(folder/'hypotheses.json') if (folder/'hypotheses.json').exists() else None
        local=read(folder/'local.json')['decisions'] if (folder/'local.json').exists() else None
        for path in sorted((folder/'requests').rglob('*.request.json')):
            payload=decoded_payload(path)
            audit_payload(payload,arm,info,plan,local)
            assert not any(k in json.dumps(payload) for k in ['training_library','candidate_q','reference_observed_delta','treated_cell_ids'])
            checked.append({'path':str(path.relative_to(experiment)),'sha256':sha(path),'arm':arm})
        statuses.extend(read(p) for p in (folder/'requests').rglob('*.status.json'))
    errors={}
    for s in statuses:
        if s['status']=='failed':errors[s['error']]=errors.get(s['error'],0)+1
    report={'status':'passed','created_asia_shanghai':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'prediction_targets_read':False,'n_actual_request_payloads':len(checked),
            'stage_status_counts':{'completed':sum(s['status']=='completed' for s in statuses),'failed':sum(s['status']=='failed' for s in statuses)},
            'format_or_transport_errors':errors,'checked_requests':checked}
    save(experiment/'live_payload_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='checked_requests'},ensure_ascii=False,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work',type=Path,default=Path(__file__).resolve().parent/'runs/pilot_seen_dose')
    parser.add_argument('--experiment',type=Path)
    parser.add_argument('--require-predictions',action='store_true')
    parser.add_argument('--score',action='store_true',help='Read evaluation targets only after a verified complete prediction seal and existing scores')
    parser.add_argument('--live-payloads',action='store_true',help='Audit currently written requests, including incomplete runs, without opening outcomes')
    args=parser.parse_args();experiment=args.experiment or args.work/'experiment_v4_pure_mechanism'
    if args.live_payloads:
        audit_live_payloads(args.work,experiment)
        return
    audit(args.work,experiment,args.require_predictions)
    if args.score:score_audit(args.work,experiment)


if __name__=='__main__':main()
