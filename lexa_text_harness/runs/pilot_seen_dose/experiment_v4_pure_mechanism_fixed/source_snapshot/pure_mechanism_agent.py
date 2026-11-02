"""Pure mechanism-to-code text agent; no perturbation-response retrieval.

Only an opaque case context, drug mechanism (when enabled), control state and
frozen code meanings are serialized. Numerical feedback checks a hypothesis's
internal consistency; it never scores treatment accuracy.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from agent import TransportError
from annotate_cards import atomic_write_json, file_sha256


ARMS = ('full', 'no_mechanism', 'no_semantics', 'neither')
ALPHAS = np.asarray([.10, .30, .60])
MAIN_ALPHA = .30
LOCAL_BATCH_SIZE = 8
PROMPT_VERSION = 'pure-mechanism-code-v1.1-source-refs'
PLAN_PROMPT = '''Form cautious transcriptional hypotheses for the supplied perturbation.
Only mechanism information and untreated cellular context are available. There
are no measured perturbation responses, dose trajectories, or correct answer
profiles. Do not invent missing drug identities. If the perturbation mechanism
is hidden, treatment-specific directions are unknown; do not infer a named drug
from an opaque identifier. Inhibiting a protein does not imply lowering its mRNA.
Distinguish direct mechanism from secondary hypotheses and context dependencies.
Code meanings are not yet supplied; do not mention axis or code identifiers.
Return only JSON:
{"hypotheses":[{"id":"H1","program":"short process","direction":"increase|decrease|uncertain","basis":"direct|secondary|unknown","conditions":"brief condition or unknown","gene_expectations":[{"gene":"symbol or gene ID","direction":"increase|decrease|uncertain"}],"evidence_refs":["M"],"reason":"brief evidence-based justification"}],"summary":"brief","limitations":["brief"]}
Use one to five hypotheses with sequential IDs H1,H2,...; at most three gene
expectations per hypothesis, and at most three short limitations. Evidence refs
must be drawn from available_evidence_refs in the payload. C refers to untreated
context; M refers to the whole mechanism card, and the exact source IDs listed
in that card may also be cited. Mechanism refs are unavailable when hidden.
An uncertain hypothesis may have no gene expectations. Keep program/conditions/
reason under 120 characters each. State explicit uncertainty rather than a
confident biological story unsupported by the supplied context.'''
DECISION_PROMPT = '''Predict code-state preferences using the supplied fixed mechanism
hypotheses and axis/state evidence. No perturbation-response examples are present.
State IDs are unordered categories. Signed effects are decoder contributions,
not causal observations; read their directions and changes from current control.
Every axis can change. Use shift, keep (supported unchanged), or unknown
(insufficient information). Do not infer a named drug when its mechanism is
hidden. Anonymous feature IDs have no supplied gene identity; do not invent one.
For shift choose one to three displayed state IDs with positive relative weights.
Local arithmetic normalizes weights to r and computes q=(1-alpha)*p0+alpha*r.
Alpha is fixed at 0.30 for the main result, not a confidence score or calibrated
dose response. Current zero occupancy does not prevent selecting a supported state.
For keep/unknown use targets: []. Supply a short evidence-based reason, not a
reasoning trace. Use at least one supplied hypothesis or current-axis evidence ref.
Local stages must return every requested axis exactly once:
{"decisions":[{"axis_id":0,"decision":"shift","targets":[[2,1]],"evidence_refs":["H1","A0/C2"],"reason":"under 80 characters"}],"summary":"brief","limitations":["brief"]}
The joint stage instead returns only axes requiring revision, using the same row
schema and field updates in place of decisions. Return updates: [] if no specific
revision is justified. Do not rewrite the frozen hypotheses. Check direction
contradictions, cross-axis cancellation and evidence support. Feedback can test
consistency with your hypotheses, never true treatment accuracy. Keep at most two
evidence refs per row, short reasons, summary under 300 characters, and at most
three short limitations. Return only JSON so that all axes fit the output budget.'''


class ValidationError(ValueError):
    pass


class StageFailure(RuntimeError):
    pass


def _plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): _plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [_plain(v) for v in value]
    return value


def _json(value):
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _array_digest(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256((str(a.dtype)+str(a.shape)).encode()+a.tobytes()).hexdigest()


def _parse(content):
    if not isinstance(content,str) or not content.strip():raise ValidationError('Return one JSON object')
    text=content.strip()
    if text.startswith('```'):
        lines=text.splitlines()
        if len(lines)<3 or lines[0].lower() not in ('```','```json') or lines[-1]!='```':
            raise ValidationError('Invalid JSON fence')
        text='\n'.join(lines[1:-1])
    try:value=json.loads(text)
    except (ValueError,TypeError):raise ValidationError('Return valid JSON without surrounding prose') from None
    if not isinstance(value,dict):raise ValidationError('Return a JSON object')
    return value


def _short(value,name,maximum=500,nonempty=True):
    if not isinstance(value,str) or (nonempty and not value.strip()) or len(value)>maximum:
        raise ValidationError(f'{name} must be a short string of at most {maximum} characters')


def _trailer(value):
    _short(value.get('summary'),'summary',1000)
    limits=value.get('limitations')
    if not isinstance(limits,list) or not 1<=len(limits)<=6:
        raise ValidationError('limitations must contain one to six short strings')
    for limit in limits:_short(limit,'limitation',500)


def parse_plan(content, mechanism_visible=True, allowed_mechanism_refs=None):
    value=_parse(content)
    if set(value)!={'hypotheses','summary','limitations'}:raise ValidationError('Plan fields: hypotheses, summary, limitations')
    _trailer(value);rows=value['hypotheses']
    if not isinstance(rows,list) or not 1<=len(rows)<=5:raise ValidationError('Return one to five hypotheses')
    fields={'id','program','direction','basis','conditions','gene_expectations','evidence_refs','reason'}
    for i,row in enumerate(rows,1):
        if not isinstance(row,dict) or set(row)!=fields:raise ValidationError('Hypothesis fields differ from declared schema')
        if row['id']!=f'H{i}':raise ValidationError('Hypothesis IDs must be sequential H1,H2,...')
        if row['direction'] not in ('increase','decrease','uncertain') or row['basis'] not in ('direct','secondary','unknown'):
            raise ValidationError('Invalid hypothesis direction or basis')
        for key in ('program','conditions','reason'):_short(row[key],key,400)
        refs=row['evidence_refs'];allowed={'C','M'} if mechanism_visible else {'C'}
        if mechanism_visible:
            allowed.update(ref for ref in (allowed_mechanism_refs or []) if isinstance(ref,str) and ref.strip())
        if not isinstance(refs,list) or not refs or any(not isinstance(x,str) for x in refs) or len(set(refs))!=len(refs) or not set(refs)<=allowed:
            raise ValidationError('Hypothesis evidence refs must cite only supplied IDs: '+', '.join(sorted(allowed)))
        genes=row['gene_expectations']
        if not isinstance(genes,list) or len(genes)>3:raise ValidationError('At most three gene expectations per hypothesis')
        for gene in genes:
            if not isinstance(gene,dict) or set(gene)!={'gene','direction'}:raise ValidationError('Gene expectation fields: gene, direction')
            _short(gene['gene'],'gene',100)
            if gene['direction'] not in ('increase','decrease','uncertain'):raise ValidationError('Invalid gene direction')
    return value


def _validate_rows(rows, axes, states, hypotheses, partial=False, train_occupancy=None):
    if not isinstance(rows,list) or (len(rows)>len(axes) if partial else len(rows)!=len(axes)):
        raise ValidationError(f'Return {"at most" if partial else "exactly"} {len(axes)} axis rows')
    seen=set();hids={h['id'] for h in hypotheses['hypotheses']}
    fields={'axis_id','decision','targets','evidence_refs','reason'}
    for row in rows:
        if not isinstance(row,dict) or set(row)!=fields:raise ValidationError('Decision fields: axis_id, decision, targets, evidence_refs, reason')
        axis=row['axis_id']
        if type(axis)is not int or axis not in axes or axis in seen:raise ValidationError('Axis IDs must be requested unique integers')
        seen.add(axis)
        if row['decision'] not in ('shift','keep','unknown'):raise ValidationError('decision must be shift, keep or unknown')
        _short(row['reason'],'reason',400)
        targets=row['targets']
        if not isinstance(targets,list):raise ValidationError('targets must be an array')
        if row['decision']=='shift':
            if not 1<=len(targets)<=3:raise ValidationError('shift requires one to three target states')
        elif targets:raise ValidationError('keep and unknown require empty targets')
        used=set()
        for target in targets:
            if not isinstance(target,list) or len(target)!=2:raise ValidationError('Each target is [state_id, positive_weight]')
            state,weight=target
            if type(state)is not int or not 0<=state<states or state in used:raise ValidationError('Target state IDs must be unique displayed integers')
            used.add(state)
            if type(weight)not in (int,float) or not math.isfinite(weight) or not 0<weight<=100:
                raise ValidationError('Target weights must be positive finite numbers at most100')
            if train_occupancy is not None and train_occupancy[axis,state]<=0:
                raise ValidationError('Selected state has no training occupancy support')
        refs=row['evidence_refs'];allowed=hids|{f'A{axis}'}|{f'A{axis}/C{s}' for s in range(states)}
        if not isinstance(refs,list) or not 1<=len(refs)<=6 or any(not isinstance(x,str) for x in refs) or len(set(refs))!=len(refs) or not set(refs)<=allowed:
            raise ValidationError('Evidence refs must cite supplied hypotheses or this axis/state')
    if not partial and seen!=set(axes):raise ValidationError('All requested axes must occur exactly once')
    return sorted(rows,key=lambda row:row['axis_id'])


def parse_decisions(content, axes, states, hypotheses, partial=False, train_occupancy=None):
    value=_parse(content);key='updates' if partial else 'decisions'
    if set(value)!={key,'summary','limitations'}:raise ValidationError(f'Response fields: {key}, summary, limitations')
    _trailer(value)
    value[key]=_validate_rows(value[key],axes,states,hypotheses,partial,train_occupancy)
    return value


def _visible(arm):
    if arm not in ARMS:raise ValueError('Unknown ablation arm')
    return arm in ('full','no_semantics'),arm in ('full','no_mechanism')


def _feature_rows(bundle,vector,arm,count=3):
    _,semantics=_visible(arm)
    vector=np.asarray(vector,dtype=float);std=np.maximum(np.asarray(bundle['std'],dtype=float),.05)
    scaled=vector/std;genes=np.asarray(bundle['genes']).astype(str)
    symbols=np.asarray(bundle.get('symbols',genes)).astype(str)
    anonymous=np.asarray(bundle['anonymous_gene_ids']).astype(str)
    def rows(indices):
        output=[]
        for i in indices:
            record={'feature':str(genes[i] if semantics else anonymous[i]),
                    'delta':round(float(vector[i]),5),'scaled_delta':round(float(scaled[i]),5)}
            if semantics:record['symbol']=str(symbols[i])
            output.append(record)
        return output
    up=np.flatnonzero(scaled>1e-8);down=np.flatnonzero(scaled < -1e-8)
    return {'up':rows(up[np.argsort(-scaled[up],kind='stable')][:count]),
            'down':rows(down[np.argsort(scaled[down],kind='stable')][:count])}


def _context(bundle,arm):
    mechanism,semantics=_visible(arm);request=bundle['request']
    value={'case':bundle.get('opaque_id',bundle.get('public_case_id','case')),
           'cell_context':request.get('cell_context','A549 human cells'),
           'dose_uM':request['dose_uM'],'exposure_hours':request['exposure_hours'],
           'context_ref':'C','n_control_cells':int(len(bundle.get('control_codes',[]))) or int(bundle.get('n_control',0)),
           'measurement':'normalized log1p(CP10K); population mean predictions, unpaired cells'}
    if mechanism:
        value['mechanism_card']=copy.deepcopy(bundle.get('mechanism_card',request.get('mechanism_card')))
        value['mechanism_ref']='M'
    else:value['mechanism_status']='Perturbation identity and mechanism withheld; no treatment-specific response is known.'
    if 'control_mean' in bundle:
        mean=np.asarray(bundle['control_mean']);indices=np.argsort(-mean,kind='stable')[:12]
        if semantics:
            value['control_expressed_features']=[{'gene':str(bundle['genes'][i]),'symbol':str(bundle['symbols'][i]),
                                                 'mean':round(float(mean[i]),4)} for i in indices]
        else:
            value['control_expressed_features']=[{'feature':str(bundle['anonymous_gene_ids'][i]),
                                                 'mean':round(float(mean[i]),4)} for i in indices]
    if not semantics:
        value['control_feature_semantics']='withheld; no mapping from anonymous features to genes is available'
    return value


def _cards_list(cards,bundle):
    if cards is None:cards=bundle.get('axis_cards')
    if isinstance(cards,dict):cards=cards.get('axes',cards.get('axis_cards'))
    if not isinstance(cards,list):raise ValueError('Expected axis cards')
    by_axis={c['axis_id']:c for c in cards}
    if len(by_axis)!=len(cards) or set(by_axis)!=set(range(len(bundle['p0']))):raise ValueError('Cards must cover every axis')
    return [by_axis[i] for i in range(len(cards))]


def _overview(card,arm):
    _,semantics=_visible(arm);axis=card['axis_id']
    value={'axis_id':axis,'evidence_ref':f'A{axis}'}
    if semantics:
        for key in ('scope','scope_support','language_annotation'):
            if key in card:value[key]=copy.deepcopy(card[key])
    else:value['scope']='anonymous expression axis'
    return value


def _axis_card(bundle,card,arm,compact=False):
    _,semantics=_visible(arm);axis=card['axis_id'];p0=np.asarray(bundle['p0'])[axis]
    effects=np.asarray(bundle['effects'])[axis];centered=effects-effects.mean(0)
    baseline=p0@effects;known={s['state_id']:s for s in card.get('states',[])}
    result=_overview(card,arm);result['control_occupancy']=np.round(p0,5).tolist();states=[]
    occupancy=np.asarray(bundle['train_occupancy'])
    for state in range(len(p0)):
        row={'state_id':state,'evidence_ref':f'A{axis}/C{state}',
             'train_occupancy':round(float(occupancy[axis,state]),5)}
        if semantics:
            item=known.get(state,{})
            row['name']=item.get('name','unannotated')
            if not compact:row['summary']=item.get('summary','')
        row['effect_from_control']=_feature_rows(bundle,effects[state]-baseline,arm,1 if compact else 2)
        if not compact:row['effect_from_axis_reference']=_feature_rows(bundle,centered[state],arm,1)
        states.append(row)
    result['states']=states
    return result


def _mechanism_refs(bundle,arm):
    mechanism,_=_visible(arm)
    if not mechanism:return []
    card=bundle.get('mechanism_card',bundle['request'].get('mechanism_card')) or {}
    return sorted({source['id'] for source in card.get('sources',[])
                   if isinstance(source,dict) and isinstance(source.get('id'),str) and source['id'].strip()})


def build_plan_payload(bundle,arm):
    mechanism,_=_visible(arm)
    return {'stage':'mechanism_hypotheses','prompt_version':PROMPT_VERSION,'context':_context(bundle,arm),
            'available_evidence_refs':['C']+(['M']+_mechanism_refs(bundle,arm) if mechanism else []),
            'instruction':'Predict possible functional directions and explicit uncertainties before inspecting any code identifiers.'}


def build_local_payload(bundle,cards,plan,axes,arm):
    cards=_cards_list(cards,bundle)
    return {'stage':'local_state_preferences','context':_context(bundle,arm),'frozen_hypotheses':plan,
            'requested_axes':axes,'main_alpha':MAIN_ALPHA,'response_field':'decisions',
            'axis_overview':[_overview(c,arm) for c in cards],
            'axes':[_axis_card(bundle,cards[a],arm) for a in axes]}


def evaluate_decisions(bundle,decisions,alpha=MAIN_ALPHA,hypotheses=None,arm='full'):
    p0=np.asarray(bundle['p0'],dtype=float);effect=np.asarray(bundle['effects'],dtype=float)
    if isinstance(decisions,dict):decisions=decisions['decisions']
    if p0.ndim!=2 or effect.ndim!=3:raise ValueError('Invalid control/effect dimensions')
    axes,states=p0.shape
    if type(alpha) not in (float,int) or not math.isfinite(alpha) or not 0<=alpha<=1 or effect.shape[:2]!=p0.shape:raise ValueError('Invalid alpha/dimensions')
    std=np.asarray(bundle['std'],dtype=float)
    if std.shape!=(effect.shape[-1],) or not np.isfinite(std).all() or np.any(std<0) or not np.isfinite(effect).all():
        raise ValueError('Invalid decoder effects or gene standard deviations')
    if not np.isfinite(p0).all() or p0.min()<0 or not np.allclose(p0.sum(1),1):raise ValueError('Invalid control occupancy')
    if len(decisions)!=axes or {x['axis_id'] for x in decisions}!=set(range(axes)):raise ValueError('Need all axis decisions')
    r=p0.copy();seen=set();counts={k:0 for k in ('shift','keep','unknown')}
    for row in decisions:
        axis=row['axis_id'];decision=row['decision']
        if type(axis)is not int or axis in seen or decision not in counts:raise ValueError('Invalid axis decision')
        seen.add(axis);counts[decision]+=1;targets=row['targets']
        if decision!='shift':
            if targets:raise ValueError('Nonshift axis contains targets')
            continue
        if not 1<=len(targets)<=3:raise ValueError('Invalid target count')
        r[axis]=0.;used=set()
        for state,weight in targets:
            if type(state)is not int or state in used or not 0<=state<states or type(weight)not in (float,int) or not np.isfinite(weight) or weight<=0:
                raise ValueError('Invalid target state/weight')
            if np.asarray(bundle['train_occupancy'])[axis,state]<=0:raise ValueError('Target state lacks training support')
            used.add(state);r[axis,state]=weight
        r[axis]/=r[axis].sum()
    q=(1-alpha)*p0+alpha*r
    for row in decisions:
        if row['decision']!='shift':q[row['axis_id']]=p0[row['axis_id']]
    axis_delta=np.einsum('as,asg->ag',q-p0,effect);delta=axis_delta.sum(0)
    tv=.5*np.abs(q-p0).sum(1);std=np.maximum(np.asarray(bundle['std']),.05)
    feedback={'definition':'Exact decoder change; consistency with hypotheses is not observed treatment accuracy.',
        'alpha':alpha,'decision_counts':counts,'delta_l2':float(np.linalg.norm(delta)),
        'standardized_delta_rms':float(np.sqrt(np.mean((delta/std)**2))),
        'axis_tv':tv.tolist(),'mean_axis_tv':float(tv.mean()),
        'top_gene_changes':_feature_rows(bundle,delta,arm,10)}
    _,semantics=_visible(arm)
    if hypotheses is not None and semantics:
        genes=np.asarray(bundle['genes']).astype(str);symbols=np.asarray(bundle['symbols']).astype(str)
        checks=[]
        for h in hypotheses['hypotheses']:
            for item in h['gene_expectations']:
                positions=np.flatnonzero((genes==item['gene'])|(symbols==item['gene']))
                if len(positions)!=1:
                    checks.append({'hypothesis':h['id'],'gene':item['gene'],'status':'unmeasured_or_ambiguous'})
                else:
                    g=positions[0];observed='increase' if delta[g]>1e-8 else 'decrease' if delta[g]<-1e-8 else 'zero'
                    checks.append({'hypothesis':h['id'],'gene':item['gene'],'expected':item['direction'],
                        'decoded_direction':observed,'delta':float(delta[g]),
                        'consistent':None if item['direction']=='uncertain' else observed==item['direction']})
        feedback['hypothesis_gene_checks']=checks
    else:feedback['hypothesis_gene_checks']='Unavailable without feature semantics; anonymous IDs cannot be mapped to hypothesis gene names.'
    cancellation=np.abs(axis_delta).sum(0)-np.abs(delta)
    idx=np.argsort(-(cancellation/std),kind='stable')[:6]
    identifiers=bundle['genes'] if semantics else bundle['anonymous_gene_ids']
    feedback['cross_axis_cancellation']=[{'feature':str(identifiers[g]),'summed_delta':float(delta[g]),
                                        'absolute_axis_sum':float(np.abs(axis_delta[:,g]).sum())} for g in idx if cancellation[g]>1e-8]
    return {'q':q,'r':r,'delta_mean':delta,'axis_delta_genes':axis_delta,'feedback':feedback}


def build_joint_payload(bundle,cards,plan,local,arm):
    cards=_cards_list(cards,bundle)
    return {'stage':'joint_consistency_review','context':_context(bundle,arm),'frozen_hypotheses':plan,
        'response_field':'updates','requested_axes':list(range(len(cards))),
        'instruction':'Return only justified axis revisions; updates may be empty. Hypotheses remain fixed.',
        'main_alpha':MAIN_ALPHA,'local_decisions':local['decisions'],'decoder_feedback':local['feedback'],
        'axes':[_axis_card(bundle,c,arm,compact=True) for c in cards]}


def _content(completion):
    try:
        message=completion['choices'][0]['message']
        if message.get('tool_calls'):raise ValidationError('Return JSON text, no tool calls')
        return message['content']
    except (TypeError,KeyError,IndexError):raise ValidationError('Missing assistant text') from None


def _run_stage(folder,payload,prompt,validator,client,fingerprint,max_retries):
    folder.mkdir(parents=True,exist_ok=True)
    sf=_digest({'run':fingerprint,'payload':payload,'prompt':prompt})
    contract={'fingerprint':sf,'max_attempts':max_retries+1}
    contract_path=folder/'contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text())!=contract:raise ValueError('Stage contract changed')
    else:atomic_write_json(contract_path,contract)
    messages=[{'role':'system','content':prompt},{'role':'user','content':_json(payload)}]
    for attempt in range(1,max_retries+2):
        prefix=folder/f'attempt_{attempt:02d}';rq=prefix.with_suffix('.request.json');rp=prefix.with_suffix('.response.json');sp=prefix.with_suffix('.status.json')
        request={'stage_fingerprint':sf,'attempt':attempt,'messages':copy.deepcopy(messages),'tools':[]}
        reused=rq.exists()
        if reused:
            if json.loads(rq.read_text())!=request:raise ValueError('Cached request differs')
        else:atomic_write_json(rq,request)
        completion=None;error=None
        if rp.exists():
            completion=json.loads(rp.read_text())
            if sp.exists() and json.loads(sp.read_text()).get('response_sha256')!=file_sha256(rp):raise ValueError('Cached response changed')
        elif reused:
            error=json.loads(sp.read_text()).get('error') if sp.exists() else 'Interrupted request with unknown outcome counts against budget'
        else:
            try:
                completion=client(messages=copy.deepcopy(messages),tools=[])
                atomic_write_json(rp,completion)
            except TransportError as exc:error=str(exc)
            except Exception:error='API call failed; details omitted'
        parsed=None
        if completion is not None:
            try:parsed=validator(_content(completion))
            except ValidationError as exc:error=str(exc)
        status={'stage_fingerprint':sf,'attempt':attempt,'status':'completed' if parsed is not None else 'failed',
                'error':error,'response_sha256':file_sha256(rp) if rp.exists() else None}
        atomic_write_json(sp,status)
        if parsed is not None:return {'response':parsed,'attempts_used':attempt,'stage_fingerprint':sf,'response_sha256':status['response_sha256']}
        if completion is not None:
            try:content=_content(completion)
            except ValidationError:content='Invalid completion'
            messages.append({'role':'assistant','content':content if isinstance(content,str) else 'Invalid completion'})
        messages.append({'role':'user','content':'Format validation failed: '+str(error)+'. Correct the complete JSON. No treatment outcome has been scored.'})
    raise StageFailure(f'{folder.name} exhausted {max_retries+1} attempts; all attempts retained')


def _atomic_npz(path,**values):
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez_compressed(f,**values);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:Path(name).unlink(missing_ok=True)


def _save_prediction(output,name,bundle,decisions,response,plan,arm,fingerprint,stages):
    evaluations=[evaluate_decisions(bundle,decisions,float(a),plan,arm) for a in ALPHAS]
    main=evaluations[1];array_path=output/f'{name}.npz';json_path=output/f'{name}.json'
    document={'status':'completed','case_id':bundle['case_id'],'opaque_id':bundle.get('opaque_id'),
        'arm':arm,'stage':name,'fingerprint':fingerprint,'decisions':decisions,'response':response,
        'feedback':main['feedback'],'main_alpha':MAIN_ALPHA,'alphas':ALPHAS.tolist(),
        'hypotheses_sha256':_digest(plan),'stages':stages,'attempts_used':sum(s['attempts_used'] for s in stages),
        'arrays':str(array_path.resolve()),'is_prediction':True,'uses_measured_perturbation_response':False}
    arrays={'q':main['q'],'p0':np.asarray(bundle['p0']),'r':main['r'],'delta_mean':main['delta_mean'],
        'genes':np.asarray(bundle['genes']).astype(str),'alphas':ALPHAS,
        'q_by_alpha':np.stack([e['q'] for e in evaluations]),
        'delta_by_alpha':np.stack([e['delta_mean'] for e in evaluations])}
    if json_path.exists():
        previous=json.loads(json_path.read_text())
        if previous.get('fingerprint')!=fingerprint or previous.get('decisions')!=decisions or previous.get('response')!=response:raise ValueError('Prediction cache changed')
        if previous.get('arrays_sha256')!=file_sha256(array_path):raise ValueError('Prediction arrays changed')
        with np.load(array_path,allow_pickle=False) as z:
            if any(not np.array_equal(z[k],v) for k,v in arrays.items()):raise ValueError('Stored arrays differ from decisions')
        if previous.get('feedback')!=main['feedback']:raise ValueError('Stored feedback differs')
        return {**previous,'q':main['q'],'delta_mean':main['delta_mean']}
    _atomic_npz(array_path,**arrays);document['arrays_sha256']=file_sha256(array_path)
    atomic_write_json(json_path,document)
    return {**document,'q':main['q'],'delta_mean':main['delta_mean']}


def run_case(bundle,cards,client,output,arm='full',max_retries=2):
    mechanism,_=_visible(arm)
    if type(max_retries)is not int or not 0<=max_retries<=2:raise ValueError('max_retries must be 0,1,2')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);cards=_cards_list(cards,bundle)
    folder=Path(__file__).resolve().parent
    sources={name:file_sha256(folder/name) for name in ['pure_mechanism_agent.py','pure_mechanism_data.py','agent.py','annotate_cards.py'] if (folder/name).exists()}
    identity={'version':PROMPT_VERSION,'arm':arm,'plan_prompt':PLAN_PROMPT,'decision_prompt':DECISION_PROMPT,
        'model':getattr(client,'model','mock'),'endpoint':getattr(client,'url','mock'),
        'thinking_disabled':getattr(client,'_disable_thinking',None),'max_retries':max_retries,
        'main_alpha':MAIN_ALPHA,'alphas':ALPHAS.tolist(),'batch_size':LOCAL_BATCH_SIZE,
        'source_sha256':sources,'context_digest':_digest(_context(bundle,arm)),'cards_digest':_digest(cards),
        'array_sha256':{key:_array_digest(bundle[key]) for key in ['p0','effects','genes','symbols','std','train_occupancy','anonymous_gene_ids']},
        'case_id':bundle['case_id']}
    fingerprint=_digest(identity);contract={'fingerprint':fingerprint,'identity':identity,'provenance':_plain(bundle.get('provenance',{}))}
    contract_path=output/'run_contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text())!=contract:raise ValueError('Case inputs/source/model/budget changed; use new output')
    else:atomic_write_json(contract_path,contract)
    plan_stage=_run_stage(output/'requests/plan',build_plan_payload(bundle,arm),PLAN_PROMPT,
                         lambda text:parse_plan(text,mechanism,_mechanism_refs(bundle,arm)),client,fingerprint,max_retries)
    plan=plan_stage['response'];atomic_write_json(output/'hypotheses.json',plan)
    stage_record=lambda s:{k:s[k] for k in ('attempts_used','stage_fingerprint','response_sha256')}
    stages=[stage_record(plan_stage)];all_decisions=[];summaries=[];limitations=[]
    axes=list(range(len(bundle['p0'])));states=bundle['p0'].shape[1]
    for first in range(0,len(axes),LOCAL_BATCH_SIZE):
        batch=axes[first:first+LOCAL_BATCH_SIZE]
        result=_run_stage(output/'requests'/f'local_{first//LOCAL_BATCH_SIZE:02d}',
             build_local_payload(bundle,cards,plan,batch,arm),DECISION_PROMPT,
             lambda text,batch=batch:parse_decisions(text,batch,states,plan,False,bundle['train_occupancy']),
             client,fingerprint,max_retries)
        stages.append(stage_record(result));all_decisions.extend(result['response']['decisions'])
        summaries.append(result['response']['summary']);limitations.extend(result['response']['limitations'])
    response={'decisions':all_decisions,'summary':' | '.join(summaries),'limitations':list(dict.fromkeys(limitations))}
    local=_save_prediction(output,'local',bundle,all_decisions,response,plan,arm,fingerprint,stages)
    joint_stage=_run_stage(output/'requests/joint',build_joint_payload(bundle,cards,plan,local,arm),DECISION_PROMPT,
          lambda text:parse_decisions(text,axes,states,plan,True,bundle['train_occupancy']),client,fingerprint,max_retries)
    revised={row['axis_id']:row for row in all_decisions}
    revised.update({row['axis_id']:row for row in joint_stage['response']['updates']})
    decisions=[revised[a] for a in axes]
    joint=_save_prediction(output,'joint',bundle,decisions,joint_stage['response'],plan,arm,fingerprint,stages+[stage_record(joint_stage)])
    complete={'status':'completed','case_id':bundle['case_id'],'opaque_id':bundle.get('opaque_id'),'arm':arm,
        'fingerprint':fingerprint,'api_attempts':joint['attempts_used'],'local':str((output/'local.json').resolve()),
        'joint':str((output/'joint.json').resolve()),'local_sha256':file_sha256(output/'local.json'),
        'joint_sha256':file_sha256(output/'joint.json'),'hypotheses_sha256':file_sha256(output/'hypotheses.json')}
    atomic_write_json(output/'COMPLETE.json',complete)
    return {'local':local,'joint':joint,'complete':complete}
