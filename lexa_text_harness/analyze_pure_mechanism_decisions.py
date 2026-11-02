#!/usr/bin/env python3
"""Describe completed pure-mechanism decisions without treatment truth or scores.

Reads the input manifest only for case/drug identity, then prediction artifacts
and the exact code cards shown in successful local requests. No decoder source,
expression data, scoring artifact or network client is loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import numpy as np


ARMS=('full','no_mechanism','no_semantics','neither')
DECISIONS=('shift','keep','unknown')
TOLERANCE=1e-12


class Reader:
    def __init__(self,root):
        self.root=Path(root).resolve();self.paths=set()

    def allow(self,path):
        path=Path(path).resolve()
        if not path.is_relative_to(self.root):raise ValueError(f'Artifact outside experiment: {path}')
        relative=path.relative_to(self.root)
        if relative!=Path('inputs/manifest.json') and relative.parts[0]!='predictions':
            raise ValueError(f'Analysis does not read scoring/data artifacts: {relative}')
        self.paths.add(str(path));return path

    def json(self,path,expected_sha256=None):
        path=self.allow(path);raw=path.read_bytes()
        if expected_sha256 and hashlib.sha256(raw).hexdigest()!=expected_sha256:
            raise ValueError(f'Artifact checksum differs: {path}')
        return json.loads(raw)

    def arrays(self,document):
        path=self.allow(document['arrays'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=document['arrays_sha256']:
            raise ValueError(f'Prediction checksum differs: {path}')
        with np.load(path,allow_pickle=False) as z:
            return {k:z[k].copy() for k in ('q','p0','delta_mean','genes')}


def summarize(values):
    a=np.asarray(values,dtype=float)
    return {'mean':float(a.mean()),'minimum':float(a.min()),'maximum':float(a.max())}


def direction_checks(feedback):
    checks=feedback.get('hypothesis_gene_checks')
    if not isinstance(checks,list):
        return {'available':False,'reason':str(checks),'directional_checks':0,
                'consistent':0,'contradicting':0,'unmeasured_or_ambiguous':0,
                'uncertain':0,'contradictions':[]}
    out={'available':True,'directional_checks':0,'consistent':0,'contradicting':0,
         'unmeasured_or_ambiguous':0,'uncertain':0,'contradictions':[]}
    for c in checks:
        if c.get('status')=='unmeasured_or_ambiguous':out['unmeasured_or_ambiguous']+=1
        elif c.get('expected') not in ('increase','decrease'):out['uncertain']+=1
        else:
            out['directional_checks']+=1
            if c.get('consistent') is True:out['consistent']+=1
            else:
                out['contradicting']+=1
                out['contradictions'].append({**c,'contradiction_kind':
                    'no_predicted_change' if c.get('decoded_direction')=='zero' else 'opposite_direction'})
    return out


def _check_key(c):
    return (c['hypothesis'],c['gene'],c['expected'])


def stage_summary(document,arrays,plan):
    rows=document['decisions'];counts=Counter(r['decision'] for r in rows)
    q=arrays['q'];p0=arrays['p0']
    changed=np.any(np.abs(q-p0)>TOLERANCE,axis=1)
    hypotheses={h['id']:h for h in plan['hypotheses']}
    support=Counter()
    for r in rows:
        if r['decision']!='shift':continue
        linked=[hypotheses[ref] for ref in r['evidence_refs'] if ref in hypotheses]
        if not linked:support['no_hypothesis_reference']+=1
        elif any(h['direction'] in ('increase','decrease') for h in linked):support['some_directional_hypothesis_reference']+=1
        else:support['only_uncertain_hypothesis_references']+=1
    return {'axis_counts':{d:counts[d] for d in DECISIONS},
            'shift_hypothesis_support':{key:support[key] for key in ('some_directional_hypothesis_reference','only_uncertain_hypothesis_references','no_hypothesis_reference')},
            'shift_axes':[r['axis_id'] for r in rows if r['decision']=='shift'],
            'keep_axes':[r['axis_id'] for r in rows if r['decision']=='keep'],
            'unknown_axes':[r['axis_id'] for r in rows if r['decision']=='unknown'],
            'numerically_changed_axes':np.flatnonzero(changed).tolist(),
            'n_numerically_changed_axes':int(changed.sum()),
            'mean_axis_tv':float(np.mean(.5*np.abs(q-p0).sum(1))),
            'predicted_delta_l2':float(np.linalg.norm(arrays['delta_mean'])),
            'predicted_delta_rms':float(np.sqrt(np.mean(arrays['delta_mean']**2))),
            'hypothesis_gene_checks':direction_checks(document['feedback'])}


def displayed_cards(reader,episode_dir):
    cards={}
    for folder in sorted((episode_dir/'requests').glob('local_*')):
        valid=[]
        for status_path in sorted(folder.glob('attempt_*.status.json')):
            status=reader.json(status_path)
            if status.get('status')=='completed':valid.append(status_path)
        if len(valid)!=1:raise ValueError(f'Expected one successful local request: {folder}')
        request=reader.json(valid[0].with_name(valid[0].name.replace('.status.','.request.')))
        payload=json.loads(request['messages'][1]['content'])
        cards.update({card['axis_id']:card for card in payload['axes']})
    return cards


def occupancy_preference_diagnostics(document,arrays,cards):
    """Compare selections to the numeric prior actually shown; never to response."""
    axes=[]
    for row in document['decisions']:
        if row['decision']!='shift':continue
        axis=row['axis_id'];state_rows=cards[axis]['states']
        occupancy=np.asarray([s['train_occupancy'] for s in state_rows],dtype=float)
        occupancy/=occupancy.sum()
        order=np.argsort(-occupancy,kind='stable');top3=set(order[:3].tolist())
        selected=[s for s,_ in row['targets']]
        # Competition ranks include ties at the displayed rounded frequency.
        ranks=[1+int(np.sum(occupancy>occupancy[s]+1e-12)) for s in selected]
        p0=arrays['p0'][axis];q=arrays['q'][axis]
        before=float(.5*np.abs(p0-occupancy).sum());after=float(.5*np.abs(q-occupancy).sum())
        axes.append({'axis_id':axis,'selected_states':selected,
            'train_frequency_ranks':ranks,'train_top3_states':sorted(top3),
            'selected_in_train_top3':sum(s in top3 for s in selected),
            'all_selected_in_train_top3':all(s in top3 for s in selected),
            'control_to_train_tv':before,'prediction_to_train_tv':after,
            'tv_to_train_change':after-before})
    total=sum(len(a['selected_states']) for a in axes)
    return {'n_shift_axes':len(axes),'n_selected_states':total,
        'n_selected_in_train_top3':sum(a['selected_in_train_top3'] for a in axes),
        'n_axes_all_selected_in_train_top3':sum(a['all_selected_in_train_top3'] for a in axes),
        'n_axes_move_toward_train_prior':sum(a['tv_to_train_change'] < -1e-5 for a in axes),
        'n_axes_move_away_from_train_prior':sum(a['tv_to_train_change'] > 1e-5 for a in axes),
        'mean_selected_train_frequency_rank':float(np.mean([rank for a in axes for rank in a['train_frequency_ranks']])) if total else None,
        'mean_tv_to_train_change':float(np.mean([a['tv_to_train_change'] for a in axes])) if axes else None,
        'train_prior_precision':'Uses the five-decimal train occupancies shown to the model; normalized to sum one.',
        'per_axis':axes}


def selected_state_examples(document,cards):
    result=[]
    for row in document['decisions']:
        if row['decision']!='shift':continue
        axis=row['axis_id'];card=cards[axis]
        states={s['state_id']:s for s in card['states']}
        weight_sum=sum(float(weight) for _,weight in row['targets'])
        choices=[]
        for state,weight in row['targets']:
            s=states[state]
            choices.append({'state_id':state,'normalized_preference_weight':float(weight)/weight_sum,
                'state_name':s.get('name'),'state_summary':s.get('summary'),
                'train_occupancy':s['train_occupancy'],
                'displayed_effect_from_control':s['effect_from_control'],
                'displayed_effect_from_axis_reference':s.get('effect_from_axis_reference')})
        result.append({'axis_id':axis,'scope':card.get('scope'),
                       'reason':row['reason'],'evidence_refs':row['evidence_refs'],
                       'state_choices':choices})
    return result


def hypothesis_conflicts(plan):
    directions=defaultdict(lambda:defaultdict(list))
    for h in plan['hypotheses']:
        for gene in h['gene_expectations']:
            if gene['direction'] in ('increase','decrease'):
                directions[gene['gene']][gene['direction']].append(h['id'])
    return [{'gene':gene,'increase_hypotheses':v['increase'],'decrease_hypotheses':v['decrease']}
            for gene,v in directions.items() if v['increase'] and v['decrease']]


def analyze(run,require_complete=False):
    reader=Reader(run);manifest=reader.json(reader.root/'inputs/manifest.json')
    by_public={c.get('opaque_id',c['public_case_id']):c for c in manifest['cases']}
    episodes=[];details={};complete_paths=sorted((reader.root/'predictions').glob('*/*/COMPLETE.json'))
    for path in complete_paths:
        complete=reader.json(path)
        if complete.get('status')!='completed':continue
        arm=complete['arm'];opaque=complete['opaque_id'];case=by_public[opaque]
        local=reader.json(complete['local'],complete['local_sha256'])
        joint=reader.json(complete['joint'],complete['joint_sha256'])
        plan=reader.json(path.parent/'hypotheses.json',complete['hypotheses_sha256'])
        la=reader.arrays(local);ja=reader.arrays(joint)
        if local['arm']!=arm or joint['arm']!=arm or local['case_id']!=case['case_id'] or joint['case_id']!=case['case_id']:
            raise ValueError('Completed episode identity mismatch')
        if len(local['decisions'])!=32 or len(joint['decisions'])!=32:raise ValueError('Expected 32 axes')
        if not np.array_equal(la['p0'],ja['p0']):raise ValueError('Local/joint control differs')
        ls=stage_summary(local,la,plan);js=stage_summary(joint,ja,plan)
        cards=displayed_cards(reader,path.parent)
        ls['occupancy_preference']=occupancy_preference_diagnostics(local,la,cards)
        js['occupancy_preference']=occupancy_preference_diagnostics(joint,ja,cards)
        ld={r['axis_id']:r for r in local['decisions']};jd={r['axis_id']:r for r in joint['decisions']}
        dq=ja['q']-la['q'];dd=ja['delta_mean']-la['delta_mean']
        changed=np.flatnonzero(np.any(np.abs(dq)>TOLERANCE,axis=1)).tolist()
        semantic_changed=[a for a in ld if (ld[a]['decision'],ld[a]['targets'])!=(jd[a]['decision'],jd[a]['targets'])]
        lc={_check_key(c) for c in ls['hypothesis_gene_checks']['contradictions']}
        jc={_check_key(c) for c in js['hypothesis_gene_checks']['contradictions']}
        transitions=Counter(ld[a]['decision']+'->'+jd[a]['decision'] for a in ld)
        episode={'opaque_id':opaque,'case_id':case['case_id'],'drug':case['drug'],'arm':arm,
            'api_attempts':complete['api_attempts'],'local':ls,'joint':js,
            'joint_revision':{'returned_update_rows':len(joint['response']['updates']),
                'decision_or_raw_target_changed_axes':semantic_changed,'q_changed_axes':changed,
                'n_q_changed_axes':len(changed),'any_q_changed':bool(changed),
                'q_l1_difference':float(np.abs(dq).sum()),'predicted_delta_difference_l2':float(np.linalg.norm(dd)),
                'predicted_delta_difference_rms':float(np.sqrt(np.mean(dd**2))),
                'any_predicted_delta_changed':bool(np.any(np.abs(dd)>TOLERANCE)),
                'decision_transitions':dict(transitions),'resolved_internal_contradictions':len(lc-jc),
                'new_internal_contradictions':len(jc-lc),'persistent_internal_contradictions':len(lc&jc)},
            'contradictory_hypotheses':hypothesis_conflicts(plan)}
        episodes.append(episode);details[(arm,opaque)]=(path.parent,local,joint,plan)
    expected=len(manifest['cases'])*len(ARMS)
    identities={(e['arm'],e['opaque_id']) for e in episodes}
    if len(identities)!=len(episodes):raise ValueError('Duplicate episode')
    missing=[{'arm':arm,'opaque_id':opaque} for arm in ARMS for opaque in by_public if (arm,opaque) not in identities]
    if require_complete and missing:raise ValueError(f'Only {len(episodes)}/{expected} complete; no final diagnostic written')
    groups=[]
    for drug in dict.fromkeys(c['drug'] for c in manifest['cases']):
        for arm in ARMS:
            group=[e for e in episodes if e['drug']==drug and e['arm']==arm]
            if not group:continue
            record={'drug':drug,'arm':arm,'n_episodes':len(group),'stages':{}}
            for stage in ('local','joint'):
                checks=[e[stage]['hypothesis_gene_checks'] for e in group]
                frequency=Counter(a for e in group for a in e[stage]['shift_axes'])
                record['stages'][stage]={'axis_counts':{d:summarize([e[stage]['axis_counts'][d] for e in group]) for d in DECISIONS},
                    'shift_hypothesis_support':{key:summarize([e[stage]['shift_hypothesis_support'][key] for e in group])
                        for key in ('some_directional_hypothesis_reference','only_uncertain_hypothesis_references','no_hypothesis_reference')},
                    'n_numerically_changed_axes':summarize([e[stage]['n_numerically_changed_axes'] for e in group]),
                    'predicted_delta_l2':summarize([e[stage]['predicted_delta_l2'] for e in group]),
                    'mean_axis_tv':summarize([e[stage]['mean_axis_tv'] for e in group]),
                    'shift_axis_frequency':{str(a):frequency[a] for a in sorted(frequency)},
                    'hypothesis_checks_available_episodes':sum(c['available'] for c in checks),
                    'directional_hypothesis_gene_checks':sum(c['directional_checks'] for c in checks),
                    'consistent_hypothesis_gene_checks':sum(c['consistent'] for c in checks),
                    'contradicting_hypothesis_gene_checks':sum(c['contradicting'] for c in checks),
                    'unmeasured_or_ambiguous_hypothesis_genes':sum(c['unmeasured_or_ambiguous'] for c in checks)}
                prefs=[e[stage]['occupancy_preference'] for e in group]
                selected=sum(p['n_selected_states'] for p in prefs)
                shifted=sum(p['n_shift_axes'] for p in prefs)
                record['stages'][stage]['occupancy_preference']={
                    key:sum(p[key] for p in prefs) for key in ('n_shift_axes','n_selected_states','n_selected_in_train_top3',
                        'n_axes_all_selected_in_train_top3','n_axes_move_toward_train_prior','n_axes_move_away_from_train_prior')}
                record['stages'][stage]['occupancy_preference'].update({
                    'fraction_selected_in_train_top3':sum(p['n_selected_in_train_top3'] for p in prefs)/selected if selected else None,
                    'mean_selected_train_frequency_rank':sum((p['mean_selected_train_frequency_rank'] or 0)*p['n_selected_states'] for p in prefs)/selected if selected else None,
                    'mean_tv_to_train_change':sum((p['mean_tv_to_train_change'] or 0)*p['n_shift_axes'] for p in prefs)/shifted if shifted else None})
            record['joint_revision']={'episodes_with_update_rows':sum(e['joint_revision']['returned_update_rows']>0 for e in group),
                'episodes_with_actual_q_change':sum(e['joint_revision']['any_q_changed'] for e in group),
                'episodes_with_actual_delta_change':sum(e['joint_revision']['any_predicted_delta_changed'] for e in group),
                'n_q_changed_axes':summarize([e['joint_revision']['n_q_changed_axes'] for e in group]),
                'delta_difference_l2':summarize([e['joint_revision']['predicted_delta_difference_l2'] for e in group]),
                'resolved_internal_contradictions':sum(e['joint_revision']['resolved_internal_contradictions'] for e in group),
                'new_internal_contradictions':sum(e['joint_revision']['new_internal_contradictions'] for e in group)}
            groups.append(record)
    ablation_examples=[]
    for arm in ARMS[1:]:
        for case in manifest['cases']:
            opaque=case.get('opaque_id',case['public_case_id']);key=(arm,opaque)
            if key not in details:continue
            _,local,joint,plan=details[key]
            ablation_examples.append({'arm':arm,'opaque_id':opaque,'drug':case['drug'],
                'selection_rule':'First completed case in manifest order; first three shifted axes',
                'frozen_hypotheses':plan,
                'joint_shift_decision_examples':[r for r in joint['decisions'] if r['decision']=='shift'][:3],
                'joint_review_summary':joint['response']['summary']})
            break
    representatives=[];selected_drugs=set()
    for case in manifest['cases']:
        opaque=case.get('opaque_id',case['public_case_id']);key=('full',opaque)
        if case['drug'] in selected_drugs or key not in details:continue
        selected_drugs.add(case['drug']);folder,local,joint,plan=details[key]
        cards=displayed_cards(reader,folder)
        representatives.append({'drug':case['drug'],'opaque_id':opaque,'case_id':case['case_id'],
            'selection_rule':'First completed full-arm case in fixed input manifest order for this drug',
            'frozen_hypotheses':plan,'local_shift_choices':selected_state_examples(local,cards),
            'joint_shift_choices':selected_state_examples(joint,cards),
            'joint_top_gene_changes':joint['feedback']['top_gene_changes'],
            'local_internal_direction_checks':direction_checks(local['feedback']),
            'joint_internal_direction_checks':direction_checks(joint['feedback']),
            'joint_review_summary':joint['response']['summary'],
            'joint_review_limitations':joint['response']['limitations']})
    return {'created_asia_shanghai':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'status':'complete' if not missing else 'partial','completed_episodes':len(episodes),'expected_episodes':expected,
        'analysis_scope':'Prediction-only diagnostics; no treatment expressions, true responses or score artifacts read',
        'main_alpha':.3,'numerical_change_threshold':TOLERANCE,'missing_episodes':missing,
        'limitations':['Agreement with a model-generated hypothesis is internal consistency, not biological or predictive accuracy.',
            'Hypothesis gene checks are deliberately unavailable when code semantics are hidden.',
            'Displayed state effects are sparse full-state-minus-control decoder contributions; final mixed predictions use alpha=0.3.',
            'Axis counts do not measure the number of true biological programs changed; axes may overlap in genes.',
            'A positive unknown count denotes missing evidence; numerically unknown retains control occupancy.',
            'Source labels direct/secondary are the model\'s claims; this diagnostic does not validate their biological evidence.',
            'Citing a directional hypothesis does not establish that a selected axis or state implements it; reference coverage is a weaker diagnostic.',
            'No formal functional enrichment or pathway consistency score is computed here.'],
        'groups':groups,'episodes':episodes,'representative_full_cases':representatives,
        'representative_ablation_choices':ablation_examples,'read_paths':sorted(reader.paths)}


def _effects_text(effects):
    parts=[]
    for direction,sign in (('up','↑'),('down','↓')):
        for gene in effects.get(direction,[]):
            parts.append(f"{gene.get('symbol',gene['feature'])}{sign}{gene['delta']:+.4g}")
    return '、'.join(parts) or '展示列表为空'


def render(result):
    lines=[f"纯机制预测：决策与内部一致性分析（{result['status']}）",result['created_asia_shanghai'],
        f"已完成 {result['completed_episodes']}/{result['expected_episodes']} case×arm；主强度 alpha=0.30。",
        '仅读取预测与原始请求；没有读取真实处理表达或评分。以下方向一致性不等于实验准确率。','',
        '各组平均轴数：shift / keep / unknown；每个case共32轴。']
    for g in result['groups']:
        def counts(stage):
            return '/'.join(f"{g['stages'][stage]['axis_counts'][d]['mean']:.2f}" for d in DECISIONS)
        jr=g['joint_revision']
        lines.append(f"{g['drug']} | {g['arm']} | n={g['n_episodes']} | local {counts('local')} | joint {counts('joint')} | 联合复核实际改变q {jr['episodes_with_actual_q_change']}/{g['n_episodes']}例、平均{jr['n_q_changed_axes']['mean']:.2f}轴")
    lines+=['','shift轴所引假设覆盖（平均轴数；引用不等于机制证据成立）：']
    for g in result['groups']:
        s=g['stages']['joint']['shift_hypothesis_support']
        lines.append(f"{g['drug']} | {g['arm']} | joint引用方向性假设 {s['some_directional_hypothesis_reference']['mean']:.2f}；只引uncertain假设 {s['only_uncertain_hypothesis_references']['mean']:.2f}；未引任何假设 {s['no_hypothesis_reference']['mean']:.2f}")
    lines+=['','冻结假设的基因方向检查（同一gene可出现在不同假设中，逐条计数）：']
    for g in result['groups']:
        a=g['stages']['local'];b=g['stages']['joint']
        if not b['hypothesis_checks_available_episodes']:continue
        lines.append(f"{g['drug']} | {g['arm']} | local不一致 {a['contradicting_hypothesis_gene_checks']}/{a['directional_hypothesis_gene_checks']}；joint不一致 {b['contradicting_hypothesis_gene_checks']}/{b['directional_hypothesis_gene_checks']}；joint未建模或名称不唯一 {b['unmeasured_or_ambiguous_hypothesis_genes']}条")
    lines+=['','是否只是偏好训练高频状态（joint；按实际展示的训练占用计算）：']
    for g in result['groups']:
        p=g['stages']['joint']['occupancy_preference']
        if not p['n_selected_states']:continue
        lines.append(f"{g['drug']} | {g['arm']} | 所选state属于训练top3比例 {p['fraction_selected_in_train_top3']:.1%}；平均频率排名 {p['mean_selected_train_frequency_rank']:.2f}/16；编辑后靠近/远离训练先验的轴 {p['n_axes_move_toward_train_prior']}/{p['n_axes_move_away_from_train_prior']}，共{p['n_shift_axes']}个shift轴")
    for r in result['representative_full_cases']:
        lines+=['',f"固定样例：{r['drug']} / {r['opaque_id']}（按manifest顺序，未按效果挑选）"]
        for h in r['frozen_hypotheses']['hypotheses']:
            genes=', '.join(g['gene']+':'+g['direction'] for g in h['gene_expectations']) or '未指定基因'
            lines.append(f"  {h['id']} {h['program']} → {h['direction']}；模型分类{h['basis']}；{genes}")
        lines.append('  joint选择（下列基因效应为单state相对当前对照的展示值，尚未乘0.30）：')
        if not r['joint_shift_choices']:lines.append('    无shift轴。')
        for choice in r['joint_shift_choices']:
            lines.append(f"    A{choice['axis_id']}: {choice['reason']}")
            for state in choice['state_choices']:
                lines.append(f"      C{state['state_id']} 权重{state['normalized_preference_weight']:.3f} {state['state_name']}；{_effects_text(state['displayed_effect_from_control'])}")
        lines.append('  最终预测的较大基因变化：'+_effects_text(r['joint_top_gene_changes']))
        for c in r['joint_internal_direction_checks']['contradictions']:
            lines.append(f"  内部不一致：{c['hypothesis']} {c['gene']} 假设{c['expected']}，解码{c['decoded_direction']}，Δ={c['delta']:+.6g}")
        lines.append('  联合复核说明：'+r['joint_review_summary'])
    lines+=['','消融组固定样例（显示其决策理由，不代表这些理由科学上成立）：']
    for r in result['representative_ablation_choices']:
        lines.append(f"{r['arm']} / {r['opaque_id']}")
        for h in r['frozen_hypotheses']['hypotheses']:
            lines.append(f"  {h['id']} {h['program']} → {h['direction']}")
        for d in r['joint_shift_decision_examples']:
            lines.append(f"  A{d['axis_id']} {d['targets']} / {d['evidence_refs']}: {d['reason']}")
        lines.append('  '+r['joint_review_summary'])
    lines+=['','解释限制：']+[f'- {x}' for x in result['limitations']]
    return '\n'.join(lines)+'\n'


def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
    finally:Path(name).unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,help='Output directory; default experiment directory')
    parser.add_argument('--require-complete',action='store_true')
    args=parser.parse_args();result=analyze(args.run,args.require_complete)
    output=args.output or args.run
    atomic_text(output/'decision_analysis.json',json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    atomic_text(output/'decision_analysis.txt',render(result))
    print(json.dumps({k:result[k] for k in ('status','completed_episodes','expected_episodes')},ensure_ascii=False))


if __name__=='__main__':main()
