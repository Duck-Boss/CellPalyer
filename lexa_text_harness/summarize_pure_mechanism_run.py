#!/usr/bin/env python3
"""Summarize sealed, scored pure-mechanism development results without API calls."""
from pathlib import Path
import argparse
import collections
import hashlib
import html
import json
import numpy as np
import pandas as pd


def read(p):return json.loads(Path(p).read_text())
def summarize(output):
    summary=pd.read_csv(output/'scores_summary.csv').set_index('method')
    bydrug=pd.read_csv(output/'scores_by_condition.csv')
    decisions=pd.read_csv(output/'decisions_by_well.csv')
    occ=pd.read_csv(output/'occupancy_by_well.csv')
    wells=pd.read_csv(output/'scores_by_well.csv')
    coverage={method:{'defined_wells':int(group.delta_pearson.notna().sum()),
                     'defined_drugs':int(group.loc[group.delta_pearson.notna(),'drug'].nunique()),
                     'total_wells':len(group),'total_drugs':int(group.drug.nunique())}
              for method,group in wells.groupby('method')}
    main=['full_joint_a0.3','no_mechanism_joint_a0.3','no_semantics_joint_a0.3','neither_joint_a0.3']
    noedit=float(summary.loc['no_edit','delta_rmse'])
    full=float(summary.loc[main[0],'delta_rmse'])
    result={'development_only':True,'primary_alpha':.3,
        'main_vs_no_edit_relative_rmse_reduction':(noedit-full)/noedit,
        'primary_by_arm':summary.loc[main].replace({np.nan:None}).to_dict('index'),
        'primary_mean_decision_counts':decisions[decisions.method.isin([m.replace('_a0.3','') for m in main])].groupby('method')[['shift_axes','keep_axes','unknown_axes']].mean().to_dict('index'),
        'mean_dominant_code_changes':occ[occ.method.isin(main)].groupby('method').predicted_changed_axes.mean().to_dict(),
        'semantics_rmse_gain_with_mechanism':float(summary.loc['no_semantics_joint_a0.3','delta_rmse']-full),
        'mechanism_rmse_gain_with_semantics':float(summary.loc['no_mechanism_joint_a0.3','delta_rmse']-full),
        'factorial_interaction_rmse':float(summary.loc['full_joint_a0.3','delta_rmse']-summary.loc['no_mechanism_joint_a0.3','delta_rmse']-summary.loc['no_semantics_joint_a0.3','delta_rmse']+summary.loc['neither_joint_a0.3','delta_rmse']),
        'interaction_note':'Descriptive signed interaction only, not a significance or causal mediation claim.',
        'correlation_coverage':coverage,
        'correlation_note':'Pearson/cosine summaries skip undefined correlations within drugs, then average available drug means. Full joint15/16, neither joint10/16, literal4/16(Nutlin only). Not identical comparison populations.'}
    execution={};usage=collections.Counter()
    for label,folder in [('fixed',output),('original_failed',output.parent/'experiment_v4_pure_mechanism')]:
        statuses=[read(p) for p in folder.glob('predictions/*/*/requests/*/*.status.json')]
        requests=list(folder.glob('predictions/*/*/requests/*/*.request.json'))
        responses=list(folder.glob('predictions/*/*/requests/*/*.response.json'))
        tokens=collections.Counter()
        for p in responses:
            u=read(p).get('usage',{})
            for k in ['prompt_tokens','completion_tokens','total_tokens']:tokens[k]+=int(u.get(k,0))
        usage.update(tokens)
        execution[label]={'requests_recorded':len(requests),'responses_recorded':len(responses),
            'complete_case_arms':len(list(folder.glob('predictions/*/*/COMPLETE.json'))),
            'status_counts':dict(collections.Counter(x['status'] for x in statuses)),
            'failure_reasons':dict(collections.Counter(x['error'] for x in statuses if x['status']=='failed')),
            'requests_without_status':len(requests)-len(statuses),'reported_usage':dict(tokens)}
    result['execution']=execution;result['reported_usage_total']=dict(usage)
    result['usage_note']='Provider-reported tokens; not a billing estimate. Original interrupted requests may have unrecorded usage.'
    oldpath=output.parent/'experiment_v3_hierarchy_fixed/scores_summary.csv'
    old=pd.read_csv(oldpath).set_index('method')
    result['historical_comparisons']={k:old.loc[k,['delta_rmse','delta_pearson','response_l2_ratio']].to_dict() for k in ['old_agent','hierarchical_joint','expression_interpolate']}
    result['historical_comparison_note']='v3 hierarchy and expression interpolation receive matched training perturbation responses; these are different information settings, not language-only baselines. Old agent also differs in editing space and API budget.'
    (output/'experiment_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    main_drug=bydrug[bydrug.method.isin(['no_edit','literal_keyword_a0.3','full_local_a0.3','full_joint_a0.3'])]
    text='纯机制 v4 实验解释\n\n'
    text+=f'主参数 alpha=0.3；full 相对 no-edit 的平均 ΔRMSE 降幅={100*result["main_vs_no_edit_relative_rmse_reduction"]:.3f}%。\n'
    text+='这是四个已揭盲药物条件的开发实验，不能从16孔或786细胞推断有16/786种独立干预。\n\n'
    text+='逐药主结果：\n'+main_drug.to_string(index=False,float_format=lambda x:f'{x:.6f}')+'\n\n'
    text+='四组决策（每例32轴）：\n'+json.dumps(result['primary_mean_decision_counts'],ensure_ascii=False,indent=2)+'\n\n'
    text+='解释边界：\n'
    text+='1. 只有 full 与隐藏组在同一次冻结实验中的比较，才能帮助判断机制/code文字是否有用；隐藏组允许弃权，因此若等于no-edit，应说明这是弃权结果。\n'
    text+='2. 词面匹配只识别明确的术语和方向，是弱的透明对照，不代表完整知识图谱或通路方法。\n'
    text+='3. alpha=.1/.6 复用以.3作反馈的同一组决策，只是幅度敏感性；不得从开发答案挑一个后声称是预设主结果。\n'
    text+='4. joint使用自己提出的假设作为检查依据，可能自洽却生物学错误；local/joint多一次调用，不是等预算反馈消融。\n'
    text+='5. 轴的shift数量、占用变化及主导code更换数量不同；群体边际预测不等于同一个细胞的前后转变。\n'
    text+='6. 词典训练含其他剂量；本实验限制推理输入，不证明表示或预训练LLM从未接触相关药物知识。\n'
    text+='7. answer-codec使用答案，是重建诊断，不是预测、理论最低误差或可提供给agent的提示。\n'
    text+='8. 零变化输出的Pearson/cosine未定义，以NaN/空值保留，不能记成相关性0。\n\n'
    text+='相关性汇总跳过未定义值：full joint为15/16孔、neither joint为10/16孔、literal为4/16孔且只Nutlin；相应均值不是相同样本范围，不能直接排序。RMSE覆盖全部16孔。\n\n'
    text+='具体失败模式：Nutlin固定样例正确提出p53/细胞周期停滞，却同时保留A24高增殖状态，理由甚至写明它与停滞矛盾。它可满足少数CDKN1A等基因检查，却在未被明确约束的程序上产生反向编辑。详见decision_analysis.txt；这属于预测内部矛盾诊断，不用实验答案给模型修正。\n\n'
    text+='执行与失败重试：\n'+json.dumps(execution,ensure_ascii=False,indent=2)+'\n'
    (output/'INTERPRETATION.txt').write_text(text)
    selected=['no_edit','literal_keyword_a0.3']+main+['answer_codec_diagnostic']
    display=summary.copy()
    display['correlation_defined_wells']=[str(coverage[m]['defined_wells'])+'/16' for m in display.index]
    display['correlation_defined_drugs']=[str(coverage[m]['defined_drugs'])+'/4' for m in display.index]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['No edit','Literal rule','Full','No mechanism','No semantics','Neither','Answer codec*']
    fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    colors=['#888888','#a2a2a2','#277da1','#90be6d','#f9c74f','#f9844a','#c8c8c8']
    for ax,col,title in zip(axes,['delta_rmse','delta_pearson','response_l2_ratio'],
            ['Delta RMSE: all 16 wells (lower)','Pearson: defined outputs only','Response norm / truth: all 16 wells']):
        values=summary.loc[selected,col].to_numpy();positions=np.arange(len(labels))
        ax.barh(positions,values,color=colors)
        ax.set_yticks(positions,labels);ax.set_ylim(len(labels)-.25,-.75);ax.set_title(title);ax.grid(axis='x',alpha=.2)
        if col=='delta_pearson':
            for i,m in enumerate(selected):
                note=f" n={coverage[m]['defined_wells']}/16, drugs={coverage[m]['defined_drugs']}/4"
                ax.text(float(values[i]) if np.isfinite(values[i]) else 0,i,note,va='center',fontsize=8)
            ax.set_xlim(0,1.15)
    fig.suptitle('Pure mechanism text pilot — alpha 0.3; previously unblinded development cases; *uses answers')
    fig.tight_layout();fig.savefig(output/'metrics_comparison.png',dpi=150);plt.close(fig)
    page='<!doctype html><meta charset="utf-8"><title>LEXA pure mechanism v4</title><style>body{font:16px system-ui;max-width:1300px;margin:32px auto}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ddd}pre{white-space:pre-wrap}img{max-width:100%}</style>'
    page+='<h1>LEXA 纯机制文字预测 v4</h1><p><strong>预设主方案未优于不编辑：ΔRMSE 0.356299 vs 0.355724。</strong>16个已揭盲开发孔，786处理细胞、951对照，四药5µM/24h/A549。</p>'
    page+='<img src="metrics_comparison.png"><p>主alpha=.3；.1/.6只作敏感性，不按答案选。相关性仅对非零、非恒定预测定义，下面标明有效孔/药数量；RMSE所有方法均覆盖16孔。</p>'
    page+=display.loc[selected].to_html(float_format=lambda x:f'{x:.6f}')
    page+='<h2>逐药结果</h2>'+main_drug.to_html(index=False,float_format=lambda x:f'{x:.6f}')
    page+='<h2>全部预设强度与local/joint</h2>'+display.to_html(float_format=lambda x:f'{x:.6f}')
    page+='<h2>解释与执行</h2><pre>'+html.escape(text)+'</pre>'
    page+='<p><a href="decision_analysis.txt">决策样例与矛盾</a> · <a href="independent_prediction_audit.json">独立预测审核</a> · <a href="review_scores.json">独立评分审核</a> · <a href="protocol.json">冻结协议</a></p>'
    (output/'RESULTS.html').write_text(page)
    (output/'RESULTS.txt').write_text('主alpha=.3，ΔRMSE未优于不编辑。\n\n'+display.loc[selected].to_string(float_format=lambda x:f'{x:.6f}')+'\n\n'+text)
    return {k:result[k] for k in ['main_vs_no_edit_relative_rmse_reduction','semantics_rmse_gain_with_mechanism','mechanism_rmse_gain_with_semantics','execution']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent/'runs/pilot_seen_dose/experiment_v4_pure_mechanism_fixed')
    args=p.parse_args();print(json.dumps(summarize(args.output.resolve()),indent=2))
