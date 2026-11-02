#!/usr/bin/env python3
"""Add audited interpretations to an already scored hierarchy pilot report."""
import argparse
import html
import json
from pathlib import Path
import pandas as pd


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();p=args.output
    table=pd.read_csv(p/'scores_summary.csv').set_index('method')
    execution=json.loads((p/'independent_execution_audit.json').read_text())
    score_audit=json.loads((p/'independent_score_audit.json').read_text())
    result={
        'role':'previously unblinded development data',
        'new_vs_old_agent_rmse_reduction_pct':100*(1-table.loc['hierarchical_joint','delta_rmse']/table.loc['old_agent','delta_rmse']),
        'new_vs_code_interpolation_rmse_increase_pct':100*(table.loc['hierarchical_joint','delta_rmse']/table.loc['code_interpolate','delta_rmse']-1),
        'new_vs_expression_interpolation_rmse_increase_pct':100*(table.loc['hierarchical_joint','delta_rmse']/table.loc['expression_interpolate','delta_rmse']-1),
        'candidate_selection_counts':execution['candidate_selection_counts'],
        'joint_changed_gene_predictions':execution['cases_with_joint_actual_gene_delta_changes'],
        'current_api':execution['current_api'],
        'previous_incomplete_annotation_run':execution['previous_incomplete_annotation_run'],
        'conclusion':'The new pipeline improves over the old restricted text agent but does not beat retrieval baselines. Joint review changes none of the sixteen local predictions. This pilot does not demonstrate an additional benefit from language or global review.',
        'remaining_limits':['All32 axes eligible but seven training-derived candidates per axis; not arbitrary occupancy editing.',
             'No formal GO enrichment; provisional language scope uses displayed training genes.',
             'Hierarchy, retrieval and edit space changed together relative to old agent.',
             'Unseen drug/time and independent test generalization remain untested.']}
    (p/'findings.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    paragraphs=[
        f"新版相对旧agent的ΔRMSE降低{result['new_vs_old_agent_rmse_reduction_pct']:.2f}%，但比纯code插值高{result['new_vs_code_interpolation_rmse_increase_pct']:.3f}%，比直接表达插值高{result['new_vs_expression_interpolation_rmse_increase_pct']:.2f}%。当前提升不能归因于语言推理。",
        '16组×32轴共512次局部选择，510次保留interpolate、2次选择half；两次均为BMS的axis11。其余14组与纯code插值的表达预测完全一致。联合复核在全部16组均未改变候选、占用分布或最终基因变化。',
        '本轮固定成功运行84次真实API请求：4次轴标注、64次局部选择、16次联合复核，全部首次校验通过；前一版本另有8次标注请求后因基因引用格式失败而停止，未运行药物预测。两次运行的源码、请求和失败记录均保留。',
        '独立审计核对全部输入、64份数值预测、实际API返回选择以及176行逐孔评分通过。21项针对测试通过；标注格式修复后6项相关测试再次通过。正式泛化结论仍需新的未触碰测试集。']
    start='<!-- AUDITED_FINDINGS_START -->';end='<!-- AUDITED_FINDINGS_END -->'
    report=(p/'RESULTS.html').read_text()
    if start in report:
        before,rest=report.split(start,1);report=before+rest.split(end,1)[1]
    block=start+'<h2>已核实的主要结果</h2>'+''.join('<p>'+html.escape(x)+'</p>' for x in paragraphs)+end
    report=report.replace('<img src="metrics_comparison.png">',block+'<img src="metrics_comparison.png">')
    (p/'RESULTS.html').write_text(report)
    text=(p/'RESULTS.txt').read_text().split('\n已核实的主要结果\n',1)[0]
    (p/'RESULTS.txt').write_text(text+'\n已核实的主要结果\n'+'\n'.join(paragraphs)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k.endswith('_pct')},ensure_ascii=False))


if __name__=='__main__':
    main()
