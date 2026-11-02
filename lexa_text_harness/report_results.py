"""Report completed, blinded perturbation predictions from existing score CSVs.

This module never opens held-out expression, credentials, or a network connection.
Only generated evaluation_NNN.npz artifacts may be opened, to distinguish an
actually revised decoded prediction from a changed textual proposal.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


METRICS = ["delta_rmse", "delta_pearson", "top100_delta_rmse",
           "top100_direction_accuracy", "top100_abs_delta_overlap"]
ORDER = ["no_edit", "training_mean", "nearest_training_dose", "one_shot", "agent",
         "agent_no_labels", "shuffled_labels"]
DISPLAY = {"no_edit": "No edit", "training_mean": "Training mean",
           "nearest_training_dose": "Nearest training dose", "one_shot": "One shot",
           "agent": "Agent", "agent_no_labels": "Agent without labels",
           "shuffled_labels": "Shuffled labels"}


def read_json(path: Path, default=None):
    if not path.is_file():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def canonical(edits):
    return json.dumps(sorted(edits, key=lambda item: item["axis_id"]), sort_keys=True)


def generated_delta_hash(value: dict, experiment: Path):
    """Read generated predictions only; explicitly reject the truth archive."""
    artifact = value.get("artifact")
    if not artifact:
        return None
    path = Path(artifact).resolve()
    if not re.fullmatch(r"evaluation_[0-9]{3,}\.npz", path.name):
        return None
    if not path.is_relative_to(experiment.resolve()) or not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as payload:
        delta = np.asarray(payload["delta_mean"])
        return hashlib.sha256(delta.tobytes()).hexdigest()


def trace_statistics(trace: dict, experiment: Path):
    calls, successful = {}, []
    api_round = 0
    for message in trace.get("messages", []):
        if message.get("role") == "assistant":
            api_round += 1
            for call in message.get("tool_calls") or []:
                if call.get("function", {}).get("name") != "evaluate_edits":
                    continue
                try:
                    arguments = json.loads(call["function"]["arguments"])
                    calls[call["id"]] = (api_round, arguments["edits"])
                except (KeyError, ValueError, TypeError):
                    continue
        elif message.get("role") == "tool" and message.get("tool_call_id") in calls:
            try:
                value = json.loads(message.get("content", ""))
            except (TypeError, ValueError):
                continue
            if "error" in value or value.get("ok") is False:
                continue
            round_number, edits = calls[message["tool_call_id"]]
            successful.append((round_number, canonical(edits), generated_delta_hash(value, experiment)))
    changed, output_changed = 0, 0
    known_output_pairs = 0
    for previous, current in zip(successful, successful[1:]):
        if current[0] > previous[0]:
            changed += current[1] != previous[1]
            if current[2] is not None and previous[2] is not None:
                known_output_pairs += 1
                output_changed += current[2] != previous[2]
    events = trace.get("events", [])
    return {
        "successful_evaluations_in_trace": len(successful),
        "cross_api_round_proposal_revisions": changed,
        "cross_api_round_decoded_output_revisions": output_changed if known_output_pairs else None,
        "cross_api_round_output_pairs_checked": known_output_pairs,
        "invalid_or_failed_tool_calls": sum("tool" in event and not event.get("success") for event in events),
        "transport_error_events": sum("error" in event for event in events),
    }


def runtime_statistics(experiment: Path, plan: dict, manifest: dict, *, allow_incomplete=False):
    rows = []
    for method in plan["methods"]:
        for case in manifest["cases"]:
            folder = experiment / method / case["slug"]
            result = read_json(folder / "agent_result.json", {}) if allow_incomplete else read_json(folder / "agent_result.json")
            attempt_folders = sorted(path for path in (folder / "attempts").glob("*") if path.is_dir())
            attempts = attempt_folders or ([folder] if (folder / "agent_result.json").is_file() or (folder / "agent_trace.json").is_file() else [])
            stats = {"api_requests_all_attempts": 0, "decoder_evaluations_all_attempts": 0,
                     "failed_attempts": 0, "attempts_without_result": 0,
                     "cross_api_round_proposal_revisions": 0,
                     "cross_api_round_decoded_output_revisions": 0,
                     "cross_api_round_output_pairs_checked": 0,
                     "invalid_or_failed_tool_calls": 0, "transport_error_events": 0}
            for attempt in attempts:
                attempt_result = read_json(attempt / "agent_result.json", {})
                trace = read_json(attempt / "agent_trace.json", {})
                stats["attempts_without_result"] += not bool(attempt_result)
                stats["api_requests_all_attempts"] += int(attempt_result.get("api_requests", 0))
                stats["decoder_evaluations_all_attempts"] += int(attempt_result.get("evaluations", 0))
                stats["failed_attempts"] += bool(attempt_result) and (not attempt_result.get("decoder_evaluated") or attempt_result.get("status") == "transport_error")
                current = trace_statistics(trace, experiment)
                for key in ("cross_api_round_proposal_revisions", "cross_api_round_decoded_output_revisions",
                            "cross_api_round_output_pairs_checked", "invalid_or_failed_tool_calls", "transport_error_events"):
                    stats[key] += current[key] or 0
            edits = (result.get("last_evaluated_proposal") or {}).get("edits")
            evaluation = result.get("evaluation") or {}
            rows.append({"case_id": case["case_id"], "drug": case["drug"], "method": method,
                         "status": result.get("status", "unknown" if attempts else "not_started"), "attempts": len(attempts),
                         "started": bool(attempts), "result_missing": not bool(result),
                         "latest_api_requests": int(result.get("api_requests", 0)),
                         "latest_decoder_evaluations": int(result.get("evaluations", 0)),
                         "latest_prediction_failed": bool(attempts) and (not result.get("decoder_evaluated") or result.get("status") == "transport_error"),
                         "abstained_no_edits": edits == [] if edits is not None else None,
                         "final_requested_axes": len(edits) if edits is not None else None,
                         "final_changed_cells": evaluation.get("changed_cells"),
                         "final_axis_cell_changes": sum(edit.get("changed_cells", 0) for edit in evaluation.get("edits", [])),
                         "final_delta_rms_standardized": evaluation.get("standardized_delta_rms"),
                         **stats})
    return pd.DataFrame(rows)


def previous_execution(previous: Path, current: Path):
    """Audit an aborted run without reading any previous efficacy scores."""
    if previous.resolve() == current.resolve():
        raise ValueError("Previous experiment must differ from the scored experiment")
    plan = read_json(previous / "run_plan.json")
    manifest = read_json(previous / "inputs/manifest.json")
    runtime = runtime_statistics(previous, plan, manifest, allow_incomplete=True)
    columns = ["started", "result_missing", "latest_prediction_failed", "attempts",
               "api_requests_all_attempts", "decoder_evaluations_all_attempts", "failed_attempts",
               "attempts_without_result", "invalid_or_failed_tool_calls", "transport_error_events"]
    summary = runtime.groupby("method", as_index=False)[columns].sum()
    summary["planned_cases"] = runtime.groupby("method").size().reindex(summary.method).to_numpy()
    score_files = [name for name in ("scores_by_well.csv", "scores_by_condition.csv", "scores_summary.csv")
                   if (previous / name).is_file()]
    protocol = read_json(current.parent / "api_run_protocol_v2.json", {})
    previous_declared = protocol.get("previous_run")
    protocol_matches = bool(previous_declared and Path(previous_declared).resolve() == previous.resolve())
    context = {
        "experiment": str(previous.resolve()), "score_files_present": score_files,
        "included_in_current_efficacy": False,
        "planned_episodes": len(runtime), "started_episodes": int(runtime.started.sum()),
        "failed_latest_episodes": int(runtime.latest_prediction_failed.sum()),
        "api_requests_all_attempts": int(runtime.api_requests_all_attempts.sum()),
        "decoder_evaluations_all_attempts": int(runtime.decoder_evaluations_all_attempts.sum()),
        "pre_registered_restart_reason": protocol.get("reason") if protocol_matches else None,
        "target_expression_accessed_at_v2_registration": protocol.get("target_expression_accessed") if protocol_matches else None,
    }
    return runtime, summary, context


def load_scores(experiment: Path, plan: dict, manifest: dict):
    well = pd.read_csv(experiment / "scores_by_well.csv")
    condition = pd.read_csv(experiment / "scores_by_condition.csv")
    summary = pd.read_csv(experiment / "scores_summary.csv")
    required = {"case_id", "drug", "dose_uM", "method", "n_treated", "n_control", *METRICS}
    if not required <= set(well):
        raise ValueError(f"Missing score columns: {sorted(required - set(well))}")
    if well.duplicated(["case_id", "method"]).any():
        raise ValueError("Duplicate method/case score rows")
    expected_cases = {case["case_id"] for case in manifest["cases"]}
    methods = set(well["method"])
    if not {"no_edit", "training_mean", "nearest_training_dose", *plan["methods"]} <= methods:
        raise ValueError("Missing planned methods or required baselines in completed scores")
    for method in methods:
        if set(well.loc[well.method == method, "case_id"]) != expected_cases:
            raise ValueError(f"Incomplete or unexpected case population for {method}; no silent exclusion allowed")
    for metric in METRICS:
        values = pd.to_numeric(well[metric], errors="raise")
        if np.isinf(values).any() or (metric != "delta_pearson" and values.isna().any()):
            raise ValueError(f"Nonfinite scores for {metric}")
    calculated_condition = well.groupby(["method", "drug", "dose_uM"], as_index=False)[METRICS].mean()
    calculated_summary = calculated_condition.groupby("method", as_index=False)[METRICS].mean()
    for provided, calculated, index in ((condition, calculated_condition, ["method", "drug", "dose_uM"]),
                                         (summary, calculated_summary, ["method"])):
        actual, expected = provided.set_index(index).sort_index(), calculated.set_index(index).sort_index()
        if not actual.index.equals(expected.index) or not np.allclose(actual[METRICS], expected[METRICS], equal_nan=True, rtol=1e-6, atol=1e-10):
            raise ValueError("Saved score aggregation differs from equal-well/equal-condition calculation")
    order = [method for method in ORDER if method in methods] + sorted(methods - set(ORDER))
    return well, condition, summary.set_index("method").loc[order].reset_index()


def relative_comparisons(well: pd.DataFrame, condition: pd.DataFrame, summary: pd.DataFrame):
    records = []
    groups = [("macro", "all", summary)]
    groups += [("drug", str(drug), table) for drug, table in condition.groupby("drug")]
    for level, name, table in groups:
        means = table.groupby("method")[METRICS].mean()
        if "agent" not in means.index:
            continue
        for baseline in ("no_edit", "nearest_training_dose", "one_shot", "training_mean"):
            if baseline not in means.index:
                continue
            agent_rmse, baseline_rmse = means.loc["agent", "delta_rmse"], means.loc[baseline, "delta_rmse"]
            records.append({"level": level, "group": name, "reference": baseline,
                            "agent_delta_rmse": agent_rmse, "reference_delta_rmse": baseline_rmse,
                            "agent_rmse_change_percent": 100 * (agent_rmse / baseline_rmse - 1) if baseline_rmse else np.nan,
                            "agent_rmse_reduction_percent": 100 * (1 - agent_rmse / baseline_rmse) if baseline_rmse else np.nan,
                            "note": "change_percent < 0 means lower RMSE (better); reduction_percent > 0 means better"})
    return pd.DataFrame(records)


def make_plots(well, condition, summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    methods = summary["method"].tolist()
    colors = {method: color for method, color in zip(ORDER, ["#87909a", "#c4a264", "#4c87a1", "#65a982", "#9a64b2", "#b77561", "#bd82a6"])}
    labels = [DISPLAY.get(method, method) for method in methods]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = [("delta_rmse", "Expression-change RMSE (lower is better)"),
              ("delta_pearson", "Expression-change Pearson (higher is better)"),
              ("top100_direction_accuracy", "Top-100 direction accuracy (higher is better)"),
              ("top100_abs_delta_overlap", "Top-100 overlap fraction (higher is better)")]
    for ax, (metric, title) in zip(axes.flat, panels):
        for position, method in enumerate(methods):
            value = summary.loc[summary.method == method, metric].iloc[0]
            if pd.notna(value):
                ax.bar(position, value, color=colors.get(method, "#777777"), width=.65)
                ax.annotate(f"{value:.3f}", (position, value), xytext=(0, 4 if value >= 0 else -12), textcoords="offset points", ha="center", fontsize=9)
            else:
                ax.text(position, 0, "undefined", rotation=90, va="bottom", ha="center", fontsize=8)
        ax.set_xticks(range(len(methods)), labels, rotation=22, ha="right")
        ax.set_title(title, loc="left", fontsize=11)
        ax.axhline(0, color="#444444", linewidth=.7)
        ax.grid(axis="y", alpha=.16)
        ax.set_axisbelow(True)
        ax.margins(y=.16)
    fig.suptitle("Held-out 5 uM / 24 h: equal-drug macro averages\nPilot point estimates; no independent-replicate confidence interval", fontsize=13)
    fig.savefig(output / "metrics_comparison.png", dpi=180)
    plt.close(fig)

    drugs = sorted(condition.drug.unique())
    fig, axes = plt.subplots(1, len(drugs), figsize=(max(7, 3.7 * len(drugs)), 4.6), squeeze=False, constrained_layout=True)
    for ax, drug in zip(axes.flat, drugs):
        for position, method in enumerate(methods):
            scores = well[(well.drug == drug) & (well.method == method)].delta_rmse.to_numpy()
            value = condition[(condition.drug == drug) & (condition.method == method)].delta_rmse.mean()
            ax.bar(position, value, color=colors.get(method, "#777777"), width=.65)
            ax.scatter(position + np.linspace(-.13, .13, len(scores)), scores, color="#30343b", s=12, zorder=3)
        ax.set_title(str(drug))
        ax.set_xticks(range(len(methods)), labels, rotation=40, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0)
    axes[0, 0].set_ylabel("Expression-change RMSE (lower is better)")
    fig.suptitle("By drug: bars = well means; dots = all selected wells (not independent biological replicates)")
    fig.savefig(output / "drug_rmse_comparison.png", dpi=180)
    plt.close(fig)


def show_table(table):
    return table.to_string(index=False, float_format=lambda value: f"{value:.6f}", na_rep="未定义")


def conclusions(well, summary, comparisons, runtime):
    if "agent" not in set(summary.method):
        return ["本次没有 agent 评分，不能判断文字 agent 的扰动预测效果。"]
    macro = comparisons[comparisons.level == "macro"].set_index("reference")
    noedit = macro.loc["no_edit", "agent_rmse_change_percent"]
    nearest = macro.loc["nearest_training_dose", "agent_rmse_change_percent"]
    lines = [f"Agent 相对不编辑的宏平均 ΔRMSE 变化 {noedit:+.2f}%；相对最近训练剂量 {nearest:+.2f}%（负值表示改善，正值表示变差）。"]
    if noedit < 0 and nearest < 0:
        lines.append("本轮点估计支持 agent 在所选留出条件上降低表达变化预测误差；样本与实验设计不足以证明普遍有效或统计显著。")
    elif noedit < 0:
        lines.append("Agent 仅优于不编辑，未超过最近训练剂量；当前结果不支持优于简单训练响应基线。")
    else:
        lines.append("Agent 未超过不编辑；当前结果不支持这套文字编辑方案提升药物响应预测效果。")
    paired = well.pivot(index="case_id", columns="method", values="delta_rmse")
    for baseline in ("no_edit", "nearest_training_dose", "one_shot"):
        if baseline in paired:
            differences = paired.agent - paired[baseline]
            lines.append(f"逐 well 比较 {baseline}：agent 严格改善 {(differences < -1e-12).sum()}/{len(differences)}，持平 {(np.abs(differences) <= 1e-12).sum()}/{len(differences)}。全部 well 均保留。")
    agent = runtime[runtime.method == "agent"]
    revised = int((agent.cross_api_round_proposal_revisions > 0).sum())
    adjusted = int((agent.cross_api_round_decoded_output_revisions > 0).sum())
    lines.append(f"Agent {len(agent)} 个 case 中，{revised} 个在收到反馈后的后续 API round 改过提案；{adjusted} 个可核验为实际改变了解码输出。修订按相邻成功评价计算，同一 API round 批量提案不算反馈修订。")
    if revised == 0:
        lines.append("本轮未观察到跨轮提案修订，不能把 agent 与 one_shot 的差异归因为反馈带来的修订收益。")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--fitted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-experiment", type=Path,
                        help="Optional aborted previous run: disclose execution cost separately, never merge its efficacy scores")
    args = parser.parse_args(argv)
    experiment, fitted, output = args.experiment.resolve(), args.fitted.resolve(), args.output.resolve()
    plan = read_json(experiment / "run_plan.json")
    manifest = read_json(experiment / "inputs/manifest.json")
    metadata = read_json(fitted / "metadata.json")
    well, condition, summary = load_scores(experiment, plan, manifest)
    runtime = runtime_statistics(experiment, plan, manifest)
    if runtime.latest_prediction_failed.any():
        raise ValueError("A latest planned prediction failed; preserve its case and finish scoring before a completed report")
    comparisons = relative_comparisons(well, condition, summary)
    scope = read_json(experiment / "evaluation_scope.json", {})
    labels_meta = read_json(fitted / "labels.json", {}).get("meta", {})
    previous_runtime = previous_summary = previous_context = None
    if args.previous_experiment:
        previous_runtime, previous_summary, previous_context = previous_execution(args.previous_experiment.resolve(), experiment)
    connectivity = read_json(experiment.parent / "api_connectivity.json", {})
    cost_rows = [{"phase": "current_prediction", "api_requests_documented": int(runtime.api_requests_all_attempts.sum())}]
    if previous_context:
        cost_rows.append({"phase": "previous_prediction_aborted", "api_requests_documented": previous_context["api_requests_all_attempts"]})
    if isinstance(labels_meta.get("api_requests_total"), int):
        cost_rows.append({"phase": "shared_state_annotation", "api_requests_documented": labels_meta["api_requests_total"]})
    if connectivity.get("status") == "connected":
        cost_rows.append({"phase": "connectivity_success_documented", "api_requests_documented": 1})
    cost_rows.append({"phase": "documented_total", "api_requests_documented": sum(row["api_requests_documented"] for row in cost_rows)})
    api_cost = pd.DataFrame(cost_rows)
    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    by_drug = well[["case_id", "drug", "n_treated", "n_control"]].drop_duplicates().groupby("drug").agg(
        wells=("case_id", "count"), treated_cells=("n_treated", "sum"), matched_control_cell_uses=("n_control", "sum")).reset_index()
    runtime_columns = ["attempts", "api_requests_all_attempts", "decoder_evaluations_all_attempts", "failed_attempts",
                       "attempts_without_result", "cross_api_round_proposal_revisions", "cross_api_round_decoded_output_revisions",
                       "cross_api_round_output_pairs_checked", "invalid_or_failed_tool_calls", "transport_error_events",
                       "abstained_no_edits", "final_requested_axes", "final_changed_cells", "final_axis_cell_changes"]
    runtime_summary = runtime.groupby("method", as_index=False)[runtime_columns].sum()
    runtime_summary["cases"] = runtime.groupby("method").size().reindex(runtime_summary.method).to_numpy()
    interpretation = conclusions(well, summary, comparisons, runtime)
    genes = int(scope.get("evaluated_gene_count", metadata.get("n_genes", 2000)))
    doses = sorted(well.dose_uM.unique().tolist())
    sections = [
        ("本轮结果", "\n".join(interpretation)),
        ("范围与盲评", f"报告时间（Asia/Shanghai）：{generated_at}\n数据：{metadata.get('provenance', {}).get('dataset', '见模型元数据')}\n"
         f"纳入 {len(manifest['cases'])} 个预先选定 ready wells，{len(by_drug)} 种药物，剂量 {doses} µM，固定 24 小时；训练选择的 {genes} 个基因。\n"
         "预测目标：处理组相对匹配零剂量对照的平均 normalized log1p(CP10K) 表达变化。预测使用编辑后解码均值减编辑前解码均值。\n"
         "报告只读取既有评分 CSV 和预测产物，不读取留出表达。是否遵守先完成全部预测再评分，以 run_plan、原始 trace 与实验协议为审计依据；评分文件本身不能证明执行时序。\n"
         "宏平均先对每个药物/剂量内 wells 等权平均，再对药物/剂量条件等权平均。对照细胞可能跨条件复用，表中 control cell uses 不是独立细胞总数。\n" + show_table(by_drug)),
        ("主要指标（ΔRMSE 越低越好；其余三项越高越好）", show_table(summary[["method", "delta_rmse", "delta_pearson", "top100_direction_accuracy", "top100_abs_delta_overlap"]]) +
         "\nΔPearson 是基因变化量的相关性；不使用绝对表达相关性论证扰动预测。零变化的 Pearson 未定义，不能当作 0。\nTop100 为真实平均变化绝对值最大的基因，不等于统计显著差异基因；overlap 为预测与真实 Top100 交集比例。"),
        ("逐药物结果", show_table(condition[["method", "drug", "dose_uM", "delta_rmse", "delta_pearson", "top100_direction_accuracy", "top100_abs_delta_overlap"]])),
        ("Agent 相对基线的 RMSE 百分比", show_table(comparisons.drop(columns=["note"])) +
         "\nchange_percent = 100×(agent RMSE/reference RMSE−1)，负值改善，正值变差；reduction_percent 的符号相反。"),
        ("实际执行与修订", show_table(runtime_summary) +
         f"\n语言标注累计 API 请求：{labels_meta.get('api_requests_total', '未知')}；此数单独列出，不含在预测请求统计中。\n"
         "预测请求、解码与修订按全部保留 attempts 累加，根目录镜像不重复计数；latest 结果用于评分。未记录到磁盘的崩溃中请求无法据此还原。\n"
         "abstained_no_edits 表示最终成功评价的 edits=[]；非空 edits 仍可能因目标已满足或选中 0 个细胞而没有实际表达变化。\n"
         "跨 API round 输出修订只在相邻预测 artifact 都存在时核验，缺失时不会推定发生。failed attempts 保留，不按效果删 case。\n"
         "one_shot 最多 1 次解码评价；agent 最多 3 次，且每 episode API 请求上限见 run_plan。预算不匹配，因此方法间差异不能单独证明工具反馈的因果收益。"),
        ("解释与限制", "1. 本轮是所见药物的留出剂量测试，4 种药物、5 µM、24 小时的 A549 pilot；不能据此宣称未见药物、其他细胞类型或时间泛化。\n"
         "2. 仅覆盖训练选择的 2,000 个基因，不代表全转录组；采用缩小模型及共表达 scaffold，不是 LEXA 论文结果复现。\n"
         "3. author culture-well/replicate 标签不自动等于独立 biological replicates；本报告不提供把单细胞当独立重复的显著性检验或置信区间。\n"
         "4. DeepSeek 的语言标签是状态卡数值证据的解释。状态贡献以轴内均匀参考中心化；正负标签不等于绝对表达，也不等于药物造成的真实因果变化。源状态→目标状态的数值差才是编辑效应。\n"
         "5. Decoder 工具只能反馈数值变化与合法性，没有真实处理组反馈。单细胞不是同细胞前后配对，编辑比例也不是经验证的剂量模型。\n"
         "6. 本轮没有预算匹配、文字打乱/去标签等完整消融，不能把性能归因于状态语言、LLM 知识或反馈中的某个因素。\n"
         "7. 若看到本轮结果后更改词典、prompt 或 agent，再用相同留出集调参，应视为开发集，最终结论需新独立测试。"),
    ]
    if previous_context:
        scored_note = ("未生成任何评分 CSV，前轮未评分。" if not previous_context["score_files_present"] else
                       "存在评分文件，但本报告未读取前轮评分，前轮不计入效果比较。")
        pre_registered_note = ("v2 预登记记录声明：登记时尚未访问留出处理组表达。" if
                               previous_context["target_expression_accessed_at_v2_registration"] is False else
                               "未取得可匹配的未揭盲预登记标记；不能仅凭文件缺失证明未访问真值。")
        failed_cases = previous_runtime[previous_runtime.latest_prediction_failed][["method", "case_id", "status", "latest_api_requests", "latest_decoder_evaluations"]]
        previous_text = (f"前轮目录：{previous_context['experiment']}\n{scored_note}\n{pre_registered_note}\n"
                         f"前轮计划 {previous_context['planned_episodes']} 个 episodes，实际启动 {previous_context['started_episodes']} 个；"
                         f"最新预测失败 {previous_context['failed_latest_episodes']} 个；实际持久化 API 请求 {previous_context['api_requests_all_attempts']} 次，"
                         f"成功解码评价 {previous_context['decoder_evaluations_all_attempts']} 次。\n"
                         "前轮在工具执行错误后中止并保留全部 artifacts。后轮统一重跑，前轮结果不补入任何后轮效果指标，也不按前轮效果选择 case。\n"
                         f"预登记的重启原因：{previous_context['pre_registered_restart_reason'] or '未记录'}\n" +
                         show_table(previous_summary) + "\n已启动但没有有效最终预测的 case：\n" + show_table(failed_cases))
        sections.insert(1, ("前轮执行中止与重跑（仅披露执行，不纳入效果）", previous_text))
    sections.append(("跨轮可核验 API 消耗", show_table(api_cost) +
                     "\n共享32轴状态标注只计一次，未因 v2 重跑重复计入；前轮失败请求包含在 previous_prediction_aborted 中。\n"
                     "连接检查按现存一条成功记录计 1。此处是已落盘请求次数，不是 token 数或账单金额；未持久化的调用及其他会话的调用不在本报告可见范围。"))
    for candidate in (experiment / "representation_diagnostic.json", experiment.parent / "representation_diagnostic.json", fitted.parent / "representation_diagnostic.json"):
        if candidate.is_file():
            diagnostic = read_json(candidate)
            sections.append(("表示诊断（独立列示，不是 agent 预测）", f"来源：{candidate}\n"
                "此诊断把真实处理组也经过编码/解码，检查响应信号是否保留；用了测试真值，不是 agent 预测成绩，也不是数学上界。\n" +
                json.dumps(diagnostic.get("aggregate", diagnostic.get("aggregates", {})), ensure_ascii=False, indent=2)))
            break
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("scores_by_well.csv", "scores_by_condition.csv", "scores_summary.csv"):
        source, destination = experiment / filename, output / filename
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
    runtime.to_csv(output / "execution_by_case.csv", index=False)
    runtime_summary.to_csv(output / "execution_summary.csv", index=False)
    comparisons.to_csv(output / "agent_rmse_comparisons.csv", index=False)
    api_cost.to_csv(output / "api_request_costs.csv", index=False)
    if previous_runtime is not None:
        previous_runtime.to_csv(output / "previous_execution_by_case.csv", index=False)
        previous_summary.to_csv(output / "previous_execution_summary.csv", index=False)
    make_plots(well, condition, summary, output)
    text = "LEXA 文字扰动预测：首轮真实实验报告\n\n" + "\n\n".join(title + "\n" + content for title, content in sections) + "\n"
    (output / "RESULTS.txt").write_text(text, encoding="utf-8")
    body = "".join(f"<section><h2>{html.escape(title)}</h2><pre>{html.escape(content)}</pre></section>" for title, content in sections)
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>LEXA 实验报告</title><style>body{font-family:system-ui,sans-serif;max-width:1300px;margin:32px auto;padding:0 24px;line-height:1.6;color:#242932}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#f5f6f8;padding:16px;border-radius:8px}img{width:100%;height:auto}h2{margin-top:32px}</style><h1>LEXA 文字扰动预测：首轮真实实验报告</h1><img src="metrics_comparison.png" alt="Macro metrics"><img src="drug_rmse_comparison.png" alt="Drug RMSE">' + body + '</html>'
    (output / "RESULTS.html").write_text(page, encoding="utf-8")
    (output / "report_manifest.json").write_text(json.dumps({
        "generated_at": generated_at, "experiment": str(experiment), "fitted": str(fitted),
        "case_count": len(manifest["cases"]), "methods": summary.method.tolist(),
        "reads_heldout_expression": False, "conclusions": interpretation,
        "previous_execution": previous_context,
        "documented_api_requests_total": int(api_cost.iloc[-1].api_requests_documented),
        "source_score_sha256": {name: hashlib.sha256((experiment / name).read_bytes()).hexdigest()
                                for name in ["scores_by_well.csv", "scores_by_condition.csv", "scores_summary.csv"]},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "report_complete", "output": str(output), "case_count": len(manifest["cases"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
