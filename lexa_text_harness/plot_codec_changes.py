"""Plot observed perturbation code occupancy and frozen codec diagnostics.

Reads only analyze_codec_changes.py's exported tables/arrays. No model fitting,
LLM calls, credentials, or raw held-out expression are used here.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import html
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


DRUGS = ["BMS", "Dex", "Nutlin", "SAHA"]
COLORS = ["#266c94", "#c27436", "#548e64"]


def correlation(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else np.nan


def vector_metrics(actual, decoded):
    length = np.linalg.norm(actual)
    return {"mean_vector_pearson": correlation(actual, decoded),
            "mean_vector_l2_retention": float(np.linalg.norm(decoded) / length) if length > 0 else np.nan,
            "mean_vector_projection_slope": float(np.dot(actual, decoded) / length ** 2) if length > 0 else np.nan,
            "mean_vector_delta_rmse": float(np.sqrt(np.square(actual - decoded).mean()))}


def format_number(value, digits=3):
    return f"{float(value):.{digits}f}" if pd.notna(value) else "未定义"


def load_inputs(directory):
    occupancy = pd.read_csv(directory / "condition_state_occupancy.csv")
    axes = pd.read_csv(directory / "condition_axis_changes.csv")
    metrics = pd.read_csv(directory / "condition_codec_metrics.csv")
    case_metrics = pd.read_csv(directory / "case_codec_metrics.csv")
    with np.load(directory / "condition_gene_deltas.npz", allow_pickle=False) as archive:
        genes = {key: archive[key].copy() for key in archive.files}
    for table, required in [(occupancy, {"condition_id", "drug", "dose_uM", "axis_id", "state_id", "control_fraction", "treated_fraction", "delta_fraction", "n_wells"}),
                            (axes, {"condition_id", "drug", "dose_uM", "axis_id", "total_variation", "axis_effect_l2"}),
                            (metrics, {"condition_id", "drug", "dose_uM", "n_wells"})]:
        if not required <= set(table):
            raise ValueError(f"Missing analysis columns: {sorted(required - set(table))}")
    if occupancy.duplicated(["condition_id", "axis_id", "state_id"]).any():
        raise ValueError("Duplicate condition-axis-state rows")
    if axes.duplicated(["condition_id", "axis_id"]).any() or metrics.duplicated("condition_id").any():
        raise ValueError("Duplicate condition-axis or condition-metric rows")
    if not np.allclose(occupancy.treated_fraction - occupancy.control_fraction, occupancy.delta_fraction, atol=1e-7):
        raise ValueError("Occupancy delta differs from treatment minus control")
    groups = occupancy.groupby(["condition_id", "axis_id"])
    for name in ("control_fraction", "treated_fraction"):
        if not np.allclose(groups[name].sum().to_numpy(), 1.0, atol=1e-5):
            raise ValueError("Each axis must contain a probability distribution over all states")
    if set(occupancy.axis_id) != set(range(32)) or set(occupancy.state_id) != set(range(16)):
        raise ValueError("This diagnostic plot expects 32 categorical axes and 16 states per axis")
    if groups.size().ne(16).any():
        raise ValueError("Missing state rows within an axis")
    needed = {"condition_ids", "drug", "dose", "genes", "symbols", "observed_delta", "decoded_delta", "axis_gene_delta"}
    if not needed <= set(genes):
        raise ValueError(f"Missing gene arrays: {sorted(needed - set(genes))}")
    if set(genes["condition_ids"].astype(str)) != set(metrics.condition_id):
        raise ValueError("Gene arrays and condition tables have different populations")
    return occupancy, axes, metrics, case_metrics, genes


def condition_for_drug(table, drug, dose=5.):
    selected = table[(table.drug == drug) & np.isclose(table.dose_uM, dose)]
    if len(selected) == 0 or selected.condition_id.nunique() != 1:
        raise ValueError(f"Expected one {drug} condition at {dose:g} uM")
    return selected


def matrix_for_condition(occupancy, condition_id, value):
    return occupancy[occupancy.condition_id == condition_id].pivot(index="axis_id", columns="state_id", values=value).reindex(index=range(32), columns=range(16)).to_numpy()


def top_axes_for_drug(axes, drug):
    return condition_for_drug(axes, drug).sort_values(["total_variation", "axis_id"], ascending=[False, True]).head(2)


def plot_axis_tv(occupancy, axes, output, plt):
    values = np.stack([condition_for_drug(axes, drug).set_index("axis_id").reindex(range(32)).total_variation.to_numpy() for drug in DRUGS])
    fig, ax = plt.subplots(figsize=(15, 3.5), constrained_layout=True)
    heat = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0)
    ax.set_xticks(range(32), [str(axis) for axis in range(32)])
    labels = [f"{drug} (n={int(condition_for_drug(occupancy, drug).n_wells.iloc[0])} wells)" for drug in DRUGS]
    ax.set_yticks(range(4), labels)
    for row, drug in enumerate(DRUGS):
        for axis in top_axes_for_drug(axes, drug).axis_id:
            from matplotlib.patches import Rectangle
            ax.add_patch(Rectangle((axis-.48, row-.48), .96, .96, fill=False, edgecolor="white", linewidth=1.4))
    ax.set_xlabel("Axis ID (categorical; zero-based)")
    ax.set_title("Observed 5 uM / 24 h: code occupancy change by axis\nTV of equal-well mean state distributions; white boxes mark each drug's top two axes", loc="left")
    fig.colorbar(heat, ax=ax, label="Total variation = 0.5 sum |p_treated - p_control|")
    fig.savefig(output / "01_axis_tv_5uM.png", dpi=180)
    plt.close(fig)


def plot_delta_occupancy(occupancy, output, plt):
    matrices = [matrix_for_condition(occupancy, condition_for_drug(occupancy, drug).condition_id.iloc[0], "delta_fraction") for drug in DRUGS]
    maximum = max(float(np.max(np.abs(value))) for value in matrices) or 1.
    fig, axs = plt.subplots(1, 4, figsize=(16, 9.5), constrained_layout=True, sharey=True)
    for ax, drug, values in zip(axs, DRUGS, matrices):
        heat = ax.imshow(values, aspect="auto", cmap="RdBu_r", vmin=-maximum, vmax=maximum)
        ax.set_xticks(range(16), [str(state) for state in range(16)], fontsize=7)
        ax.set_yticks(range(32), [str(axis) for axis in range(32)], fontsize=8)
        ax.set_title(drug)
        ax.set_xlabel("State ID (categorical)")
    axs[0].set_ylabel("Axis ID (categorical)")
    fig.colorbar(heat, ax=axs, shrink=.65, label="Treated fraction - control fraction (same scale in all panels)")
    fig.suptitle("Observed 5 uM / 24 h: signed population occupancy shifts\nRed = more cells occupying this state; blue = fewer. This is not a gene up/down map.")
    fig.savefig(output / "02_state_occupancy_delta_5uM.png", dpi=180)
    plt.close(fig)


def plot_state_bars(occupancy, axes, output, plt):
    fig, axs = plt.subplots(4, 2, figsize=(16, 14), constrained_layout=True)
    positions = np.arange(16)
    for row, drug in enumerate(DRUGS):
        for col, (_, axis_row) in enumerate(top_axes_for_drug(axes, drug).iterrows()):
            axis = int(axis_row.axis_id)
            table = condition_for_drug(occupancy, drug)
            table = table[table.axis_id == axis].set_index("state_id").reindex(range(16))
            ax = axs[row, col]
            ax.bar(positions-.18, table.control_fraction, .36, label="Matched control", color="#87909a")
            ax.bar(positions+.18, table.treated_fraction, .36, label="Observed treatment", color="#c46c43")
            ax.set_xticks(positions, [str(value) for value in positions], fontsize=8)
            ax.set_ylim(0, min(1., max(table.control_fraction.max(), table.treated_fraction.max()) * 1.22 + .01))
            ax.set_xlabel("All 16 categorical state IDs")
            ax.set_ylabel("Fraction of cells")
            ax.set_title(f"{drug} | Axis {axis} | TV={axis_row.total_variation:.3f} | gene-effect L2={axis_row.axis_effect_l2:.3f}\nEqual-well means, n={int(table.n_wells.iloc[0])} wells", fontsize=10)
            ax.grid(axis="y", alpha=.15)
            ax.set_axisbelow(True)
    axs[0, 0].legend(frameon=False, ncol=2, fontsize=9)
    fig.suptitle("Observed 5 uM / 24 h: two axes with largest occupancy TV per drug\nPopulation proportions; individual cell transitions are not measured. TV size does not establish gene-effect size.")
    fig.savefig(output / "03_top_axes_state_distributions_5uM.png", dpi=180)
    plt.close(fig)


def plot_dose_trends(occupancy, axes, metrics, output, plt, directory):
    case_axes = pd.read_csv(directory / "case_axis_changes.csv") if (directory / "case_axis_changes.csv").is_file() else None
    case_occupancy = pd.read_csv(directory / "case_state_occupancy.csv") if (directory / "case_state_occupancy.csv").is_file() else None
    eligibility = pd.read_csv(directory / "condition_eligibility.csv") if (directory / "condition_eligibility.csv").is_file() else None
    fig, axs = plt.subplots(4, 2, figsize=(16, 15), constrained_layout=True)
    for row, drug in enumerate(DRUGS):
        conditions = metrics[metrics.drug == drug].sort_values("dose_uM")
        if eligibility is not None:
            missing = eligibility[(eligibility.drug == drug) & ~eligibility.condition_id.isin(conditions.condition_id)].copy()
            if len(missing):
                missing["n_wells"] = missing.n_ready_wells
                missing["split"] = "not_ready"
                conditions = pd.concat([conditions, missing], ignore_index=True).sort_values("dose_uM")
        doses = conditions.dose_uM.to_numpy(float)
        positions = np.log10(1 + doses)
        top_axes = top_axes_for_drug(axes, drug).axis_id.astype(int).tolist()
        state_table = condition_for_drug(occupancy, drug)
        for col in range(2):
            ax = axs[row, col]
            if len(positions) > 1:
                edges = np.r_[positions[0]-(positions[1]-positions[0])/2, (positions[:-1]+positions[1:])/2, positions[-1]+(positions[-1]-positions[-2])/2]
            else:
                edges = np.array([positions[0]-.1, positions[0]+.1])
            tick_labels=[]
            for index, (_, condition) in enumerate(conditions.iterrows()):
                heldout = "test" in str(condition.get("split", "")) or np.isclose(condition.dose_uM, 5.)
                missing = condition.n_wells == 0
                ax.axvspan(edges[index], edges[index+1], facecolor="#ffffff" if missing else ("#fae5e6" if heldout else "#f0f2f3"), edgecolor="#cccccc" if missing else "none", linewidth=0, alpha=.6, zorder=0, hatch="///" if missing else None)
                tick_labels.append(f"{condition.dose_uM:g}{'*' if heldout else ''}\nn={int(condition.n_wells)}")
            for index, axis in enumerate(top_axes):
                if col == 0:
                    table = axes[(axes.drug == drug) & (axes.axis_id == axis)].set_index("condition_id").reindex(conditions.condition_id)
                    values = table.total_variation.to_numpy()
                    label = f"Axis {axis}"
                    individual = case_axes[(case_axes.drug == drug) & (case_axes.axis_id == axis)] if case_axes is not None else None
                    individual_metric = "total_variation"
                else:
                    state = int(state_table[state_table.axis_id == axis].assign(magnitude=lambda value: value.delta_fraction.abs()).sort_values(["magnitude", "state_id"], ascending=[False, True]).state_id.iloc[0])
                    table = occupancy[(occupancy.drug == drug) & (occupancy.axis_id == axis) & (occupancy.state_id == state)].set_index("condition_id").reindex(conditions.condition_id)
                    values = table.delta_fraction.to_numpy()
                    label = f"Axis {axis}, state {state}"
                    individual = case_occupancy[(case_occupancy.drug == drug) & (case_occupancy.axis_id == axis) & (case_occupancy.state_id == state)] if case_occupancy is not None else None
                    individual_metric = "delta_fraction"
                if individual is not None and {"plate", "replicate"} <= set(individual):
                    for _, well in individual.groupby(["plate", "replicate"]):
                        well = well.sort_values("dose_uM")
                        ax.plot(np.log10(1 + well.dose_uM.to_numpy(float)), well[individual_metric], color=COLORS[index], alpha=.16, linewidth=.8, zorder=1)
                ax.plot(positions, values, "o-", color=COLORS[index], label=label, linewidth=1.6, markersize=5)
            ax.set_xticks(positions, tick_labels, fontsize=8)
            ax.set_xlabel("Dose (uM); horizontal coordinate = log10(1 + dose/uM)")
            ax.set_ylabel("Axis total variation" if col == 0 else "State occupancy delta")
            ax.set_title(f"{drug}: {'axis distribution shift' if col == 0 else 'largest-shift states at 5 uM'}", fontsize=11)
            ax.axhline(0, color="#777777", linewidth=.6)
            ax.grid(axis="y", alpha=.15)
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Observed matched-control dose trends at 24 h (descriptive; lines only guide the eye)\nAxes/states selected at 5 uM. Pale lines = available wells; pink = held-out; gray = seen dose; hatching = no ready wells.")
    fig.savefig(output / "04_observed_dose_trends.png", dpi=180)
    plt.close(fig)


def plot_gene_deltas(metrics, genes, output, plt):
    lookup = {str(value): index for index, value in enumerate(genes["condition_ids"])}
    fig, axs = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    rows=[]
    for drug, ax in zip(DRUGS, axs.flat):
        condition = condition_for_drug(metrics, drug).iloc[0]
        index = lookup[condition.condition_id]
        actual, decoded = genes["observed_delta"][index], genes["decoded_delta"][index]
        values = vector_metrics(actual, decoded)
        rows.append({"condition_id":condition.condition_id,"drug":drug,"dose_uM":float(condition.dose_uM),"n_wells":int(condition.n_wells), **values})
        ax.scatter(actual, decoded, s=9, alpha=.28, color="#41799b", edgecolor="none", rasterized=True)
        lower, upper = min(actual.min(), decoded.min()), max(actual.max(), decoded.max())
        padding = max((upper-lower)*.06, .01)
        limits = lower-padding, upper+padding
        ax.plot(limits, limits, linestyle="--", color="#777777", linewidth=.8, label="Perfect retention")
        ax.axhline(0, color="#aaaaaa", linewidth=.5)
        ax.axvline(0, color="#aaaaaa", linewidth=.5)
        ax.set_xlim(limits); ax.set_ylim(limits); ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{drug} | n={int(condition.n_wells)} wells | {len(actual)} selected genes", fontsize=10)
        ax.set_xlabel("Observed treatment-control mean delta")
        ax.set_ylabel("Decoded treatment-control mean delta")
        ax.text(.04,.96,f"Mean-vector r = {format_number(values['mean_vector_pearson'])}\nL2 retention = {format_number(values['mean_vector_l2_retention'])}\nProjection slope = {format_number(values['mean_vector_projection_slope'])}", transform=ax.transAxes, va="top", fontsize=9, bbox={"facecolor":"white","alpha":.8,"edgecolor":"none"})
    fig.suptitle("Observed 5 uM / 24 h: gene response retained through E -> discrete codes -> D\nTrue treated cells are encoded here. This is a codec diagnostic, not a perturbation prediction.")
    fig.savefig(output / "05_gene_delta_retention_5uM.png", dpi=180)
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_top_genes(metrics, genes, output, plt):
    lookup = {str(value): index for index, value in enumerate(genes["condition_ids"])}
    symbols = genes["symbols"].astype(str)
    gene_ids = genes["genes"].astype(str)
    fig, axs = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)
    top_rows=[]
    for drug, ax in zip(DRUGS, axs.flat):
        condition = condition_for_drug(metrics, drug).iloc[0]
        index = lookup[condition.condition_id]
        actual, decoded = genes["observed_delta"][index], genes["decoded_delta"][index]
        selected = np.argsort(-np.abs(actual), kind="stable")[:12]
        selected = selected[np.argsort(actual[selected])]
        positions = np.arange(len(selected))
        names=[symbols[gene] if symbols[gene] not in ("", "nan", "None") else gene_ids[gene] for gene in selected]
        ax.barh(positions-.18, actual[selected], height=.36, color="#87909a", label="Observed")
        ax.barh(positions+.18, decoded[selected], height=.36, color="#c46c43", label="Decoded")
        ax.set_yticks(positions, names, fontsize=9)
        ax.axvline(0, color="#555555", linewidth=.7)
        ax.set_xlabel("Treatment-control delta in normalized log1p(CP10K)")
        ax.set_title(f"{drug}: 12 largest absolute observed gene changes", fontsize=11)
        for gene in selected:
            top_rows.append({"drug":drug,"dose_uM":5.,"gene_id":gene_ids[gene],"symbol":symbols[gene],"observed_delta":float(actual[gene]),"decoded_delta":float(decoded[gene])})
    axs[0,0].legend(frameon=False)
    fig.suptitle("Observed 5 uM / 24 h: signed gene changes before and after compression\nGenes selected descriptively by observed absolute response; no DE significance test.")
    fig.savefig(output / "06_top_signed_gene_deltas_5uM.png", dpi=180)
    plt.close(fig)
    return pd.DataFrame(top_rows)


def text_table(table):
    return table.to_string(index=False, float_format=lambda value:f"{value:.5f}", na_rep="未定义")


def create_report(directory, output, occupancy, axes, metrics, case_metrics, genes, mean_vector, top_genes):
    top_rows=[]
    for drug in DRUGS:
        for _, axis in top_axes_for_drug(axes, drug).iterrows():
            distribution=condition_for_drug(occupancy, drug)
            distribution=distribution[distribution.axis_id==axis.axis_id]
            increase=distribution.loc[distribution.delta_fraction.idxmax()]
            decrease=distribution.loc[distribution.delta_fraction.idxmin()]
            top_rows.append({"drug":drug,"axis_id":int(axis.axis_id),"TV_of_mean_occupancy":axis.total_variation,
                "mean_well_TV":axis.get("mean_well_total_variation",np.nan),
                "axis_gene_effect_L2":axis.axis_effect_l2,"state_increase":int(increase.state_id),"increase_fraction":increase.delta_fraction,
                "state_decrease":int(decrease.state_id),"decrease_fraction":decrease.delta_fraction})
    top_axes=pd.DataFrame(top_rows)
    heldout_cases=case_metrics[np.isclose(case_metrics.dose_uM,5.)]
    metric_names=[name for name in ["delta_pearson","response_l2_ratio","response_projection_slope","delta_rmse","direction_agreement","top100_direction_agreement"] if name in heldout_cases]
    mean_well=heldout_cases.groupby("drug",as_index=False)[metric_names].mean()
    mean_well=mean_well.rename(columns={name:"mean_well_"+name for name in metric_names})
    comparison=mean_vector.merge(mean_well,on="drug",how="left")
    narrative=[]
    dose_observations=[]
    for drug in DRUGS:
        row=mean_vector[mean_vector.drug==drug].iloc[0]
        axes_row=top_axes[top_axes.drug==drug]
        descriptive="、".join(f"轴{int(item.axis_id)}（TV={item.TV_of_mean_occupancy:.3f}）" for _,item in axes_row.iterrows())
        narrative.append(f"{drug} 5 µM：占用变化最大的两个轴为{descriptive}；孔等权平均基因变化向量经过编解码后 Pearson={format_number(row.mean_vector_pearson)}，L2幅度保留={format_number(row.mean_vector_l2_retention)}，沿真实响应方向的投影系数={format_number(row.mean_vector_projection_slope)}。")
        for _, top in axes_row.iterrows():
            axis, state = int(top.axis_id), int(top.state_increase)
            series=occupancy[(occupancy.drug==drug)&(occupancy.axis_id==axis)&(occupancy.state_id==state)].sort_values("dose_uM")
            peak=series.loc[series.delta_fraction.idxmax()]
            last=series.iloc[-1]
            if last.dose_uM > peak.dose_uM and peak.delta_fraction-last.delta_fraction > .2:
                last_axis=axes[(axes.drug==drug)&(axes.axis_id==axis)&np.isclose(axes.dose_uM,last.dose_uM)].iloc[0]
                dose_observations.append(f"{drug} 轴{axis}状态{state}：相对匹配对照的占用差在{peak.dose_uM:g} µM为{peak.delta_fraction:+.3f}，到{last.dose_uM:g} µM为{last.delta_fraction:+.3f}；该高剂量轴TV仍为{last_axis.total_variation:.3f}。单一状态的占用并不随剂量一直增加，其他状态可能承担高剂量响应；这里只描述观测，不推定单细胞转移或剂量因果关系。")
    additive_error=float(np.max(np.abs(genes["axis_gene_delta"].sum(axis=1)-genes["decoded_delta"])))
    generated=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    eligibility = pd.read_csv(directory / "condition_eligibility.csv") if (directory / "condition_eligibility.csv").is_file() else None
    eligibility_text = ("\n完整条件可用性（无ready孔的条件只列缺失，不计算响应指标）：\n"+text_table(eligibility) if eligibility is not None else "")
    sections=[("这次检查了什么", "直接比较真实处理细胞与匹配对照的离散code占用，再把两组都送入同一个冻结LEXA编解码器，检查真实基因变化有多少保留下来。这里没有DeepSeek生成或修改code，也不评价LLM的扰动预测。\n"+
        f"来源：{directory}\n报告时间（Asia/Shanghai）：{generated}\n固定24小时，A549，4种药物，32轴×16状态，训练选出的{len(genes['genes'])}个基因。优先看留出的5 µM；其余剂量仅用于观察数据中的剂量趋势。\n"
        f"复现绘图命令：python plot_codec_changes.py --analysis {directory} --output {output}"),
        ("5 µM的可核验观察", "\n".join(narrative)),
        ("全剂量中的状态变化", "\n".join(dose_observations) if dose_observations else "所展示状态的全剂量变化见图；本报告不根据曲线判断因果剂量关系。"),
        ("各剂量样本范围", text_table(metrics[[name for name in ["drug","dose_uM","split","seen_in_training","n_wells","n_control","n_treated","training_wells","validation_wells","test_wells"] if name in metrics]].sort_values(["drug","dose_uM"]))+eligibility_text),
        ("变化最大的轴与状态", text_table(top_axes)+"\n轴排名按5 µM条件的TV选择，属于描述性展示。TV_of_mean_occupancy先孔等权平均占用分布再算TV；mean_well_TV先逐孔算TV再平均，两者不同。state_increase/decrease表示群体占用比例增减，不表示同一个细胞的转移，也不直接表示基因上调/下调。"),
        ("编解码保留：两种聚合方式分开看", text_table(comparison)+
        "\nmean_vector：先对每个well的基因变化向量等权平均，再计算一次相关和幅度；散点图使用此定义。mean_well：先逐well计算metric，再对well等权平均，与原表示诊断口径对齐。两者一般不相等，不能相互替代。\n"
        "L2 retention=||decoded delta||₂/||observed delta||₂，只看幅度；接近1仍可能方向错误，需结合Pearson和projection slope。projection slope=<observed,decoded>/||observed||₂²。\n"
        f"各轴基因效应相加与总decoded delta的一致性最大绝对误差：{additive_error:.3g}。"),
        ("解释图时要注意", "1. 所有状态编号都是类别，没有大小、剂量或时间顺序；折线只连接同一轴/状态在各剂量的观测，不把状态编号当连续坐标。\n"
        "2. 占用变化：先算每个well中16种状态的比例，再孔等权平均；TV=0.5×Σ|p处理−p对照|。群体占用变化不能恢复单个细胞的前后转移，所以不画Sankey。\n"
        "3. TV大表示分布变得多，不能直接说明基因效应更大；gene-effect L2需单独查看，轴间贡献可能叠加或抵消。\n"
        "4. 正负占用变化不是基因表达上调/下调。基因响应必须经解码器的源/目标状态贡献差及群体占用共同计算。文字状态名只是数值证据的解释。\n"
        "5. 5 µM在词典训练时留出；其余剂量的训练/验证细胞已用于模型拟合或选择。全剂量图不构成模型跨剂量泛化成绩，连线也不是拟合的剂量响应模型。\n"
        "各剂量可用ready well集合和细胞样本数可能不同，趋势也受这些变化影响；不能据单调线断言剂量因果关系或编辑fraction已经校准。浅色well轨迹按可用plate/replicate连接，不代表同一个细胞前后配对。\n"
        "6. 这次检查编码了真实处理组，属于观察性编解码诊断，既不是药物预测，也不是编辑器最优效果的数学上界。\n"
        "7. 单细胞测量没有同一细胞前后配对；culture well标签不自动证明独立biological replicate。本报告不把单细胞当作独立重复做显著性检验。\n"
        "8. 只分析2000个训练选择基因与pilot缩小架构；重建或响应相关良好不证明状态具有因果机制，不能推广为全转录组或论文复现。\n"
        "9. 看过这些真实留出响应后再改词典或prompt会消耗测试集；后续效果结论需新的独立测试。")]
    figures=[("01_axis_tv_5uM.png","先看哪些轴的状态分布改变：四种药物×32轴；各轴TV由孔等权平均占用计算。白框为各药物的前两个轴。"),
        ("02_state_occupancy_delta_5uM.png","再看哪些状态变多/变少：每个药物展示32×16状态，共用同一正负色标；颜色是比例差，不是基因表达方向。"),
        ("03_top_axes_state_distributions_5uM.png","每种药物的前两个轴展示全部16状态，对照与真实处理组并排；同时标TV及基因效应L2。"),
        ("04_observed_dose_trends.png","描述性剂量趋势：相对各自matched control的TV和状态占用差；横轴log10(1+剂量/µM)，每点标well数。浅色线为可用plate/replicate轨迹；粉红为留出5 µM，灰色为已见剂量。轴/状态由5 µM变化选择，不用于预测评分。"),
        ("05_gene_delta_retention_5uM.png","真实基因变化与编解码后变化的散点；每点一个基因。标的是孔等权平均变化向量的相关、幅度比和投影，不是逐孔metric均值。"),
        ("06_top_signed_gene_deltas_5uM.png","每药物真实变化幅度最大的12个基因，保留正负方向比较原始响应与重建响应；不是统计显著差异基因名单。")]
    (output/"RESULTS.txt").write_text("真实扰动前后code及编解码趋势\n\n"+"\n\n".join(title+"\n"+content for title,content in sections)+"\n\n图文件\n"+"\n".join(name+"："+caption for name,caption in figures)+"\n",encoding="utf-8")
    body="".join(f"<section><h2>{html.escape(title)}</h2><pre>{html.escape(content)}</pre></section>" for title,content in sections[:2])
    body+="".join(f'<figure><a href="{name}"><img src="{name}" alt="{html.escape(caption)}"></a><figcaption>{html.escape(caption)}</figcaption></figure>' for name,caption in figures)
    body+="".join(f"<section><h2>{html.escape(title)}</h2><pre>{html.escape(content)}</pre></section>" for title,content in sections[2:])
    style="body{font-family:system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;line-height:1.6;color:#24303c}figure{margin:28px 0}img{width:100%;height:auto;border:1px solid #edf0f4}figcaption{color:#52616e;font-size:14px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;background:#f4f6f8;padding:18px;border-radius:8px}h2{margin-top:32px}"
    (output/"RESULTS.html").write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>真实扰动code与编解码诊断</title><style>'+style+'</style><h1>真实扰动前后code及编解码趋势</h1>'+body+'</html>',encoding="utf-8")
    top_axes.to_csv(output/"top_axes_5uM.csv",index=False)
    comparison.to_csv(output/"codec_metrics_two_aggregations_5uM.csv",index=False)
    top_genes.to_csv(output/"top_signed_genes_5uM.csv",index=False)
    (output/"report_manifest.json").write_text(json.dumps({"generated_at":generated,"analysis":str(directory),"kind":"observed_codec_diagnostic_not_llm_prediction","reads_raw_heldout_expression":False,"n_genes":len(genes["genes"]),"n_conditions":len(metrics),"axis_additivity_max_abs_error":additive_error,"observations":narrative},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return narrative


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis","--input",type=Path,required=True,help="Directory exported by analyze_codec_changes.py")
    parser.add_argument("--output",type=Path,help="Report directory (default: analysis/report)")
    args=parser.parse_args(argv)
    directory=args.analysis.resolve(); output=(args.output or directory/"report").resolve()
    occupancy,axes,metrics,case_metrics,genes=load_inputs(directory)
    output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.facecolor":"white","savefig.facecolor":"white"})
    plot_axis_tv(occupancy,axes,output,plt)
    plot_delta_occupancy(occupancy,output,plt)
    plot_state_bars(occupancy,axes,output,plt)
    plot_dose_trends(occupancy,axes,metrics,output,plt,directory)
    mean_vector=plot_gene_deltas(metrics,genes,output,plt)
    top_genes=plot_top_genes(metrics,genes,output,plt)
    observations=create_report(directory,output,occupancy,axes,metrics,case_metrics,genes,mean_vector,top_genes)
    print(json.dumps({"status":"plots_complete","output":str(output),"observations":observations},ensure_ascii=False))


if __name__=="__main__":
    main()
