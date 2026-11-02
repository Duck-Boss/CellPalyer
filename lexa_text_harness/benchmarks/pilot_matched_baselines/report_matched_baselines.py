"""Render already-scored outputs without changing sealed source or predictions.

The report subcommand in the sealed scorer uses Series.view attribute access;
this standalone presentation step uses explicit column indexing instead.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from score_baselines import verify_seal
    verify_seal()
    frame = pd.read_csv(BASE / "scores_summary.csv")
    labels = {
        "no_edit": "No edit", "literal_rule": "Literal rule", "lexa_pure_mechanism": "LEXA + DeepSeek (mechanism)",
        "expression_nearest": "Expression nearest dose", "expression_interpolate": "Expression interpolation",
        "lexa_response_retrieval": "LEXA + DeepSeek (response retrieval)", "cpa": "CPA", "codex": "CODEX"}
    frame["method"] = frame.apply(lambda r: labels[r["model"]] +
        (" / " + r["view"] if r["view"] != "existing" else ""), axis=1)
    frame = frame.sort_values("delta_rmse_mean")
    show = frame[["method", "n_seeds", "delta_rmse_mean", "delta_rmse_std", "delta_pearson_mean",
                  "absolute_r2_mean", "delta_pearson_defined_wells_min", "delta_pearson_defined_wells_max"]]
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#2878a5" if r.model in ("cpa", "codex") else "#e69f00" if r.model == "lexa_pure_mechanism" else "#888888"
              for r in frame.itertuples()]
    ax.barh(frame["method"], frame.delta_rmse_mean, xerr=frame.delta_rmse_std.fillna(0), color=colors, capsize=3)
    ax.invert_yaxis()
    ax.set_xlabel("Delta RMSE (lower is better); error bars: SD over 3 training seeds")
    ax.set_title("Matched sciPlex2 pilot: 16 development wells / 4 drugs / 5 uM / 2,000 genes")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(BASE / "comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for model, ax in zip(("cpa", "codex"), axes):
        for seed in (0, 1, 2):
            directory = BASE / "models" / model / f"seed_{seed}"
            if model == "cpa":
                curve = pd.DataFrame([json.loads(x) for x in (directory / "history.jsonl").read_text().splitlines()])
                valkey = "val_rmse"
            else:
                curve = pd.read_csv(directory / "training.csv")
                valkey = "val_rmse"
            ax.plot(curve.epoch, curve[valkey], label=f"seed {seed}", alpha=.8)
        ax.set_title(f"{model.upper()} validation RMSE")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch (CPA 1-based, CODEX 0-based)")
        ax.set_ylabel("Matched validation RMSE (log scale)")
        ax.grid(alpha=.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(BASE / "training_curves.png", dpi=150)
    plt.close(fig)
    notes = (
        "同一原实验：四药BMS/Dex/Nutlin/SAHA，5µM/24h/A549，16孔、786处理细胞、951唯一对照、2000个训练来源基因。\n"
        "CPA/CODEX为作者模型在本地数据协议下的适配，每种3个独立训练种子；旧LEXA/DeepSeek及简单参考沿用封存预测，n=1无种子标准差。\n"
        "native：模型直接预测处理表达；calibrated：模型处理预测减同模型零剂量预测，再加真实对照均值。两种锚点预先指定并全部报告。\n"
        "CODEX的calibrated零剂量预测经同药物网络分支计算，是补充诊断；该分支未专门以零剂量训练，不是作者标准的直接预测路径。\n"
        "ΔRMSE逐孔计算后药内等权、四药等权；相关性只在有定义孔计算，需同时看有效孔数；absolute R²是基因维度1-SSE/SST。\n"
        "CPA/CODEX训练可见其他剂量真实响应；纯机制主方法推理使用机制与code语义，无响应检索。比较说明相同测试集下表现，差异不能全部归因于架构或语言模型。\n"
        "此5µM数据已经用于前期开发，不是新的独立泛化测试；不是论文/公开榜单原协议的复现分数。\n"
        "CODEX三次均达到预设1000epoch上限，seed1最佳在最后一轮，不能声称完全收敛；CPA由验证早停结束。\n"
        "表达插值/响应检索属于响应监督参考；no-edit与文字规则为辅助基线。本轮没有新DeepSeek调用。\n"
        "图中误差条为3个训练种子的sample SD，不是细胞或生物学置信区间。\n"
    )
    native = frame[(frame["view"].isin(["native", "existing"])) & (frame.model != "literal_rule")]
    takeaway = (
        "已验证结果：CODEX native ΔRMSE约0.195562±0.002547，CPA native约0.242347±0.013473。\n"
        "简单表达剂量插值0.194145，CODEX未证明超过它；纯机制LEXA+DeepSeek约0.356299。\n"
        "预先指定的calibrated变体CODEX约0.279644、CPA约0.280974，两者均劣于各自native。\n\n"
    )
    (BASE / "RESULTS.txt").write_text("原sciPlex2实验的同协议模型比较\n\n" + takeaway +
        show.to_string(index=False, float_format=lambda x: f"{x:.6f}") + "\n\n" + notes)
    page = '<!doctype html><meta charset="utf-8"><title>Matched perturbation baselines</title><style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 16px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}img{max-width:100%}pre{white-space:pre-wrap}h1{font-size:26px}</style>'
    page += '<h1>原 sciPlex2 实验：CPA、CODEX 与 LEXA＋DeepSeek</h1><pre>' + html.escape(takeaway) + '</pre>'
    page += '<img src="comparison.png">' + show.to_html(index=False, float_format=lambda x: f"{x:.6f}")
    page += '<h2>训练过程</h2><img src="training_curves.png"><h2>评价口径和限制</h2><pre>' + html.escape(notes) + '</pre>'
    page += '<p><a href="scores_by_seed.csv">逐种子</a> · <a href="scores_by_drug.csv">逐药</a> · <a href="scores_by_well.csv">逐孔</a></p>'
    (BASE / "RESULTS.html").write_text(page)
    (BASE / "README.txt").write_text(
        "用户要求：在原来已经做过的5µM开发实验上重跑其他扰动模型，获得横向可比分数。\n"
        "状态：CPA/CODEX各3seed真实训练及预测完成，所有预测先全局seal再评分。独立审核见对应audit.json。\n"
        "主要结果：RESULTS.html、RESULTS.txt、scores_summary.csv。详细sample定义见protocol.json。\n"
        "复现环境：/cpfs_speech/user/hejia.hj/code/anaconda/bin/python；没有安装或改动全局依赖。\n"
        "项目根目录下命令：\n"
        "python benchmarks/pilot_matched_baselines/prepare_matched.py\n"
        "python benchmarks/pilot_matched_baselines/run_cpa.py --seed 0 --output <新目录>\n"
        "python benchmarks/pilot_matched_baselines/run_codex.py --seed 0 --output <新目录>\n"
        "两个训练脚本对seed1/2重复；线程环境OMP/OPENBLAS/MKL_NUM_THREADS=4，CPU显式指定。\n"
        "已有输出应直接审核，无需重复训练；训练脚本拒绝覆盖完成预测。\n"
        "python benchmarks/pilot_matched_baselines/score_baselines.py verify\n"
        "python benchmarks/pilot_matched_baselines/report_matched_baselines.py\n"
        "首次score_baselines.py report因pandas Series.view名称冲突失败；保留该已封存源码，独立报告脚本修正展示，不改预测或评分。\n"
    )
    print(str(BASE / "RESULTS.html"))


if __name__ == "__main__":
    main()
