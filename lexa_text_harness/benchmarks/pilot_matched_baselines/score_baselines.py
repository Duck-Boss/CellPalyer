"""Seal CPA/CODEX predictions, then compare them on the frozen existing pilot."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
WORK = ROOT / "runs/pilot_seen_dose"


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def contract():
    p = read(BASE / "protocol.json")
    c = read(BASE / "data/prediction_cases.json")
    assert sha(BASE / "data/prediction_cases.json") == p["prediction_cases_sha256"]
    for src in p["inputs"].values():
        assert sha(src["path"]) == src["sha256"]
    with np.load(p["inputs"]["prepared.npz"]["path"], allow_pickle=False) as z:
        genes = z["genes"].astype(str)
    assert hashlib.sha256("\n".join(genes).encode()).hexdigest() == c["genes_sha256"]
    return p, c, genes


def references():
    """Repackage existing sealed deltas without inspecting biological targets."""
    _, cases, genes = contract()
    v4 = WORK / "experiment_v4_pure_mechanism_fixed"
    v3 = WORK / "experiment_v3_hierarchy_fixed"
    sources = [v4 / "prediction_seal.json", v3 / "prediction_seal.json",
               v4 / "inputs/manifest.json", v3 / "inputs/manifest.json"]
    a = {c["case_id"]: c for c in read(v4 / "inputs/manifest.json")["cases"]}
    b = {c["case_id"]: c for c in read(v3 / "inputs/manifest.json")["cases"]}
    methods = ["no_edit", "literal_rule", "lexa_pure_mechanism",
               "expression_nearest", "expression_interpolate", "lexa_response_retrieval"]
    delta = np.zeros((len(methods), 16, len(genes)), dtype=np.float64)
    for i, case in enumerate(cases["cases"]):
        cid = case["case_id"]
        for method, path in [
            ("literal_rule", v4 / "baselines" / a[cid]["slug"] / "literal_keyword.npz"),
            ("lexa_pure_mechanism", v4 / "predictions/full" / a[cid]["slug"] / "joint.npz"),
        ]:
            sources.append(path)
            with np.load(path, allow_pickle=False) as z:
                assert np.array_equal(z["genes"].astype(str), genes)
                idx = np.flatnonzero(np.isclose(z["alphas"], .3))
                assert len(idx) == 1
                delta[methods.index(method), i] = z["delta_by_alpha"][idx[0]]
        inp = Path(b[cid]["npz"])
        sources.append(inp)
        with np.load(inp, allow_pickle=False) as z:
            assert np.array_equal(z["genes"].astype(str), genes)
            delta[methods.index("expression_nearest"), i] = z["nearest_observed_delta"]
            delta[methods.index("expression_interpolate"), i] = z["interpolate_observed_delta"]
        path = v3 / "predictions" / b[cid]["slug"] / "joint.npz"
        sources.append(path)
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z["genes"].astype(str), genes)
            delta[methods.index("lexa_response_retrieval"), i] = z["delta_mean"]
    assert np.isfinite(delta).all()
    folder = BASE / "references"
    folder.mkdir(exist_ok=True)
    path = folder / "predictions.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            assert np.array_equal(z["delta"], delta)
            assert np.array_equal(z["model_names"], methods)
    else:
        np.savez_compressed(path, case_ids=np.array([c["case_id"] for c in cases["cases"]]),
                            model_names=np.array(methods), genes=genes, delta=delta)
    document = {"created_at": now(), "sources": {str(p.resolve()): sha(p) for p in sources},
                "main_alpha": .3, "new_api_calls": 0,
                "note": "Existing frozen predictions only; no new fitting or test target access."}
    if not (folder / "provenance.json").exists():
        save(folder / "provenance.json", document)
    else:
        assert read(folder / "provenance.json")["sources"] == document["sources"]
    print(json.dumps({"status": "historical_predictions_materialized", "models": methods, "cases": 16}))


def check_predictions(path, ids, genes):
    with np.load(path, allow_pickle=False) as z:
        assert np.array_equal(z["case_ids"].astype(str), ids), path
        assert np.array_equal(z["genes"].astype(str), genes), path
        for key in ("pred_treated_mean", "pred_control_mean"):
            assert z[key].shape == (len(ids), len(genes)) and np.isfinite(z[key]).all(), (path, key)


def seal():
    protocol, cases, genes = contract()
    ids = np.array([c["case_id"] for c in cases["cases"]])
    for model in protocol["models"]:
        for seed in protocol["training_seeds"]:
            folder = BASE / "models" / model / f"seed_{seed}"
            check_predictions(folder / "predictions.npz", ids, genes)
            assert (folder / "best.pt").is_file()
    for name, digest in read(BASE / "references/provenance.json")["sources"].items():
        assert sha(name) == digest
    old = BASE / "prediction_seal.json"
    if old.exists():
        verify_seal()
        return
    assert not list(BASE.glob("scores*.csv")), "Scores existed before all-model seal"
    files = {}
    for directory in (BASE / "models", BASE / "references"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".log"):
                files[str(path.relative_to(BASE))] = sha(path)
    for path in [BASE / "protocol.json", BASE / "data/prediction_cases.json",
                 BASE / "run_cpa.py", BASE / "run_codex.py", BASE / "prepare_matched.py",
                 BASE / "score_baselines.py"]:
        files[str(path.relative_to(BASE))] = sha(path)
    source_files = {}
    for path in sorted((BASE / "vendor").rglob("*.py")):
        source_files[str(path.relative_to(BASE))] = sha(path)
    for path in sorted((ROOT / "literature/tdc_leaderboard/CODEX-master/codex").glob("*.py")):
        source_files[str(path.resolve())] = sha(path)
    save(old, {"status": "all_six_model_predictions_sealed", "sealed_at": now(),
               "score_exists_at_seal": False, "files": files, "source_files": source_files,
               "n_fitted_models": 6, "n_prediction_cases_per_model": 16})
    print(json.dumps({"status": "all_six_models_sealed", "n_files": len(files)}))


def verify_seal():
    s = read(BASE / "prediction_seal.json")
    assert s["status"] == "all_six_model_predictions_sealed" and not s["score_exists_at_seal"]
    for key in ("files", "source_files"):
        for name, digest in s[key].items():
            p = Path(name)
            if not p.is_absolute():
                p = BASE / p
            assert sha(p) == digest, str(p)
    return s


def metrics(delta, true, treated, control):
    predicted = control + delta
    residual = predicted - treated
    denom = np.sum((treated - treated.mean()) ** 2)
    std_ok = np.std(delta) > 1e-12 and np.std(true) > 1e-12
    top = np.argsort(-np.abs(true))[:100]
    return {
        "delta_rmse": float(np.sqrt(np.mean((delta - true) ** 2))),
        "delta_pearson": float(np.corrcoef(delta, true)[0, 1]) if std_ok else np.nan,
        "absolute_rmse": float(np.sqrt(np.mean(residual ** 2))),
        "absolute_r2": float(1 - np.sum(residual ** 2) / denom) if denom > 1e-15 else np.nan,
        "response_l2_ratio": float(np.linalg.norm(delta) / np.linalg.norm(true)),
        "top100_delta_rmse": float(np.sqrt(np.mean((delta[top] - true[top]) ** 2))),
        "top100_direction_accuracy": float(np.mean(np.sign(delta[top]) == np.sign(true[top]))),
    }


def score():
    verify_seal()
    protocol, cases, genes = contract()
    evaluation = Path(protocol["prepared_dir"]) / "evaluation.npz"
    assert sha(evaluation) == protocol["evaluation_archive_sha256"]
    # First target-value access in the new model comparison, after all six seals.
    with np.load(evaluation, allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    assert np.array_equal(genes, data["genes"].astype(str))
    original = {c["case_id"]: c for c in read(Path(protocol["prepared_dir"]) / "cases.json")["cases"]}
    cache = {}
    for model in protocol["models"]:
        for seed in protocol["training_seeds"]:
            with np.load(BASE / "models" / model / f"seed_{seed}" / "predictions.npz", allow_pickle=False) as z:
                cache[model, seed] = (z["pred_treated_mean"].astype(float), z["pred_control_mean"].astype(float))
    with np.load(BASE / "references/predictions.npz", allow_pickle=False) as z:
        refs = {str(name): z["delta"][j].astype(float) for j, name in enumerate(z["model_names"])}
    rows = []
    for i, case in enumerate(cases["cases"]):
        cid = case["case_id"]
        tx = data["x_test"][data["test_case_ids"].astype(str) == cid].astype(float)
        ctrl = data["control_x"][np.isin(data["control_cell_ids"].astype(str), case["control_cell_ids"])].astype(float)
        assert (len(tx), len(ctrl)) == (original[cid]["n_treated"], original[cid]["n_control"])
        t, c = tx.mean(0), ctrl.mean(0)
        true = t - c
        all_predictions = []
        for (model, seed), (pt, pc) in cache.items():
            all_predictions.extend([(model, seed, "native", pt[i] - c),
                                    (model, seed, "calibrated", pt[i] - pc[i])])
        all_predictions.extend((model, -1, "existing", delta[i]) for model, delta in refs.items())
        for model, seed, view, delta in all_predictions:
            rows.append({"model": model, "seed": seed, "view": view, "case_id": cid,
                         "drug": case["drug"], "n_treated": len(tx), "n_control": len(ctrl),
                         **metrics(delta, true, t, c)})
    frame = pd.DataFrame(rows)
    cols = list(metrics(np.ones(2), np.ones(2), np.array([1., 2.]), np.array([0., 1.])).keys())
    keys = ["model", "seed", "view"]
    per_drug = frame.groupby(keys + ["drug"], as_index=False)[cols].mean()
    per_seed = per_drug.groupby(keys, as_index=False)[cols].mean()
    coverage = frame.groupby(keys).delta_pearson.count().rename("delta_pearson_defined_wells").reset_index()
    per_seed = per_seed.merge(coverage, on=keys)
    per_seed["n_wells"] = 16
    summary = per_seed.groupby(["model", "view"])[cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns]
    count = per_seed.groupby(["model", "view"]).agg(n_seeds=("seed", "size"),
        delta_pearson_defined_wells_min=("delta_pearson_defined_wells", "min"),
        delta_pearson_defined_wells_max=("delta_pearson_defined_wells", "max")).reset_index()
    summary = summary.merge(count, on=["model", "view"])
    for name, table in [("scores_by_well.csv", frame), ("scores_by_drug.csv", per_drug),
                        ("scores_by_seed.csv", per_seed), ("scores_summary.csv", summary)]:
        table.to_csv(BASE / name, index=False)
    save(BASE / "score_manifest.json", {"status": "complete", "scored_at": now(),
        "prediction_seal_sha256": sha(BASE / "prediction_seal.json"),
        "evaluation_sha256": sha(evaluation), "n_wells": 16, "n_treated": 786,
        "n_unique_controls": 951, "n_genes": 2000, "new_models": 6,
        "n_well_rows": len(frame), "aggregation": "equal wells within drug then equal drugs; SD across seeds ddof1",
        "outputs_sha256": {p.name: sha(p) for p in BASE.glob("scores*.csv")},
        "development_only": True})
    print(summary.to_string(index=False))


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.read_csv(BASE / "scores_summary.csv")
    labels = {
        "no_edit": "No edit", "literal_rule": "Literal rule", "lexa_pure_mechanism": "LEXA + DeepSeek (mechanism)",
        "expression_nearest": "Expression nearest dose", "expression_interpolate": "Expression interpolation",
        "lexa_response_retrieval": "LEXA + DeepSeek (response retrieval)", "cpa": "CPA", "codex": "CODEX"}
    frame["method"] = frame.apply(lambda r: labels[r.model] + (" / " + r.view if r.view != "existing" else ""), axis=1)
    frame = frame.sort_values("delta_rmse_mean")
    show = frame[["method", "n_seeds", "delta_rmse_mean", "delta_rmse_std", "delta_pearson_mean",
                  "absolute_r2_mean", "delta_pearson_defined_wells_min", "delta_pearson_defined_wells_max"]]
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = ["#2878a5" if r.model in ("cpa", "codex") else "#e69f00" if r.model == "lexa_pure_mechanism" else "#888888"
              for r in frame.itertuples()]
    ax.barh(frame.method, frame.delta_rmse_mean, xerr=frame.delta_rmse_std.fillna(0), color=colors, capsize=3)
    ax.invert_yaxis()
    ax.set_xlabel("Delta RMSE (lower is better); error bars: SD over 3 training seeds")
    ax.set_title("Matched sciPlex2 pilot: 16 development wells / 4 drugs / 5 uM / 2,000 genes")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    fig.savefig(BASE / "comparison.png", dpi=160)
    plt.close(fig)
    notes = (
        "同一原实验：四药BMS/Dex/Nutlin/SAHA，5µM/24h/A549，16孔、786处理细胞、951唯一对照、2000个训练来源基因。\n"
        "CPA/CODEX为作者模型在本地数据协议下的适配，每种3个独立训练种子；旧LEXA/DeepSeek及简单参考沿用封存预测，n=1无种子标准差。\n"
        "native：模型直接预测处理表达；calibrated：模型处理预测减同模型零剂量预测，再加真实对照均值。两种锚点预先指定并全部报告。\n"
        "ΔRMSE逐孔计算后药内等权、四药等权；相关性只在有定义孔计算，需同时看有效孔数；absolute R²是基因维度1-SSE/SST。\n"
        "CPA/CODEX训练可见其他剂量真实响应；纯机制主方法推理使用机制与code语义，无响应检索。对比说明相同测试集下表现，不能把差异全部归因于架构或语言模型。\n"
        "此5µM数据已经用于前期开发，不是新的独立泛化测试；不是论文/公开榜单原协议的复现分数。\n"
        "表达插值/响应检索属于响应监督参考；no-edit与文字规则为辅助基线。本轮没有新DeepSeek调用。\n"
    )
    (BASE / "RESULTS.txt").write_text("已完成的同协议模型比较\n\n" + show.to_string(index=False, float_format=lambda x: f"{x:.6f}") + "\n\n" + notes)
    page = '<!doctype html><meta charset="utf-8"><title>Matched perturbation baselines</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ddd}img{max-width:100%}pre{white-space:pre-wrap}</style>'
    page += '<h1>原 sciPlex2 实验：CPA、CODEX 与 LEXA＋DeepSeek 横向比较</h1>'
    page += '<img src="comparison.png">' + show.to_html(index=False, float_format=lambda x: f"{x:.6f}")
    page += '<pre>' + html.escape(notes) + '</pre><p><a href="scores_by_seed.csv">逐种子</a> · <a href="scores_by_drug.csv">逐药</a> · <a href="scores_by_well.csv">逐孔</a></p>'
    (BASE / "RESULTS.html").write_text(page)
    print(str(BASE / "RESULTS.html"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["references", "seal", "score", "report", "verify"])
    args = p.parse_args()
    {"references": references, "seal": seal, "score": score, "report": report, "verify": verify_seal}[args.stage]()
