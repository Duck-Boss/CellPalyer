"""Recompute the shipped tables from frozen group means and predicted changes.

No model checkpoint, original filesystem, credential or API call is required.
The replay archive includes measured held-out group means for scoring only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def calculate(delta, treated, control):
    truth = treated - control
    error = delta - truth
    top = np.argsort(-np.abs(truth))[:100]
    variance = np.sum((treated - np.mean(treated)) ** 2)
    defined = np.std(delta) > 1e-12 and np.std(truth) > 1e-12
    return {
        "delta_rmse": float(np.sqrt(np.mean(error ** 2))),
        "delta_pearson": float(np.corrcoef(delta, truth)[0, 1]) if defined else np.nan,
        "absolute_rmse": float(np.sqrt(np.mean((control + delta - treated) ** 2))),
        "absolute_r2": float(1 - np.sum((control + delta - treated) ** 2) / variance),
        "response_l2_ratio": float(np.linalg.norm(delta) / np.linalg.norm(truth)),
        "top100_delta_rmse": float(np.sqrt(np.mean(error[top] ** 2))),
        "top100_direction_accuracy": float(np.mean(np.sign(delta[top]) == np.sign(truth[top]))),
    }


def compare(calculated, filename, keys):
    expected = pd.read_csv(RESULTS / filename).set_index(keys).sort_index()
    actual = calculated.set_index(keys).sort_index()
    assert expected.index.equals(actual.index), f"Row identity mismatch: {filename}"
    errors = {}
    for column in expected.columns:
        assert column in actual.columns, (filename, column)
        a, b = actual[column].to_numpy(float), expected[column].to_numpy(float)
        assert np.array_equal(np.isnan(a), np.isnan(b)), (filename, column, "NaN coverage")
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12, equal_nan=True, err_msg=f"{filename}:{column}")
        difference = np.abs(a - b)
        errors[column] = float(np.nanmax(difference)) if np.isfinite(difference).any() else 0.0
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-dir", type=Path, help="Optionally write recomputed tables outside the frozen results directory")
    args = parser.parse_args()
    manifest = json.loads((RESULTS / "export_manifest.json").read_text())
    assert sha(RESULTS / "replay.npz") == manifest["replay_sha256"]
    original = json.loads((RESULTS / "score_manifest.json").read_text())
    for name, digest in original["outputs_sha256"].items():
        assert sha(RESULTS / name) == digest, f"Original table hash mismatch: {name}"
    with np.load(RESULTS / "replay.npz", allow_pickle=False) as z:
        d = {k: z[k].copy() for k in z.files}
    assert d["predicted_delta"].shape == (18, 16, 2000)
    assert d["treated_mean"].shape == d["control_mean"].shape == (16, 2000)
    assert len(set(d["case_ids"])) == 16 and len(set(d["genes"])) == 2000
    assert int(d["n_treated"].sum()) == 786
    rows = []
    metric_columns = None
    for j, (model, seed, view) in enumerate(zip(d["model_names"], d["seeds"], d["views"])):
        for i, cid in enumerate(d["case_ids"]):
            values = calculate(d["predicted_delta"][j, i], d["treated_mean"][i], d["control_mean"][i])
            metric_columns = list(values)
            rows.append({"model": str(model), "seed": int(seed), "view": str(view),
                         "case_id": str(cid), "drug": str(d["drugs"][i]),
                         "n_treated": int(d["n_treated"][i]), "n_control": int(d["n_control"][i]), **values})
    wells = pd.DataFrame(rows)
    group = ["model", "seed", "view"]
    drugs = wells.groupby(group + ["drug"], as_index=False)[metric_columns].mean()
    seeds = drugs.groupby(group, as_index=False)[metric_columns].mean()
    coverage = wells.groupby(group).delta_pearson.count().rename("delta_pearson_defined_wells").reset_index()
    seeds = seeds.merge(coverage, on=group)
    seeds["n_wells"] = 16
    summary = seeds.groupby(["model", "view"])[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in summary.columns]
    counts = seeds.groupby(["model", "view"]).agg(n_seeds=("seed", "size"),
        delta_pearson_defined_wells_min=("delta_pearson_defined_wells", "min"),
        delta_pearson_defined_wells_max=("delta_pearson_defined_wells", "max")).reset_index()
    summary = summary.merge(counts, on=["model", "view"])
    tables = [("scores_by_well.csv", wells, group + ["case_id", "drug"]),
              ("scores_by_drug.csv", drugs, group + ["drug"]),
              ("scores_by_seed.csv", seeds, group),
              ("scores_summary.csv", summary, ["model", "view"])]
    errors = {name: compare(frame, name, keys) for name, frame, keys in tables}
    report = {"status": "passed", "new_training": False, "new_api_calls": 0,
              "well_rows": len(wells), "drug_rows": len(drugs), "seed_rows": len(seeds),
              "summary_rows": len(summary), "max_absolute_error": max(x for t in errors.values() for x in t.values()),
              "errors": errors}
    if args.write_dir:
        out = args.write_dir.resolve()
        assert out != RESULTS.resolve() and RESULTS.resolve() not in out.parents
        out.mkdir(parents=True, exist_ok=True)
        for name, frame, _ in tables:
            frame.to_csv(out / name, index=False)
        (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
