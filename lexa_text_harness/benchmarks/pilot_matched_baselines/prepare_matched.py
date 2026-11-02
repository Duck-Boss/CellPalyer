"""Freeze the existing sciPlex2 pilot as inputs for external model baselines.

This writes no treated test expression. Existing data and sealed predictions
are read only. The target archive is hashed for later scoring provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREPARED = ROOT / "runs/pilot_seen_dose/prepared"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_once(path: Path, value: dict) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists():
        old = json.loads(path.read_text())
        comparable = dict(value)
        comparable["created_at"] = old.get("created_at", value.get("created_at"))
        if old != comparable:
            raise ValueError(f"Frozen contract differs: {path}")
        return
    path.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    data = OUT / "data"
    data.mkdir(parents=True, exist_ok=True)
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    metadata = pd.read_csv(PREPARED / "metadata.csv").set_index("cell_id")
    with np.load(PREPARED / "prepared.npz", allow_pickle=False) as z:
        genes = z["genes"].astype(str)
        train_ids = z["train_cell_ids"].astype(str)
        val_ids = z["val_cell_ids"].astype(str)
        control_ids = z["control_cell_ids"].astype(str)
        train = metadata.loc[train_ids]
        val = metadata.loc[val_ids]
        assert z["x_train"].shape == (5285, 2000)
        assert z["x_val"].shape == (2742, 2000)
        assert not set(train_ids) & set(val_ids)
        assert not np.isclose(train.dose_uM, 5).any()
        assert not np.isclose(val.dose_uM, 5).any()
        assert float(train.dose_uM.max()) == 100.0
    original = json.loads((PREPARED / "cases.json").read_text())["cases"]
    ready = [c for c in original if c["status"] == "ready"]
    drugs = sorted({c["drug"] for c in ready})
    groups = {d: sorted([c for c in ready if c["drug"] == d], key=lambda c: c["case_id"])
              for d in drugs}
    ordered = [groups[d][i] for i in range(4) for d in drugs]
    assert len(ordered) == 16 and all(len(groups[d]) == 4 for d in drugs)
    fields = ("case_id", "drug", "dose_uM", "plate", "replicate", "vehicle", "control_cell_ids")
    cases = [{key: c[key] for key in fields} for c in ordered]
    all_ctrl = set(control_ids)
    assert all(set(c["control_cell_ids"]).issubset(all_ctrl) for c in cases)
    assert len(set(x for c in cases for x in c["control_cell_ids"])) == 951
    genes_sha = hashlib.sha256("\n".join(genes).encode()).hexdigest()
    inputs = {name: {"path": str((PREPARED / name).resolve()), "sha256": sha(PREPARED / name)}
              for name in ("prepared.npz", "metadata.csv", "manifest.json", "cases.json")}
    prediction = {
        "created_at": now, "prepared_dir": str(PREPARED.resolve()),
        "genes_sha256": genes_sha, "n_genes": len(genes),
        "dose_model_input": "dose_uM / 100; training maximum fixes denominator",
        "inputs": inputs, "cases": cases,
        "contains_treated_test_expression": False,
    }
    save_once(data / "prediction_cases.json", prediction)
    protocol = {
        "created_at": now, "task": "existing seen-drug unseen-5uM pilot",
        "dataset": "sciPlex2 A549, 24 hours, BMS/Dex/Nutlin/SAHA",
        "prepared_dir": str(PREPARED.resolve()), "inputs": inputs,
        "prediction_cases_sha256": sha(data / "prediction_cases.json"),
        "evaluation_archive_sha256": sha(PREPARED / "evaluation.npz"),
        "models": ["cpa", "codex"], "training_seeds": [0, 1, 2],
        "device": "cpu", "threads_per_process": 4,
        "normalization": "reuse log1p(CP10K) over original genes then train-selected 2000 features",
        "dose_scaling": {"denominator_uM": 100, "source": "maximum in training metadata"},
        "n_train": len(train_ids), "n_val": len(val_ids), "n_genes": len(genes),
        "n_training_controls": int(train.is_control.sum()),
        "n_validation_controls": int(val.is_control.sum()),
        "n_cases": 16, "n_treated_scored": 786, "n_unique_prediction_controls": 951,
        "selection": "the same 16 ready cases; no new sample exclusion",
        "validation": "matched validation-condition mean expression RMSE, equal conditions within drug then equal drugs; max1000 epochs/patience50",
        "test_target_access": "only shared scorer after all six model predictions are sealed",
        "prediction_contract": {
            "file": "models/{model}/seed_{seed}/predictions.npz",
            "fields": ["case_ids", "genes", "pred_treated_mean", "pred_control_mean"],
            "shape": [16, 2000],
            "zero_dose": "same inference control population and same model; drug-specific zero-dose branch where required",
        },
        "prespecified_prediction_views": {
            "native": "delta = model_treated_mean - observed_control_mean; absolute = model_treated_mean",
            "calibrated": "delta = model_treated_mean - model_zero_dose_mean; absolute = observed_control_mean + delta",
            "reporting": "report both; no test-based selection of anchor or checkpoint",
        },
        "metrics": {
            "delta_rmse": "per well sqrt(mean_g((pred_delta - (true_treated_mean - observed_control_mean))**2)); equal wells within drug then equal drugs",
            "delta_pearson": "correlation across gene-wise changes; undefined zero-variance vectors stay NaN, report valid-well coverage",
            "absolute_r2": "sklearn definition 1-SSE/SST across gene-wise treated means, not squared Pearson",
            "aggregation_of_seeds": "compute each full experiment separately; mean and sample SD (ddof=1) across three seeds",
        },
        "references": {
            "lexa_pure_v4": "reuse sealed full_joint_a0.3 predictions",
            "lexa_retrieval_v3": "historical supervised-response reference, separate information condition",
            "expression_interpolation": "same training-response interpolation as existing experiment",
        },
        "limitations": [
            "5uM targets already unblinded in prior work: development comparison, not new independent generalization test",
            "CPA/CODEX supervised on other-dose responses; pure LEXA+DeepSeek consumes mechanism/code semantics at inference, not matched-response retrieval",
            "Matched data protocol is ours; models are adapted author implementations, not reproduction of paper or leaderboard scores",
            "No matched single-cell pre/post pairing; outputs and metrics are population means",
        ],
    }
    save_once(OUT / "protocol.json", protocol)
    print(json.dumps({"status": "input_contract_frozen", "cases": 16, "genes": len(genes),
                      "train": len(train_ids), "val": len(val_ids), "new_api_calls": 0}))


if __name__ == "__main__":
    main()
