"""Reload all CODEX checkpoints and verify sealed target-free predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_codex import CODEXDose, ROOT, sha256, zero_dose_predict


def main():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    work = ROOT / "benchmarks/pilot_matched_baselines"
    cases = json.loads((work / "data/prediction_cases.json").read_text())["cases"]
    prepared = np.load(ROOT / "runs/pilot_seen_dose/prepared/prepared.npz", allow_pickle=False)
    control_ids = prepared["control_cell_ids"].astype(str)
    lookup = {k: i for i, k in enumerate(control_ids)}
    controls = torch.tensor(prepared["control_x"], dtype=torch.float32)
    audits = []
    for seed in range(3):
        directory = work / f"models/codex/seed_{seed}"
        seal = json.loads((directory / "prediction_seal.json").read_text())
        assert not seal["test_target_access"]
        for name, expected in seal["files"].items():
            assert sha256(directory / name) == expected, (seed, name)
        ckpt = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        config = ckpt["config"]
        assert config["adapter_sha256"] == sha256(Path(__file__).with_name("run_codex.py"))
        assert config["adapter_sha256"] == seal["adapter_sha256"]
        assert not config["test_target_access"]
        for path, expected in config["input_files"].items():
            assert sha256(path) == expected
        for relative_path, expected in config["author_files"].items():
            assert sha256(ROOT / relative_path) == expected
        curve = pd.read_csv(directory / "training.csv")
        best_row = curve.iloc[int(curve.val_rmse.argmin())]
        assert int(best_row.epoch) == ckpt["best_epoch"]
        assert abs(float(best_row.val_rmse) - ckpt["best_val_rmse"]) < 1e-12
        assert np.isfinite(curve.select_dtypes(include="number").to_numpy()).all()
        model = CODEXDose(2000, [512, 128, 64], 4, batch_norm=False, dropout=0.0, random_seed=seed).cpu()
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
        saved = np.load(directory / "predictions.npz", allow_pickle=False)
        assert np.array_equal(saved["genes"], prepared["genes"])
        assert saved["case_ids"].tolist() == [c["case_id"] for c in cases]
        assert saved["pred_treated_mean"].shape == saved["pred_control_mean"].shape == (16, 2000)
        predictions, zeroes = [], []
        with torch.no_grad():
            for case in cases:
                x = controls[[lookup[cell] for cell in case["control_cell_ids"]]]
                drug = config["drug_order"].index(case["drug"])
                condition = torch.zeros(len(x), 4)
                condition[:, drug] = float(case["dose_uM"]) / 100.0
                predictions.append(model(x, condition)[0][:, :2000].mean(0).numpy())
                zeroes.append(zero_dose_predict(model, x, drug).mean(0).numpy())
        error_t = float(np.abs(np.stack(predictions) - saved["pred_treated_mean"]).max())
        error_c = float(np.abs(np.stack(zeroes) - saved["pred_control_mean"]).max())
        assert error_t == error_c == 0.0
        assert np.isfinite(saved["pred_treated_mean"]).all() and np.isfinite(saved["pred_control_mean"]).all()
        audits.append({"seed": seed, "best_epoch": ckpt["best_epoch"], "best_val_rmse": ckpt["best_val_rmse"],
                       "epochs_ran": len(curve), "prediction_max_abs_diff": error_t,
                       "zero_dose_max_abs_diff": error_c, "prediction_seal_valid": True,
                       "input_and_author_source_hashes_valid": True})
    out = {"status": "passed", "test_target_access": False, "models": audits,
           "audit_source_sha256": sha256(__file__)}
    (work / "codex_integrity_audit.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))


if __name__ == "__main__":
    main()
