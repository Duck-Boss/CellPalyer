"""Run the original authors' CPA on the frozen local sciPlex2 pilot split.

The CPA architecture, Gaussian loss, adversarial losses, optimizers and update
function are imported unchanged from facebookresearch/CPA. This wrapper replaces
only dataset plumbing, validation-based checkpoint selection and prediction I/O.
It never loads prepared/evaluation.npz or held-out treatment expression.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = HERE / "vendor/CPA-main/cpa/model.py"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_source():
    spec = importlib.util.spec_from_file_location("original_cpa_model", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validation_cases(metadata, val_ids, minimum=1):
    """Use validation wells with the same drug/plate/replicate controls."""
    sub = metadata.loc[val_ids]
    result = []
    treated = sub[~sub.is_control.astype(bool)]
    for (drug, dose, plate, replicate), group in treated.groupby(
        ["drug", "dose_uM", "plate", "replicate"], sort=True
    ):
        controls = sub[
            sub.is_control.astype(bool)
            & (sub.drug == drug)
            & (sub.plate == plate)
            & (sub.replicate == replicate)
        ]
        if len(group) < minimum or len(controls) < minimum:
            continue
        if set(group.vehicle) != set(controls.vehicle):
            raise ValueError("Validation vehicle matching failed")
        result.append({
            "case_id": f"drug={drug}|dose_uM={dose:g}|plate={plate}|replicate={int(replicate)}",
            "drug": drug,
            "dose_uM": float(dose),
            "plate": str(plate),
            "replicate": int(replicate),
            "vehicle": str(group.vehicle.iloc[0]),
            "treated_cell_ids": group.index.tolist(),
            "control_cell_ids": controls.index.tolist(),
        })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--prepared", type=Path, default=ROOT / "runs/pilot_seen_dose/prepared")
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--prediction-cases", type=Path, default=HERE / "data/prediction_cases.json")
    ap.add_argument("--validation-cases", type=Path, default=HERE / "data/validation_cases.json")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    output = args.output or HERE / "models/cpa" / f"seed_{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "predictions.npz").exists():
        raise RuntimeError("Refusing to overwrite completed predictions")
    prepared = args.prepared.resolve()
    archive_path = prepared / "prepared.npz"
    metadata_path = prepared / "metadata.csv"
    z = np.load(archive_path, allow_pickle=False)
    metadata = pd.read_csv(metadata_path).set_index("cell_id")
    train_ids = z["train_cell_ids"].astype(str)
    val_ids = z["val_cell_ids"].astype(str)
    control_ids = z["control_cell_ids"].astype(str)
    genes = z["genes"].astype(str)
    train_meta = metadata.loc[train_ids]
    x_train = torch.from_numpy(z["x_train"].astype(np.float32))
    x_val = z["x_val"].astype(np.float32)
    control_x = z["control_x"].astype(np.float32)
    if set(train_ids) & set(val_ids):
        raise ValueError("Train/validation overlap")
    if np.any(train_meta.dose_uM.to_numpy() == 5):
        raise ValueError("Held-out dose present in training")
    drugs = sorted(train_meta.loc[~train_meta.is_control.astype(bool), "drug"].unique().tolist())
    drug_to_index = {d: i for i, d in enumerate(drugs)}
    scale = float(train_meta.dose_uM.max())
    assert scale == 100.0
    treatment = np.zeros((len(train_ids), len(drugs)), np.float32)
    for i, (_, row) in enumerate(train_meta.iterrows()):
        if not row.is_control:
            treatment[i, drug_to_index[row.drug]] = row.dose_uM / scale
    treatment = torch.from_numpy(treatment)
    # CPA expects one-hot categorical covariates. A549 is constant, so the
    # associated adversarial classification has exactly one class and zero loss.
    covariate = torch.ones((len(train_ids), 1), dtype=torch.float32)
    if args.validation_cases.exists():
        document = json.loads(args.validation_cases.read_text())
        vcases = document["cases"]
    else:
        vcases = validation_cases(metadata, val_ids)
    if not vcases:
        raise ValueError("No eligible validation cases")
    val_lookup = {v: i for i, v in enumerate(val_ids)}
    vprepared = []
    for c in vcases:
        ci = np.asarray([val_lookup[v] for v in c["control_cell_ids"]])
        ti = np.asarray([val_lookup[v] for v in c["treated_cell_ids"]])
        vprepared.append((c, torch.from_numpy(x_val[ci]), x_val[ti].mean(0, dtype=np.float64)))
    cases = json.loads(args.prediction_cases.read_text())["cases"]
    control_lookup = {v: i for i, v in enumerate(control_ids)}
    module = load_source()
    model = module.CPA(
        num_genes=len(genes), num_drugs=len(drugs), num_covariates=[1],
        device="cpu", seed=args.seed, patience=args.patience,
        loss_ae="gauss", doser_type="mlp", decoder_activation="linear",
        hparams="",
    )
    config = {
        "method": "CPA (original author implementation)",
        "seed": args.seed, "device": "cpu", "threads": args.threads,
        "max_epochs": args.max_epochs, "patience_epochs": args.patience,
        "scheduler_interval_epochs": 20,
        "scheduler_interval_source": "Original CPA train.py default checkpoint_freq=20; original early_stopping steps LR only at these checkpoints",
        "selection": "native counterfactual validation expression RMSE; well within drug then drug equal weight",
        "dose_transform": "dose_uM / training_max_dose_uM", "dose_scale": scale,
        "drug_order": drugs, "covariates": ["constant A549"],
        "loss_ae": "gauss", "doser_type": "mlp", "decoder_activation": "linear",
        "hparams": model.hparams,
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": sha256(SOURCE),
        "source_provenance": json.loads((HERE / "vendor/cpa_original_provenance.json").read_text()),
        "wrapper_sha256": sha256(__file__),
        "input_sha256": {
            "prepared.npz": sha256(archive_path), "metadata.csv": sha256(metadata_path),
            "prediction_cases.json": sha256(args.prediction_cases),
        },
        "n_train": len(train_ids), "n_val": len(val_ids), "n_genes": len(genes),
        "validation_cases": len(vcases), "test_cases": len(cases),
        "no_test_targets_loaded": True,
        "adaptations": [
            "Direct import of original model.py avoids optional AnnData/Scanpy API wrappers",
            "Original model/update/loss/optimizer/default hyperparameters unchanged",
            "Validation matched-well RMSE replaces original OOD-reporting training harness",
            "Original learning-rate scheduler called at the original default 20-epoch interval; early stopping checks validation each epoch",
            "Explicit train-only fitting and metadata/ID-based frozen data plumbing",
        ],
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "validation_cases_used.json").write_text(json.dumps({"cases": vcases}, indent=2) + "\n")
    if args.validation_cases.exists():
        config["input_sha256"]["validation_cases.json"] = sha256(args.validation_cases)
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    def means(controls, drug, dose):
        d = torch.zeros((len(controls), len(drugs)), dtype=torch.float32)
        if dose > 0:
            d[:, drug_to_index[drug]] = float(dose) / scale
        cov = [torch.ones((len(controls), 1))]
        predictions = []
        for i in range(0, len(controls), 256):
            predictions.append(model.predict(controls[i:i+256], d[i:i+256], [cov[0][i:i+256]])[:, :len(genes)].numpy())
        return np.concatenate(predictions).mean(0, dtype=np.float64)
    def validation():
        per_drug = {}
        with torch.no_grad():
            for c, controls, target in vprepared:
                pred = means(controls, c["drug"], c["dose_uM"])
                rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
                per_drug.setdefault(c["drug"], []).append(rmse)
        return float(np.mean([np.mean(v) for v in per_drug.values()]))
    started = time.time()
    best = float("inf")
    best_epoch = 0
    bad = 0
    batch_size = model.hparams["batch_size"]
    log = (output / "history.jsonl").open("a", buffering=1)
    for epoch in range(1, args.max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(x_train))
        stats = []
        for start in range(0, len(x_train), batch_size):
            idx = permutation[start:start + batch_size]
            if len(idx) < 2:
                continue
            row = model.update(x_train[idx], treatment[idx], [covariate[idx]])
            if not np.isfinite(list(row.values())).all():
                raise FloatingPointError(f"Nonfinite CPA training statistics: {row}")
            stats.append(row)
        model.eval()
        score = validation()
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite validation RMSE")
        improved = score < best - 1e-8
        if improved:
            best = score
            best_epoch = epoch
            bad = 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_rmse": score, "config": config}, output / "best.pt")
        else:
            bad += 1
        # Original train.py calls early_stopping (which steps all three
        # schedulers) only at checkpoint_freq=20, rather than every epoch.
        if epoch % 20 == 0:
            model.scheduler_autoencoder.step()
            model.scheduler_adversary.step()
            model.scheduler_dosers.step()
        event = {"epoch": epoch, "val_rmse": score, "best_val_rmse": best, "best_epoch": best_epoch,
                 "bad_epochs": bad, "seconds": time.time() - started,
                 "train": {key: float(np.mean([r[key] for r in stats])) for key in stats[0]}}
        log.write(json.dumps(event) + "\n")
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(event), flush=True)
        if bad >= args.patience:
            break
    log.close()
    best_checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_checkpoint["state_dict"])
    model.eval()
    treated_means, control_means = [], []
    with torch.no_grad():
        for c in cases:
            controls = torch.from_numpy(control_x[[control_lookup[v] for v in c["control_cell_ids"]]])
            treated_means.append(means(controls, c["drug"], c["dose_uM"]))
            control_means.append(means(controls, c["drug"], 0.0))
    treated_means, control_means = np.asarray(treated_means), np.asarray(control_means)
    if not np.isfinite(treated_means).all() or not np.isfinite(control_means).all():
        raise FloatingPointError("Nonfinite final prediction")
    np.savez_compressed(output / "predictions.npz", case_ids=np.asarray([c["case_id"] for c in cases]),
                        genes=genes, pred_treated_mean=treated_means, pred_control_mean=control_means)
    summary = {"method": "CPA", "seed": args.seed, "epochs": epoch, "best_epoch": best_epoch,
               "best_validation_rmse": best, "wall_seconds": time.time() - started,
               "prediction_shape": list(treated_means.shape),
               "predictions_sha256": sha256(output / "predictions.npz"),
               "config_sha256": sha256(output / "config.json"),
               "source_sha256": sha256(SOURCE), "no_test_targets_loaded": True}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
