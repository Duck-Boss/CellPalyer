"""Run the author's CODEXDose architecture on the existing frozen pilot split.

This adapter reads prepared train/validation arrays and a target-free prediction
manifest. It deliberately does not import the author's GPU-selecting fit helper,
whose epoch loop also evaluates OOD outcomes. No test-expression file is opened.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
AUTHOR = ROOT / "literature/tdc_leaderboard/CODEX-master"
sys.path.insert(0, str(AUTHOR))
from codex.CODEX_Dose import CODEXDose  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def drug_group(row):
    return (str(row.drug), str(row.plate), int(row.replicate))


def zero_dose_predict(model, x, drug_id):
    """Evaluate a known treatment branch at exact dose 0.

    The author's forward routes with argmax(dose-weighted one-hot); an all-zero
    vector would always select branch 0. Explicit routing keeps the chosen drug
    branch and all trained layers unchanged for this supplemental calibration.
    """
    z = model.encoder(x)
    dose = torch.zeros((len(x), 1), dtype=x.dtype, device=x.device)
    latent = model.T_rep[drug_id](torch.cat((z, dose), 1))
    return model.decoder(latent)[:, :x.shape[1]]


def build_partition(x, cell_ids, metadata, drugs, dose_scale):
    rows = metadata.loc[np.asarray(cell_ids).astype(str)].copy()
    assert rows.index.is_unique and len(rows) == len(x)
    is_control = rows.is_control.to_numpy(dtype=bool)
    controls = {}
    for i, row in enumerate(rows.itertuples()):
        if is_control[i]:
            controls.setdefault(drug_group(row), []).append(i)
    treated = np.flatnonzero(~is_control)
    t_rows = rows.iloc[treated]
    keys = [drug_group(row) for row in t_rows.itertuples()]
    assert all(key in controls for key in keys)
    conditions = {}
    for j, row in enumerate(t_rows.itertuples()):
        key = (*drug_group(row), float(row.dose_uM))
        conditions.setdefault(key, []).append(j)
    d = np.zeros((len(treated), len(drugs)), dtype=np.float32)
    for j, row in enumerate(t_rows.itertuples()):
        assert float(row.dose_uM) != 5.0
        d[j, drugs.index(str(row.drug))] = float(row.dose_uM) / dose_scale
    return {
        "all_x": torch.tensor(np.asarray(x), dtype=torch.float32),
        "treated_x": torch.tensor(np.asarray(x)[treated], dtype=torch.float32),
        "doses": torch.tensor(d),
        "keys": keys,
        "controls": {k: np.asarray(v, dtype=np.int64) for k, v in controls.items()},
        "conditions": conditions,
        "rows": rows,
    }


@torch.no_grad()
def validation_rmse(model, part, drugs, dose_scale):
    """Observed validation condition means, matched controls, drug-macro RMSE."""
    model.eval()
    by_drug = {d: [] for d in drugs}
    for key, ids in part["conditions"].items():
        drug, plate, rep, dose = key
        x = part["all_x"][part["controls"][(drug, plate, rep)]]
        dt = torch.zeros((len(x), len(drugs)), dtype=torch.float32)
        dt[:, drugs.index(drug)] = dose / dose_scale
        pred = model(x, dt)[0][:, :x.shape[1]].mean(0)
        truth = part["treated_x"][ids].mean(0)
        by_drug[drug].append(float(torch.mean((pred - truth) ** 2).sqrt()))
    model.train()
    drug_means = {k: float(np.mean(v)) for k, v in by_drug.items() if v}
    return float(np.mean(list(drug_means.values()))), drug_means


def load_inputs(args):
    prepared = args.prepared.resolve()
    source = np.load(prepared / "prepared.npz", allow_pickle=False)
    metadata = pd.read_csv(prepared / "metadata.csv").set_index("cell_id")
    cases_document = json.loads(args.case_file.read_text())
    cases = cases_document["cases"]
    assert len(cases) == 16 and len({c["case_id"] for c in cases}) == 16
    assert all(float(c["dose_uM"]) == 5.0 for c in cases)
    genes = source["genes"].astype(str)
    train_ids = source["train_cell_ids"].astype(str)
    val_ids = source["val_cell_ids"].astype(str)
    assert not set(train_ids) & set(val_ids)
    rows = metadata.loc[train_ids]
    drugs = sorted(rows.loc[~rows.is_control, "drug"].unique())
    dose_scale = float(rows.dose_uM.max())
    train = build_partition(source["x_train"], train_ids, metadata, drugs, dose_scale)
    val = build_partition(source["x_val"], val_ids, metadata, drugs, dose_scale)
    assert all(set(metadata.loc[c["control_cell_ids"]].split).issubset({"train", "val"}) for c in cases)
    return source, metadata, cases, genes, drugs, dose_scale, train, val


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "prediction_seal.json").exists():
        raise RuntimeError("Output is already sealed; choose a new directory")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    source, metadata, cases, genes, drugs, dose_scale, train, val = load_inputs(args)
    config = {
        "model": "CODEXDose author architecture",
        "author_commit": "0e5f8ff3b10d21b582cf91ea6cf5398fe8eeab99",
        "author_files": {str(p.relative_to(ROOT)): sha256(p) for p in [AUTHOR / "codex/CODEX_Dose.py", AUTHOR / "codex/Network_base.py"]},
        "adapter_sha256": sha256(__file__),
        "seed": args.seed, "device": "cpu", "threads": args.threads,
        "layers": [512, 128, 64], "batch_size": args.batch_size,
        "learning_rate": 1e-4, "weight_decay": 1e-6,
        "dropout": 0.0, "batch_norm": False, "fine_tuning": False,
        "epochs_maximum": args.epochs, "patience_epochs": args.patience,
        "loss": "torch.nn.GaussianNLLLoss, mean reduction",
        "selection": "validation drug-macro condition-mean all-gene absolute RMSE (no OOD outcomes)",
        "drug_order": drugs, "dose_scale_uM_from_train": dose_scale,
        "n_train": len(train["rows"]), "n_train_treated": len(train["treated_x"]),
        "n_val": len(val["rows"]), "n_val_treated": len(val["treated_x"]),
        "n_genes": len(genes), "n_prediction_cases": len(cases),
        "adaptations": [
            "Author CODEXDose layers and GaussianNLLLoss reused without modification; isolated CPU training wrapper.",
            "Frozen pilot 2000-gene log1pCP10K input and existing train/val/well split replace author's 4999-gene 50uM protocol.",
            "Training control cells sampled with replacement within same drug+plate+replicate to match each treated batch size; no single-cell pairing assumed.",
            "Only train/validation outcomes used for fitting and selection; author's evaluate_r2/fit_DNMLVAE_ES not called.",
            "All training treated cells retained including final incomplete batch; no test-condition filtering by model performance.",
            "For supplemental calibrated delta, exact zero-dose inference explicitly routes to the case's trained drug branch.",
        ],
        "input_files": {str(p): sha256(p) for p in [args.prepared / "prepared.npz", args.prepared / "metadata.csv", args.case_file]},
        "prediction_control_cell_ids_sha256": {c["case_id"]: hashlib.sha256("\n".join(c["control_cell_ids"]).encode()).hexdigest() for c in cases},
        "test_target_access": False,
    }
    dump(args.output / "config.json", config)
    model = CODEXDose(len(genes), [512, 128, 64], len(drugs), batch_norm=False, dropout=0.0, random_seed=args.seed).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    loss_fn = torch.nn.GaussianNLLLoss()
    records, best_score, best_epoch, best_state = [], float("inf"), -1, None
    start = time.monotonic()
    n = len(train["treated_x"])
    for epoch in range(args.epochs):
        model.train()
        shuffled = rng.permutation(n)
        loss_sum, samples = 0.0, 0
        for begin in range(0, n, args.batch_size):
            ids = shuffled[begin:begin + args.batch_size]
            cids = np.asarray([rng.choice(train["controls"][train["keys"][int(i)]]) for i in ids])
            x = train["all_x"][cids]
            y = train["treated_x"][ids]
            d = train["doses"][ids]
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(x, d)
            mean, variance = reconstruction[:, :len(genes)], reconstruction[:, len(genes):]
            loss = loss_fn(mean, y, variance)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
            samples += len(ids)
        score, drug_scores = validation_rmse(model, val, drugs, dose_scale)
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite validation score")
        if score < best_score:
            best_score, best_epoch = score, epoch
            best_state = copy.deepcopy(model.state_dict())
        row = {"epoch": epoch, "train_nll": loss_sum / samples, "val_rmse": score,
               "best_val_rmse": best_score, "best_epoch": best_epoch,
               "wall_seconds": time.monotonic() - start, **{f"val_{k}": v for k, v in drug_scores.items()}}
        records.append(row)
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(json.dumps(row), flush=True)
            pd.DataFrame(records).to_csv(args.output / "training.csv", index=False)
        if epoch - best_epoch >= args.patience:
            print(f"Early stopping after epoch {epoch}, best {best_epoch}", flush=True)
            break
    pd.DataFrame(records).to_csv(args.output / "training.csv", index=False)
    assert best_state is not None
    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "config": config, "best_epoch": best_epoch, "best_val_rmse": best_score}, args.output / "best.pt")
    control_ids = source["control_cell_ids"].astype(str)
    control_lookup = {cell: i for i, cell in enumerate(control_ids)}
    control_x = torch.tensor(source["control_x"], dtype=torch.float32)
    pred_treated, pred_control = [], []
    model.eval()
    with torch.no_grad():
        for case in cases:
            x = control_x[[control_lookup[c] for c in case["control_cell_ids"]]]
            assert len(x) >= 20
            t = drugs.index(case["drug"])
            d = torch.zeros((len(x), len(drugs)), dtype=torch.float32)
            d[:, t] = float(case["dose_uM"]) / dose_scale
            treated = model(x, d)[0][:, :len(genes)]
            baseline = zero_dose_predict(model, x, t)
            pred_treated.append(treated.mean(0).numpy())
            pred_control.append(baseline.mean(0).numpy())
    np.savez_compressed(args.output / "predictions.npz", case_ids=np.asarray([c["case_id"] for c in cases]),
                        genes=genes, pred_treated_mean=np.stack(pred_treated), pred_control_mean=np.stack(pred_control))
    stats = {"status": "complete_predictions_unscored", "best_epoch": best_epoch,
             "epochs_ran": len(records), "best_validation_rmse": best_score,
             "training_wall_seconds": time.monotonic() - start,
             "parameter_count": sum(p.numel() for p in model.parameters()),
             "seed": args.seed, "test_target_access": False}
    dump(args.output / "training_summary.json", stats)
    protected = [args.output / n for n in ["config.json", "best.pt", "training.csv", "predictions.npz", "training_summary.json"]]
    dump(args.output / "prediction_seal.json", {"created_unix": time.time(), "test_target_access": False,
         "files": {p.name: sha256(p) for p in protected}, "adapter_sha256": sha256(__file__)})
    print(json.dumps(stats), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, default=ROOT / "runs/pilot_seen_dose/prepared")
    p.add_argument("--case-file", type=Path, default=ROOT / "benchmarks/pilot_matched_baselines/data/prediction_cases.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1024)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
