#!/usr/bin/env python3
"""Fit the original LEXA on prepared perturbation data and freeze exact effects.

This is a small *pilot adaptation*, not a reproduction of the LEXA paper:
encoder=(512,), decoder=256, and the scaffold is train-only expression
coexpression (SVD + KMeans), not the paper's cell-type/donor scaffold.
No language model or trainable perturbation editor is trained here.

Required prepared.npz fields (allow_pickle=False):
  x_train[N,G], x_val[M,G], control_x[C,G]: normalized log1p expression;
  genes[G]: ordered gene names. Values must already be log1p(CP10K).
Optional: synthetic (bool), provenance_json (JSON str),
  train_cell_ids[N], val_cell_ids[M], control_cell_ids[C].

The caller owns biological replicate splitting. This script never constructs
a random cell split. Test expression is neither requested nor used.
"""

from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

sys.dont_write_bytecode = True

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.utils.extmath import randomized_svd
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_REPO = Path(__file__).resolve().parent.parent / "LEXA-5913"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_torch(path: Path, payload: Any) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def import_original(repo: Path):
    repo = repo.resolve()
    if not (repo / "lexa" / "lexa_model.py").is_file():
        raise FileNotFoundError(f"Original LEXA source missing in {repo}")
    sys.path.insert(0, str(repo))
    from lexa.lexa_model import LEXA
    # The original effects CLI imports h5py/data-bundle helpers at module level,
    # although its exact contributions function only needs torch and F. Load
    # that unchanged function's AST, avoiding unrelated CLI dependencies. The
    # full source SHA256 is recorded in every checkpoint for reproducibility.
    effects_path = repo / "lexa" / "compute_state_effects.py"
    module = ast.parse(effects_path.read_text(encoding="utf-8"), filename=str(effects_path))
    functions = [node for node in module.body
                 if isinstance(node, ast.FunctionDef) and node.name == "contributions"]
    if len(functions) != 1:
        raise RuntimeError("Original source must expose one contributions function")
    namespace = {"torch": torch, "F": torch.nn.functional}
    code = compile(ast.Module(body=functions, type_ignores=[]), str(effects_path), "exec")
    exec(code, namespace)
    return LEXA, namespace["contributions"]


def load_prepared(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        missing = {"x_train", "x_val", "control_x", "genes"} - set(data.files)
        if missing:
            raise ValueError(f"Prepared data missing keys: {sorted(missing)}")
        result = {key: np.asarray(data[key], dtype=np.float32) for key in
                  ("x_train", "x_val", "control_x")}
        result["genes"] = np.asarray(data["genes"], dtype=str)
        result["gene_symbols"] = np.asarray(data["gene_symbols"], dtype=str) if "gene_symbols" in data else result["genes"].copy()
        result["synthetic"] = bool(np.asarray(data["synthetic"]).item()) if "synthetic" in data else False
        result["provenance"] = json.loads(str(np.asarray(data["provenance_json"]).item())) if "provenance_json" in data else {}
        for key in ("train_cell_ids", "val_cell_ids", "control_cell_ids"):
            if key in data:
                result[key] = np.asarray(data[key], dtype=str)

    genes = result["genes"]
    if genes.ndim != 1 or len(genes) < 32 or len(np.unique(genes)) != len(genes):
        raise ValueError("genes must be a unique ordered 1D vocabulary of at least 32 genes")
    if result["gene_symbols"].shape != genes.shape:
        raise ValueError("gene_symbols must match the gene ID vocabulary shape")
    for matrix_key, ids_key, min_rows in (("x_train", "train_cell_ids", 2),
                                           ("x_val", "val_cell_ids", 1),
                                           ("control_x", "control_cell_ids", 1)):
        x = result[matrix_key]
        if x.ndim != 2 or x.shape[1] != len(genes) or len(x) < min_rows:
            raise ValueError(f"{matrix_key} has incompatible shape {x.shape}")
        if not np.isfinite(x).all() or (x < 0).any():
            raise ValueError(f"{matrix_key} must contain finite nonnegative log1p values")
        if ids_key in result and result[ids_key].shape != (len(x),):
            raise ValueError(f"{ids_key} length must equal {matrix_key} rows")
        if ids_key in result and len(np.unique(result[ids_key])) != len(x):
            raise ValueError(f"{ids_key} must contain unique stable cell IDs")
    if "train_cell_ids" in result and "val_cell_ids" in result:
        if np.intersect1d(result["train_cell_ids"], result["val_cell_ids"]).size:
            raise ValueError("Training and validation cell IDs overlap")
    return result


def build_pilot_scaffold(x_train: np.ndarray, seed: int, max_cells: int = 4096,
                         svd_dim: int = 24, temperature: float = 0.15):
    """Expression-only scaffold; every statistic is computed from x_train.

    Row-standardized gene profiles are projected into a low-rank cell space;
    normalized loadings keep positive and negative coexpression distinct.
    Softmax(-squared distance / T) assigns each gene mass to 32 clusters.
    """
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(x_train), min(max_cells, len(x_train)), replace=False))
    x = x_train[selected].copy()
    x -= x.mean(axis=0, keepdims=True)
    x /= np.maximum(x.std(axis=0, keepdims=True), 0.05)
    rank = min(svd_dim, min(x.shape) - 1)
    if rank < 1:
        raise ValueError("Too few training cells/genes for the scaffold")
    _, singular_values, right = randomized_svd(x, n_components=rank, random_state=seed)
    features = (right.T * singular_values).astype(np.float32)
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-8)
    km = KMeans(n_clusters=32, n_init=10, random_state=seed).fit(features)
    squared_distances = np.maximum(
        np.square(features).sum(axis=1, keepdims=True)
        + np.square(km.cluster_centers_).sum(axis=1)[None, :]
        - 2 * features @ km.cluster_centers_.T, 0)
    logits = -squared_distances / temperature
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    prior = weights.T.astype(np.float32)
    meta = {"type": "train_only_gene_coexpression_svd_kmeans_pilot",
            "not_paper_celltype_donor_scaffold": True,
            "training_cells_available": len(x_train), "training_cells_used": len(selected),
            "svd_dim": rank, "n_clusters": 32, "distance_softmax_temperature": temperature,
            "gene_profile_std_floor": 0.05, "seed": seed,
            "occupied_gene_clusters": int(len(np.unique(km.labels_)))}
    return prior, meta


def model_config(input_dim: int) -> dict[str, Any]:
    # Match published core geometry/loss scales; reduce widths for this pilot.
    return dict(input_dim=input_dim, hidden_dims=[512], num_axes=32, code_dim=64,
                codebook_size=16, decoder_hidden_dim=256, dropout=0.1,
                reconstruction_distribution="mse_log1p", count_normalized_total=10000.0,
                quantization_mode="spherical", codebook_parameterization="free_spherical",
                encoder_fixed_norm=True, vq_loss_normalization="per_radius",
                commitment_weight=1.0, vq_aux_loss_weight=0.003125,
                diversity_loss_weight=0.1, module_mask_parameterization="prior_guided_softmax",
                assignment_temperature=0.1, module_mask_temperature=0.5,
                module_mask_logit_init_std=0.0, module_mask_prior_floor=0.001,
                module_mask_prior_weight=0.1, module_effective_axes_target=0.0,
                module_sparsity_weight=0.0, module_balance_weight=1.0,
                module_overlap_weight=0.0, codebook_sep_weight=0.0,
                anchor_loss_weight=0.0, center_norm_weight=0.0,
                center_norm_target=4.0, codebook_compact_weight=0.0,
                intra_codebook_div_weight=0.005, cluster_radius=0.0,
                residual_center_weight=0.0, technical_covariate_fields=[],
                nuisance_decoder_weight=0.0, batch_adversary_weight=0.0,
                conditional_code_usage_weight=0.0)


@torch.no_grad()
def validation_mse(model, values: np.ndarray, batch_size: int, device) -> float:
    model.eval()
    total = 0.0
    for start in range(0, len(values), batch_size):
        x = torch.from_numpy(values[start:start + batch_size]).to(device)
        z_q = model.quantize(model.encode(x))[0]
        total += torch.square(model.decode(z_q) - x).sum().item()
    return total / values.size


@torch.no_grad()
def encode_codes(model, values: np.ndarray, batch_size: int, device) -> np.ndarray:
    model.eval()
    rows = []
    for start in range(0, len(values), batch_size):
        x = torch.from_numpy(values[start:start + batch_size]).to(device)
        rows.append(model.quantize(model.encode(x))[1].cpu().numpy().astype(np.int16))
    return np.concatenate(rows)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lexa-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--smoke", nargs="?", const=2, type=int, metavar="MAX_STEPS",
                        help="Stop after MAX_STEPS updates (default 2); exports tagged smoke_only")
    parser.add_argument("--resume", action="store_true", help="Resume an epoch checkpoint in output/last.pt")
    return parser.parse_args(argv)


def run(args) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".fit.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another fit is already using this output directory") from exc
        return _run_locked(args)


def _run_locked(args) -> dict[str, Any]:
    if min(args.epochs, args.batch_size, args.patience, args.threads) < 1:
        raise ValueError("epochs, batch-size, patience and threads must be positive")
    if args.smoke is not None and args.max_steps is not None:
        raise ValueError("Use only one of --smoke or --max-steps")
    max_steps = args.smoke if args.smoke is not None else args.max_steps
    if max_steps is not None and max_steps < 1:
        raise ValueError("max-steps must be positive")
    if (args.output / "last.pt").exists() and not args.resume:
        raise FileExistsError(f"{args.output}/last.pt exists; use --resume or a new output directory")
    prepared = args.prepared.resolve()
    fingerprint = file_sha256(prepared)
    data = load_prepared(prepared)
    original_model, original_contributions = import_original(args.lexa_repo)
    config = model_config(len(data["genes"]))
    source_hashes = {name: file_sha256(args.lexa_repo / "lexa" / name)
                     for name in ("lexa_model.py", "compute_state_effects.py")}
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device_name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu")
    # Remove completion before mutating any previously exported artifacts.
    (args.output / "COMPLETE.json").unlink(missing_ok=True)
    started = time.time()
    history = []
    start_epoch = updates = cell_updates = stale = 0
    best_mse = float("inf")
    best_epoch = -1
    train_cfg = {"lr": args.lr, "weight_decay": args.weight_decay,
                 "batch_size": args.batch_size, "seed": args.seed,
                 "vq_warmup_epochs": 3, "gradient_clip_norm": 1.0}
    last_path = args.output / "last.pt"
    if args.resume:
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=True)
        if checkpoint["prepared_sha256"] != fingerprint or checkpoint["config"] != config:
            raise ValueError("Cannot resume: prepared data or model config changed")
        if checkpoint["training_config"] != train_cfg or checkpoint["source_sha256"] != source_hashes:
            raise ValueError("Cannot resume: training settings or original source changed")
        prior = checkpoint["module_mask_prior"].cpu().numpy()
        scaffold_meta = checkpoint["scaffold_metadata"]
        start_epoch = int(checkpoint["next_epoch"])
        updates, cell_updates = int(checkpoint["updates"]), int(checkpoint["cell_updates"])
        best_mse, best_epoch = float(checkpoint["best_val_mse"]), int(checkpoint["best_epoch"])
        stale, history = int(checkpoint["stale_epochs"]), checkpoint["history"]
    else:
        prior, scaffold_meta = build_pilot_scaffold(data["x_train"], args.seed)
        atomic_npz(args.output / "pilot_scaffold.npz", module_mask=prior, genes=data["genes"])
    model = original_model(**config, module_mask_prior=torch.from_numpy(prior)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.resume:
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_states"]])

    common = {"config": config, "training_config": train_cfg, "gene_vocab": data["genes"].tolist(),
              "genes": data["genes"].tolist(),
              "gene_symbols": data["gene_symbols"].tolist(),
              "module_mask_prior": torch.from_numpy(prior), "prepared_sha256": fingerprint,
              "prepared_path": str(prepared), "source_sha256": source_hashes,
              "scaffold_metadata": scaffold_meta, "synthetic": data["synthetic"],
              "provenance": data["provenance"]}
    stopped = "epochs_exhausted"
    for epoch in range(start_epoch, args.epochs):
        if max_steps is not None and updates >= max_steps:
            stopped = "max_steps"
            break
        model.train()
        warmup = min(1.0, (epoch + 1) / 3.0)
        model.vq_aux_loss_weight = config["vq_aux_loss_weight"] * warmup
        model.intra_codebook_div_weight = config["intra_codebook_div_weight"] * warmup
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(TensorDataset(torch.from_numpy(data["x_train"])),
                            batch_size=args.batch_size, shuffle=True, generator=generator,
                            num_workers=0, drop_last=False)
        train_loss = train_recon = 0.0
        epoch_cells = 0
        for (x,) in loader:
            x = x.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(x)
            if not torch.isfinite(result["loss"]):
                raise RuntimeError("Nonfinite LEXA training loss")
            result["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            train_loss += float(result["loss"].detach()) * len(x)
            train_recon += float(result["recon_loss"].detach()) * len(x)
            epoch_cells += len(x)
            cell_updates += len(x)
            updates += 1
            if max_steps is not None and updates >= max_steps:
                stopped = "max_steps"
                break
        val_mse = validation_mse(model, data["x_val"], args.batch_size, device)
        if not np.isfinite(val_mse):
            raise RuntimeError("Nonfinite validation MSE")
        improved = val_mse < best_mse
        if improved:
            best_mse, best_epoch, stale = val_mse, epoch + 1, 0
        else:
            stale += 1
        row = {"epoch": epoch + 1, "train_loss": train_loss / epoch_cells,
               "train_recon_mse": train_recon / epoch_cells, "val_recon_mse": val_mse,
               "updates": updates, "cell_updates": cell_updates,
               "epoch_cells": epoch_cells, "partial_epoch": epoch_cells < len(data["x_train"]),
               "vq_warmup_scale": warmup}
        history.append(row)
        print(json.dumps(row), flush=True)
        checkpoint = {**common, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                      "epoch": epoch + 1, "next_epoch": epoch + 1, "updates": updates,
                      "cell_updates": cell_updates, "best_val_mse": best_mse,
                      "best_epoch": best_epoch, "stale_epochs": stale, "history": history,
                      "torch_rng_state": torch.get_rng_state(),
                      "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        if improved:
            atomic_torch(args.output / "best.pt", checkpoint)
        atomic_torch(last_path, checkpoint)
        atomic_json(args.output / "history.json", history)
        if stopped == "max_steps":
            break
        if stale >= args.patience:
            stopped = "early_stopping"
            break

    best_path = args.output / "best.pt"
    if not best_path.exists():
        raise RuntimeError("No best checkpoint was produced; increase --epochs")
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state"])
    model.eval()
    model.requires_grad_(False)
    with torch.no_grad():
        effect = original_contributions(model).float().cpu().numpy()
        mask = model.module_mask().float().cpu().numpy()
    train_codes = encode_codes(model, data["x_train"], args.batch_size, device)
    control_codes = encode_codes(model, data["control_x"], args.batch_size, device)
    occupancy = np.stack([np.bincount(train_codes[:, axis], minlength=16) for axis in range(32)]).astype(np.float32)
    occupancy /= occupancy.sum(axis=1, keepdims=True)
    train_gene_std = data["x_train"].std(axis=0, dtype=np.float64).astype(np.float32)
    occupied_mean = np.einsum("ak,akg->ag", occupancy, effect)
    atomic_npz(args.output / "exact_code_effects.npz", genes=data["genes"], gene_symbols=data["gene_symbols"], contributions=effect,
               uniform_centered=effect - effect.mean(axis=1, keepdims=True),
               occupancy_centered=effect - occupied_mean[:, None, :],
               module_mask=mask, train_occupancy=occupancy, train_gene_std=train_gene_std,
               synthetic=np.asarray(data["synthetic"]), smoke_only=np.asarray(args.smoke is not None),
               effect_units=np.asarray("normalized log1p expression (CP10K)"))
    baseline_ids = data.get("control_cell_ids", np.asarray([f"control_row_{i}" for i in range(len(control_codes))]))
    baseline_recon = np.zeros((len(control_codes), len(data["genes"])), dtype=np.float32)
    for axis in range(32):
        baseline_recon += effect[axis, control_codes[:, axis]]
    atomic_npz(args.output / "baseline.npz", codes=control_codes, cell_ids=baseline_ids,
               control_cell_ids=baseline_ids, genes=data["genes"], gene_symbols=data["gene_symbols"], reconstructed_x=baseline_recon,
               observed_x=data["control_x"], synthetic=np.asarray(data["synthetic"]))
    code_payload = {"codes": train_codes, "sampled": np.asarray(False), "genes": data["genes"]}
    if "train_cell_ids" in data:
        code_payload["cell_ids"] = data["train_cell_ids"]
    atomic_npz(args.output / "train_codes.npz", **code_payload)

    # Verify exact additive identity against the original decoder on held-out
    # controls; the observed expression is not assumed to be reconstructed exactly.
    probe_codes = control_codes[:min(16, len(control_codes))]
    with torch.no_grad():
        codebook = model.quantization_codebook()
        z = torch.stack([codebook[axis, torch.as_tensor(probe_codes[:, axis], dtype=torch.long, device=device)]
                         for axis in range(32)], dim=1)
        direct = model.decode(z).float().cpu().numpy()
    identity_error = float(np.max(np.abs(direct - baseline_recon[:len(probe_codes)])))
    if identity_error > 2e-5:
        raise RuntimeError(f"Exact effect additivity check failed: {identity_error}")
    metadata = {"status": "smoke_complete" if args.smoke is not None else "fit_and_export_complete",
                "not_paper_reproduction": True, "synthetic": data["synthetic"],
                "smoke_only": args.smoke is not None, "device": str(device),
                "prepared_path": str(prepared), "prepared_sha256": fingerprint,
                "provenance": data["provenance"], "source_sha256": source_hashes,
                "source_repository": str(args.lexa_repo.resolve()),
                "architecture": config, "scaffold": scaffold_meta,
                "training": {**train_cfg, "epochs_requested": args.epochs,
                             "epochs_visited": len(history), "updates": updates,
                             "cell_updates": cell_updates, "stopped": stopped,
                             "resume_semantics": "epoch boundary; interrupted partial epoch is not replayed"},
                "n_train": len(data["x_train"]), "n_val": len(data["x_val"]),
                "n_control": len(data["control_x"]), "n_genes": len(data["genes"]),
                "gene_ids": data["genes"].tolist(), "gene_symbols": data["gene_symbols"].tolist(),
                "best_epoch": best_epoch, "best_val_reconstruction_mse": best_mse,
                "selection_metric": "mean squared error per cell-gene in validation normalized log1p expression",
                "normalization_total": 10000, "effect_units": "normalized log1p expression",
                "uniform_centering": "subtract unweighted per-axis state mean; not an untreated state",
                "transition_effect": "contributions[a,new,g] - contributions[a,old,g]",
                "additivity_max_abs_error": identity_error,
                "median_axis_active_states": float(np.median((occupancy > 0).sum(axis=1))),
                "split_verification": "caller provided train/val; biological replicate independence must be established in preparation provenance",
                "wall_seconds": round(time.time() - started, 3),
                "limitations": ["Reduced-width architecture and coexpression scaffold are pilot changes.",
                                "Low reconstruction MSE alone does not validate state semantics or perturbation prediction.",
                                "Uniformly centered states are relative effects, not absolute gene expression.",
                                "MSE decoder output is unbounded and may be negative; no clipping is applied to preserve exact additivity.",
                                "Observed single-cell baseline expression contains reconstruction residual not explained by the codes."]}
    atomic_json(args.output / "metadata.json", metadata)
    output_files = ("best.pt", "last.pt", "history.json", "pilot_scaffold.npz",
                    "exact_code_effects.npz", "baseline.npz", "train_codes.npz", "metadata.json")
    if any(not (args.output / name).is_file() for name in output_files):
        raise RuntimeError("Export missing required artifacts")
    atomic_json(args.output / "COMPLETE.json", {"status": metadata["status"],
                "prepared_sha256": fingerprint, "synthetic": data["synthetic"],
                "smoke_only": args.smoke is not None, "files": list(output_files),
                "exact_code_effects_sha256": file_sha256(args.output / "exact_code_effects.npz")})
    return metadata


def main():
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
