"""Stream full-split discrete codes and a bounded latent sample to disk."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from lexa.lexa_model import build_lexa_model
from data_pipeline.bundle import load_data_bundle, warm_csr_caches
from data_pipeline.checkpoints import checkpoint_metadata, load_checkpoint_and_config
from data_pipeline.data import H5CSRDataset, make_dataloader
from data_pipeline.runtime import categorical_indices_for_model, load_config, make_output_dirs, pick_device


LABELS = [
    "dataset_id", "group_id", "donor_id", "sample_id", "technical_block", "batch_id", "cell_type",
    "coarse_cell_type", "author_cell_type", "sex", "ethnicity", "geography",
    "tissue", "disease", "assay", "suspension_type", "development_stage", "age_parse_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-run-id", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-latent-samples", type=int, default=30000)
    parser.add_argument("--gene-scale-samples", type=int, default=100000)
    parser.add_argument("--analysis-seed", type=int, default=2701)
    parser.add_argument("--output-name", default="representations")
    return parser.parse_args()


class LabelEncoder:
    def __init__(self, n: int) -> None:
        self.values = np.empty(n, dtype=np.int32)
        self.mapping: Dict[str, int] = {}
        self.categories: List[str] = []

    def put(self, start: int, values: List[Any]) -> None:
        out = np.empty(len(values), dtype=np.int32)
        for idx, raw in enumerate(values):
            value = str(raw)
            code = self.mapping.get(value)
            if code is None:
                code = len(self.categories)
                self.mapping[value] = code
                self.categories.append(value)
            out[idx] = code
        self.values[start : start + len(values)] = out


@torch.no_grad()
def extract_split(
    model: torch.nn.Module,
    dataset: H5CSRDataset,
    config: Dict[str, Any],
    device: torch.device,
    split: str,
    output: Path,
    max_latent_samples: int,
    gene_scale_samples: int,
    analysis_seed: int,
) -> Dict[str, Any]:
    n = len(dataset)
    loader = make_dataloader(dataset, int(config["training"]["batch_size"]), False, int(config["training"]["num_workers"]))
    num_axes = int(getattr(model, "num_axes", getattr(model, "latent_tokens", 1)))
    codes = np.empty((n, num_axes), dtype=np.uint8)
    row_indices = np.empty(n, dtype=np.int32)
    ages = np.empty(n, dtype=np.int16)
    cell_mse = np.empty(n, dtype=np.float32)
    cell_mae = np.empty(n, dtype=np.float32)
    cell_cosine = np.empty(n, dtype=np.float32)
    encoders = {key: LabelEncoder(n) for key in LABELS}
    rng = np.random.default_rng(int(analysis_seed) + {"train": 101, "val": 102, "test": 103}[split])
    take = min(n, max(0, int(max_latent_samples)))
    latent_positions = np.sort(rng.choice(n, take, replace=False)) if take else np.empty(0, dtype=np.int64)
    scale_take = min(n, max(0, int(gene_scale_samples))) if split == "train" else 0
    scale_positions = np.sort(rng.choice(n, scale_take, replace=False)) if scale_take else np.empty(0, dtype=np.int64)
    latent_ze: List[np.ndarray] = []
    latent_zq: List[np.ndarray] = []
    latent_rows: List[np.ndarray] = []
    gene_sum = np.zeros(len(dataset.gene_vocab), dtype=np.float64) if scale_take else None
    gene_sum_sq = np.zeros(len(dataset.gene_vocab), dtype=np.float64) if scale_take else None
    gene_count = 0
    recon_sse = np.zeros(len(dataset.gene_vocab), dtype=np.float64)
    recon_observed = np.zeros(len(dataset.gene_vocab), dtype=np.float64)
    cursor = 0
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.eval()
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        count_target = batch.get("x_counts")
        if count_target is not None:
            count_target = count_target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda", dtype=amp_dtype):
            categorical = categorical_indices_for_model(batch, model, device)
            result = model(x, count_target=count_target, categorical_indices=categorical)
        batch_n = len(x)
        end = cursor + batch_n
        index = result["indices"].detach().cpu().numpy()
        if index.ndim == 1:
            index = index[:, None]
        codes[cursor:end] = index.astype(np.uint8, copy=False)
        pred = result["x_recon"].float()
        diff = pred - x.float()
        observed = batch.get("x_mask")
        if observed is None:
            observed_float = torch.ones_like(diff)
        else:
            observed_float = observed.to(device, non_blocking=True).to(diff.dtype)
        elements = observed_float.sum(dim=1).clamp_min(1.0)
        scored_diff = diff * observed_float
        scored_x = x.float() * observed_float
        scored_pred = pred * observed_float
        cell_mse[cursor:end] = (scored_diff.square().sum(dim=1) / elements).cpu().numpy().astype(np.float32)
        cell_mae[cursor:end] = (scored_diff.abs().sum(dim=1) / elements).cpu().numpy().astype(np.float32)
        cosine = torch.sum(scored_x * scored_pred, dim=1) / (
            torch.linalg.vector_norm(scored_x, dim=1).clamp_min(1e-8)
            * torch.linalg.vector_norm(scored_pred, dim=1).clamp_min(1e-8)
        )
        cell_cosine[cursor:end] = cosine.cpu().numpy().astype(np.float32)
        recon_sse += scored_diff.square().sum(dim=0).cpu().numpy().astype(np.float64)
        recon_observed += observed_float.sum(dim=0).cpu().numpy().astype(np.float64)
        row_indices[cursor:end] = np.asarray(batch["row_idx"], dtype=np.int32)
        ages[cursor:end] = np.asarray(batch.get("age", [-1] * batch_n), dtype=np.int16)
        for key, encoder in encoders.items():
            encoder.put(cursor, list(batch.get(key, ["unknown"] * batch_n)))

        left = int(np.searchsorted(latent_positions, cursor, side="left"))
        right = int(np.searchsorted(latent_positions, end, side="left"))
        if right > left:
            local = latent_positions[left:right] - cursor
            latent_ze.append(result["z_e"][local].reshape(len(local), -1).float().cpu().numpy().astype(np.float16))
            latent_zq.append(result["z_q"][local].reshape(len(local), -1).float().cpu().numpy().astype(np.float16))
            latent_rows.append(latent_positions[left:right].astype(np.int64))
        if scale_take:
            left = int(np.searchsorted(scale_positions, cursor, side="left"))
            right = int(np.searchsorted(scale_positions, end, side="left"))
            if right > left:
                local = scale_positions[left:right] - cursor
                values = x[local].float().cpu().numpy().astype(np.float64)
                assert gene_sum is not None and gene_sum_sq is not None
                gene_sum += values.sum(axis=0)
                gene_sum_sq += np.square(values).sum(axis=0)
                gene_count += len(values)
        cursor = end
        if cursor == batch_n or cursor % 200000 < batch_n:
            print(f"extract {split}: {cursor}/{n}", flush=True)
    if cursor != n:
        raise RuntimeError(f"Extracted {cursor} cells from a {n}-cell split")

    payload = {
        "codes": codes,
        "row_idx": row_indices,
        "age": ages,
        "cell_mse": cell_mse,
        "cell_mae": cell_mae,
        "cell_cosine": cell_cosine,
        **{f"label_{key}": encoder.values for key, encoder in encoders.items()},
    }
    np.savez_compressed(output / f"{split}_codes_and_metadata.npz", **payload)
    categories = {key: encoder.categories for key, encoder in encoders.items()}
    with (output / f"{split}_categories.json").open("w", encoding="utf-8") as handle:
        json.dump(categories, handle, indent=2)
    if take:
        np.savez_compressed(
            output / f"{split}_latent_sample.npz",
            manifest_position=np.concatenate(latent_rows),
            z_e=np.concatenate(latent_ze),
            z_q=np.concatenate(latent_zq),
        )
    if gene_count:
        assert gene_sum is not None and gene_sum_sq is not None
        mean = gene_sum / gene_count
        variance = np.maximum(0.0, gene_sum_sq / gene_count - mean**2)
        np.savez_compressed(
            output / "train_gene_scale.npz",
            genes=np.asarray(dataset.gene_vocab, dtype=str),
            mean=mean.astype(np.float32),
            std=np.sqrt(variance).astype(np.float32),
            num_cells=np.asarray([gene_count], dtype=np.int64),
        )
    return {
        "split": split,
        "num_cells": n,
        "mean_cell_mse": float(cell_mse.mean()),
        "median_cell_mse": float(np.median(cell_mse)),
        "latent_samples": take,
        "gene_scale_samples": gene_count,
        "gene_mse_mean": float(recon_sse.sum() / max(recon_observed.sum(), 1)),
        "observed_gene_elements": int(recon_observed.sum()),
    }


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    out = make_output_dirs(base, args.run_id)
    ckpt, config, ckpt_path = load_checkpoint_and_config(args.config, out["root"], args.checkpoint, return_path=True)
    config.setdefault("data", {})["_data_run_id"] = args.data_run_id
    bundle_root = Path(config["output"]["root"]) / args.data_run_id
    bundle_path = bundle_root / "data_bundle.slim.json"
    if not bundle_path.exists():
        bundle_path = bundle_root / "data_bundle.json"
    bundle = load_data_bundle(bundle_path, config)
    device = pick_device(config["training"]["device"])
    model = build_lexa_model(len(bundle["gene_vocab"]), config, gene_vocab=bundle["gene_vocab"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    output = out["root"] / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    split_names = [x.strip() for x in args.splits.split(",") if x.strip()]
    warm_csr_caches(bundle, config, split_names=split_names)
    rows = []
    for split in split_names:
        ds = H5CSRDataset(bundle["splits"][split], bundle["file_gene_maps"], bundle["gene_vocab"], config, bundle["file_metas"])
        try:
            rows.append(extract_split(model, ds, config, device, split, output, args.max_latent_samples, args.gene_scale_samples, args.analysis_seed))
        finally:
            ds.close()
    summary = {
        "run_id": args.run_id,
        "data_run_id": args.data_run_id,
        "checkpoint": checkpoint_metadata(ckpt, config, ckpt_path),
        "analysis_sampling_seed": int(args.analysis_seed),
        "splits": rows,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
