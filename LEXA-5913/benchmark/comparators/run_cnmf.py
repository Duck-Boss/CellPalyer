"""Fit cNMF and project held-out cells onto its factors."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from cnmf import cNMF, load_df_from_npz
from sklearn.decomposition import non_negative_factorization
import torch

from benchmark.comparators.run_pca_harmony_scvi import LABELS, probe_suite, sampled_dataset
from data_pipeline.bundle import load_data_bundle
from data_pipeline.data import H5CSRDataset, make_dataloader
from data_pipeline.runtime import load_config


def collect(dataset, config):
    counts = []
    labels = {key: [] for key in LABELS}
    loader = make_dataloader(dataset, int(config["training"]["batch_size"]), False, int(config["training"]["num_workers"]))
    for batch in loader:
        counts.append(sp.csr_matrix(batch["x_counts"].numpy().astype(np.float32, copy=False)))
        for key in LABELS:
            labels[key].extend(str(value) for value in batch[key])
    return {"counts": sp.vstack(counts, format="csr"), **{key: np.asarray(value, dtype=object) for key, value in labels.items()}}


def factorize_one(output_dir: str, name: str, worker: int, workers: int) -> None:
    cNMF(output_dir=output_dir, name=name).factorize(worker_i=worker, total_workers=workers)


def grouped_sample(dataset, maximum: int, seed: int, group_fraction: float):
    """Sample cells from a fraction of donor or sample groups."""
    rng = np.random.default_rng(seed)
    if group_fraction < 1.0:
        groups = dataset.records.label_indices("group_id")
        labels = np.asarray(sorted(groups), dtype=object)
        keep_n = max(2, int(math.ceil(len(labels) * group_fraction)))
        kept = rng.choice(labels, size=keep_n, replace=False)
        eligible = np.concatenate([groups[str(label)] for label in kept])
    else:
        kept = np.asarray([], dtype=object)
        eligible = np.arange(len(dataset), dtype=np.int64)
    if len(eligible) > maximum:
        eligible = rng.choice(eligible, size=maximum, replace=False)
    indices = np.sort(np.asarray(eligible, dtype=np.int64))
    return torch.utils.data.Subset(dataset, indices.tolist()), indices, kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-run-id", default="iclr260814_core")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train", type=int, default=100_000)
    parser.add_argument("--max-test", type=int, default=50_000)
    parser.add_argument("--components", type=int, default=32)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-nmf-iter", type=int, default=200)
    parser.add_argument("--fit-seed", type=int, default=20260815)
    parser.add_argument("--sample-seed", type=int, default=20260815)
    parser.add_argument("--test-sample-seed", type=int, default=20260816)
    parser.add_argument("--group-fraction", type=float, default=1.0)
    parser.add_argument("--model-dir", default=None, help="Reuse an existing fitted cNMF directory.")
    args = parser.parse_args()
    started = time.time()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config["data"]["_data_run_id"] = args.data_run_id
    config["data"]["return_raw_counts"] = True
    bundle = load_data_bundle(Path(config["output"]["root"]) / args.data_run_id / "data_bundle.slim.json", config)
    train_base = H5CSRDataset(bundle["splits"]["train"], bundle["file_gene_maps"], bundle["gene_vocab"], config, bundle["file_metas"])
    test_base = H5CSRDataset(bundle["splits"]["test"], bundle["file_gene_maps"], bundle["gene_vocab"], config, bundle["file_metas"])
    try:
        train_ds, train_indices, kept_groups = grouped_sample(
            train_base, args.max_train, args.sample_seed, args.group_fraction,
        )
        test_ds, test_indices, _ = grouped_sample(test_base, args.max_test, args.test_sample_seed, 1.0)
        np.save(output / "train_indices.npy", train_indices)
        np.save(output / "test_indices.npy", test_indices)
        train = collect(train_ds, config)
        test = collect(test_ds, config)
    finally:
        train_base.close()
        test_base.close()

    genes = np.asarray(bundle["gene_vocab"], dtype=str)
    name = "cnmf32"
    model_root = Path(args.model_dir) if args.model_dir else output
    model = cNMF(output_dir=str(model_root), name=name)
    if not args.model_dir:
        train_obs = pd.DataFrame(index=[f"train_{i}" for i in range(train["counts"].shape[0])])
        counts_path = output / "train_counts.h5ad"
        ad.AnnData(train["counts"], obs=train_obs, var=pd.DataFrame(index=genes)).write_h5ad(counts_path, compression="gzip")
        model.prepare(
            counts_fn=str(counts_path), components=[args.components], n_iter=args.replicates,
            seed=args.fit_seed, num_highvar_genes=2000, max_NMF_iter=args.max_nmf_iter,
        )
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(factorize_one, str(output), name, worker, args.workers) for worker in range(args.workers)]
            for future in futures:
                future.result()
        model.combine(components=[args.components])
        model.consensus(args.components, density_threshold=0.5, show_clustering=False, build_ref=True, close_clustergram_fig=True)
    train_usage, _scores, _spectra_tpm, _top = model.load_results(args.components, 0.5, norm_usage=True)

    normalized = ad.read_h5ad(model.paths["normalized_counts"])
    selected_genes = np.asarray(normalized.var_names, dtype=str)
    gene_index = {gene: i for i, gene in enumerate(genes)}
    selected = np.asarray([gene_index[gene] for gene in selected_genes], dtype=np.int64)
    train_selected = train["counts"][:, selected].astype(np.float64)
    mean = np.asarray(train_selected.mean(axis=0)).ravel()
    mean_sq = np.asarray(train_selected.power(2).mean(axis=0)).ravel()
    scale = np.sqrt(np.maximum(mean_sq - mean * mean, 1e-12))
    test_scaled = test["counts"][:, selected].astype(np.float64).multiply(1.0 / scale)
    density = str(0.5).replace(".", "_")
    spectra = load_df_from_npz(model.paths["consensus_spectra"] % (args.components, density))
    h = np.asarray(spectra, dtype=np.float64)
    if h.shape[0] != args.components:
        h = h.T
    test_usage, _, _ = non_negative_factorization(
        test_scaled, H=h, n_components=args.components, init="custom", update_H=False,
        solver="cd", max_iter=args.max_nmf_iter,
    )
    test_usage /= np.maximum(test_usage.sum(axis=1, keepdims=True), 1e-12)
    train_z = np.asarray(train_usage, dtype=np.float32)
    test_z = test_usage.astype(np.float32)
    results = {
        "method": "cnmf_32",
        "train_cells": len(train_z), "test_cells": len(test_z), "components": args.components,
        "replicates": args.replicates, "fit_seconds": time.time() - started,
        "fit_seed": args.fit_seed, "sample_seed": args.sample_seed,
        "group_fraction": args.group_fraction,
        "sampled_group_count": int(len(kept_groups)) if args.group_fraction < 1.0 else None,
        "reused_model_dir": str(model_root) if args.model_dir else None,
        "frozen_test_projection": True,
        "probes": probe_suite(train_z, test_z, train, test, 20260815),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
