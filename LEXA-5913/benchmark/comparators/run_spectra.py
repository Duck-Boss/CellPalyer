"""Fit and evaluate Spectra with global pathway priors."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import non_negative_factorization
import torch
import Spectra

from data_pipeline.bundle import load_data_bundle
from data_pipeline.data import H5CSRDataset, make_dataloader
from data_pipeline.runtime import load_config
from benchmark.comparators.run_pca_harmony_scvi import fine_scope_mask, fit_linear_probe


LABELS = ("cell_type", "coarse_cell_type", "assay", "dataset_id", "donor_id")


def sampled_dataset(dataset, maximum, seed):
    if len(dataset) <= maximum:
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=maximum, replace=False)).tolist()
    return torch.utils.data.Subset(dataset, indices)


def loader(dataset, config):
    return make_dataloader(dataset, int(config["training"]["batch_size"]), False, int(config["training"]["num_workers"]))


def select_hvg(dataset, config, number):
    total = total_sq = None
    cells = 0
    for batch in loader(dataset, config):
        x = batch["x"].numpy().astype(np.float64, copy=False)
        if total is None:
            total = np.zeros(x.shape[1], dtype=np.float64)
            total_sq = np.zeros(x.shape[1], dtype=np.float64)
        total += x.sum(axis=0)
        total_sq += np.square(x).sum(axis=0)
        cells += len(x)
    variance = np.maximum(total_sq / cells - np.square(total / cells), 0.0)
    return np.argsort(variance)[::-1][:number].copy()


def collect(dataset, config, hvg):
    expression = []
    counts = []
    labels = defaultdict(list)
    for batch in loader(dataset, config):
        expression.append(batch["x"][:, hvg].numpy().astype(np.float32, copy=False))
        counts.append(sp.csr_matrix(batch["x_counts"][:, hvg].numpy().astype(np.float32, copy=False)))
        for key in LABELS:
            labels[key].extend(str(value) for value in batch[key])
    return {"x": np.concatenate(expression), "counts": sp.vstack(counts, format="csr"),
            **{key: np.asarray(value, dtype=object) for key, value in labels.items()}}


def probe_suite(train_z, test_z, train, test, seed):
    probes = {}
    for label in ("coarse_cell_type", "assay", "dataset_id"):
        probes[label] = fit_linear_probe(
            train_z, train[label], test_z, test[label], min_class_count=32,
            max_samples_per_class=256, seed=seed,
        )
    for scope in ("all", "immune", "t_cell"):
        train_scope = fine_scope_mask(train["cell_type"], train["coarse_cell_type"], scope)
        test_scope = fine_scope_mask(test["cell_type"], test["coarse_cell_type"], scope)
        key = "cell_type" if scope == "all" else f"cell_type@{scope}"
        probes[key] = fit_linear_probe(
            train_z, train["cell_type"], test_z, test["cell_type"],
            min_class_count=32, max_samples_per_class=256, seed=seed,
            train_scope=train_scope, test_scope=test_scope, min_test_class_count=16,
        )
    return probes


def load_global_priors(root: Path, genes: set[str]) -> dict[str, list[str]]:
    priors: dict[str, list[str]] = {}
    for path in sorted(root.glob("*.gmt")):
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.rstrip().split("\t")
            if len(fields) < 3:
                continue
            members = sorted({gene for gene in fields[2:] if gene in genes})
            if len(members) >= 3:
                priors[f"{path.stem}:{fields[0]}"] = members
    if not priors:
        raise RuntimeError(f"no usable global priors in {root}")
    return priors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-run-id", default="iclr260814_core")
    parser.add_argument("--gene-set-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train", type=int, default=20_000)
    parser.add_argument("--max-test", type=int, default=10_000)
    parser.add_argument("--hvg", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--sample-seed", type=int, default=20260815)
    parser.add_argument("--model-dir", default=None, help="Reuse fitted Spectra factors and train scores.")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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
        train_ds = sampled_dataset(train_base, args.max_train, args.sample_seed)
        test_ds = sampled_dataset(test_base, args.max_test, args.sample_seed + 1)
        if args.model_dir:
            saved = np.load(Path(args.model_dir) / "programs_and_markers.npz", allow_pickle=False)
            saved_genes = saved["genes"].astype(str)
            gene_index = {gene: index for index, gene in enumerate(bundle["gene_vocab"])}
            hvg = np.asarray([gene_index[gene] for gene in saved_genes], dtype=np.int64)
        else:
            hvg = select_hvg(train_ds, config, args.hvg)
        train = collect(train_ds, config, hvg)
        test = collect(test_ds, config, hvg)
    finally:
        train_base.close()
        test_base.close()
    genes = np.asarray(bundle["gene_vocab"], dtype=str)[hvg]
    if args.model_dir:
        saved = np.load(Path(args.model_dir) / "programs_and_markers.npz", allow_pickle=False)
        factors = saved["factors"].astype(np.float64)
        train_z = np.load(Path(args.model_dir) / "train_scores_and_labels.npz", allow_pickle=False)["scores"].astype(np.float32)
        priors = {}
        if len(train_z) != len(train["x"]):
            raise RuntimeError("reused Spectra train scores do not match deterministic train sample")
    else:
        priors = load_global_priors(Path(args.gene_set_dir), set(genes.tolist()))
        adata = ad.AnnData(
            X=train["x"].astype(np.float32, copy=False),
            obs=pd.DataFrame(index=[f"train_{i}" for i in range(len(train["x"]))]),
            var=pd.DataFrame({"highly_variable": True}, index=genes),
        )
        model = Spectra.est_spectra(
            adata=adata, gene_set_dictionary=priors, L=32,
            use_highly_variable=True, use_weights=True, lam=0.01, delta=0.001,
            kappa=None, rho=0.001, use_cell_types=False, n_top_vals=0,
            filter_sets=True, clean_gs=True, min_gs_num=3, label_factors=False,
            num_epochs=args.epochs, verbose=True,
        )
        train_z = np.asarray(adata.obsm["SPECTRA_cell_scores"], dtype=np.float32)
        factors = np.asarray(adata.uns["SPECTRA_factors"], dtype=np.float64)
        if factors.shape[0] != 32:
            factors = factors.T
    test_z, _, _ = non_negative_factorization(
        test["x"].astype(np.float64, copy=False), H=factors, n_components=32,
        init="custom", update_H=False, solver="cd", max_iter=200,
    )
    train_z /= np.maximum(train_z.sum(axis=1, keepdims=True), 1e-12)
    test_z /= np.maximum(test_z.sum(axis=1, keepdims=True), 1e-12)
    np.savez_compressed(output / "programs_and_markers.npz", genes=genes, factors=factors.astype(np.float32))
    np.savez_compressed(
        output / "train_scores_and_labels.npz",
        scores=train_z,
        coarse_cell_type=train["coarse_cell_type"].astype(str),
        assay=train["assay"].astype(str),
        dataset_id=train["dataset_id"].astype(str),
    )
    result = {
        "method": "spectra_32_global_priors", "source_commit": "9c079a584ab71ea83e1e0ed471f678d2f2a997a2",
        "train_cells": len(train_z), "test_cells": len(test_z), "hvg": len(genes),
        "epochs": args.epochs, "num_global_pathway_priors": len(priors) if priors else None,
        "reused_model_dir": args.model_dir,
        "cell_type_labels_used": False, "frozen_test_projection": True,
        "seed": args.seed, "sample_seed": args.sample_seed,
        "fit_seconds": time.time() - started,
        "probes": probe_suite(train_z, test_z, train, test, args.seed),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
