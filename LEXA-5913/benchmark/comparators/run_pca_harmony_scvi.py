"""Fit and evaluate PCA, Harmony, and scVI representations."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import torch

from data_pipeline.bundle import load_data_bundle
from data_pipeline.data import H5CSRDataset, make_dataloader
from data_pipeline.runtime import load_config

LABELS = ("cell_type", "coarse_cell_type", "assay", "dataset_id", "donor_id")


def sampled_dataset(dataset, maximum, seed):
    if len(dataset) <= maximum:
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=maximum, replace=False)).tolist()
    return torch.utils.data.Subset(dataset, indices)


def fine_scope_mask(fine_labels, coarse_labels, scope):
    fine = np.asarray([str(value).strip().casefold() for value in fine_labels], dtype=object)
    coarse = np.asarray([str(value).strip().casefold() for value in coarse_labels], dtype=object)
    if scope in {"", "all"}:
        return np.ones(len(fine), dtype=bool)
    if scope == "immune":
        return coarse == "immune"
    if scope == "t_cell":
        is_t = np.asarray([
            ("t cell" in value)
            or ("t-cell" in value)
            or ("t lymphocyte" in value)
            or ("thymocyte" in value)
            or ("natural killer t" in value)
            or ("nkt" in value)
            for value in fine
        ], dtype=bool)
        return (coarse == "immune") & is_t
    raise ValueError(f"unsupported fine-cell scope: {scope}")


def fit_linear_probe(
    train_x, train_y, test_x, test_y, min_class_count,
    max_samples_per_class=None, seed=0, train_scope=None, test_scope=None,
    min_test_class_count=1,
):
    train_y = np.asarray([str(value) for value in train_y], dtype=object)
    test_y = np.asarray([str(value) for value in test_y], dtype=object)
    train_scope = np.ones(len(train_y), dtype=bool) if train_scope is None else np.asarray(train_scope, dtype=bool)
    test_scope = np.ones(len(test_y), dtype=bool) if test_scope is None else np.asarray(test_scope, dtype=bool)
    invalid = {"", "na", "unknown", "none", "nan"}
    train_counter = Counter(train_y[train_scope].tolist())
    test_counter = Counter(test_y[test_scope].tolist())
    classes = {
        label for label, count in train_counter.items()
        if count >= min_class_count
        and test_counter.get(label, 0) >= min_test_class_count
        and label.strip().casefold() not in invalid
    }
    train_mask = train_scope & np.isin(train_y, list(classes))
    test_mask = test_scope & np.isin(test_y, list(classes))
    if train_mask.sum() < 2 or test_mask.sum() < 2 or len(classes) < 2:
        return None
    train_indices = np.where(train_mask)[0]
    if max_samples_per_class is not None and int(max_samples_per_class) > 0:
        rng = np.random.default_rng(seed)
        selected = []
        for label in sorted(classes):
            indices = train_indices[train_y[train_indices] == label]
            if len(indices) > int(max_samples_per_class):
                indices = rng.choice(indices, size=int(max_samples_per_class), replace=False)
            selected.append(indices)
        train_indices = np.concatenate(selected)
        rng.shuffle(train_indices)
    test_indices = np.where(test_mask)[0]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_x[train_indices])
    x_test = scaler.transform(test_x[test_indices])
    classifier = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=1e-4, max_iter=10, tol=None,
        random_state=seed, n_jobs=1, average=True,
    )
    classifier.fit(x_train, train_y[train_indices])
    predicted = classifier.predict(x_test)
    observed = test_y[test_indices]
    return {
        "accuracy": float(accuracy_score(observed, predicted)),
        "macro_f1": float(f1_score(observed, predicted, average="macro", zero_division=0)),
        "num_classes": float(len(classes)),
        "num_train_samples": float(len(train_indices)),
        "num_test_samples": float(len(test_indices)),
        "train_scope_cells": float(train_scope.sum()),
        "test_scope_cells": float(test_scope.sum()),
        "test_cell_coverage": float(test_mask.sum() / max(int(test_scope.sum()), 1)),
    }


def probe_suite(train_z, test_z, train, test, seed):
    def required_probe(train_labels, test_labels, train_scope=None, test_scope=None, fine=False):
        result = fit_linear_probe(
            train_z, train_labels, test_z, test_labels,
            min_class_count=32, max_samples_per_class=256, seed=seed,
            train_scope=train_scope, test_scope=test_scope,
            min_test_class_count=16 if fine else 1,
        )
        if result is None:
            raise RuntimeError("probe has fewer than two supported classes")
        return result

    probes = {
        label: required_probe(train[label], test[label])
        for label in ("coarse_cell_type", "assay", "dataset_id")
    }
    for scope in ("all", "immune", "t_cell"):
        train_scope = fine_scope_mask(train["cell_type"], train["coarse_cell_type"], scope)
        test_scope = fine_scope_mask(test["cell_type"], test["coarse_cell_type"], scope)
        key = "cell_type" if scope == "all" else f"cell_type@{scope}"
        probes[key] = required_probe(
            train["cell_type"], test["cell_type"], train_scope, test_scope, fine=True,
        )
    return probes


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
    return {
        "x": np.concatenate(expression),
        "counts": sp.vstack(counts, format="csr"),
        **{key: np.asarray(value, dtype=object) for key, value in labels.items()},
    }


def evaluate(name, train_z, test_z, train, test, seconds, extra=None):
    return {
        "method": name,
        "latent_dim": int(train_z.shape[1]),
        "fit_seconds": float(seconds),
        "probes": probe_suite(train_z, test_z, train, test, 20260815),
        **(extra or {}),
    }


def main():
    import harmonypy as hm
    import scvi

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-run-id", default="iclr260814_core")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train", type=int, default=100_000)
    parser.add_argument("--max-test", type=int, default=50_000)
    parser.add_argument("--hvg", type=int, default=2_000)
    parser.add_argument("--scvi-epochs", type=int, default=20)
    parser.add_argument("--scarches-epochs", type=int, default=10)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config["data"]["_data_run_id"] = args.data_run_id
    config["data"]["return_raw_counts"] = True
    bundle = load_data_bundle(Path(config["output"]["root"]) / args.data_run_id / "data_bundle.slim.json", config)
    train_base = H5CSRDataset(bundle["splits"]["train"], bundle["file_gene_maps"], bundle["gene_vocab"], config, bundle["file_metas"])
    test_base = H5CSRDataset(bundle["splits"]["test"], bundle["file_gene_maps"], bundle["gene_vocab"], config, bundle["file_metas"])
    try:
        train_ds = sampled_dataset(train_base, args.max_train, 20260815)
        test_ds = sampled_dataset(test_base, args.max_test, 20260816)
        hvg = select_hvg(train_ds, config, args.hvg)
        train = collect(train_ds, config, hvg)
        test = collect(test_ds, config, hvg)
    finally:
        train_base.close()
        test_base.close()
    np.save(output / "hvg_indices.npy", hvg)
    results = []

    started = time.time()
    pca = PCA(n_components=128, svd_solver="randomized", random_state=20260815)
    train_pca = pca.fit_transform(train["x"])
    test_pca = pca.transform(test["x"])
    pca_seconds = time.time() - started
    for dimensions in (32, 128):
        reconstructed = test_pca[:, :dimensions] @ pca.components_[:dimensions] + pca.mean_
        mse = float(np.mean(np.square(reconstructed - test["x"])))
        cosine = float(np.mean(np.sum(reconstructed * test["x"], axis=1) / np.maximum(np.linalg.norm(reconstructed, axis=1) * np.linalg.norm(test["x"], axis=1), 1e-8)))
        results.append(evaluate(f"pca_{dimensions}", train_pca[:, :dimensions], test_pca[:, :dimensions], train, test, pca_seconds, {"mse": mse, "cosine": cosine}))

    started = time.time()
    combined = np.concatenate([train_pca[:, :50], test_pca[:, :50]])
    obs = pd.DataFrame({
        "dataset_id": pd.Categorical(np.concatenate([train["dataset_id"], test["dataset_id"]])),
        "assay": pd.Categorical(np.concatenate([train["assay"], test["assay"]])),
    })
    harmony = hm.run_harmony(combined, obs, ["dataset_id", "assay"], max_iter_harmony=10, verbose=False)
    corrected = np.asarray(harmony.Z_corr)
    if corrected.shape[0] != len(combined):
        corrected = corrected.T
    results.append(evaluate("harmony_50_transductive", corrected[: len(train["x"])], corrected[len(train["x"]):], train, test, time.time() - started, {"transductive_unlabeled_test_adaptation": True, "reconstruction": None}))

    genes = np.asarray(bundle["gene_vocab"], dtype=object)[hvg].astype(str)
    train_obs = pd.DataFrame({key: pd.Categorical(train[key]) for key in LABELS})
    test_obs = pd.DataFrame({key: pd.Categorical(test[key]) for key in LABELS})
    train_obs.index = [f"train_{idx}" for idx in range(len(train_obs))]
    test_obs.index = [f"test_{idx}" for idx in range(len(test_obs))]
    var = pd.DataFrame(index=genes)
    train_ad = ad.AnnData(train["counts"], obs=train_obs, var=var.copy())
    test_ad = ad.AnnData(test["counts"], obs=test_obs, var=var.copy())
    query_ad = ad.AnnData(test["counts"].copy(), obs=test_obs.copy(), var=var.copy())
    scvi.model.SCVI.setup_anndata(train_ad, batch_key="dataset_id")
    started = time.time()
    model = scvi.model.SCVI(train_ad, n_latent=32, n_layers=2, n_hidden=256, gene_likelihood="nb")
    model.train(max_epochs=args.scvi_epochs, accelerator="gpu", devices=1, batch_size=1024, enable_progress_bar=False)
    train_scvi = model.get_latent_representation(train_ad)
    test_manager = model.adata_manager.transfer_fields(test_ad, extend_categories=True)
    model._register_manager_for_instance(test_manager)
    test_scvi = model.get_latent_representation(test_ad)
    results.append(evaluate("scvi_32_frozen", train_scvi, test_scvi, train, test, time.time() - started, {"reconstruction": "not_run_in_screen"}))
    started = time.time()
    query_model = scvi.model.SCVI.load_query_data(query_ad, model, freeze_dropout=True)
    query_model.train(
        max_epochs=args.scarches_epochs,
        accelerator="gpu",
        devices=1,
        batch_size=1024,
        enable_progress_bar=False,
    )
    test_scarches = query_model.get_latent_representation()
    results.append(evaluate(
        "scarches_32_adapted",
        train_scvi,
        test_scarches,
        train,
        test,
        time.time() - started,
        {"unlabeled_test_adaptation": True, "reconstruction": "not_run_in_screen"},
    ))
    payload = {
        "screen": {"train_cells": len(train["x"]), "test_cells": len(test["x"]), "hvg": len(hvg)},
        "results": results,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
