"""Prepare CellVQ inputs and evaluate its token representation."""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from benchmark.comparators.run_pca_harmony_scvi import LABELS, probe_suite, sampled_dataset
from data_pipeline.bundle import load_data_bundle
from data_pipeline.data import H5CSRDataset, make_dataloader
from data_pipeline.runtime import load_config


def export(dataset, config, genes, output: Path) -> int:
    blocks = []
    labels = {key: [] for key in LABELS}
    rows = []
    loader = make_dataloader(dataset, int(config["training"]["batch_size"]), False, int(config["training"]["num_workers"]))
    for batch in loader:
        blocks.append(sp.csr_matrix(batch["x_counts"].numpy().astype(np.float32, copy=False)))
        rows.extend(np.asarray(batch["row_idx"], dtype=np.int64).tolist())
        for key in LABELS:
            labels[key].extend(str(value) for value in batch[key])
    obs = pd.DataFrame({key: values for key, values in labels.items()})
    obs["row_idx"] = rows
    adata = ad.AnnData(sp.vstack(blocks, format="csr"), obs=obs, var=pd.DataFrame(index=np.asarray(genes, dtype=str)))
    adata.write_h5ad(output, compression="gzip")
    return adata.n_obs


def export_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-run-id", default="iclr260814_core")
    parser.add_argument("--bundle-root", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-train", type=int, default=20000)
    parser.add_argument("--max-test", type=int, default=20000)
    parser.add_argument("--sample-seed", type=int, default=20260815)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config["data"]["_data_run_id"] = args.data_run_id
    config["data"]["return_raw_counts"] = True
    bundle_root = Path(args.bundle_root) if args.bundle_root else Path(config["output"]["root"]) / args.data_run_id
    bundle = load_data_bundle(bundle_root / "data_bundle.slim.json", config)
    counts = {}
    for split, maximum, seed in [("train", args.max_train, args.sample_seed), ("test", args.max_test, args.sample_seed + 1)]:
        base = H5CSRDataset(bundle["splits"][split], bundle["file_gene_maps"], bundle["gene_vocab"], config, bundle["file_metas"])
        try:
            counts[split] = export(sampled_dataset(base, maximum, seed), config, bundle["gene_vocab"], output / f"{split}.h5ad")
        finally:
            base.close()
    summary = {
        "data_run_id": args.data_run_id,
        "sample_seed": args.sample_seed,
        "train_cells": counts["train"],
        "test_cells": counts["test"],
        "genes": len(bundle["gene_vocab"]),
        "format": "raw counts in sparse CSR AnnData; CellVQ adapter performs official normalize-total/log1p preprocessing",
    }
    print(json.dumps(summary, indent=2))

def load_run(path: Path):
    data = np.load(path / "representations.npz", allow_pickle=False)
    metadata = pd.read_csv(path / "metadata.csv.gz", keep_default_na=False)
    labels = {key: metadata[key].astype(str).to_numpy() for key in ["coarse_cell_type", "cell_type", "assay", "dataset_id"]}
    return data["embedding"].astype(np.float32), data["code"].astype(np.int64), labels


def bits(code: np.ndarray, width: int) -> np.ndarray:
    offsets = np.arange(width - 1, -1, -1, dtype=np.int64)
    return ((code[:, None] >> offsets[None, :]) & 1).astype(np.float32)


def evaluate_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260815)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_embedding, train_code, train = load_run(Path(args.train_dir))
    test_embedding, test_code, test = load_run(Path(args.test_dir))
    codebook_size = int(max(train_code.max(), test_code.max()) + 1)
    configured_codebook = 4096
    train_onehot = np.eye(configured_codebook, dtype=np.float32)[train_code]
    test_onehot = np.eye(configured_codebook, dtype=np.float32)[test_code]
    counts = np.bincount(train_code, minlength=configured_codebook)
    probabilities = counts[counts > 0] / counts.sum()
    results = {
        "method": "CellVQ",
        "train_cells": int(len(train_code)),
        "test_cells": int(len(test_code)),
        "native_codebook_size": configured_codebook,
        "observed_max_code_plus_one": codebook_size,
        "active_codes_train": int((counts > 0).sum()),
        "active_code_ratio_train": float((counts > 0).mean()),
        "normalized_code_entropy_train": float(-(probabilities * np.log(probabilities)).sum() / np.log(configured_codebook)),
        "bits_per_cell": 12,
        "spaces": {
            "continuous_embedding": {
                "dimensions": int(train_embedding.shape[1]),
                "probes": probe_suite(train_embedding, test_embedding, train, test, args.seed),
            },
            "single_token_onehot": {
                "dimensions": configured_codebook,
                "probes": probe_suite(train_onehot, test_onehot, train, test, args.seed),
            },
            "binary_code": {
                "dimensions": 12,
                "probes": probe_suite(bits(train_code, 12), bits(test_code, 12), train, test, args.seed),
            },
        },
        "reconstruction": None,
        "reconstruction_note": "The released inference interface does not provide reconstruction output.",
    }
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] not in {"export", "evaluate"}:
        raise SystemExit("usage: cellvq.py {export|evaluate} [arguments]")
    mode = sys.argv.pop(1)
    (export_main if mode == "export" else evaluate_main)()
