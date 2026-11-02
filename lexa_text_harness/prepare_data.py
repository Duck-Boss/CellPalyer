#!/usr/bin/env python3
"""Prepare original sciPlex2 counts without using held-out expression for fitting.

The two pilot protocols are dose interpolation within seen drugs and leave-one-
drug-out. A well is never split across train/validation/test. The author's
``replicate`` labels are derived from culture-well columns; they are not proof
of independent donors or experiments. The malformed SAHA_100_E09 annotation is
retained in metadata but excluded because its plate is unknown.

Feature IDs are unique, versioned Ensembl IDs from the original annotation.
Duplicate *symbols* are kept as distinct genes and are never silently merged.
HVGs are ranked by log1p-CP10K variance using training cells only (pilot rule,
not a reproduction of the paper's scaffold or an author HVG protocol).
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse


DRUGS = ("BMS", "Dex", "Nutlin", "SAHA")
AUTHOR_SOURCE = "https://github.com/cole-trapnell-lab/sci-plex/blob/master/small_screen/Notebook_1_small_screen_analysis.R"
CONDITION_RE = re.compile(r"^(?P<drug>BMS|Dex|Nutlin|SAHA)_(?P<dose>\d+(?:\.\d+)?)_(?P<well>[A-Z]+\d{2})$")
WELL_RE = re.compile(r"^(?P<plate>[AB])(?P<row>[A-H])(?P<column>0[1-9]|1[0-2])$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_npz(path: Path, **payload: Any) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def atomic_csv(path: Path, data: pd.DataFrame) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    data.to_csv(temp, index=False)
    os.replace(temp, path)


def unique_input(raw: Path, suffix: str) -> Path:
    found = sorted(raw.glob("*" + suffix))
    if len(found) != 1:
        raise FileNotFoundError(f"Expected one completed *{suffix} in {raw}; found {len(found)} (a .part download is not accepted)")
    return found[0]


def parse_condition(value: Any) -> dict[str, Any]:
    empty = {"drug": None, "dose_uM": np.nan, "well": None, "plate": None,
             "replicate": None, "is_control": False, "vehicle": None,
             "parse_status": "unassigned"}
    if pd.isna(value) or not str(value).strip():
        return empty
    match = CONDITION_RE.fullmatch(str(value))
    if match is None:
        return {**empty, "parse_status": "invalid_condition"}
    fields = match.groupdict()
    dose = float(fields["dose"])
    result = {**empty, "drug": fields["drug"], "dose_uM": dose,
              "well": fields["well"], "is_control": dose == 0,
              "vehicle": "Ethanol" if fields["drug"] == "Dex" else "DMSO"}
    well = WELL_RE.fullmatch(fields["well"])
    if well is None:
        return {**result, "parse_status": "invalid_well"}
    column = int(well["column"])
    return {**result, "plate": well["plate"], "replicate": 1 + ((column - 1) % 3),
            "parse_status": "ok"}


def read_triplet_counts(path: Path, n_cells: int, n_genes: int,
                         chunk_size: int = 1_000_000) -> sparse.csr_matrix:
    """Read headerless one-based gene,cell,integer-UMI triples (not MM format)."""
    row_parts, col_parts, value_parts = [], [], []
    for chunk in pd.read_csv(path, sep=r"\s+", header=None,
                             names=["gene", "cell", "count"], dtype=np.int64,
                             chunksize=chunk_size, compression="gzip"):
        genes = chunk["gene"].to_numpy() - 1
        cells = chunk["cell"].to_numpy() - 1
        values = chunk["count"].to_numpy()
        if (len(genes) == 0 or genes.min() < 0 or genes.max() >= n_genes
                or cells.min() < 0 or cells.max() >= n_cells or (values <= 0).any()):
            raise ValueError("Triplet counts contain out-of-range indices or nonpositive counts")
        if values.max() > np.iinfo(np.int32).max:
            raise ValueError("Unexpected UMI count larger than int32")
        row_parts.append(cells.astype(np.int32))
        col_parts.append(genes.astype(np.int32))
        value_parts.append(values.astype(np.int32))
    if not value_parts:
        raise ValueError("The count matrix is empty")
    matrix = sparse.coo_matrix((np.concatenate(value_parts),
                                (np.concatenate(row_parts), np.concatenate(col_parts))),
                               shape=(n_cells, n_genes), dtype=np.int32).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def calculate_qc(counts: sparse.csr_matrix, gene_symbols: np.ndarray,
                 metadata: pd.DataFrame, min_umi: int = 500,
                 min_genes: int = 720, max_mito_pct: float = 10.0) -> pd.DataFrame:
    if len(metadata) != counts.shape[0] or len(gene_symbols) != counts.shape[1]:
        raise ValueError("Counts, cell metadata and gene symbols are not aligned")
    missing = {"top_to_second_best_ratio", "qval", "hash_umis"} - set(metadata.columns)
    if missing:
        raise ValueError(f"Hash assignment columns missing: {sorted(missing)}")
    counts = counts.tocsr(copy=True)
    counts.sum_duplicates()
    counts.eliminate_zeros()
    umi = np.asarray(counts.sum(axis=1)).ravel().astype(np.int64)
    detected = counts.getnnz(axis=1)
    mito_mask = np.char.startswith(np.char.upper(np.asarray(gene_symbols, dtype=str)), "MT-")
    mito_umi = np.asarray(counts[:, mito_mask].sum(axis=1)).ravel()
    mito_pct = np.divide(mito_umi * 100.0, umi, out=np.zeros(len(umi), dtype=float), where=umi > 0)
    ratio = pd.to_numeric(metadata["top_to_second_best_ratio"], errors="coerce").to_numpy()
    qval = pd.to_numeric(metadata["qval"], errors="coerce").to_numpy()
    hash_umi = pd.to_numeric(metadata["hash_umis"], errors="coerce").to_numpy()
    # Strict inequalities exactly preserve the original author's assignment QC.
    hash_ok = (ratio > 10) & (qval < 0.01) & (hash_umi > 30)
    expression_ok = (umi >= min_umi) & (detected >= min_genes) & (mito_pct <= max_mito_pct)
    return pd.DataFrame({"total_umi": umi, "detected_genes": detected,
                         "mito_pct": mito_pct, "pass_hash_qc": hash_ok,
                         "pass_expression_qc": expression_ok, "pass_qc": hash_ok & expression_ok})


def assign_splits(metadata: pd.DataFrame, protocol: str = "seen_drug_unseen_dose",
                   holdout_drug: str | None = None):
    """Select holdouts by annotated condition, without reading expression values."""
    if protocol not in ("seen_drug_unseen_dose", "leave_drug_out"):
        raise ValueError(f"Unknown protocol {protocol}")
    if protocol == "leave_drug_out" and holdout_drug not in DRUGS:
        raise ValueError("leave_drug_out needs --holdout-drug BMS/Dex/Nutlin/SAHA")
    result = metadata.copy()
    result["split"] = "excluded_qc"
    valid = result["parse_status"].eq("ok")
    result.loc[~valid, "split"] = "excluded_" + result.loc[~valid, "parse_status"]
    eligible = valid & result["pass_qc"].astype(bool)
    heldout_doses = {}
    if protocol == "seen_drug_unseen_dose":
        for drug in DRUGS:
            # Use the annotated experimental dose grid before expression QC.
            # Choosing from expression-surviving doses would make the protocol
            # depend on response/viability of held-out samples.
            doses = np.sort(result.loc[valid & result["drug"].eq(drug)
                                       & result["dose_uM"].gt(0), "dose_uM"].unique())
            if len(doses) == 0:
                raise ValueError(f"No valid nonzero experimental dose for {drug}")
            heldout_doses[drug] = float(doses[len(doses) // 2])
        test = np.zeros(len(result), dtype=bool)
        for drug, dose in heldout_doses.items():
            test |= (result["drug"].eq(drug) & result["dose_uM"].eq(dose)).to_numpy()
    else:
        test = (result["drug"].eq(holdout_drug) & result["dose_uM"].gt(0)).to_numpy()
    result.loc[eligible & ~test & result["replicate"].isin([1, 2]), "split"] = "train"
    result.loc[eligible & ~test & result["replicate"].eq(3), "split"] = "val"
    result.loc[eligible & test, "split"] = "test"
    used = result[result["split"].isin(["train", "val", "test"])]
    if (used.groupby("top_oligo")["split"].nunique() > 1).any():
        raise RuntimeError("A physical condition/well crossed split boundaries")
    if (used.groupby("well")["split"].nunique() > 1).any():
        raise RuntimeError("A physical well crossed split boundaries")
    for name in ("train", "val", "test"):
        if not result["split"].eq(name).any():
            raise ValueError(f"No cells in {name}; check QC and split protocol")
    return result, heldout_doses


def normalize_log1p(counts: sparse.csr_matrix, total: float = 10000) -> sparse.csr_matrix:
    """Per-cell library normalization over all genes, before HVG selection."""
    expression = counts.astype(np.float32, copy=True)
    libraries = np.asarray(expression.sum(axis=1)).ravel()
    factor = np.divide(total, libraries, out=np.zeros(len(libraries), dtype=np.float32), where=libraries > 0)
    expression = sparse.diags(factor) @ expression
    expression = expression.tocsr()
    expression.data = np.log1p(expression.data)
    return expression


def select_training_hvg(log_expression: sparse.csr_matrix, counts: sparse.csr_matrix,
                         train_mask: np.ndarray, n_genes: int = 2000,
                         min_gene_cells: int = 10):
    train = log_expression[train_mask]
    if train.shape[0] == 0:
        raise ValueError("Cannot fit HVGs without training cells")
    train64 = train.astype(np.float64)
    means = np.asarray(train64.mean(axis=0)).ravel()
    second = np.asarray(train64.power(2).mean(axis=0)).ravel()
    variance = np.maximum(second - means ** 2, 0)
    detected = counts[train_mask].getnnz(axis=0)
    candidates = np.flatnonzero((detected >= min_gene_cells) & (variance > 1e-12))
    if not len(candidates):
        raise ValueError("No train-expressed variable genes satisfy the selection rule")
    # Stable sorting resolves variance ties by original annotation order.
    selected = candidates[np.argsort(-variance[candidates], kind="stable")[:n_genes]]
    return selected, means, variance, detected


def control_group_id(drug: str, plate: str, replicate: int) -> str:
    return f"drug={drug}|plate={plate}|replicate={int(replicate)}"


def treatment_case_id(drug: str, dose: float, plate: str, replicate: int) -> str:
    return f"drug={drug}|dose_uM={float(dose):g}|plate={plate}|replicate={int(replicate)}"


def build_cases(metadata: pd.DataFrame, min_case_cells: int = 20):
    controls = metadata[metadata["pass_qc"] & metadata["parse_status"].eq("ok") & metadata["is_control"]]
    test = metadata[metadata["split"].eq("test")]
    rows = []
    for (drug, dose, plate, replicate), treated in test.groupby(["drug", "dose_uM", "plate", "replicate"], sort=True):
        matched = controls[controls["drug"].eq(drug) & controls["plate"].eq(plate)
                           & controls["replicate"].eq(replicate)]
        n_treated, n_control = len(treated), len(matched)
        status = ("missing_matched_control" if n_control == 0 else
                  "small_sample" if min(n_treated, n_control) < min_case_cells else "ready")
        rows.append({"case_id": treatment_case_id(drug, dose, plate, replicate),
                     "drug": drug, "dose_uM": float(dose), "dose_unit": "uM",
                     "plate": plate, "replicate": int(replicate),
                     "cell_line": "A549", "exposure_time_hours": 24,
                     "exposure_time_source": AUTHOR_SOURCE,
                     "vehicle": "Ethanol" if drug == "Dex" else "DMSO",
                     "n_treated": n_treated, "n_control": n_control,
                     "control_group_id": control_group_id(drug, plate, replicate),
                     "control_cell_ids": matched["cell_id"].astype(str).tolist(),
                     "treated_cell_ids": treated["cell_id"].astype(str).tolist(),
                     "status": status})
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path(__file__).resolve().parent / "data/raw/sciplex2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=["seen_drug_unseen_dose", "leave_drug_out"], default="seen_drug_unseen_dose")
    parser.add_argument("--holdout-drug", choices=DRUGS)
    parser.add_argument("--hvg", type=int, default=2000)
    parser.add_argument("--min-gene-cells", type=int, default=10)
    parser.add_argument("--min-umi", type=int, default=500)
    parser.add_argument("--min-genes", type=int, default=720)
    parser.add_argument("--max-mito-pct", type=float, default=10)
    parser.add_argument("--min-case-cells", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=1_000_000)
    return parser.parse_args(argv)


def run(args) -> dict[str, Any]:
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".prepare.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another preparation is writing this directory") from exc
        return _run_locked(args)


def _run_locked(args) -> dict[str, Any]:
    if min(args.hvg, args.min_gene_cells, args.min_umi, args.min_genes,
           args.min_case_cells, args.chunk_size) < 1 or args.max_mito_pct < 0:
        raise ValueError("Invalid QC or preparation limits")
    started = time.time()
    raw = args.raw.resolve()
    inputs = {"counts": unique_input(raw, "UMI.count.matrix.gz"),
              "genes": unique_input(raw, "gene.annotations.txt.gz"),
              "cells": unique_input(raw, "cell.annotations.txt.gz"),
              "metadata": unique_input(raw, "pData.txt.gz")}
    gene_info = pd.read_csv(inputs["genes"], sep="\t", header=None,
                            names=["gene_id", "gene_symbol"], dtype=str, keep_default_na=False)
    cells = pd.read_csv(inputs["cells"], sep="\t", header=None,
                       names=["cell_id", "batch"], dtype=str, keep_default_na=False)
    if gene_info["gene_id"].duplicated().any() or cells["cell_id"].duplicated().any():
        raise ValueError("Original Ensembl IDs and cell IDs must each be unique")
    info = pd.read_csv(inputs["metadata"], sep=r"\s+", index_col=0)
    if "Cell" not in info or info["Cell"].duplicated().any():
        raise ValueError("pData must provide unique Cell IDs")
    info = info.set_index("Cell", drop=False)
    if set(cells["cell_id"]) != set(info.index):
        raise ValueError("Count cell annotation IDs differ from pData cell IDs")
    info = info.loc[cells["cell_id"]].reset_index(drop=True)
    metadata = pd.DataFrame({"cell_id": cells["cell_id"], "batch": cells["batch"],
                              "top_oligo": info["top_oligo"].astype("string")})
    parsed = pd.DataFrame([parse_condition(value) for value in metadata["top_oligo"]])
    metadata = pd.concat([metadata, parsed], axis=1)
    metadata["replicate"] = metadata["replicate"].astype("Int64")
    print(json.dumps({"stage": "read_counts", "cells": len(cells), "genes": len(gene_info)}), flush=True)
    counts = read_triplet_counts(inputs["counts"], len(cells), len(gene_info), args.chunk_size)
    qc = calculate_qc(counts, gene_info["gene_symbol"].to_numpy(), info,
                       args.min_umi, args.min_genes, args.max_mito_pct)
    metadata = pd.concat([metadata, qc], axis=1)
    metadata, heldout_doses = assign_splits(metadata, args.protocol, args.holdout_drug)
    train_mask = metadata["split"].eq("train").to_numpy()
    val_mask = metadata["split"].eq("val").to_numpy()
    test_mask = metadata["split"].eq("test").to_numpy()
    control_mask = (metadata["pass_qc"] & metadata["parse_status"].eq("ok") & metadata["is_control"]).to_numpy()
    print(json.dumps({"stage": "normalize_and_select_train_hvg", "splits": metadata["split"].value_counts().to_dict()}), flush=True)
    normalized = normalize_log1p(counts)
    selected, means, variances, detected = select_training_hvg(normalized, counts, train_mask,
                                                                args.hvg, args.min_gene_cells)
    if len(selected) < 32:
        raise ValueError("Fewer than 32 training HVGs; the 32-axis pilot cannot be fitted")
    genes = gene_info["gene_id"].to_numpy(dtype=str)[selected]
    gene_symbols = gene_info["gene_symbol"].to_numpy(dtype=str)[selected]
    expression = normalized[:, selected].tocsr()
    controls = metadata[control_mask]
    test = metadata[test_mask]
    cases = build_cases(metadata, args.min_case_cells)
    control_ids = np.asarray([control_group_id(row.drug, row.plate, row.replicate)
                               for row in controls.itertuples()], dtype=str)
    test_case_ids = np.asarray([treatment_case_id(row.drug, row.dose_uM, row.plate, row.replicate)
                                 for row in test.itertuples()], dtype=str)
    raw_provenance = {key: {"path": str(path), "size_bytes": path.stat().st_size,
                            "sha256": sha256_file(path)} for key, path in inputs.items()}
    for name in ("Notebook_1_small_screen_analysis.source.R", "author_analysis_provenance.json"):
        path = raw / name
        if path.exists():
            raw_provenance[name] = {"path": str(path), "size_bytes": path.stat().st_size,
                                     "sha256": sha256_file(path)}
    provenance = {"dataset": "GSE139944 / GSM4150377 sciPlex2 A549 Transcription Modulators",
                  "raw_files": raw_provenance, "protocol": args.protocol,
                  "holdout_drug": args.holdout_drug, "heldout_doses_uM": heldout_doses,
                  "normalization": "per-cell sum over all original genes to 10000, then log1p, before HVG selection",
                  "hvg": {"method": "train-only log1p-CP10K variance descending; original annotation order breaks ties",
                          "requested": args.hvg, "selected": len(selected), "min_train_cells_expressed": args.min_gene_cells},
                  "feature_namespace": "original unique versioned Ensembl gene IDs",
                  "duplicate_symbols": "not merged; symbol is a display annotation only",
                  "source_population_for_fitted_statistics": "split=train only; no validation/test expression used for gene selection",
                  "baseline_controls": "all quality-passing zero-dose cells, with exact drug+plate+author replicate matching",
                  "replicates": "author culture-well column labels; independence beyond author-defined triplicates unverified",
                  "time_hours": 24, "dose_unit": "uM", "author_source": AUTHOR_SOURCE,
                  "synthetic": False, "not_paper_reproduction": True}
    control_x = expression[control_mask].toarray().astype(np.float32)
    (args.output / "COMPLETE.json").unlink(missing_ok=True)
    atomic_npz(args.output / "prepared.npz", x_train=expression[train_mask].toarray().astype(np.float32),
               x_val=expression[val_mask].toarray().astype(np.float32),
               control_x=control_x, genes=genes, gene_symbols=gene_symbols,
               train_cell_ids=metadata.loc[train_mask, "cell_id"].to_numpy(dtype=str),
               val_cell_ids=metadata.loc[val_mask, "cell_id"].to_numpy(dtype=str),
               control_cell_ids=controls["cell_id"].to_numpy(dtype=str),
               train_gene_mean=means[selected].astype(np.float32),
               train_gene_variance=variances[selected].astype(np.float32),
               synthetic=np.asarray(False), provenance_json=np.asarray(json.dumps(provenance)))
    atomic_npz(args.output / "evaluation.npz", x_test=expression[test_mask].toarray().astype(np.float32),
               test_cell_ids=test["cell_id"].to_numpy(dtype=str), test_case_ids=test_case_ids,
               control_x=control_x, control_cell_ids=controls["cell_id"].to_numpy(dtype=str),
               control_case_ids=control_ids, genes=genes, gene_symbols=gene_symbols,
               synthetic=np.asarray(False))
    atomic_csv(args.output / "metadata.csv", metadata)
    gene_table = gene_info.iloc[selected].copy()
    gene_table["train_mean_log1p_cp10k"] = means[selected]
    gene_table["train_variance_log1p_cp10k"] = variances[selected]
    gene_table["train_cells_expressed"] = detected[selected]
    atomic_csv(args.output / "selected_genes.csv", gene_table)
    atomic_json(args.output / "cases.json", {"protocol": args.protocol, "dose_unit": "uM",
                "exposure_time_hours": 24, "min_case_cells": args.min_case_cells,
                "cases": cases})
    split_counts = {str(key): int(value) for key, value in metadata["split"].value_counts().items()}
    split_wells = {str(key): int(group["top_oligo"].nunique()) for key, group in
                   metadata[metadata["split"].isin(["train", "val", "test"])].groupby("split")}
    manifest = {"status": "prepared", **provenance, "n_raw_cells": len(metadata),
                "n_raw_genes": len(gene_info), "n_selected_genes": len(selected),
                "raw_duplicate_symbol_excess": int(len(gene_info) - gene_info["gene_symbol"].nunique()),
                "count_nnz": int(counts.nnz), "split_cell_counts": split_counts,
                "split_condition_well_counts": split_wells,
                "n_conditioning_control_cells": int(control_mask.sum()),
                "case_status_counts": dict(Counter(case["status"] for case in cases)),
                "qc": {"min_umi_inclusive": args.min_umi, "min_detected_genes_inclusive": args.min_genes,
                       "max_mito_pct_inclusive": args.max_mito_pct,
                       "hash_assignment": "top_to_second_best_ratio > 10 and qval < 0.01 and hash_umis > 30",
                       "expression_metrics_source": "recomputed from original integer UMI counts",
                       "gene_detection_count": "distinct original Ensembl features, before HVG selection",
                       "mitochondrial_definition": "gene_symbol starts with MT- (case insensitive)"},
                "excluded_condition_annotations": sorted(metadata.loc[metadata["parse_status"].ne("ok"),
                                                                         "top_oligo"].dropna().unique().tolist()),
                "evaluation_boundary": "evaluation.npz contains held-out targets; fit_lexa and DeepSeek annotation/edit prompts must not read it",
                "wall_seconds": round(time.time() - started, 3)}
    atomic_json(args.output / "manifest.json", manifest)
    output_files = ("prepared.npz", "evaluation.npz", "metadata.csv", "selected_genes.csv", "cases.json", "manifest.json")
    atomic_json(args.output / "COMPLETE.json", {"status": "prepared", "files": list(output_files),
                "prepared_sha256": sha256_file(args.output / "prepared.npz"),
                "protocol": args.protocol, "synthetic": False})
    return manifest


def main():
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
