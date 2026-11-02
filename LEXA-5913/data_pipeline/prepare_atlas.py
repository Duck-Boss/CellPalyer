"""Prepare a CELLxGENE data bundle with a shared protein-coding gene vocabulary."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import pandas as pd
import yaml

from data_pipeline.clean_common import (
    GENCODE_GTF_URL,
    decode_scalar,
    ensure_dir,
    expression_semantics_from_values,
    cell_type_family,
    harmonize_assay,
    harmonize_cell_type,
    harmonize_disease,
    harmonize_tissue,
    load_protein_coding_symbols,
    matrix_encoding,
    matrix_shape,
    progress,
    read_index,
    read_string_array,
    sample_matrix_values,
    select_first_present,
    stable_split,
    write_csv,
    write_json,
)


HUMAN_VALUES = {"Homo sapiens", "NCBITaxon:9606"}
COUNT_LAYERS = ["counts", "raw_counts", "UMI", "umis", "raw", "count", "data"]
RECORD_FIELDNAMES = [
    "file_path",
    "row_idx",
    "group_id",
    "dataset_id",
    "cell_type",
    "coarse_cell_type",
    "tissue",
    "disease",
    "assay",
    "donor_id",
    "sample_id",
    "batch_id",
    "technical_block",
    "author_cell_type",
    "development_stage",
    "age",
    "age_parse_source",
    "sex",
    "ethnicity",
    "suspension_type",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lexa_r4.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--h5ad-dir", default=None)
    parser.add_argument("--file-manifest", default=None, help="CSV produced by atlas_audit_and_selection.py")
    parser.add_argument("--path-column", default="source_file_path_absolute")
    parser.add_argument("--role", default=None, help="Optional exact role filter, e.g. train or ood_disease")
    parser.add_argument("--reference-bundle", default=None, help="Reuse this bundle's gene_vocab (required for OOD)")
    parser.add_argument("--require-raw-counts", action="store_true")
    parser.add_argument("--annotation-dir", default="resources/gene_annotations")
    parser.add_argument("--gencode-url", default=GENCODE_GTF_URL)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-cells-per-file", type=int, default=None)
    parser.add_argument("--target-sum", type=float, default=10000.0)
    parser.add_argument("--min-detected-genes", type=int, default=20)
    parser.add_argument("--min-total-counts", type=float, default=0.0)
    parser.add_argument("--split-group-key", help="Explicit obs field used for group-safe splitting")
    parser.add_argument("--donor-key", help="Explicit obs field exported as donor_id")
    parser.add_argument("--cell-type-key", help="Explicit obs field exported as cell_type without ontology remapping")
    parser.add_argument("--ignore-is-primary-data", action="store_true", help="Application mode: keep rows even when a catalog marks this file as non-primary")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def obs_first(handle: h5py.File, key: str) -> str:
    values = read_string_array(handle.get("obs"), key, limit=1)
    return str(values[0]) if values.size else ""


def is_human_file(handle: h5py.File) -> bool:
    return obs_first(handle, "organism") in HUMAN_VALUES or obs_first(handle, "organism_ontology_term_id") in HUMAN_VALUES


def matrix_candidates(handle: h5py.File) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    if "raw" in handle and "X" in handle["raw"]:
        candidates.append(("raw.X", "raw/X"))
    if "layers" in handle:
        for layer in COUNT_LAYERS:
            if layer in handle["layers"]:
                candidates.append((f"layers/{layer}", f"layers/{layer}"))
    if "X" in handle:
        candidates.append(("X", "X"))
    seen = set()
    out = []
    for label, key in candidates:
        if key not in seen:
            seen.add(key)
            out.append((label, key))
    return out


def classify_transform(semantics: str) -> Optional[str]:
    if semantics in {"raw_count_like", "integer_low_dynamic_range"}:
        return "normalize_total_10000_then_log1p"
    if semantics in {"log1p_like", "log_or_normalized_like"}:
        return "use_as_is"
    if semantics == "normalized_count_like":
        return "log1p_only"
    return None


def choose_matrix(handle: h5py.File, rng: np.random.Generator) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    priority = {
        "raw.X": 0,
        "layers/counts": 1,
        "layers/raw_counts": 1,
        "layers/UMI": 2,
        "layers/umis": 2,
        "layers/raw": 2,
        "layers/count": 2,
        "layers/data": 4,
        "X": 5,
    }
    for label, key in sorted(matrix_candidates(handle), key=lambda item: priority.get(item[0], 10)):
        try:
            obj = handle[key]
            fmt = matrix_encoding(obj)
            n_obs, n_vars = matrix_shape(obj)
            values = sample_matrix_values(obj, fmt, rng)
            sem = expression_semantics_from_values(values)
            transform = classify_transform(str(sem["semantics"]))
            row = {
                "candidate_label": label,
                "matrix_key": key,
                "matrix_format": fmt,
                "n_obs": n_obs,
                "n_vars": n_vars,
                "transform": transform or "exclude",
                **sem,
            }
            rows.append(row)
            if row.get("semantics") in {"raw_count_like", "integer_low_dynamic_range"}:
                return row, rows
        except Exception as exc:
            rows.append({"candidate_label": label, "matrix_key": key, "error": repr(exc), "transform": "exclude"})
    usable = [row for row in rows if row.get("transform") != "exclude"]
    if not usable:
        return None, rows
    raw_like = [row for row in usable if row.get("semantics") in {"raw_count_like", "integer_low_dynamic_range"}]
    if raw_like:
        raw_like.sort(key=lambda row: priority.get(str(row["candidate_label"]), 10))
        return raw_like[0], rows
    usable.sort(key=lambda row: priority.get(str(row["candidate_label"]), 10))
    return usable[0], rows


def normalize_biotype(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def read_feature_table_from_var(var: h5py.Group, n_vars: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_names = read_string_array(var, "feature_name", limit=n_vars)
    if feature_names.size == 0:
        feature_names = read_index(var, n_vars)
    biotype = read_string_array(var, "feature_biotype", limit=n_vars)
    if biotype.size == 0:
        biotype = np.asarray(["unknown"] * n_vars, dtype=object)
    filtered = read_string_array(var, "feature_is_filtered", limit=n_vars)
    if filtered.size == 0:
        filtered = np.asarray(["False"] * n_vars, dtype=object)
    return feature_names[:n_vars], biotype[:n_vars], filtered[:n_vars]


def feature_table_score(feature_names: np.ndarray, biotype: np.ndarray, protein_symbols: set) -> int:
    score = 0
    for name, bio in zip(feature_names, biotype):
        norm_bio = normalize_biotype(bio)
        if str(name) in protein_symbols:
            score += 3
        elif norm_bio == "protein_coding":
            score += 2
    return score


def var_table_rows(var: h5py.Group) -> int:
    for key in ["feature_name", "_index", "index", "feature_id", "gene_ids", "gene_symbols"]:
        if key in var and hasattr(var[key], "shape") and var[key].shape:
            return int(var[key].shape[0])
        if key in var and hasattr(var[key], "keys") and "codes" in var[key]:
            return int(var[key]["codes"].shape[0])
    for key in var.keys():
        obj = var[key]
        if hasattr(obj, "shape") and obj.shape:
            return int(obj.shape[0])
        if hasattr(obj, "keys") and "codes" in obj:
            return int(obj["codes"].shape[0])
    return 0


def read_feature_table(handle: h5py.File, matrix_key: str, n_vars: int, protein_symbols: set) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    candidates: List[Tuple[str, h5py.Group]] = []
    if matrix_key == "raw/X" and "raw" in handle and "var" in handle["raw"]:
        candidates.append(("raw/var", handle["raw/var"]))
    if "var" in handle:
        if var_table_rows(handle["var"]) == n_vars:
            candidates.append(("var", handle["var"]))
    if not candidates:
        raise ValueError("No var table found for selected matrix")

    best: Optional[Tuple[int, str, np.ndarray, np.ndarray, np.ndarray]] = None
    for label, var in candidates:
        feature_names, biotype, filtered = read_feature_table_from_var(var, n_vars)
        score = feature_table_score(feature_names, biotype, protein_symbols)
        if best is None or score > best[0]:
            best = (score, label, feature_names, biotype, filtered)
    assert best is not None
    _, label, feature_names, biotype, filtered = best
    return feature_names, biotype, filtered, label


def protein_coding_indices(
    feature_names: np.ndarray,
    biotype: np.ndarray,
    filtered: np.ndarray,
    protein_symbols: set,
) -> Tuple[np.ndarray, Dict[str, int]]:
    keep: List[int] = []
    stats = Counter()
    for idx, raw_name in enumerate(feature_names):
        name = str(raw_name)
        if not name:
            stats["missing_feature_name"] += 1
            continue
        if str(filtered[idx]).lower() in {"true", "1", "yes"}:
            stats["feature_is_filtered"] += 1
            continue
        biotype_value = normalize_biotype(biotype[idx])
        if name in protein_symbols:
            keep.append(idx)
            stats["kept_by_external_annotation"] += 1
        elif biotype_value == "protein_coding":
            keep.append(idx)
            stats["kept_by_cellxgene_feature_biotype"] += 1
        else:
            stats[f"excluded_biotype:{biotype_value or 'missing'}"] += 1
    stats["kept_total"] = len(keep)
    stats["duplicate_kept_symbols"] = len(keep) - len({str(feature_names[idx]) for idx in keep})
    stats["kept_symbols_missing_external_annotation"] = sum(1 for idx in keep if str(feature_names[idx]) not in protein_symbols)
    return np.asarray(keep, dtype=np.int64), dict(stats)


def sample_qc_metrics(obj: Any, matrix_format: str, rng: np.random.Generator, max_rows: int = 256) -> Dict[str, Any]:
    try:
        n_obs, n_vars = matrix_shape(obj)
        if n_obs <= 0 or n_vars <= 0:
            return {"qc_sample_rows": 0, "qc_reason": "empty_matrix"}
        rows = np.sort(rng.choice(n_obs, size=min(n_obs, max_rows), replace=False))
        if matrix_format == "dense":
            values = np.asarray(obj[rows, :], dtype=np.float64)
            totals = values.sum(axis=1)
            detected = np.count_nonzero(values > 0, axis=1)
        elif matrix_format == "csr_matrix":
            indptr = np.asarray(obj["indptr"][:], dtype=np.int64)
            data = obj["data"]
            totals = []
            detected = []
            for row_idx in rows:
                start = int(indptr[int(row_idx)])
                end = int(indptr[int(row_idx) + 1])
                chunk = np.asarray(data[start:end], dtype=np.float64)
                totals.append(float(chunk.sum()))
                detected.append(int(np.count_nonzero(chunk > 0)))
            totals = np.asarray(totals, dtype=np.float64)
            detected = np.asarray(detected, dtype=np.float64)
        else:
            return {"qc_sample_rows": 0, "qc_reason": f"skipped_for_{matrix_format}"}
        finite_totals = totals[np.isfinite(totals)]
        finite_detected = detected[np.isfinite(detected)]
        if finite_totals.size == 0:
            return {"qc_sample_rows": int(len(rows)), "qc_reason": "no_finite_totals"}
        return {
            "qc_sample_rows": int(len(rows)),
            "qc_total_min": float(np.min(finite_totals)),
            "qc_total_median": float(np.median(finite_totals)),
            "qc_total_p99": float(np.quantile(finite_totals, 0.99)),
            "qc_detected_min": float(np.min(finite_detected)) if finite_detected.size else None,
            "qc_detected_median": float(np.median(finite_detected)) if finite_detected.size else None,
            "qc_detected_p99": float(np.quantile(finite_detected, 0.99)) if finite_detected.size else None,
            "qc_zero_total_fraction": float(np.mean(finite_totals <= 0)),
            "qc_reason": "ok",
        }
    except Exception as exc:
        return {"qc_sample_rows": 0, "qc_reason": f"error:{exc!r}"}


def build_gene_vocab(
    file_rows: Sequence[Dict[str, Any]], reference_vocab: Optional[Sequence[str]] = None
) -> Tuple[List[str], Dict[str, np.ndarray]]:
    if reference_vocab is not None:
        vocab = [str(gene) for gene in reference_vocab]
    else:
        per_file = [
            {str(row["_feature_names"][idx]) for idx in row["_keep_idx"]}
            for row in file_rows
        ]
        if not per_file:
            raise ValueError("cannot build a gene vocabulary from zero files")
        common = set.intersection(*per_file)
        if not common:
            raise ValueError("selected training files have no common protein-coding genes")
        vocab = sorted(common)
    vocab_index = {gene: i for i, gene in enumerate(vocab)}
    maps: Dict[str, np.ndarray] = {}
    for row in file_rows:
        names = row["_feature_names"]
        gene_map = np.full(len(names), -1, dtype=np.int32)
        for idx in row["_keep_idx"]:
            target_idx = vocab_index.get(str(names[int(idx)]))
            if target_idx is not None:
                gene_map[int(idx)] = target_idx
        maps[row["file_path"]] = gene_map
    return vocab, maps


def save_file_gene_maps(path: Path, maps: Dict[str, np.ndarray]) -> None:
    ensure_dir(path.parent)
    payload = {f"arr_{idx}": arr for idx, arr in enumerate(maps.values())}
    np.savez_compressed(path, file_paths=np.asarray(list(maps.keys()), dtype=str), **payload)


def read_obs_values(obs: h5py.Group, key: Optional[str], n_rows: int) -> Optional[np.ndarray]:
    if key is None:
        return None
    if key == "_index":
        return read_index(obs, n_rows)
    values = read_string_array(obs, key, limit=n_rows)
    return values if values.size else None


def read_obs_values_selected(
    obs: h5py.Group,
    key: Optional[str],
    rows: np.ndarray,
) -> Optional[np.ndarray]:
    """Read only selected, sorted obs rows instead of decoding a whole atlas file."""
    if key is None or key not in obs:
        return None
    obj = obs[key]

    def dataset_strings(dataset: Any) -> List[str]:
        values = dataset[:]
        return [str(decode_scalar(value)) for value in values]

    if hasattr(obj, "keys") and "categories" in obj and "codes" in obj:
        categories = dataset_strings(obj["categories"])
        codes = np.asarray(obj["codes"][:], dtype=np.int64)[rows]
        return np.asarray([categories[code] if 0 <= code < len(categories) else "" for code in codes], dtype=object)
    if "__categories" in obs and key in obs["__categories"] and hasattr(obj, "shape"):
        categories = dataset_strings(obs["__categories"][key])
        codes = np.asarray(obj[:], dtype=np.int64)[rows]
        return np.asarray([categories[code] if 0 <= code < len(categories) else "" for code in codes], dtype=object)
    if hasattr(obj, "shape") and obj.shape != ():
        values = obj[:][rows]
        return np.asarray([str(decode_scalar(value)) for value in values], dtype=object)
    return None


def read_obs_numeric(obs: h5py.Group, key: Optional[str], n_rows: int) -> Optional[np.ndarray]:
    if key is None or key not in obs:
        return None
    obj = obs[key]
    if hasattr(obj, "keys") and "codes" in obj:
        return None
    try:
        values = np.asarray(obj[:n_rows], dtype=np.float64)
    except Exception:
        return None
    if values.shape[0] != n_rows:
        return None
    return values


def explicit_false_obs_mask(obs: h5py.Group, key: str, n_rows: int) -> Optional[np.ndarray]:
    """Return rows explicitly marked false without expanding categorical strings."""
    if key not in obs:
        return None
    obj = obs[key]
    false_words = {"false", "0", "no"}

    def category_labels(dataset: Any) -> List[str]:
        return [str(decode_scalar(value)).strip().lower() for value in dataset[:]]

    if hasattr(obj, "keys") and "categories" in obj and "codes" in obj:
        labels = category_labels(obj["categories"])
        false_codes = [idx for idx, value in enumerate(labels) if value in false_words]
        codes = np.asarray(obj["codes"][:n_rows], dtype=np.int64)
        return np.isin(codes, false_codes) if false_codes else np.zeros(len(codes), dtype=bool)
    if "__categories" in obs and key in obs["__categories"] and hasattr(obj, "shape"):
        labels = category_labels(obs["__categories"][key])
        false_codes = [idx for idx, value in enumerate(labels) if value in false_words]
        codes = np.asarray(obj[:n_rows], dtype=np.int64)
        return np.isin(codes, false_codes) if false_codes else np.zeros(len(codes), dtype=bool)
    if hasattr(obj, "shape") and obj.shape != ():
        values = np.asarray(obj[:n_rows])
        if values.dtype.kind == "b":
            return ~values.astype(bool)
        if values.dtype.kind in "iu":
            return values == 0
        normalized = np.char.lower(np.char.strip(values.astype(str)))
        return np.isin(normalized, list(false_words))
    return None


def choose_numeric_obs_key(obs_keys: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    available = set(obs_keys)
    for key in candidates:
        if key in available:
            return key
    return None


def qc_keep_mask_from_obs_or_matrix(
    handle: h5py.File,
    row: Dict[str, Any],
    min_detected_genes: int,
    min_total_counts: float,
    ignore_is_primary_data: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n_cells = int(row["n_obs"])
    obs = handle["obs"]
    obs_keys = list(obs.keys())
    detected_key = choose_numeric_obs_key(
        obs_keys,
        ["n_genes", "nFeature_RNA", "total_genes", "n_genes_by_counts", "n_features", "nFeature"],
    )
    total_key = choose_numeric_obs_key(
        obs_keys,
        ["total_counts", "n_counts", "nCount_RNA", "total_UMIs", "n_counts_UMIs", "n_UMIs"],
    )
    detected = read_obs_numeric(obs, detected_key, n_cells)
    totals = read_obs_numeric(obs, total_key, n_cells)
    source_parts = []
    primary_keep = np.ones(n_cells, dtype=bool)
    if "is_primary_data" in obs and not ignore_is_primary_data:
        explicit_false = explicit_false_obs_mask(obs, "is_primary_data", n_cells)
        if explicit_false is not None and explicit_false.size == n_cells:
            primary_keep = ~explicit_false
            source_parts.append("obs:is_primary_data_explicit_false_removed")
    if detected is not None:
        source_parts.append(f"obs:{detected_key}")
    if totals is not None:
        source_parts.append(f"obs:{total_key}")

    need_matrix_totals = totals is None and float(min_total_counts) > 0.0
    if detected is None or need_matrix_totals:
        matrix = handle[row["selected_matrix_key"]]
        fmt = str(row["matrix_format"])
        if fmt == "csr_matrix":
            indptr = np.asarray(matrix["indptr"][:], dtype=np.int64)
            if detected is None:
                detected = np.diff(indptr).astype(np.float64)
                source_parts.append("csr:indptr_detected")
            if need_matrix_totals:
                data = np.asarray(matrix["data"][:], dtype=np.float64)
                row_counts = np.diff(indptr)
                totals = np.zeros(n_cells, dtype=np.float64)
                nonempty = row_counts > 0
                starts = indptr[:-1][nonempty]
                totals[nonempty] = np.add.reduceat(data, starts)
                source_parts.append("csr:data_totals")
        else:
            keep = primary_keep.copy()
            return keep, {
                "qc_filter_source": "+".join(source_parts + [f"expression_qc_not_applied_for_{fmt}"]),
                "qc_cells_before": n_cells,
                "qc_cells_after": int(np.sum(keep)),
                "qc_cells_filtered": int(n_cells - np.sum(keep)),
            }

    keep = primary_keep.copy()
    if totals is not None:
        keep &= np.isfinite(totals) & (totals > float(min_total_counts))
    if detected is not None:
        keep &= np.isfinite(detected) & (detected >= int(min_detected_genes))
    return keep, {
        "qc_filter_source": "+".join(source_parts) if source_parts else "none",
        "qc_cells_before": n_cells,
        "qc_cells_after": int(np.sum(keep)),
        "qc_cells_filtered": int(n_cells - np.sum(keep)),
        "qc_min_detected_genes": int(min_detected_genes),
        "qc_min_total_counts": float(min_total_counts),
    }


def rows_for_file(n_cells: int, max_cells: Optional[int], seed: int) -> np.ndarray:
    if max_cells is None or n_cells <= max_cells:
        return np.arange(n_cells, dtype=np.int64)
    rng = np.random.default_rng(seed)
    rows = rng.choice(n_cells, size=max_cells, replace=False)
    rows.sort()
    return rows.astype(np.int64)


def parse_age_years(stage: object) -> Tuple[int, str]:
    value = str(stage or "").strip().lower()
    interval = re.search(r"(\d+(?:\.\d+)?)[-–](\d+(?:\.\d+)?)\s*year", value)
    if interval:
        return int(round((float(interval.group(1)) + float(interval.group(2))) / 2.0)), "development_stage_year_interval_midpoint"
    exact = re.search(r"(\d+(?:\.\d+)?)[- ]year[- ]old", value)
    if exact:
        return int(round(float(exact.group(1)))), "development_stage_exact_year"
    decade_words = {
        "second": 15, "third": 25, "fourth": 35, "fifth": 45, "sixth": 55,
        "seventh": 65, "eighth": 75, "ninth": 85, "tenth": 95,
    }
    for word, midpoint in decade_words.items():
        if f"{word} decade" in value:
            return midpoint, "development_stage_decade_midpoint"
    return -1, "unresolved"


def write_split_manifests(
    file_rows: Sequence[Dict[str, Any]],
    run_root: Path,
    config: Dict[str, Any],
    seed: int,
    max_cells_per_file: Optional[int],
    min_detected_genes: int,
    min_total_counts: float,
    ignore_is_primary_data: bool = False,
) -> Dict[str, Any]:
    manifest_dir = ensure_dir(run_root / "manifests")
    paths = {split: manifest_dir / f"split_{split}.csv" for split in ["train", "val", "test"]}
    handles = {split: paths[split].open("w", encoding="utf-8", newline="") for split in paths}
    writers = {split: csv.DictWriter(handles[split], fieldnames=RECORD_FIELDNAMES) for split in paths}
    for writer in writers.values():
        writer.writeheader()
    split_sizes = Counter()
    split_groups = {split: set() for split in paths}
    qc_filter_stats = Counter()
    label_cfg = config["data"]["label_priority"]
    split_cfg = config["data"]["split"]
    try:
        for file_idx, row in enumerate(file_rows):
            with h5py.File(row["file_path"], "r") as handle:
                obs = handle["obs"]
                obs_keys = list(obs.keys())
                n_cells = int(row["n_obs"])
                qc_keep, qc_stats = qc_keep_mask_from_obs_or_matrix(
                    handle, row, min_detected_genes, min_total_counts, ignore_is_primary_data
                )
                row.update(qc_stats)
                qc_filter_stats["cells_before_qc"] += int(qc_stats["qc_cells_before"])
                qc_filter_stats["cells_after_qc"] += int(qc_stats["qc_cells_after"])
                qc_filter_stats["cells_filtered_qc"] += int(qc_stats["qc_cells_filtered"])
                row_cap = row.get("planned_max_cells")
                if row_cap is None or int(row_cap) <= 0:
                    row_cap = max_cells_per_file
                valid_rows = np.flatnonzero(qc_keep).astype(np.int64)
                positions = rows_for_file(
                    len(valid_rows), int(row_cap) if row_cap is not None else None, seed + file_idx
                )
                rows = valid_rows[positions]
                group_values = read_obs_values_selected(obs, row["group_key"], rows)
                cell_type = read_obs_values_selected(
                    obs,
                    row.get("cell_type_key") or select_first_present(label_cfg["cell_type"], obs_keys),
                    rows,
                )
                tissue = read_obs_values_selected(obs, select_first_present(label_cfg["tissue"], obs_keys), rows)
                disease = read_obs_values_selected(obs, select_first_present(label_cfg["disease"], obs_keys), rows)
                assay = read_obs_values_selected(obs, select_first_present(label_cfg["assay"], obs_keys), rows)
                author_cell_type = read_obs_values_selected(
                    obs, select_first_present(["author_cell_type", "cell_state", "celltype"], obs_keys), rows
                )
                development_stage = read_obs_values_selected(
                    obs, select_first_present(["development_stage"], obs_keys), rows
                )
                sex = read_obs_values_selected(obs, select_first_present(["sex"], obs_keys), rows)
                ethnicity = read_obs_values_selected(
                    obs, select_first_present(["self_reported_ethnicity", "ethnicity", "Ethnicity"], obs_keys), rows
                )
                suspension_type = read_obs_values_selected(
                    obs, select_first_present(["suspension_type"], obs_keys), rows
                )
                donor = read_obs_values_selected(
                    obs,
                    row.get("donor_key") or select_first_present(["donor_id"], obs_keys),
                    rows,
                )
                sample = read_obs_values_selected(
                    obs,
                    select_first_present(["sample_id", "sample", "sample_uuid", "orig.ident"], obs_keys),
                    rows,
                )
                batch = read_obs_values_selected(
                    obs,
                    select_first_present(
                        ["batch", "batch_id", "pool_number", "library_uuid", "suspension_uuid"],
                        obs_keys,
                    ),
                    rows,
                )
                stem = Path(row["file_path"]).stem
                for local_idx, row_idx in enumerate(rows):
                    group_value = str(group_values[local_idx]) if group_values is not None else str(row_idx)
                    group_id = f"{stem}::{group_value}"
                    split = stable_split(
                        group_id,
                        seed,
                        train_frac=float(split_cfg.get("train_frac", 0.8)),
                        val_frac=float(split_cfg.get("val_frac", 0.1)),
                    )
                    raw_cell_type = cell_type[local_idx] if cell_type is not None else None
                    harmonized_cell_type = (
                        str(raw_cell_type) if row.get("cell_type_key") else harmonize_cell_type(raw_cell_type)
                    )
                    stage_value = str(development_stage[local_idx]) if development_stage is not None else "unknown"
                    age, age_source = parse_age_years(stage_value)
                    batch_value = str(batch[local_idx]) if batch is not None else "unknown"
                    record = {
                        "file_path": row["file_path"],
                        "row_idx": int(row_idx),
                        "group_id": group_id,
                        "dataset_id": stem,
                        "cell_type": harmonized_cell_type,
                        "coarse_cell_type": cell_type_family(harmonized_cell_type),
                        "tissue": harmonize_tissue(tissue[local_idx] if tissue is not None else None),
                        "disease": harmonize_disease(disease[local_idx] if disease is not None else None),
                        "assay": harmonize_assay(assay[local_idx] if assay is not None else None),
                        "donor_id": str(donor[local_idx]) if donor is not None else "unknown",
                        "sample_id": str(sample[local_idx]) if sample is not None else "unknown",
                        "batch_id": batch_value,
                        "technical_block": batch_value if batch_value != "unknown" else (
                            str(sample[local_idx]) if sample is not None else "unknown"
                        ),
                        "author_cell_type": str(author_cell_type[local_idx]) if author_cell_type is not None else "unknown",
                        "development_stage": stage_value,
                        "age": age,
                        "age_parse_source": age_source,
                        "sex": str(sex[local_idx]) if sex is not None else "unknown",
                        "ethnicity": str(ethnicity[local_idx]) if ethnicity is not None else "unknown",
                        "suspension_type": str(suspension_type[local_idx]) if suspension_type is not None else "unknown",
                    }
                    writers[split].writerow(record)
                    split_sizes[split] += 1
                    split_groups[split].add(group_id)
    finally:
        for handle in handles.values():
            handle.close()
    return {
        "split_paths": {split: str(path) for split, path in paths.items()},
        "split_sizes": {split: int(split_sizes[split]) for split in paths},
        "split_group_counts": {split: len(split_groups[split]) for split in paths},
        "qc_filter_stats": {key: int(value) for key, value in qc_filter_stats.items()},
    }


def scan_files(
    files: Sequence[Path],
    config: Dict[str, Any],
    protein_symbols: set,
    seed: int,
    require_raw_counts: bool = False,
    planned_caps: Optional[Dict[str, int]] = None,
    split_group_key: Optional[str] = None,
    donor_key: Optional[str] = None,
    cell_type_key: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    file_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    excluded_rows: List[Dict[str, Any]] = []
    started = time.time()
    for idx, path in enumerate(files, start=1):
        file_started = time.time()
        if idx == 1 or idx % 10 == 0:
            progress(f"inclusive scan: starting {idx}/{len(files)}, file={path.name}")
        try:
            with h5py.File(path, "r") as handle:
                if not is_human_file(handle):
                    excluded_rows.append({"file_path": str(path), "exclude_reason": "non_human"})
                    continue
                selected, candidates = choose_matrix(handle, rng)
                for cand in candidates:
                    candidate_rows.append({"file_path": str(path), **cand})
                if selected is None:
                    excluded_rows.append({"file_path": str(path), "exclude_reason": "no_inclusive_log_compatible_matrix"})
                    continue
                if require_raw_counts and selected.get("semantics") not in {"raw_count_like", "integer_low_dynamic_range"}:
                    excluded_rows.append({
                        "file_path": str(path),
                        "exclude_reason": "no_validated_raw_count_matrix",
                        "selected_matrix_key": selected.get("matrix_key"),
                        "selected_semantics": selected.get("semantics"),
                    })
                    continue
                feature_names, biotype, filtered, feature_annotation_source = read_feature_table(
                    handle,
                    str(selected["matrix_key"]),
                    int(selected["n_vars"]),
                    protein_symbols,
                )
                keep_idx, pc_stats = protein_coding_indices(feature_names, biotype, filtered, protein_symbols)
                if keep_idx.size == 0:
                    excluded_rows.append({
                        "file_path": str(path),
                        "exclude_reason": "no_protein_coding_features",
                        "selected_matrix_key": selected["matrix_key"],
                        "feature_annotation_source": feature_annotation_source,
                        "protein_coding_filter_stats": json.dumps(pc_stats, sort_keys=True),
                    })
                    continue
                qc = sample_qc_metrics(handle[str(selected["matrix_key"])], str(selected["matrix_format"]), rng)
                obs_keys = list(handle["obs"].keys())
                requested = [key for key in (split_group_key, donor_key, cell_type_key) if key]
                missing = [key for key in requested if key not in obs_keys]
                if missing:
                    excluded_rows.append({
                        "file_path": str(path),
                        "exclude_reason": f"requested_obs_fields_missing:{','.join(missing)}",
                    })
                    continue
                group_key = split_group_key or select_first_present(config["data"]["split"]["group_priority"], obs_keys) or "_index"
                row = {
                    "file_path": str(path),
                    "selected_matrix_source": selected["candidate_label"],
                    "selected_matrix_key": selected["matrix_key"],
                    "source_semantics": selected["semantics"],
                    "confidence": selected["confidence"],
                    "semantics_reason": selected["reason"],
                    "transform": selected["transform"],
                    "include_for_training": True,
                    "n_obs": int(selected["n_obs"]),
                    "n_vars": int(selected["n_vars"]),
                    "matrix_format": selected["matrix_format"],
                    "group_key": group_key,
                    "donor_key": donor_key,
                    "cell_type_key": cell_type_key,
                    "feature_annotation_source": feature_annotation_source,
                    "protein_coding_features": int(keep_idx.size),
                    "protein_coding_filter_stats": json.dumps(pc_stats, sort_keys=True),
                    "planned_max_cells": (planned_caps or {}).get(str(path)),
                    **qc,
                    "_feature_names": feature_names,
                    "_keep_idx": keep_idx,
                }
                file_rows.append(row)
        except Exception as exc:
            excluded_rows.append({"file_path": str(path), "exclude_reason": repr(exc)})
        if idx == 1 or idx % 10 == 0 or idx == len(files):
            progress(
                "inclusive scan: "
                f"{idx}/{len(files)} scanned, kept={len(file_rows)}, excluded={len(excluded_rows)}, "
                f"elapsed={time.time() - started:.1f}s, last_file_s={time.time() - file_started:.2f}, "
                f"last={path.name}"
            )
    return file_rows, candidate_rows, excluded_rows


def files_from_manifest(
    manifest_path: str,
    path_column: str,
    role: Optional[str],
) -> Tuple[List[Path], Dict[str, int]]:
    table = pd.read_csv(os.path.expandvars(manifest_path))
    if path_column not in table.columns:
        raise KeyError(f"manifest has no path column {path_column!r}: {manifest_path}")
    if role is not None:
        if "role" not in table.columns:
            raise KeyError(f"--role requires a role column: {manifest_path}")
        table = table[table["role"].astype(str) == role]
    if table.empty:
        raise ValueError(f"no files selected from manifest={manifest_path}, role={role!r}")
    files: List[Path] = []
    caps: Dict[str, int] = {}
    for _, row in table.iterrows():
        path = Path(os.path.expandvars(str(row[path_column]))).resolve()
        files.append(path)
        if "planned_max_cells" in table.columns and pd.notna(row.get("planned_max_cells")):
            caps[str(path)] = int(row["planned_max_cells"])
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        preview = "\n".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} manifest files are missing on this server; first paths:\n{preview}")
    return sorted(set(files)), caps


def bundle_file_meta(row: Dict[str, Any], target_sum: float) -> Dict[str, Any]:
    return {
        "file_path": row["file_path"],
        "matrix_key": row["selected_matrix_key"],
        "matrix_format": row["matrix_format"],
        "use_raw": row["selected_matrix_key"] == "raw/X",
        "n_cells": int(row["n_obs"]),
        "n_genes": int(row["n_vars"]),
        "group_key": row["group_key"],
        "cell_type_key": row.get("cell_type_key"),
        "tissue_key": None,
        "disease_key": None,
        "assay_key": None,
        "feature_info": {
            "expression_policy": "inclusive_log_compatible",
            "expression_transform": row["transform"],
            "source_semantics": row["source_semantics"],
            "semantics_confidence": row["confidence"],
            "semantics_reason": row["semantics_reason"],
            "target_sum_for_raw": float(target_sum),
            "feature_source_counts": {"column:feature_name": int(row["n_vars"])},
            "feature_annotation_source": row.get("feature_annotation_source"),
            "protein_coding_features": int(row["protein_coding_features"]),
            "protein_coding_filter_stats": json.loads(row["protein_coding_filter_stats"]),
            "qc_filter_source": row.get("qc_filter_source"),
            "qc_cells_before": row.get("qc_cells_before"),
            "qc_cells_after": row.get("qc_cells_after"),
            "qc_cells_filtered": row.get("qc_cells_filtered"),
            "qc_min_detected_genes": row.get("qc_min_detected_genes"),
            "qc_min_total_counts": row.get("qc_min_total_counts"),
        },
    }


def write_compatible_config(config: Dict[str, Any], run_root: Path, output_root: Path) -> None:
    cfg = dict(config)
    cfg["output"] = dict(config.get("output", {}))
    cfg["output"]["root"] = str(output_root)
    cfg["data"] = dict(config.get("data", {}))
    cfg["data"]["normalize_total"] = None
    cfg["data"]["log1p"] = False
    cfg["data"]["feature_column"] = "feature_name"
    cfg["data"]["feature_namespace"] = "column"
    cfg["data"]["expression_policy"] = "inclusive_log_compatible"
    write_json(run_root / "training_config.inclusive_log_compatible.json", cfg)
    with (run_root / "training_config.inclusive_log_compatible.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("seed", 20260706))
    h5ad_dir_value = args.h5ad_dir or config["data"].get("h5ad_dir")
    output_root = Path(os.path.expandvars(args.output_root or config["output"]["root"])).resolve()
    run_root = output_root / args.run_id
    if run_root.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {run_root}; pass --overwrite to rebuild")
    ensure_dir(run_root)

    protein_symbols, annotation_cache = load_protein_coding_symbols(Path(args.annotation_dir), url=args.gencode_url)
    if args.file_manifest:
        files, planned_caps = files_from_manifest(args.file_manifest, args.path_column, args.role)
        source_description = f"manifest={Path(args.file_manifest).resolve()}, role={args.role!r}"
    else:
        if not h5ad_dir_value:
            raise ValueError("provide --file-manifest or --h5ad-dir (or data.h5ad_dir in config)")
        h5ad_dir = Path(os.path.expandvars(str(h5ad_dir_value))).resolve()
        files = sorted(h5ad_dir.glob("*.h5ad"))
        planned_caps = {}
        source_description = str(h5ad_dir)
    if args.max_files is not None:
        files = files[: args.max_files]
    progress(f"inclusive preprocess: scanning {len(files)} files from {source_description}")
    file_rows, candidate_rows, excluded_rows = scan_files(
        files,
        config,
        protein_symbols,
        seed,
        require_raw_counts=args.require_raw_counts,
        planned_caps=planned_caps,
        split_group_key=args.split_group_key,
        donor_key=args.donor_key,
        cell_type_key=args.cell_type_key,
    )
    if not file_rows:
        write_csv(run_root / "matrix_candidate_manifest.csv", candidate_rows)
        write_csv(run_root / "excluded_files.csv", excluded_rows)
        progress(f"inclusive preprocess: no files passed; exclude_reason_counts={dict(Counter(row.get('exclude_reason', 'unknown') for row in excluded_rows))}")
        raise RuntimeError("No files passed inclusive_log_compatible preprocessing policy")

    reference_vocab = None
    if args.reference_bundle:
        with Path(os.path.expandvars(args.reference_bundle)).open("r", encoding="utf-8") as handle:
            reference_vocab = json.load(handle)["gene_vocab"]
        if not reference_vocab:
            raise ValueError(f"reference bundle has an empty gene_vocab: {args.reference_bundle}")
    gene_vocab, file_gene_maps = build_gene_vocab(file_rows, reference_vocab=reference_vocab)
    maps_path = run_root / "file_gene_maps.npz"
    save_file_gene_maps(maps_path, file_gene_maps)
    split_info = write_split_manifests(
        file_rows,
        run_root,
        config,
        seed,
        args.max_cells_per_file,
        args.min_detected_genes,
        args.min_total_counts,
        args.ignore_is_primary_data,
    )

    manifest_rows = [{k: v for k, v in row.items() if not k.startswith("_")} for row in file_rows]
    write_csv(run_root / "selected_expression_manifest.csv", manifest_rows)
    write_csv(run_root / "matrix_candidate_manifest.csv", candidate_rows)
    write_csv(run_root / "excluded_files.csv", excluded_rows)

    summary = {
        "bundle_schema_version": "manifest-v1",
        "clean_pipeline": "inclusive_log_compatible_v1",
        "num_files": len(file_rows),
        "num_excluded_files": len(excluded_rows),
        "num_cells_in_included_files": int(sum(int(row["n_obs"]) for row in file_rows)),
        "num_records": int(sum(split_info["split_sizes"].values())),
        "num_genes": len(gene_vocab),
        "split_sizes": split_info["split_sizes"],
        "split_group_counts": split_info["split_group_counts"],
        "qc_filter_stats": split_info["qc_filter_stats"],
        "qc_policy": {
            "min_detected_genes": args.min_detected_genes,
            "min_total_counts": args.min_total_counts,
            "note": "Light hard filter only: remove empty or near-empty cells; no global mito/ribo/max-count filtering.",
            "ignore_is_primary_data": bool(args.ignore_is_primary_data),
        },
        "split_manifest_paths": split_info["split_paths"],
        "matrix_key_counts": dict(Counter(row["selected_matrix_key"] for row in manifest_rows)),
        "matrix_format_counts": dict(Counter(row["matrix_format"] for row in manifest_rows)),
        "source_semantics_counts": dict(Counter(row["source_semantics"] for row in manifest_rows)),
        "expression_transform_counts": dict(Counter(row["transform"] for row in manifest_rows)),
        "expression_policy": "inclusive_log_compatible",
        "gene_space": (
            "frozen_training_reference_vocab_with_missing_gene_masks"
            if reference_vocab is not None
            else "strict_intersection_of_measured_human_protein_coding_feature_name_across_training_files"
        ),
        "protein_coding_annotation": str(annotation_cache),
        "target_sum_for_raw": args.target_sum,
        "file_source": source_description,
        "role_filter": args.role,
        "require_raw_counts": bool(args.require_raw_counts),
        "reference_bundle": str(Path(os.path.expandvars(args.reference_bundle)).resolve()) if args.reference_bundle else None,
        "user_obs_mapping": {
            "split_group_key": args.split_group_key,
            "donor_key": args.donor_key,
            "cell_type_key": args.cell_type_key,
        },
    }
    bundle_payload = {
        "gene_vocab": gene_vocab,
        "split_paths": split_info["split_paths"],
        "file_gene_maps_path": str(maps_path),
        "summary": summary,
        "file_metas": [bundle_file_meta(row, args.target_sum) for row in file_rows],
    }
    write_json(run_root / "data_bundle.json", bundle_payload)
    write_json(run_root / "data_bundle.slim.json", bundle_payload)
    write_compatible_config(config, run_root, output_root)
    progress(f"inclusive preprocess complete: run_root={run_root}")
    progress(f"summary: {summary}")


if __name__ == "__main__":
    main()
