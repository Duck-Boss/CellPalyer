#!/usr/bin/env python3
"""Download and audit the original GEO sci-Plex 2 count matrix and metadata.

Usage: python download_data.py [--output-dir data/raw/sciplex2] [--audit-only]
No API key is required. Existing complete files are reused; unfinished downloads
are staged as .part files, and a manifest records exact bytes and SHA256 hashes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import json
import shutil
from pathlib import Path
import time

import numpy as np
import pandas as pd
import requests


BASE_URL = "https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/GSM4150377/suppl/"
PREFIX = "GSM4150377_sciPlex2_A549_Transcription_Modulators_"
FILES = {
    "counts": PREFIX + "UMI.count.matrix.gz",
    "cells": PREFIX + "cell.annotations.txt.gz",
    "genes": PREFIX + "gene.annotations.txt.gz",
    "hash_counts": PREFIX + "hashTable.out.txt.gz",
    "sample_sheet": "GSM4150377_sciPlex2_hashSampleSheet.txt.gz",
    "metadata": "GSM4150377_sciPlex2_pData.txt.gz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parallel_download(path: Path, url: str, expected: int, workers: int = 6):
    """Resume a verified byte prefix and fetch its suffix with HTTP ranges."""
    temporary = path.with_suffix(path.suffix + ".part")
    prefix_size = temporary.stat().st_size if temporary.exists() else 0
    if prefix_size > expected:
        raise RuntimeError("Temporary file exceeds the source length")
    width = max(1, (expected - prefix_size + workers - 1) // workers)
    ranges = [(start, min(start + width, expected) - 1)
              for start in range(prefix_size, expected, width)]

    def get_range(bounds):
        start, end = bounds
        segment = path.with_name(f"{path.name}.range-{start}-{end}.part")
        for attempt in range(4):
            offset = segment.stat().st_size if segment.exists() else 0
            if offset == end - start + 1:
                return segment
            if offset > end - start + 1:
                raise RuntimeError("Range temporary file exceeds its source interval")
            try:
                headers = {"Range": f"bytes={start+offset}-{end}"}
                with requests.get(url, headers=headers, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    required_range = f"bytes {start+offset}-{end}/{expected}"
                    if response.status_code != 206 or response.headers.get("Content-Range") != required_range:
                        raise RuntimeError(f"Source did not honor HTTP Range: {response.status_code}, {response.headers.get('Content-Range')}")
                    with segment.open("ab") as output:
                        for chunk in response.iter_content(1 << 20):
                            if chunk:
                                output.write(chunk)
                if segment.stat().st_size != end - start + 1:
                    raise RuntimeError("Incomplete range download")
                print(f"Downloaded range {start:,}-{end:,}", flush=True)
                return segment
            except (requests.RequestException, OSError, EOFError, RuntimeError):
                if attempt == 3:
                    raise
                time.sleep(min(2 ** attempt, 8))

    print(f"Range download {path.name}: {prefix_size:,}/{expected:,} bytes, {len(ranges)} streams", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        segments = list(pool.map(get_range, ranges))
    with temporary.open("ab") as output:
        for segment in segments:
            with segment.open("rb") as stream:
                shutil.copyfileobj(stream, output, length=8 << 20)
    if temporary.stat().st_size != expected:
        raise RuntimeError("Assembled download size mismatch")
    # Decompression checks gzip CRC32 and length against the source trailer.
    with gzip.open(temporary, "rb") as stream:
        for _ in iter(lambda: stream.read(8 << 20), b""):
            pass
    temporary.replace(path)
    for segment in segments:
        segment.unlink()


def download(path: Path, url: str) -> dict:
    """Atomically fetch one file, and verify gzip integrity before accepting it."""
    if not path.exists():
        if "UMI.count.matrix" in path.name:
            response = requests.head(url, timeout=(30, 60))
            response.raise_for_status()
            expected = int(response.headers["Content-Length"])
            if expected > 2_000_000_000:
                raise RuntimeError("Refusing a source file larger than 2 GB")
            parallel_download(path, url, expected)
            return {"file": path.name, "url": url, "size_bytes": path.stat().st_size,
                    "sha256": sha256(path)}
        temporary = path.with_suffix(path.suffix + ".part")
        for attempt in range(4):
            try:
                with requests.get(url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    expected = response.headers.get("Content-Length")
                    if expected and int(expected) > 2_000_000_000:
                        raise RuntimeError("Refusing a source file larger than 2 GB")
                    size = 0
                    with temporary.open("wb") as output:
                        for chunk in response.iter_content(4 << 20):
                            if chunk:
                                output.write(chunk)
                                size += len(chunk)
                    if expected and size != int(expected):
                        raise RuntimeError(f"Incomplete file: {size} != {expected}")
                with gzip.open(temporary, "rb") as stream:
                    for _ in iter(lambda: stream.read(8 << 20), b""):
                        pass
                temporary.replace(path)
                print(f"Downloaded {path.name}: {path.stat().st_size:,} bytes", flush=True)
                break
            except (requests.RequestException, OSError, EOFError, RuntimeError):
                if attempt == 3:
                    raise
                time.sleep(min(2 ** attempt, 8))
    else:
        print(f"Using existing {path.name}: {path.stat().st_size:,} bytes", flush=True)
    return {"file": path.name, "url": url, "size_bytes": path.stat().st_size,
            "sha256": sha256(path)}


def frame(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", compression="gzip", **kwargs)


def matrix_audit(path: Path, n_genes: int, n_cells: int) -> dict:
    """Stream the matrix, avoiding any dense gene-by-cell representation."""
    with gzip.open(path, "rt", encoding="ascii") as stream:
        first_line = stream.readline()
        if first_line.startswith("%%MatrixMarket"):
            banner = first_line.strip()
            line = stream.readline()
            while line.startswith("%"):
                line = stream.readline()
            matrix_n_genes, matrix_n_cells, expected_nnz = map(int, line.split())
            if (matrix_n_genes, matrix_n_cells) != (n_genes, n_cells):
                raise ValueError("Matrix shape does not match annotation files")
            block = []
        else:
            # GEO describes MatrixMarket, but the actual source file is only
            # coordinate triples without a banner or a dimensions record.
            banner = "headerless TSV: 1-based gene_index, cell_index, integer_UMI_count"
            expected_nnz = None
            block = [first_line]
        cell_counts = np.zeros(n_cells, dtype=np.int64)
        detected_genes = np.zeros(n_cells, dtype=np.int64)
        actual_nnz = 0
        minimum, maximum = None, None

        def consume(lines):
            nonlocal actual_nnz, minimum, maximum
            values = np.fromstring("".join(lines), sep=" ", dtype=np.int64)
            if values.size % 3:
                raise ValueError("Malformed matrix coordinates")
            triples = values.reshape(-1, 3)
            if not ((triples[:, 0] >= 1).all() and (triples[:, 0] <= n_genes).all()
                    and (triples[:, 1] >= 1).all() and (triples[:, 1] <= n_cells).all()
                    and (triples[:, 2] > 0).all()):
                raise ValueError("Out-of-range coordinates or nonpositive UMI count")
            np.add.at(cell_counts, triples[:, 1] - 1, triples[:, 2])
            np.add.at(detected_genes, triples[:, 1] - 1, 1)
            low, high = int(triples[:, 2].min()), int(triples[:, 2].max())
            minimum = low if minimum is None else min(minimum, low)
            maximum = high if maximum is None else max(maximum, high)
            actual_nnz += len(triples)

        for line in stream:
            if line.strip():
                block.append(line)
            if len(block) == 65536:
                consume(block)
                block = []
        if block:
            consume(block)
    if expected_nnz is not None and actual_nnz != expected_nnz:
        raise ValueError(f"nnz mismatch: {actual_nnz} != {expected_nnz}")
    return {
        "format": banner,
        "orientation": "genes_by_cells; transpose to cells_by_genes",
        "shape": [n_genes, n_cells], "nnz": actual_nnz,
        "values": "positive integer UMI counts, no library normalization or log transform",
        "minimum_nonzero_count": minimum, "maximum_count": maximum,
        "total_umi": int(cell_counts.sum()),
        "cell_umi_min_median_max": [int(cell_counts.min()), float(np.median(cell_counts)),
                                     int(cell_counts.max())],
        "cell_detected_genes_min_median_max": [int(detected_genes.min()),
            float(np.median(detected_genes)), int(detected_genes.max())],
        "_cell_counts": cell_counts,
    }


def audit(folder: Path, file_manifest: dict) -> dict:
    cells = frame(folder / FILES["cells"], header=None, names=["cell_id", "sample"])
    genes = frame(folder / FILES["genes"], header=None, names=["ensembl_id", "gene_symbol"])
    # R write.table includes an unnamed rowname field that is absent in the header.
    obs = frame(folder / FILES["metadata"], index_col=0)
    sample_sheet = frame(folder / FILES["sample_sheet"], header=None,
                         names=["top_oligo", "hash_sequence", "axis"])
    parts = obs["top_oligo"].str.split("_", expand=True)
    parts.columns = ["drug_alias", "dose_source_value", "well_id"]
    parts["dose_source_value"] = pd.to_numeric(parts["dose_source_value"])
    obs = obs.join(parts)
    matrix = matrix_audit(folder / FILES["counts"], len(genes), len(cells))
    counts = matrix.pop("_cell_counts")
    assert len(genes) == matrix["shape"][0]
    assert len(cells) == len(obs) == matrix["shape"][1]
    assert obs.index.tolist() == cells["cell_id"].tolist()
    assert obs["Cell"].tolist() == cells["cell_id"].tolist()
    n_umi_matches = bool(np.array_equal(counts, obs["n.umi"].to_numpy(dtype=np.int64)))
    # Missing hashes mean unassigned cells, not DMSO controls.
    valid = obs["top_oligo"].notna()
    zero = valid & obs["dose_source_value"].eq(0)
    positive = valid & obs["dose_source_value"].gt(0)
    well_counts = obs.loc[valid].groupby(["drug_alias", "dose_source_value", "well_id"],
                                        observed=True).size()
    condition_counts = obs.loc[valid].groupby(["drug_alias", "dose_source_value"],
                                             observed=True).size()
    obs["dose_um"] = obs["dose_source_value"]
    obs["solvent"] = np.where(obs["drug_alias"].eq("Dex"), "ethanol", "DMSO")
    obs.loc[~valid, "solvent"] = None
    # Exact formula from the author's Notebook_1_small_screen_analysis.R.
    # Retain well_id as the split unit; replicate is a plate-local designation.
    obs["culture_plate_author"] = obs["well_id"].str[0]
    obs["culture_well_author"] = obs["well_id"].str[1:4]
    well_column = pd.to_numeric(obs["well_id"].str[-2:], errors="coerce")
    obs["replicate_author"] = well_column.map(
        {column: f"rep{(column-1) % 3 + 1}" for column in range(1, 13)})
    obs.loc[valid, ["drug_alias", "dose_source_value", "dose_um", "well_id", "solvent",
                   "culture_plate_author", "culture_well_author", "replicate_author", "qval",
                   "hash_umis", "top_to_second_best_ratio"]].to_csv(folder / "assigned_cell_metadata.tsv", sep="\t")
    well_counts.rename("n_cells").to_csv(folder / "well_cell_counts.tsv", sep="\t")
    qualified = valid & obs["qval"].lt(0.05) & obs["top_to_second_best_ratio"].ge(2)
    author_qualified = valid & obs["qval"].lt(0.01) & obs["top_to_second_best_ratio"].gt(10) & obs["hash_umis"].gt(30)
    return {
        "dataset": "sci-Plex 2 / GSM4150377 / A549 transcription modulators",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "geo": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSM4150377",
            "file_directory": BASE_URL,
            "original_analysis": "https://github.com/cole-trapnell-lab/sci-plex/tree/master/small_screen",
            "author_metadata_and_units": "https://github.com/cole-trapnell-lab/sci-plex/blob/master/small_screen/Notebook_1_small_screen_analysis.R",
            "paper": "https://doi.org/10.1126/science.aax6234",
        },
        "files": file_manifest, "matrix": matrix,
        "metadata": {
            "obs_columns": list(obs.columns), "n_cells": len(obs),
            "n_genes": len(genes), "n_unique_gene_symbols": genes["gene_symbol"].nunique(),
            "n_unique_versioned_ensembl_ids": genes["ensembl_id"].nunique(),
            "cell_metadata_order_matches_matrix": True,
            "matrix_cell_umi_equals_metadata_n_umi": n_umi_matches,
            "batch_sample_counts": obs["sample"].value_counts().to_dict(),
            "n_assigned_cells": int(valid.sum()), "n_unassigned_cells": int((~valid).sum()),
            "n_explicit_zero_dose_controls": int(zero.sum()),
            "n_positive_dose_cells": int(positive.sum()),
            "n_hash_quality_cells_qval_lt_0p05_ratio_ge_2": int(qualified.sum()),
            "n_author_hash_quality_cells_qval_lt_0p01_ratio_gt_10_hash_umis_gt_30": int(author_qualified.sum()),
            "treatment_duration_h": 24,
            "dose_source_values": sorted(obs.loc[valid, "dose_source_value"].unique().tolist()),
            "dose_unit": "micromolar (uM), confirmed in original author R notebook plot labels e.g. line366/509",
            "vehicle_solvents": {"Dex": "ethanol", "Nutlin": "DMSO", "BMS": "DMSO", "SAHA": "DMSO"},
            "drug_aliases": {"BMS": "BMS-345541", "Dex": "dexamethasone",
                             "Nutlin": "nutlin-3a", "SAHA": "vorinostat"},
            "n_sample_sheet_wells": len(sample_sheet), "n_observed_wells": int(obs["well_id"].nunique()),
            "condition_cell_counts": [{"drug": k[0], "dose": k[1], "n_cells": int(v)}
                                      for k, v in condition_counts.items()],
            "author_hash_qc_condition_cell_counts": [{"drug": k[0], "dose": k[1], "n_cells": int(v)}
                 for k, v in obs.loc[author_qualified].groupby(["drug_alias", "dose_source_value"]).size().items()],
            "observed_wells_per_condition": [{"drug": k[0], "dose": k[1], "n_wells": int(v)}
                 for k, v in obs.loc[valid].groupby(["drug_alias", "dose_source_value"])["well_id"].nunique().items()],
            "replicate_fields": ["well_id derived from top_oligo", "culture_plate_author = first character of well_id", "replicate_author = rep1/rep2/rep3 according to (last_two_digit_column - 1) % 3 + 1", "sample (single sequencing pool; not a biological replicate)"],
            "recommended_split_group": "well_id (keep all cells from one treatment well in one split)",
            "replicate_caveat": "Published text reports triplicate; the original R analysis defines three plate-local replicate labels from well columns. Source includes A/B plate prefixes and one malformed 3-character E09 well (SAHA_100_E09). Preserve exact top_oligo/well_id; do not invent donor independence or silently map E09 to AE09.",
            "nonstandard_well_labels": sorted(obs.loc[valid & obs["well_id"].str.len().ne(4), "top_oligo"].unique().tolist()),
        },
        "preprocessing_guidance": [
            "The actual count file lacks a MatrixMarket header despite GEO's description. Read three columns gene_index,cell_index,count from gzip TSV, then create scipy.sparse.coo_matrix((count,(cell_index-1,gene_index-1)),shape=(n_cells,n_genes)).tocsr().",
            "Read pData with pandas.read_csv(path, sep=r'\\s+', index_col=0); preserve original row order.",
            "Use well_id from the third top_oligo component to avoid cell-level train/test leakage.",
            "Use explicit assigned dose==0 wells as controls; do not relabel missing hash assignments as controls.",
            "Match each drug to its own zero-dose solvent controls, especially ethanol for Dex versus DMSO for the other three compounds.",
            "Decide hash-assignment quality and cell QC before modeling; raw data include low confidence hashes.",
            "Keep Ensembl IDs as unique feature identifiers; gene symbols can repeat.",
            "Toxic high-dose conditions have very few surviving cells under author hash QC (e.g. BMS 50/100 uM and Nutlin 100 uM); report sample sizes and restrict confirmatory evaluation to sufficiently sampled groups.",
            "Fit gene filters, normalization conventions, HVG selection and LEXA on training data only.",
            "There is only one cell line and one measured treatment duration; this dataset cannot test time generalization or transfer between cell types.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "data/raw/sciplex2")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def process(item):
        key, name = item
        path = args.output_dir / name
        if args.audit_only and not path.exists():
            raise FileNotFoundError(path)
        return key, download(path, BASE_URL + name)
    with ThreadPoolExecutor(max_workers=3) as pool:
        manifest = dict(pool.map(process, FILES.items()))
    result = audit(args.output_dir, manifest)
    destination = args.output_dir / "audit.json"
    temporary = destination.with_suffix(".json.part")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"audit": str(destination), "shape_genes_by_cells": result["matrix"]["shape"],
                      "n_assigned_cells": result["metadata"]["n_assigned_cells"],
                      "n_observed_wells": result["metadata"]["n_observed_wells"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
