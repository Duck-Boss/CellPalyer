"""Shared CELLxGENE data utilities."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import time
import urllib.request
from urllib.error import URLError
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np


ENSEMBL_RE = re.compile(r"^ENS[A-Z]*G[0-9]+(?:\.[0-9]+)?$")
GENCODE_GTF_URL = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_44/gencode.v44.annotation.gtf.gz"


def progress(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    rows = list(rows)
    ensure_dir(path.parent)
    if fieldnames is None:
        fieldnames = sorted(set().union(*(row.keys() for row in rows))) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def decode_scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        try:
            return decode_scalar(value.item())
        except Exception:
            pass
    return value


def read_string_array(group: h5py.Group, key: str, limit: Optional[int] = None) -> np.ndarray:
    if group is None or key is None or key not in group:
        return np.asarray([], dtype=object)
    obj = group[key]

    def _read_dataset_strings(dataset: Any) -> List[str]:
        if hasattr(dataset, "keys"):
            if "categories" in dataset:
                return _read_dataset_strings(dataset["categories"])
            if "values" in dataset:
                return _read_dataset_strings(dataset["values"])
            if "codes" in dataset:
                return []
            keys = list(dataset.keys())
            if len(keys) == 1:
                return _read_dataset_strings(dataset[keys[0]])
            return []
        values = dataset[:]
        return [str(decode_scalar(x)) for x in values]

    if hasattr(obj, "keys") and "categories" in obj and "codes" in obj:
        codes = np.asarray(obj["codes"][: limit or obj["codes"].shape[0]], dtype=np.int64)
        cats = _read_dataset_strings(obj["categories"])
        return np.asarray([cats[c] if 0 <= c < len(cats) else "" for c in codes], dtype=object)

    if "__categories" in group and key in group["__categories"] and hasattr(obj, "shape"):
        try:
            codes = np.asarray(obj[: min(obj.shape[0], limit) if limit else obj.shape[0]], dtype=np.int64)
            cats = _read_dataset_strings(group["__categories"][key])
            return np.asarray([cats[c] if 0 <= c < len(cats) else "" for c in codes], dtype=object)
        except Exception:
            pass

    if hasattr(obj, "shape"):
        if obj.shape == ():
            values = [obj[()]]
        else:
            n = obj.shape[0]
            values = obj[: min(n, limit) if limit else n]
        return np.asarray([str(decode_scalar(x)) for x in values], dtype=object)
    return np.asarray([], dtype=object)


def read_index(group: h5py.Group, n_rows: Optional[int] = None) -> np.ndarray:
    for key in ["_index", "index"]:
        values = read_string_array(group, key, limit=n_rows)
        if values.size:
            return values
    n = int(n_rows or 0)
    return np.asarray([str(i) for i in range(n)], dtype=object)


def select_first_present(keys: Sequence[str], available: Iterable[str]) -> Optional[str]:
    available_set = set(available)
    for key in keys:
        if key in available_set:
            return key
    return None


def normalize_label(value: Any, missing: str = "unknown") -> str:
    text = str(value if value is not None else "").strip()
    if text == "" or text.lower() in {"nan", "none", "na", "n/a", "unknown", "unspecified"}:
        return missing
    text = re.sub(r"\s+", " ", text.replace("_", " ").replace("-", " ")).strip()
    return text


def harmonize_disease(value: Any) -> str:
    text = normalize_label(value)
    low = text.lower()
    if low in {"normal", "healthy", "control", "no disease", "non diseased", "unaffected"}:
        return "normal"
    return text


def harmonize_cell_type(value: Any) -> str:
    return normalize_label(value)


def harmonize_tissue(value: Any) -> str:
    return normalize_label(value)


def harmonize_assay(value: Any) -> str:
    return normalize_label(value)


def cell_type_family(value: Any) -> str:
    text = harmonize_cell_type(value)
    low = text.lower()
    if text == "unknown":
        return "unknown"
    if any(x in low for x in ["neuron", "interneuron", "glutamatergic", "gabaergic", "cholinergic", "dopaminergic"]):
        return "neuron"
    if any(x in low for x in ["photoreceptor", " rod cell", " cone cell", "retinal rod", "retinal cone"]):
        return "photoreceptor"
    if any(x in low for x in ["astrocyte", "oligodendrocyte", "microglial", "microglia", "ependymal", "schwann"]):
        return "glia"
    if any(x in low for x in ["t cell", "b cell", "nk cell", "natural killer", "monocyte", "macrophage", "dendritic", "lymphocyte", "leukocyte", "myeloid", "neutrophil", "basophil", "eosinophil", "mast cell", "plasma cell"]):
        return "immune"
    if "endothelial" in low:
        return "endothelial"
    if any(x in low for x in ["epithelial", "keratinocyte", "club cell", "goblet cell", "basal cell", "secretory cell", "ciliated cell"]):
        return "epithelial"
    if any(x in low for x in ["fibroblast", "stromal", "mesenchymal", "pericyte", "smooth muscle", "myofibroblast"]):
        return "stromal"
    if any(x in low for x in ["cardiac myocyte", "cardiomyocyte", "muscle", "myoblast", "myotube"]):
        return "muscle"
    if any(x in low for x in ["stem cell", "progenitor", "precursor", "blast cell"]):
        return "stem_progenitor"
    if any(x in low for x in ["erythrocyte", "erythroid", "megakaryocyte", "platelet"]):
        return "erythroid_megakaryocyte"
    if "hepatocyte" in low:
        return "hepatocyte"
    if any(x in low for x in ["podocyte", "mesangial", "kidney collecting duct", "loop of henle"]):
        return "kidney_specialized"
    return "other"


def matrix_encoding(obj: Any) -> str:
    if hasattr(obj, "shape") and len(getattr(obj, "shape", ())) == 2:
        return "dense"
    encoding = str(decode_scalar(obj.attrs.get("encoding-type", ""))) if hasattr(obj, "attrs") else ""
    if "csr" in encoding:
        return "csr_matrix"
    if "csc" in encoding:
        return "csc_matrix"
    if all(key in obj for key in ["data", "indices", "indptr"]):
        return "csr_matrix"
    return "unknown"


def matrix_shape(obj: Any) -> Tuple[int, int]:
    if hasattr(obj, "shape") and len(getattr(obj, "shape", ())) == 2:
        return int(obj.shape[0]), int(obj.shape[1])
    shape = obj.attrs.get("shape") if hasattr(obj, "attrs") else None
    if shape is None and "shape" in obj:
        shape = obj["shape"][:]
    if shape is None:
        raise ValueError("Sparse matrix has no shape metadata")
    return int(shape[0]), int(shape[1])


def sample_matrix_values(obj: Any, fmt: str, rng: np.random.Generator, max_values: int = 20_000) -> np.ndarray:
    if fmt == "dense":
        n_obs, n_vars = matrix_shape(obj)
        n_rows = min(n_obs, 8)
        rows = np.sort(rng.choice(n_obs, size=n_rows, replace=False)) if n_obs else np.asarray([], dtype=np.int64)
        if rows.size == 0:
            return np.asarray([], dtype=np.float64)
        if n_vars > 2048:
            cols = np.sort(rng.choice(n_vars, size=2048, replace=False))
            values = np.asarray(obj[rows][:, cols], dtype=np.float64).reshape(-1)
        else:
            values = np.asarray(obj[rows, :], dtype=np.float64).reshape(-1)
        return values[np.isfinite(values)]
    if fmt in {"csr_matrix", "csc_matrix"} and "data" in obj:
        data = obj["data"]
        n = int(data.shape[0])
        if n == 0:
            return np.asarray([], dtype=np.float64)
        take = min(n, max_values)
        if take == n:
            values = np.asarray(data[:], dtype=np.float64)
        else:
            start = int(rng.integers(0, n - take + 1))
            values = np.asarray(data[start : start + take], dtype=np.float64)
        return values[np.isfinite(values)]
    return np.asarray([], dtype=np.float64)


def expression_semantics_from_values(values: np.ndarray) -> Dict[str, Any]:
    if values.size == 0:
        return {
            "semantics": "unknown_or_mixed",
            "confidence": "low",
            "reason": "no finite sampled values",
            "value_min": None,
            "value_max": None,
            "fraction_integer_like": None,
            "fraction_negative": None,
        }
    vmin = float(values.min())
    vmax = float(values.max())
    frac_int = float(np.mean(np.isclose(values, np.round(values), atol=1e-6)))
    frac_neg = float(np.mean(values < 0))
    if frac_neg > 0.001:
        semantics, confidence, reason = (
            "scaled_or_residual_possible",
            "high",
            f"negative_fraction={frac_neg:.4g}",
        )
    elif frac_int > 0.98 and vmax > 20:
        semantics, confidence, reason = (
            "raw_count_like",
            "high",
            f"integer_fraction={frac_int:.4g}, max={vmax:.4g}",
        )
    elif frac_int > 0.98 and vmax <= 20:
        semantics, confidence, reason = (
            "integer_low_dynamic_range",
            "medium",
            f"integer_fraction={frac_int:.4g}, max={vmax:.4g}",
        )
    elif vmin >= 0 and vmax <= 30:
        semantics, confidence, reason = (
            "log1p_like",
            "medium",
            f"non_negative_non_integer, max={vmax:.4g}",
        )
    elif vmin >= 0 and vmax <= 50:
        semantics, confidence, reason = (
            "log_or_normalized_like",
            "medium",
            f"non_negative_processed, max={vmax:.4g}",
        )
    else:
        semantics, confidence, reason = (
            "unknown_or_mixed",
            "low",
            f"integer_fraction={frac_int:.4g}, min={vmin:.4g}, max={vmax:.4g}",
        )
    return {
        "semantics": semantics,
        "confidence": confidence,
        "reason": reason,
        "value_min": vmin,
        "value_max": vmax,
        "fraction_integer_like": frac_int,
        "fraction_negative": frac_neg,
    }


def stable_split(group_id: str, seed: int, train_frac: float, val_frac: float) -> str:
    digest = hashlib.sha1(f"{seed}::{group_id}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12)
    if value < train_frac:
        return "train"
    if value < train_frac + val_frac:
        return "val"
    return "test"


def cache_key_for_matrix(file_path: str, matrix_key: str) -> str:
    digest = hashlib.sha1(f"{file_path}::{matrix_key}".encode("utf-8")).hexdigest()[:12]
    return f"{Path(file_path).stem}_{digest}"


def download_gencode_gtf(annotation_dir: Path, url: str = GENCODE_GTF_URL) -> Path:
    ensure_dir(annotation_dir)
    local_source = Path(str(url)).expanduser()
    if local_source.exists() and local_source.stat().st_size > 0:
        progress(f"using local gene annotation: {local_source}")
        return local_source
    target = annotation_dir / Path(url).name
    if target.exists() and target.stat().st_size > 0:
        progress(f"gene annotation exists: {target}")
        return target
    progress(f"downloading gene annotation: {url}")
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
    except URLError as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download GENCODE gene annotation. "
            f"URL={url!r}. Put gencode.v44.annotation.gtf.gz in the annotation directory "
            "or pass --gencode-url /path/to/gencode.v44.annotation.gtf.gz."
        ) from exc
    os.replace(tmp, target)
    return target


def load_protein_coding_symbols(annotation_dir: Path, url: str = GENCODE_GTF_URL) -> Tuple[set, Path]:
    cache_path = annotation_dir / "protein_coding_gene_symbols.json"
    if cache_path.exists():
        payload = read_json(cache_path)
        return set(payload["symbols"]), cache_path
    gtf_path = download_gencode_gtf(annotation_dir, url=url)
    symbols = set()
    opener = gzip.open if str(gtf_path).endswith(".gz") else open
    attr_re = re.compile(r'(\S+) "([^"]+)"')
    with opener(gtf_path, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attrs = dict(attr_re.findall(fields[8]))
            if attrs.get("gene_type") == "protein_coding" or attrs.get("gene_biotype") == "protein_coding":
                name = attrs.get("gene_name")
                if name:
                    symbols.add(name)
    write_json(cache_path, {"source": str(gtf_path), "num_symbols": len(symbols), "symbols": sorted(symbols)})
    progress(f"loaded protein-coding symbols: {len(symbols)} from {gtf_path}")
    return symbols, cache_path


@dataclass
class FileMeta:
    file_path: str
    matrix_key: str
    matrix_format: str
    use_raw: bool
    n_cells: int
    n_genes: int
    feature_names: List[str]
    group_key: str
    cell_type_key: Optional[str]
    tissue_key: Optional[str]
    disease_key: Optional[str]
    assay_key: Optional[str]
    feature_info: Optional[Dict[str, Any]] = None


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_config(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(os.path.expandvars(handle.read()))


def parse_scalar(value: str) -> Any:
    import yaml

    return yaml.safe_load(value)


def apply_config_overrides(config: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Config override must be key=value, got {item!r}")
        key, value = item.split("=", 1)
        target = config
        parts = [part for part in key.split(".") if part]
        if not parts:
            raise ValueError(f"Empty config override key in {item!r}")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        target[parts[-1]] = parse_scalar(value)
    return config


def now_run_id(config: Dict[str, Any], override: Optional[str] = None) -> str:
    if override:
        return override
    model_cfg = config["model"]
    arch = model_cfg.get("architecture", "fixed_axes_vqvae")
    axes = model_cfg.get("num_axes", model_cfg.get("latent_tokens", 1))
    return f"cellxgene_{arch}_a{axes}_s{config['seed']}"


def build_model(input_dim: int, config: Dict[str, Any], gene_vocab: Optional[Sequence[str]] = None):
    model_cfg = dict(config["model"])
    architecture = model_cfg.pop("architecture", "fixed_axes_vqvae")
    if architecture in {"fixed_axes_vqvae", "modular_vqvae"}:
        from lexa.lexa_model import LEXA

        mask_prior_path = model_cfg.pop("module_mask_prior_path", None)
        mask_mode = str(model_cfg.get("module_mask_parameterization", "learned_softmax")).lower()
        if mask_mode in {"fixed_prior", "fixed", "prior_guided_softmax", "learned_prior_softmax"}:
            if not mask_prior_path:
                raise ValueError("model.module_mask_prior_path is required for a fixed or prior-guided module mask")
            prior_payload = np.load(mask_prior_path, allow_pickle=False)
            if isinstance(prior_payload, np.lib.npyio.NpzFile):
                try:
                    if "module_mask" not in prior_payload.files:
                        raise ValueError(f"{mask_prior_path} does not contain a module_mask array")
                    module_mask_prior = prior_payload["module_mask"].astype(np.float32)
                    prior_genes = prior_payload["genes"].astype(str).tolist() if "genes" in prior_payload.files else None
                finally:
                    prior_payload.close()
            else:
                module_mask_prior = np.asarray(prior_payload, dtype=np.float32)
                prior_genes = None
            expected = (int(model_cfg.get("num_axes", 32)), int(input_dim))
            if tuple(module_mask_prior.shape) != expected:
                raise ValueError(
                    f"Fixed module mask {mask_prior_path} has shape {module_mask_prior.shape}; expected {expected}"
                )
            if prior_genes is not None:
                if gene_vocab is None:
                    raise ValueError(
                        "gene_vocab is required to check the module-mask prior gene order"
                    )
                if prior_genes != [str(gene) for gene in gene_vocab]:
                    raise ValueError(
                        "The module-mask prior gene order does not exactly match the selected data/checkpoint gene vocabulary"
                    )
            model_cfg["module_mask_prior"] = module_mask_prior

        return LEXA(input_dim=input_dim, **model_cfg)
    if architecture == "product_vqvae":
        from benchmark.comparators.product_vq_models import ProductVQ

        model_cfg.pop("module_mask_prior_path", None)
        return ProductVQ(input_dim=input_dim, **model_cfg)
    if architecture in {"continuous_ae", "modular_continuous_ae"}:
        from benchmark.comparators.continuous_autoencoders import ContinuousAE, ModularContinuousAE

        cls = ContinuousAE if architecture == "continuous_ae" else ModularContinuousAE
        mask_prior_path = model_cfg.pop("module_mask_prior_path", None)
        mask_mode = str(model_cfg.get("module_mask_parameterization", "uniform")).lower()
        if architecture == "modular_continuous_ae" and mask_mode in {
            "fixed_prior", "fixed", "prior_guided_softmax", "learned_prior_softmax"
        }:
            if not mask_prior_path:
                raise ValueError("model.module_mask_prior_path is required for the requested continuous module mask")
            payload = np.load(mask_prior_path, allow_pickle=False)
            try:
                model_cfg["module_mask_prior"] = payload["module_mask"].astype(np.float32)
            finally:
                if hasattr(payload, "close"):
                    payload.close()
        return cls(input_dim=input_dim, **model_cfg)
    if architecture == "single_codebook_vqvae":
        from benchmark.comparators.product_vq_models import SingleCodebookVQVAE

        return SingleCodebookVQVAE(input_dim=input_dim, **model_cfg)
    if architecture == "additive_codebook_vqvae":
        from benchmark.comparators.product_vq_models import AdditiveCodebookVQVAE

        return AdditiveCodebookVQVAE(input_dim=input_dim, **model_cfg)
    raise ValueError(f"Unknown model architecture: {architecture}")


def matrix_cache_dir(config: Dict[str, Any]) -> Path:
    return ensure_dir(Path(config["output"]["root"]) / "_matrix_cache")


def cached_csr_paths(config: Dict[str, Any], file_path: str, matrix_key: str) -> Dict[str, Path]:
    base = matrix_cache_dir(config) / cache_key_for_matrix(file_path, matrix_key)
    return {
        "data": base.with_name(base.name + "_csr_data.npy"),
        "indices": base.with_name(base.name + "_csr_indices.npy"),
        "indptr": base.with_name(base.name + "_csr_indptr.npy"),
        "meta": base.with_name(base.name + "_csr_meta.json"),
        "lock": base.with_name(base.name + ".lock"),
    }


def cached_csr_ready(paths: Dict[str, Path]) -> bool:
    return all(paths[key].exists() for key in ["data", "indices", "indptr", "meta"])


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    tmp_path = path.with_name(path.name + f".tmp.{os.getpid()}.npy")
    tmp_path.unlink(missing_ok=True)
    np.save(tmp_path, array)
    os.replace(tmp_path, path)


def build_dense_csr_cache_atomic(matrix: h5py.Dataset, shape: Tuple[int, int], file_path: str, matrix_key: str, config: Dict[str, Any], paths: Dict[str, Path], started: float) -> Dict[str, Path]:
    from scipy import sparse

    chunk_rows = int(config.get("data", {}).get("dense_csr_cache_chunk_rows", 4096))
    pid = os.getpid()
    part_paths: List[Tuple[Path, Path, int]] = []
    tmp_data_path = paths["data"].with_name(paths["data"].name + f".tmp.{pid}.npy")
    tmp_indices_path = paths["indices"].with_name(paths["indices"].name + f".tmp.{pid}.npy")
    tmp_indptr_path = paths["indptr"].with_name(paths["indptr"].name + f".tmp.{pid}.npy")
    tmp_meta_path = paths["meta"].with_name(paths["meta"].name + f".tmp.{pid}")
    tmp_paths = [tmp_data_path, tmp_indices_path, tmp_indptr_path, tmp_meta_path]
    try:
        for tmp_path in tmp_paths:
            tmp_path.unlink(missing_ok=True)
        indptr_mm = np.lib.format.open_memmap(tmp_indptr_path, mode="w+", dtype=np.int64, shape=(shape[0] + 1,))
        indptr_mm[0] = 0
        row_cursor = 1
        nnz_total = 0
        for part_idx, start_row in enumerate(range(0, shape[0], chunk_rows)):
            end_row = min(shape[0], start_row + chunk_rows)
            chunk_csr = sparse.csr_matrix(np.asarray(matrix[start_row:end_row, :]))
            n_rows = end_row - start_row
            chunk_nnz = int(chunk_csr.nnz)
            indptr_mm[row_cursor : row_cursor + n_rows] = np.asarray(chunk_csr.indptr[1:] + nnz_total, dtype=np.int64)
            row_cursor += n_rows
            if chunk_nnz:
                data_part = paths["data"].with_name(paths["data"].name + f".part.{pid}.{part_idx}.npy")
                indices_part = paths["indices"].with_name(paths["indices"].name + f".part.{pid}.{part_idx}.npy")
                np.save(data_part, np.asarray(chunk_csr.data, dtype=np.float32))
                np.save(indices_part, np.asarray(chunk_csr.indices, dtype=np.int32))
                part_paths.append((data_part, indices_part, chunk_nnz))
            nnz_total += chunk_nnz
            progress(f"building CSR cache for dense matrix: rows {end_row}/{shape[0]}, nnz={nnz_total} from {file_path}")
        indptr_mm.flush()
        del indptr_mm
        data_mm = np.lib.format.open_memmap(tmp_data_path, mode="w+", dtype=np.float32, shape=(nnz_total,))
        indices_mm = np.lib.format.open_memmap(tmp_indices_path, mode="w+", dtype=np.int32, shape=(nnz_total,))
        cursor = 0
        for data_part, indices_part, part_nnz in part_paths:
            next_cursor = cursor + part_nnz
            data_mm[cursor:next_cursor] = np.load(data_part, mmap_mode="r")
            indices_mm[cursor:next_cursor] = np.load(indices_part, mmap_mode="r")
            cursor = next_cursor
            data_part.unlink(missing_ok=True)
            indices_part.unlink(missing_ok=True)
        data_mm.flush()
        indices_mm.flush()
        del data_mm, indices_mm
        write_json(tmp_meta_path, {"file_path": file_path, "matrix_key": matrix_key, "source_format": "dense", "target_format": "csr_matrix", "shape": list(shape), "nnz": int(nnz_total), "elapsed_s": round(time.time() - started, 3)})
        os.replace(tmp_data_path, paths["data"])
        os.replace(tmp_indices_path, paths["indices"])
        os.replace(tmp_indptr_path, paths["indptr"])
        os.replace(tmp_meta_path, paths["meta"])
        progress(f"built CSR cache for {file_path} in {time.time() - started:.1f}s")
        return paths
    finally:
        for data_part, indices_part, _ in part_paths:
            data_part.unlink(missing_ok=True)
            indices_part.unlink(missing_ok=True)
        for tmp_path in tmp_paths:
            tmp_path.unlink(missing_ok=True)


def ensure_csr_cache_for_matrix(file_path: str, matrix_key: str, config: Dict[str, Any]) -> Dict[str, Path]:
    from scipy import sparse

    paths = cached_csr_paths(config, file_path, matrix_key)
    if cached_csr_ready(paths):
        return paths
    lock_fd: Optional[int] = None
    while lock_fd is None:
        try:
            lock_fd = os.open(paths["lock"], os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if cached_csr_ready(paths):
                return paths
            progress(f"waiting for CSR cache build: {file_path}")
            time.sleep(5)
    try:
        if cached_csr_ready(paths):
            return paths
        progress(f"building CSR cache for {file_path} [{matrix_key}]")
        started = time.time()
        with h5py.File(file_path, "r") as handle:
            matrix = handle[matrix_key]
            shape = matrix_shape(matrix)
            fmt = matrix_encoding(matrix)
            if fmt == "csc_matrix":
                source = sparse.csc_matrix((np.asarray(matrix["data"]), np.asarray(matrix["indices"]), np.asarray(matrix["indptr"])), shape=shape)
                csr = source.tocsr()
            elif fmt == "dense":
                return build_dense_csr_cache_atomic(matrix, shape, file_path, matrix_key, config, paths, started)
            else:
                raise ValueError(f"CSR cache requested for unsupported matrix format: {file_path} fmt={fmt}")
        save_npy_atomic(paths["data"], np.asarray(csr.data))
        save_npy_atomic(paths["indices"], np.asarray(csr.indices))
        save_npy_atomic(paths["indptr"], np.asarray(csr.indptr))
        write_json(paths["meta"], {"file_path": file_path, "matrix_key": matrix_key, "source_format": fmt, "target_format": "csr_matrix", "shape": list(csr.shape), "nnz": int(csr.nnz), "elapsed_s": round(time.time() - started, 3)})
        progress(f"built CSR cache for {file_path} in {time.time() - started:.1f}s")
        return paths
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        paths["lock"].unlink(missing_ok=True)


def load_cached_csr_matrix(paths: Dict[str, Path]) -> Dict[str, np.ndarray]:
    return {
        "data": np.load(paths["data"], mmap_mode="r"),
        "indices": np.load(paths["indices"], mmap_mode="r"),
        "indptr": np.load(paths["indptr"], mmap_mode="r"),
    }


def csr_row_to_dense(data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, row_idx: int, out_dim: int, gene_map: np.ndarray) -> np.ndarray:
    start = int(indptr[row_idx])
    end = int(indptr[row_idx + 1])
    row = np.zeros(out_dim, dtype=np.float32)
    src_idx = indices[start:end]
    values = data[start:end].astype(np.float32, copy=False)
    dest = gene_map[src_idx]
    keep = dest >= 0
    np.add.at(row, dest[keep], values[keep])
    return row


def csr_rows_to_dense(matrix: Any, row_indices: np.ndarray, out_dim: int, gene_map: np.ndarray) -> np.ndarray:
    rows = np.asarray(row_indices, dtype=np.int64)
    out = np.zeros((len(rows), out_dim), dtype=np.float32)
    if len(rows) == 0:
        return out
    data = matrix["data"]
    indices = matrix["indices"]
    indptr = matrix["indptr"]
    min_row = int(rows.min())
    max_row = int(rows.max())
    if max_row - min_row <= max(262_144, len(rows) * 64):
        indptr_block = np.asarray(indptr[min_row : max_row + 2], dtype=np.int64)
        rel_rows = rows - min_row
        starts = indptr_block[rel_rows]
        ends = indptr_block[rel_rows + 1]
        if len(starts):
            data_start = int(starts.min())
            data_end = int(ends.max())
            if data_end - data_start <= max(5_000_000, len(rows) * 5000):
                data_block = np.asarray(data[data_start:data_end], dtype=np.float32)
                indices_block = np.asarray(indices[data_start:data_end])
                lengths = (ends - starts).astype(np.int64, copy=False)
                source_positions = np.concatenate(
                    [np.arange(int(start - data_start), int(end - data_start), dtype=np.int64) for start, end in zip(starts, ends)]
                ) if int(lengths.sum()) else np.empty(0, dtype=np.int64)
                if len(source_positions):
                    output_rows = np.repeat(np.arange(len(rows), dtype=np.int64), lengths)
                    dest = gene_map[indices_block[source_positions]]
                    keep = dest >= 0
                    np.add.at(
                        out,
                        (output_rows[keep], dest[keep]),
                        data_block[source_positions[keep]],
                    )
                return out
    for out_idx, row_idx in enumerate(rows):
        out[out_idx] = csr_row_to_dense(data, indices, indptr, int(row_idx), out_dim, gene_map)
    return out


def matrix_row_to_dense(matrix: Any, matrix_format: str, row_idx: int, out_dim: int, gene_map: np.ndarray) -> np.ndarray:
    if matrix_format == "dense_shard":
        values = np.asarray(matrix[row_idx, :], dtype=np.float32)
        return values.reshape(-1)
    if matrix_format == "dense":
        values = np.asarray(matrix[row_idx, :], dtype=np.float32).reshape(-1)
        row = np.zeros(out_dim, dtype=np.float32)
        keep = gene_map >= 0
        np.add.at(row, gene_map[keep], values[keep])
        return row
    if matrix_format == "csr_matrix":
        return csr_row_to_dense(matrix["data"], matrix["indices"], matrix["indptr"], row_idx, out_dim, gene_map)
    if matrix_format == "csc_matrix":
        src_row = np.zeros(len(gene_map), dtype=np.float32)
        data = matrix["data"]
        indices = matrix["indices"]
        indptr = matrix["indptr"]
        for col_idx in range(len(indptr) - 1):
            start = int(indptr[col_idx])
            end = int(indptr[col_idx + 1])
            hit = np.where(indices[start:end] == row_idx)[0]
            if hit.size:
                src_row[col_idx] = float(data[start + int(hit[0])])
        row = np.zeros(out_dim, dtype=np.float32)
        keep = gene_map >= 0
        np.add.at(row, gene_map[keep], src_row[keep])
        return row
    raise ValueError(f"Unsupported matrix_format for row reads: {matrix_format}")


def matrix_rows_to_dense(matrix: Any, matrix_format: str, row_indices: Sequence[int], out_dim: int, gene_map: np.ndarray) -> np.ndarray:
    rows = np.asarray(row_indices, dtype=np.int64)
    if matrix_format == "dense_shard":
        values = np.asarray(matrix[rows, :], dtype=np.float32)
        return values.reshape(1, -1) if values.ndim == 1 else values
    if matrix_format == "csr_matrix":
        return csr_rows_to_dense(matrix, rows, out_dim, gene_map)
    if matrix_format == "dense":
        out = np.zeros((len(rows), out_dim), dtype=np.float32)
        if len(rows) == 0:
            return out
        unique_rows, inverse = np.unique(rows, return_inverse=True)
        values = np.asarray(matrix[unique_rows, :], dtype=np.float32)[inverse]
        keep = gene_map >= 0
        kept_dest = gene_map[keep]
        for out_idx in range(len(rows)):
            np.add.at(out[out_idx], kept_dest, values[out_idx, keep])
        return out
    return np.stack([matrix_row_to_dense(matrix, matrix_format, int(row_idx), out_dim, gene_map) for row_idx in rows], axis=0)


def optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def normalize_expression(row: np.ndarray, normalize_total: Optional[float], log1p: bool) -> np.ndarray:
    if np.any(row < 0):
        return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    total = float(row.sum())
    if normalize_total is not None and total > 0:
        row = row * (normalize_total / total)
    if log1p:
        row = np.log1p(row)
    return np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def normalize_expression_rows(rows: np.ndarray, normalize_total: Optional[float], log1p: bool) -> np.ndarray:
    has_negative = np.any(rows < 0, axis=1)
    if not np.any(has_negative):
        if normalize_total is not None:
            totals = rows.sum(axis=1, keepdims=True)
            factors = np.divide(
                float(normalize_total), totals,
                out=np.ones_like(totals, dtype=np.float32),
                where=totals > 0,
            )
            rows *= factors
        if log1p:
            np.log1p(rows, out=rows)
        return np.nan_to_num(rows, copy=False, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    totals = rows.sum(axis=1, keepdims=True)
    nonzero = (totals[:, 0] > 0) & (~has_negative) & (normalize_total is not None)
    if normalize_total is not None and np.any(nonzero):
        rows[nonzero] *= normalize_total / totals[nonzero]
    if log1p:
        log_rows = ~has_negative
        if np.any(log_rows):
            rows[log_rows] = np.log1p(rows[log_rows])
    return np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def load_file_gene_maps(path: Path) -> Dict[str, np.ndarray]:
    try:
        archive = np.load(path, allow_pickle=False)
    except ValueError as exc:
        if "Object arrays cannot be loaded" not in str(exc):
            raise
        archive = np.load(path, allow_pickle=True)
    if "__index__" in archive:
        index = json.loads(str(archive["__index__"]))
        return {file_path: archive[key].astype(np.int32, copy=False) for file_path, key in index.items()}
    if "file_paths" in archive:
        file_paths = [str(path_value) for path_value in archive["file_paths"].tolist()]
        return {file_path: archive[f"arr_{idx}"].astype(np.int32, copy=False) for idx, file_path in enumerate(file_paths)}
    raise ValueError(f"Unsupported file_gene_maps archive format: {path}")


def hydrate_file_meta(item: Dict[str, Any]) -> FileMeta:
    payload = dict(item)
    payload.setdefault("feature_names", [])
    return FileMeta(**payload)


def load_data_bundle(path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    payload = read_json(path)
    metas = [hydrate_file_meta(item) for item in payload["file_metas"]]
    maps_path = payload.get("file_gene_maps_path")
    if maps_path is None:
        raise ValueError(f"{path} is missing file_gene_maps_path; rerun clean preprocessing.")
    maps_path = str((path.parent / maps_path).resolve()) if not Path(maps_path).is_absolute() else str(maps_path)
    file_gene_maps = load_file_gene_maps(Path(maps_path))
    split_paths = payload.get("split_paths")
    if split_paths is not None:
        split_paths = {
            split: str((path.parent / rel_path).resolve()) if not Path(rel_path).is_absolute() else str(rel_path)
            for split, rel_path in split_paths.items()
        }
    return {
        "file_metas": metas,
        "gene_vocab": payload["gene_vocab"],
        "file_gene_maps": file_gene_maps,
        "file_gene_maps_path": maps_path,
        "records": payload.get("records"),
        "splits": {},
        "split_paths": split_paths,
        "summary": payload["summary"],
    }
