from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Set

from data_pipeline.clean_common import FileMeta, ensure_csr_cache_for_matrix, load_data_bundle as _load_data_bundle, progress

from data_pipeline.data import CSVRecordStore


def files_used_by_splits(splits: Dict[str, Any], split_names: Optional[Iterable[str]] = None) -> Set[str]:
    wanted = list(split_names or splits.keys())
    files: Set[str] = set()
    for split in wanted:
        store = splits.get(split)
        if store is None:
            continue
        if hasattr(store, "file_paths"):
            files.update(store.file_paths())
            continue
        for record in store:
            files.add(record["file_path"])
    return files


def warm_csr_caches(bundle: Dict[str, Any], config: Dict[str, Any], split_names: Optional[Iterable[str]] = None) -> None:
    """Build matrix caches before DataLoader workers start."""

    data_cfg = config.get("data", {})
    precompute_csc = bool(data_cfg.get("precompute_csr_cache", True))
    precompute_dense = bool(data_cfg.get("precompute_dense_csr_cache", False))
    if not precompute_csc and not precompute_dense:
        progress("warm_csr_caches: disabled by data.precompute_csr_cache=false")
        return

    metas = list(bundle.get("file_metas", []))
    cache_metas = [
        meta
        for meta in metas
        if (getattr(meta, "matrix_format", None) == "csc_matrix" and precompute_csc)
        or (getattr(meta, "matrix_format", None) == "dense" and precompute_dense)
    ]
    if not cache_metas:
        progress(f"warm_csr_caches: no cacheable CSC/dense matrices among {len(metas)} files")
        return

    counts = {}
    for meta in cache_metas:
        counts[meta.matrix_format] = counts.get(meta.matrix_format, 0) + 1
    progress(f"warm_csr_caches: preparing {len(cache_metas)} matrices before training, formats={counts}")
    for idx, meta in enumerate(cache_metas, start=1):
        ensure_csr_cache_for_matrix(meta.file_path, meta.matrix_key, config)
        if idx == 1 or idx % 10 == 0 or idx == len(cache_metas):
            progress(f"warm_csr_caches: {idx}/{len(cache_metas)} ready")


def _with_structured_splits(bundle: Dict[str, Any]) -> Dict[str, Any]:
    split_paths = bundle.get("split_paths")
    if split_paths:
        bundle = dict(bundle)
        bundle["splits"] = {split: CSVRecordStore(path) for split, path in split_paths.items()}
    return bundle


def load_data_bundle(path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    return _with_structured_splits(_load_data_bundle(path, config))


__all__ = [
    "FileMeta",
    "load_data_bundle",
    "warm_csr_caches",
]
