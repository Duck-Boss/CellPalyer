import csv
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from data_pipeline.clean_common import (
    ensure_csr_cache_for_matrix,
    load_cached_csr_matrix,
    matrix_encoding,
    matrix_row_to_dense,
    matrix_rows_to_dense,
    normalize_expression,
    normalize_expression_rows,
    optional_float,
    progress,
    cell_type_family,
)


RECORD_FIELDNAMES = [
    "file_path",
    "row_idx",
    "group_id",
    "dataset_id",
    "cell_type",
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


def gene_map_fingerprint(gene_map: np.ndarray, vocab_size: int) -> str:
    arr = np.asarray(gene_map, dtype=np.int32)
    digest = hashlib.sha1()
    digest.update(str(int(vocab_size)).encode("utf-8"))
    digest.update(str(tuple(arr.shape)).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()[:16]


class CSVRecordStore:
    """Random-access CSV manifest with cached offsets and file ranges."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self.fieldnames: List[str] = []
        self.offsets: Optional[np.ndarray] = None
        self._file_ranges: Optional[List[Tuple[str, int, int]]] = None
        self._dataset_indices: Optional[Dict[str, np.ndarray]] = None
        self._handle = None
        self._pid = None
        self._read_header()

    def _read_header(self) -> None:
        with open(self.path, "r", encoding="utf-8", newline="") as handle:
            header = handle.readline()
            self.fieldnames = next(csv.reader([header]))

    def _ensure_offsets(self) -> np.ndarray:
        if self.offsets is not None:
            return self.offsets
        offset_path = Path(self.path + ".offsets.npy")
        with open(self.path, "r", encoding="utf-8", newline="") as handle:
            header = handle.readline()
            self.fieldnames = next(csv.reader([header]))
            if offset_path.exists():
                self.offsets = np.load(offset_path)
                progress(f"CSVRecordStore: loaded {len(self.offsets)} cached offsets from {offset_path}")
                return self.offsets
            progress(f"CSVRecordStore: building offsets for {self.path}")
            offsets = []
            started = time.time()
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                offsets.append(offset)
                if len(offsets) % 1_000_000 == 0:
                    progress(f"CSVRecordStore: indexed {len(offsets)} rows from {self.path}, elapsed={time.time() - started:.1f}s")
            self.offsets = np.array(offsets, dtype=np.int64)
            np.save(offset_path, self.offsets)
            progress(f"CSVRecordStore: saved {len(self.offsets)} offsets to {offset_path}")
            return self.offsets

    def __len__(self) -> int:
        return len(self._ensure_offsets())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        offsets = self._ensure_offsets()
        if idx < 0:
            idx += len(offsets)
        if idx < 0 or idx >= len(offsets):
            raise IndexError(idx)
        handle = self._ensure_handle()
        handle.seek(int(offsets[idx]))
        return next(csv.DictReader([handle.readline()], fieldnames=self.fieldnames))

    def __getitems__(self, indices: Sequence[int]) -> List[Dict[str, Any]]:
        offsets = self._ensure_offsets()
        n = len(offsets)
        normalized = []
        for out_idx, idx in enumerate(indices):
            idx = int(idx)
            if idx < 0:
                idx += n
            if idx < 0 or idx >= n:
                raise IndexError(idx)
            normalized.append((out_idx, idx))

        rows: List[Optional[Dict[str, Any]]] = [None] * len(normalized)
        if not normalized:
            return []

        handle = self._ensure_handle()
        ordered = sorted(normalized, key=lambda item: item[1])
        cursor = 0
        while cursor < len(ordered):
            run = [ordered[cursor]]
            cursor += 1
            while cursor < len(ordered) and ordered[cursor][1] == run[-1][1] + 1:
                run.append(ordered[cursor])
                cursor += 1

            handle.seek(int(offsets[run[0][1]]))
            lines = [handle.readline() for _ in run]
            parsed = csv.DictReader(lines, fieldnames=self.fieldnames)
            for (out_idx, _), row in zip(run, parsed):
                rows[out_idx] = row

        if any(row is None for row in rows):
            raise RuntimeError(f"CSVRecordStore failed to read a full batch from {self.path}")
        return [row for row in rows if row is not None]

    def _ensure_handle(self):
        pid = os.getpid()
        if self._handle is None or self._pid != pid:
            self._handle = open(self.path, "r", encoding="utf-8", newline="")
            self._pid = pid
        return self._handle

    def file_ranges(self) -> List[Tuple[str, int, int]]:
        if self._file_ranges is not None:
            return self._file_ranges
        ranges: List[Tuple[str, int, int]] = []
        current_path: Optional[str] = None
        start = 0
        row_idx = 0
        with open(self.path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                file_path = row["file_path"]
                if current_path is None:
                    current_path = file_path
                    start = row_idx
                elif file_path != current_path:
                    ranges.append((current_path, start, row_idx))
                    current_path = file_path
                    start = row_idx
                row_idx += 1
        if current_path is not None:
            ranges.append((current_path, start, row_idx))
        self._file_ranges = ranges
        progress(f"CSVRecordStore: indexed {len(ranges)} contiguous file ranges from {self.path}")
        return ranges

    def file_paths(self) -> List[str]:
        return [file_path for file_path, _, _ in self.file_ranges()]

    def label_indices(self, label_key: str) -> Dict[str, np.ndarray]:
        if label_key == "dataset_id" and self._dataset_indices is not None:
            return self._dataset_indices
        source_key = "cell_type" if label_key == "cell_type_family" else label_key
        if source_key not in self.fieldnames:
            raise KeyError(f"CSV manifest has no {label_key} column: {self.path}")

        index_path = Path(self.path + f".{label_key}_indices.npz")
        meta_path = Path(self.path + f".{label_key}_indices.json")
        offset_mtime = Path(self.path).stat().st_mtime
        if index_path.exists() and meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as handle:
                    meta = json.load(handle)
                if meta.get("source_path") == self.path and meta.get("label_key") == label_key and float(meta.get("source_mtime", -1.0)) == float(offset_mtime):
                    payload = np.load(index_path, allow_pickle=False)
                    labels = [str(item) for item in payload["labels"]]
                    groups = {label: payload[f"indices_{idx}"] for idx, label in enumerate(labels)}
                    if label_key == "dataset_id":
                        self._dataset_indices = groups
                    progress(f"CSVRecordStore: loaded {label_key} index for {len(groups)} labels from {index_path}")
                    return groups
            except Exception as exc:
                progress(f"CSVRecordStore: ignoring stale {label_key} index {index_path}: {exc}")

        progress(f"CSVRecordStore: building {label_key} index for {self.path}")
        groups: Dict[str, List[int]] = defaultdict(list)
        started = time.time()
        with open(self.path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_idx, row in enumerate(reader):
                raw_label = row.get(source_key) or f"__missing_{source_key}__"
                label = cell_type_family(raw_label) if label_key == "cell_type_family" else raw_label
                groups[label].append(row_idx)
                if (row_idx + 1) % 1_000_000 == 0:
                    progress(f"CSVRecordStore: {label_key}-indexed {row_idx + 1} rows from {self.path}, elapsed={time.time() - started:.1f}s")

        labels = sorted(groups)
        arrays = {label: np.asarray(groups[label], dtype=np.int64) for label in labels}
        save_payload = {"labels": np.asarray(labels, dtype="U")}
        for idx, label in enumerate(labels):
            save_payload[f"indices_{idx}"] = arrays[label]
        np.savez(index_path, **save_payload)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump({"source_path": self.path, "source_mtime": offset_mtime, "label_key": label_key, "num_labels": len(labels)}, handle)
        if label_key == "dataset_id":
            self._dataset_indices = arrays
        progress(f"CSVRecordStore: saved {label_key} index for {len(arrays)} labels to {index_path}")
        return arrays

    def dataset_indices(self) -> Dict[str, np.ndarray]:
        return self.label_indices("dataset_id")

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handle"] = None
        state["_pid"] = None
        return state

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class H5CSRDataset(Dataset):
    def __init__(
        self,
        records: Any,
        file_gene_maps: Dict[str, np.ndarray],
        gene_vocab: Sequence[str],
        config: Dict[str, Any],
        file_metas: Optional[Sequence[Any]] = None,
    ) -> None:
        self.records = records
        self.file_gene_maps = file_gene_maps
        self.gene_vocab = list(gene_vocab)
        self.config = config
        self.file_meta_by_path = {str(meta.file_path): meta for meta in (file_metas or [])}
        self.return_raw_counts = bool(config.get("data", {}).get("return_raw_counts", False))
        observation_masks: Dict[str, torch.Tensor] = {}
        any_partial_coverage = False
        for file_path, gene_map in self.file_gene_maps.items():
            observed = np.zeros(len(self.gene_vocab), dtype=bool)
            mapped = np.asarray(gene_map, dtype=np.int64)
            mapped = mapped[mapped >= 0]
            if mapped.size:
                observed[np.unique(mapped)] = True
            any_partial_coverage = any_partial_coverage or not bool(observed.all())
            observation_masks[str(file_path)] = torch.from_numpy(observed)
        self.observation_masks = observation_masks if any_partial_coverage else {}
        self.dense_shards = load_dense_shard_manifest(config)
        self.handles: Dict[str, h5py.File] = {}
        self.matrix_cache: Dict[str, Tuple[Any, str]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _ensure_handle(self, file_path: str) -> h5py.File:
        if file_path not in self.handles:
            self.handles[file_path] = h5py.File(file_path, "r")
        return self.handles[file_path]

    def _ensure_dense_shard(self, file_path: str) -> Optional[Tuple[Any, str]]:
        item = self.dense_shards.get(file_path)
        if item is None:
            return None
        gene_map = self.file_gene_maps.get(file_path)
        if gene_map is None:
            progress(f"H5CSRDataset: ignoring dense shard for {file_path}; no gene map in current bundle")
            return None
        expected_hash = gene_map_fingerprint(gene_map, len(self.gene_vocab))
        if item.get("gene_map_hash") != expected_hash:
            progress(f"H5CSRDataset: ignoring stale dense shard for {file_path}; gene_map_hash mismatch")
            return None
        if file_path not in self.matrix_cache:
            path = item["path"]
            progress(f"H5CSRDataset: loading dense shard for {file_path}")
            array = np.load(path, mmap_mode="r")
            if array.ndim != 2 or int(array.shape[1]) != len(self.gene_vocab):
                progress(f"H5CSRDataset: ignoring dense shard for {file_path}; shape {array.shape} does not match vocab {len(self.gene_vocab)}")
                return None
            self.matrix_cache[file_path] = (array, "dense_shard")
        return self.matrix_cache[file_path]

    def _ensure_matrix(self, file_path: str) -> Tuple[Any, str]:
        if file_path not in self.matrix_cache:
            meta = self.file_meta_by_path.get(file_path)
            if getattr(meta, "matrix_format", None) == "dense":
                shard = self._ensure_dense_shard(file_path)
                if shard is not None:
                    return shard
            handle = self._ensure_handle(file_path)
            matrix_key = getattr(meta, "matrix_key", None) if meta is not None else None
            if matrix_key is None:
                matrix_key = str(self.config["data"].get("matrix_key", "X")).replace("raw.X", "raw/X")
            matrix = handle[matrix_key]
            matrix_format = getattr(meta, "matrix_format", None) if meta is not None else None
            if matrix_format is None:
                matrix_format = matrix_encoding(matrix)
            cache_dense = bool(self.config.get("data", {}).get("precompute_dense_csr_cache", False))
            if matrix_format == "csc_matrix" or (matrix_format == "dense" and cache_dense):
                progress(f"H5CSRDataset: loading CSR cache for {file_path}")
                paths = ensure_csr_cache_for_matrix(file_path=file_path, matrix_key=matrix_key, config=self.config)
                self.matrix_cache[file_path] = (load_cached_csr_matrix(paths), "csr_matrix")
            else:
                self.matrix_cache[file_path] = (matrix, matrix_format)
        return self.matrix_cache[file_path]

    def _expression_transform_for_file(self, file_path: str) -> Tuple[Optional[float], bool]:
        meta = self.file_meta_by_path.get(file_path)
        info = getattr(meta, "feature_info", None) or {}
        transform = info.get("expression_transform")
        if transform == "normalize_total_10000_then_log1p":
            return float(info.get("target_sum_for_raw", 10000.0)), True
        if transform == "log1p_only":
            return None, True
        if transform == "use_as_is":
            return None, False
        return optional_float(self.config["data"].get("normalize_total")), bool(self.config["data"]["log1p"])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        matrix, matrix_format = self._ensure_matrix(record["file_path"])
        gene_map = self.file_gene_maps[record["file_path"]]
        row = matrix_row_to_dense(
            matrix=matrix,
            matrix_format=matrix_format,
            row_idx=int(record["row_idx"]),
            out_dim=len(self.gene_vocab),
            gene_map=gene_map,
        )
        raw_counts = row.copy() if self.return_raw_counts else None
        normalize_total, log1p = self._expression_transform_for_file(record["file_path"])
        if self.return_raw_counts and (normalize_total is None or not log1p):
            raise ValueError(
                "data.return_raw_counts=true requires a raw-count source with the "
                "normalize_total_then_log1p transform; refusing to treat an already-normalized "
                f"matrix as counts for {record['file_path']}"
            )
        row = normalize_expression(
            row,
            normalize_total=normalize_total,
            log1p=log1p,
        )
        return _record_to_item(
            record,
            torch.from_numpy(row),
            None if raw_counts is None else torch.from_numpy(raw_counts.astype(np.float32, copy=False)),
            self.observation_masks.get(record["file_path"]),
        )

    def __getitems__(self, indices: Sequence[int]) -> List[Dict[str, Any]]:
        if hasattr(self.records, "__getitems__"):
            records = self.records.__getitems__([int(idx) for idx in indices])
        else:
            records = [self.records[int(idx)] for idx in indices]
        out_dim = len(self.gene_vocab)
        rows = np.zeros((len(records), out_dim), dtype=np.float32)
        raw_rows = np.zeros((len(records), out_dim), dtype=np.float32) if self.return_raw_counts else None
        by_file: Dict[str, List[int]] = defaultdict(list)
        for batch_idx, record in enumerate(records):
            by_file[record["file_path"]].append(batch_idx)

        for file_path, batch_indices in by_file.items():
            matrix, matrix_format = self._ensure_matrix(file_path)
            gene_map = self.file_gene_maps[file_path]
            file_rows = [int(records[batch_idx]["row_idx"]) for batch_idx in batch_indices]
            file_values = matrix_rows_to_dense(
                matrix=matrix,
                matrix_format=matrix_format,
                row_indices=file_rows,
                out_dim=out_dim,
                gene_map=gene_map,
            )
            normalize_total, log1p = self._expression_transform_for_file(file_path)
            if self.return_raw_counts:
                if normalize_total is None or not log1p:
                    raise ValueError(
                        "data.return_raw_counts=true requires a raw-count source with the "
                        "normalize_total_then_log1p transform; refusing to treat an already-normalized "
                        f"matrix as counts for {file_path}"
                    )
                raw_rows[np.asarray(batch_indices, dtype=np.int64)] = file_values
                file_values = file_values.copy()
            file_values = normalize_expression_rows(file_values, normalize_total=normalize_total, log1p=log1p)
            rows[np.asarray(batch_indices, dtype=np.int64)] = file_values
        return [
            _record_to_item(
                record,
                torch.from_numpy(rows[batch_idx]),
                None if raw_rows is None else torch.from_numpy(raw_rows[batch_idx]),
                self.observation_masks.get(record["file_path"]),
            )
            for batch_idx, record in enumerate(records)
        ]

    def close(self) -> None:
        for handle in self.handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self.handles.clear()
        self.matrix_cache.clear()
        if hasattr(self.records, "close"):
            self.records.close()


def _record_to_item(
    record: Dict[str, Any],
    x: torch.Tensor,
    x_counts: Optional[torch.Tensor] = None,
    x_mask: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    item = {key: value for key, value in record.items()}
    item["x"] = x
    if x_counts is not None:
        item["x_counts"] = x_counts
    if x_mask is not None:
        item["x_mask"] = x_mask
    for key in ["cell_type", "tissue", "disease", "assay", "group_id", "dataset_id"]:
        item.setdefault(key, "unknown")
    for key in ["donor_id", "sample_id", "batch_id"]:
        item.setdefault(key, "unknown")
    return item


def load_dense_shard_manifest(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    data_cfg = config.get("data", {})
    if not bool(data_cfg.get("use_dense_shards", True)):
        return {}
    data_run_id = data_cfg.get("_data_run_id")
    if data_run_id is None:
        return {}
    path = Path(config["output"]["root"]) / str(data_run_id) / "dense_shards" / "manifest.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    out = {}
    for item in payload.get("files", []):
        if item.get("path") and Path(item["path"]).exists():
            out[str(item["file_path"])] = item
    progress(f"H5CSRDataset: loaded {len(out)} dense shard entries from {path}")
    return out


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    xs = torch.stack([item["x"] for item in batch], dim=0)
    tensor_keys = {key for key, value in batch[0].items() if torch.is_tensor(value)}
    meta = {
        key: torch.stack([item[key] for item in batch], dim=0)
        if key in tensor_keys
        else [item[key] for item in batch]
        for key in batch[0]
        if key != "x"
    }
    meta["x"] = xs
    return meta


class FileBatchSampler(Sampler[List[int]]):
    """Yield file-local batches, optionally shuffling the batch order."""

    def __init__(self, records: Any, batch_size: int, shuffle: bool, seed: int) -> None:
        self.records = records
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        self._batches: Optional[List[List[int]]] = None

    def _build_batches(self) -> List[List[int]]:
        if self._batches is not None:
            return self._batches
        ranges = _record_file_ranges(self.records)
        batches: List[List[int]] = []
        for _, start, end in ranges:
            for batch_start in range(start, end, self.batch_size):
                batches.append(list(range(batch_start, min(end, batch_start + self.batch_size))))
        self._batches = batches
        progress(f"FileBatchSampler: built {len(batches)} file-local batches from {len(ranges)} file ranges")
        return batches

    def __iter__(self):
        batches = list(self._build_batches())
        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            rng.shuffle(batches)
        self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return len(self._build_batches())


class LabelBalancedBatchSampler(Sampler[List[int]]):
    """Sample manifest labels approximately uniformly with replacement."""

    def __init__(
        self,
        records: Any,
        batch_size: int,
        shuffle: bool,
        seed: int,
        label_key: str = "dataset_id",
        steps_per_epoch: Optional[int] = None,
        epoch_fraction: Optional[float] = None,
        labels_per_batch: Optional[int] = None,
        drop_last: bool = False,
    ) -> None:
        self.records = records
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.label_key = str(label_key)
        if steps_per_epoch is None:
            fraction = 1.0 if epoch_fraction is None else float(epoch_fraction)
            steps_per_epoch = int(np.ceil(len(records) * fraction / max(1, self.batch_size)))
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.labels_per_batch = None if labels_per_batch is None else max(1, int(labels_per_batch))
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self._labels: Optional[List[str]] = None
        self._indices_by_label: Optional[Dict[str, np.ndarray]] = None

    def _load_groups(self) -> Tuple[List[str], Dict[str, np.ndarray]]:
        if self._labels is not None and self._indices_by_label is not None:
            return self._labels, self._indices_by_label
        if hasattr(self.records, "label_indices"):
            groups = self.records.label_indices(self.label_key)
        else:
            temp: Dict[str, List[int]] = defaultdict(list)
            for idx in range(len(self.records)):
                record = self.records[idx]
                temp[record.get(self.label_key) or f"__missing_{self.label_key}__"].append(idx)
            groups = {label: np.asarray(indices, dtype=np.int64) for label, indices in temp.items()}
        labels = sorted(label for label, indices in groups.items() if len(indices) > 0)
        if not labels:
            raise ValueError(f"LabelBalancedBatchSampler found no non-empty {self.label_key} groups")
        self._labels = labels
        self._indices_by_label = {label: groups[label] for label in labels}
        progress(
            "LabelBalancedBatchSampler: "
            f"label={self.label_key}, groups={len(labels)}, batch_size={self.batch_size}, "
            f"labels_per_batch={self.labels_per_batch or len(labels)}, steps_per_epoch={self.steps_per_epoch}"
        )
        return self._labels, self._indices_by_label

    def __iter__(self):
        labels, groups = self._load_groups()
        rng = np.random.default_rng(self.seed + self.epoch)
        label_count = len(labels)
        label_positions = np.arange(label_count)
        labels_per_batch = min(label_count, self.labels_per_batch or label_count)
        epoch_label_order = rng.permutation(label_positions) if self.shuffle else label_positions
        for step_idx in range(self.steps_per_epoch):
            if labels_per_batch < label_count:
                start = (step_idx * labels_per_batch) % label_count
                selected = epoch_label_order[
                    (start + np.arange(labels_per_batch)) % label_count
                ]
            else:
                selected = label_positions
            repeats = int(np.ceil(self.batch_size / max(1, len(selected))))
            positions = np.tile(selected, repeats)[: self.batch_size].copy()
            if self.shuffle and label_count > 1:
                rng.shuffle(positions)

            batch = []
            for pos in positions:
                label = labels[int(pos)]
                indices = groups[label]
                picked = indices[int(rng.integers(0, len(indices)))]
                batch.append(int(picked))
            if batch or not self.drop_last:
                yield batch
        self.epoch += 1

    def __len__(self) -> int:
        return self.steps_per_epoch


DatasetBalancedBatchSampler = LabelBalancedBatchSampler


def _record_file_ranges(records: Any) -> List[Tuple[str, int, int]]:
    if hasattr(records, "file_ranges"):
        return records.file_ranges()
    ranges: List[Tuple[str, int, int]] = []
    current_path: Optional[str] = None
    start = 0
    for idx, record in enumerate(records):
        file_path = record["file_path"]
        if current_path is None:
            current_path = file_path
            start = idx
        elif file_path != current_path:
            ranges.append((current_path, start, idx))
            current_path = file_path
            start = idx
    if current_path is not None:
        ranges.append((current_path, start, len(records)))
    return ranges


def make_dataloader(dataset: H5CSRDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    kwargs: Dict[str, Any] = {}
    config = getattr(dataset, "config", getattr(getattr(dataset, "dataset", None), "config", {}))
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(config.get("training", {}).get("prefetch_factor", 2))

    records = getattr(dataset, "records", None)
    training_cfg = config.get("training", {})
    balanced = (
        bool(training_cfg.get("label_balanced_batches", False))
        or bool(training_cfg.get("dataset_balanced_batches", False))
    ) and bool(shuffle) and records is not None
    if balanced:
        label_key = str(training_cfg.get("balanced_batch_label", "dataset_id"))
        if bool(training_cfg.get("dataset_balanced_batches", False)) and "balanced_batch_label" not in training_cfg:
            label_key = "dataset_id"
        batch_sampler = LabelBalancedBatchSampler(
            records,
            batch_size,
            shuffle=True,
            seed=int(config["seed"]),
            label_key=label_key,
            steps_per_epoch=training_cfg.get("balanced_steps_per_epoch", training_cfg.get("dataset_balanced_steps_per_epoch")),
            epoch_fraction=training_cfg.get("balanced_epoch_fraction"),
            labels_per_batch=training_cfg.get("balanced_labels_per_batch"),
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
            **kwargs,
        )

    group_by_file = bool(config.get("training", {}).get("group_batches_by_file", True)) and bool(shuffle) and records is not None
    if group_by_file:
        batch_sampler = FileBatchSampler(records, batch_size, shuffle, int(config["seed"]))
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate_batch,
            **kwargs,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_batch,
        **kwargs,
    )


__all__ = ["CSVRecordStore", "DatasetBalancedBatchSampler", "LabelBalancedBatchSampler", "FileBatchSampler", "H5CSRDataset", "collate_batch", "make_dataloader"]
