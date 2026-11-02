"""Sample benchmark manifests from group-disjoint data splits."""

import argparse
import csv
import hashlib
import heapq
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from data_pipeline.clean_common import cell_type_family


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-bundle", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--train-cells", type=int, default=500_000)
    p.add_argument("--val-cells", type=int, default=100_000)
    p.add_argument("--test-cells", type=int, default=200_000)
    p.add_argument("--seed", type=int, default=20260814)
    return p.parse_args()


def apportion(counts, total):
    """Equalize datasets, then preserve coarse lineage proportions within each dataset."""
    by_dataset = defaultdict(Counter)
    for (dataset, coarse), count in counts.items():
        by_dataset[dataset][coarse] = count
    datasets = sorted(by_dataset)
    remaining = min(total, sum(counts.values()))
    dataset_quota = {d: 0 for d in datasets}
    active = set(datasets)
    while remaining and active:
        share = max(1, math.ceil(remaining / len(active)))
        progressed = False
        for dataset in sorted(active):
            capacity = sum(by_dataset[dataset].values()) - dataset_quota[dataset]
            take = min(share, capacity, remaining)
            dataset_quota[dataset] += take
            remaining -= take
            progressed |= take > 0
            if remaining == 0:
                break
        active = {d for d in active if dataset_quota[d] < sum(by_dataset[d].values())}
        if not progressed:
            break
    quota = {}
    for dataset, coarse_counts in by_dataset.items():
        wanted = dataset_quota[dataset]
        denom = sum(coarse_counts.values())
        raw = {c: wanted * n / denom for c, n in coarse_counts.items()}
        alloc = {c: min(coarse_counts[c], int(math.floor(v))) for c, v in raw.items()}
        left = wanted - sum(alloc.values())
        order = sorted(coarse_counts, key=lambda c: (raw[c] - alloc[c], coarse_counts[c], c), reverse=True)
        for coarse in order:
            if not left:
                break
            if alloc[coarse] < coarse_counts[coarse]:
                alloc[coarse] += 1
                left -= 1
        for coarse, value in alloc.items():
            quota[(dataset, coarse)] = value
    return quota


def rank(seed, split, row_number, row):
    identity = "\x1f".join([
        str(seed), split, str(row_number), row.get("file_path", ""), row.get("row_idx", ""), row.get("group_id", "")
    ])
    return int.from_bytes(hashlib.blake2b(identity.encode(), digest_size=8).digest(), "big")


def sample_manifest(source, destination, target, seed, split):
    counts = Counter()
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            counts[(row["dataset_id"], cell_type_family(row.get("cell_type", "")))] += 1
    quotas = apportion(counts, target)
    reservoirs = defaultdict(list)
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader):
            key = (row["dataset_id"], cell_type_family(row.get("cell_type", "")))
            capacity = quotas.get(key, 0)
            if capacity <= 0:
                continue
            score = rank(seed, split, row_number, row)
            heap = reservoirs[key]
            item = (-score, row_number, row)
            if len(heap) < capacity:
                heapq.heappush(heap, item)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, item)
    selected = sorted((line, row) for heap in reservoirs.values() for _, line, row in heap)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(row for _, row in selected)
    return len(selected), len({row["dataset_id"] for _, row in selected}), len({row["group_id"] for _, row in selected})


def audit_group_disjoint(split_paths):
    groups = {}
    for split, path in split_paths.items():
        with Path(path).open(newline="", encoding="utf-8") as handle:
            groups[split] = {row["group_id"] for row in csv.DictReader(handle)}
    overlaps = {
        f"{left}-{right}": sorted(groups[left] & groups[right])
        for index, left in enumerate(groups)
        for right in list(groups)[index + 1:]
        if groups[left] & groups[right]
    }
    if overlaps:
        preview = {key: values[:10] for key, values in overlaps.items()}
        raise ValueError(f"benchmark manifest leaks group_id across splits: {preview}")
    return {split: len(values) for split, values in groups.items()}


def main():
    args = parse_args()
    source_path = Path(args.source_bundle)
    output = Path(args.output_dir)
    source = json.loads(source_path.read_text())
    targets = {"train": args.train_cells, "val": args.val_cells, "test": args.test_cells}
    split_paths = {}
    summary = {}
    for split, target in targets.items():
        path = output / "manifests" / f"split_{split}.csv"
        n, datasets, groups = sample_manifest(Path(source["split_paths"][split]), path, target, args.seed, split)
        split_paths[split] = str(path)
        summary[split] = {"cells": n, "datasets": datasets, "groups": groups}
    audited_counts = audit_group_disjoint(split_paths)
    if audited_counts != {key: value["groups"] for key, value in summary.items()}:
        raise RuntimeError("group audit counts disagree with sampling summary")
    bundle = dict(source)
    bundle["split_paths"] = split_paths
    bundle["summary"] = dict(source.get("summary", {}))
    bundle["summary"].update({
        "bundle_schema_version": "manifest-v1-iclr-core",
        "num_records": sum(v["cells"] for v in summary.values()),
        "split_sizes": {k: v["cells"] for k, v in summary.items()},
        "split_group_counts": {k: v["groups"] for k, v in summary.items()},
        "split_manifest_paths": split_paths,
        "benchmark_core": {"seed": args.seed, "source_bundle": str(source_path), "sampling": "dataset-equal_then_coarse-proportional_hash_reservoir"},
    })
    output.mkdir(parents=True, exist_ok=True)
    with (output / "data_bundle.slim.json").open("w", encoding="utf-8") as handle:
        json.dump(bundle, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
