"""Build a gene-responsibility scaffold from training cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_pipeline.clean_common import cell_type_family, progress
from data_pipeline.bundle import load_data_bundle
from data_pipeline.data import H5CSRDataset, make_dataloader
from data_pipeline.runtime import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cells", type=int, default=300_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-axes", type=int, default=32)
    parser.add_argument("--donor-components", type=int, default=32)
    parser.add_argument("--min-group-cells", type=int, default=16)
    parser.add_argument("--max-groups", type=int, default=6000)
    parser.add_argument("--top-memberships", type=int, default=3)
    parser.add_argument("--primary-membership-weight", type=float, default=0.8)
    parser.add_argument("--primary-membership-weight-min", type=float, default=0.55)
    parser.add_argument("--uniform-low-signal-fraction", type=float, default=0.15)
    parser.add_argument("--min-gene-mean-log-expression", type=float, default=0.002)
    parser.add_argument("--min-gene-sd", type=float, default=0.03)
    parser.add_argument("--kmeans-n-init", type=int, default=20)
    parser.add_argument("--kmeans-max-iter", type=int, default=300)
    parser.add_argument("--balanced-refinement-iterations", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3101, 3102, 3103])
    return parser.parse_args()


def sha256_strings(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def write_csv(path: Path, rows) -> None:
    rows = list(rows)
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def equal_dataset_indices(records, max_cells: int, seed: int) -> np.ndarray:
    groups = records.dataset_indices()
    if not groups:
        raise ValueError("Training manifest has no dataset_id index")
    if int(max_cells) <= 0:
        raise ValueError("max_cells must be positive")

    rng = np.random.default_rng(seed)
    datasets = sorted(groups)
    capacities = {
        dataset: int(np.asarray(groups[dataset]).size)
        for dataset in datasets
    }
    total_available = sum(capacities.values())
    budget = min(int(max_cells), total_available)

    tie_order = [datasets[int(idx)] for idx in rng.permutation(len(datasets))]
    allocation = {dataset: 0 for dataset in datasets}
    remaining = budget
    while remaining > 0:
        active = [
            dataset
            for dataset in tie_order
            if allocation[dataset] < capacities[dataset]
        ]
        if not active:
            break
        round_take = max(1, remaining // len(active))
        for dataset in active:
            if remaining == 0:
                break
            take = min(
                round_take,
                capacities[dataset] - allocation[dataset],
                remaining,
            )
            allocation[dataset] += take
            remaining -= take

    chosen = []
    for dataset in datasets:
        indices = np.asarray(groups[dataset], dtype=np.int64)
        take = allocation[dataset]
        if take:
            chosen.append(rng.choice(indices, size=take, replace=False))
        progress(f"scaffold sample {dataset}: {take}/{indices.size}")
    if not chosen:
        raise ValueError("No cells were available for the scaffold sample")
    merged = np.concatenate(chosen)
    rng.shuffle(merged)
    if merged.size != budget or merged.size > int(max_cells):
        raise RuntimeError(
            f"Internal sampling error: selected {merged.size}, budget={budget}, max_cells={max_cells}"
        )
    return merged


def count_sample_groups(records, indices: np.ndarray, chunk_size: int = 50_000):
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start : start + chunk_size]
        if hasattr(records, "__getitems__"):
            rows = records.__getitems__([int(idx) for idx in chunk])
        else:
            rows = [records[int(idx)] for idx in chunk]
        for row in rows:
            coarse = cell_type_family(str(row.get("cell_type", "unknown")))
            key = (str(row.get("dataset_id", "unknown")), str(row.get("donor_id", "unknown")), coarse)
            counts[key] += 1
    return counts


def choose_groups(counts, min_group_cells: int, max_groups: int):
    by_dataset: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for key, count in counts.items():
        if count >= min_group_cells:
            by_dataset[key[0]].append(key)
    for dataset in by_dataset:
        by_dataset[dataset].sort(key=lambda key: (-counts[key], key))
    selected: list[tuple[str, str, str]] = []
    cursor = {dataset: 0 for dataset in by_dataset}
    datasets = sorted(by_dataset)
    while len(selected) < max_groups:
        added = False
        for dataset in datasets:
            idx = cursor[dataset]
            if idx < len(by_dataset[dataset]):
                selected.append(by_dataset[dataset][idx])
                cursor[dataset] += 1
                added = True
                if len(selected) == max_groups:
                    break
        if not added:
            break
    return set(selected)


def aggregate_groups(loader, selected_groups):
    sums: dict[tuple[str, str, str], np.ndarray] = {}
    counts: dict[tuple[str, str, str], int] = defaultdict(int)
    total_sum = None
    total_square_sum = None
    seen = 0
    for batch in loader:
        x = batch["x"].numpy()
        batch_sum = x.sum(axis=0, dtype=np.float64)
        batch_square_sum = np.square(x, dtype=np.float64).sum(axis=0)
        total_sum = batch_sum if total_sum is None else total_sum + batch_sum
        total_square_sum = batch_square_sum if total_square_sum is None else total_square_sum + batch_square_sum
        coarse_types = [cell_type_family(str(value)) for value in batch["cell_type"]]
        keys = list(zip(batch["dataset_id"], batch["donor_id"], coarse_types))
        positions: dict[tuple[str, str, str], list[int]] = defaultdict(list)
        for idx, key in enumerate(keys):
            normalized_key = tuple(str(v) for v in key)
            if normalized_key in selected_groups:
                positions[normalized_key].append(idx)
        for key, local_indices in positions.items():
            values = x[np.asarray(local_indices, dtype=np.int64)].sum(axis=0, dtype=np.float32)
            if key in sums:
                sums[key] += values
            else:
                sums[key] = values
            counts[key] += len(local_indices)
        seen += x.shape[0]
        if seen == x.shape[0] or seen % 50_000 < x.shape[0]:
            progress(f"scaffold aggregation: {seen} cells, {len(sums)} donor/cell-type groups")
    if not sums:
        raise ValueError("No groups were aggregated")
    gene_mean = total_sum / max(seen, 1)
    gene_variance = total_square_sum / max(seen, 1) - np.square(gene_mean)
    gene_sd = np.sqrt(np.maximum(gene_variance, 0.0))
    return sums, counts, seen, gene_mean.astype(np.float32), gene_sd.astype(np.float32)


def build_features(sums, counts, min_group_cells: int, donor_components: int, seed: int):
    keys = [key for key in sorted(sums) if counts[key] >= min_group_cells]
    if len(keys) < 4:
        raise ValueError(f"Only {len(keys)} donor/cell-type groups pass min_group_cells")
    group_mean = np.stack([sums[key] / max(counts[key], 1) for key in keys]).astype(np.float32)

    by_dataset_celltype: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, (dataset, _donor, celltype) in enumerate(keys):
        by_dataset_celltype[(dataset, celltype)].append(idx)
    dataset_celltype_mean = {
        key: group_mean[np.asarray(indices, dtype=np.int64)].mean(axis=0)
        for key, indices in by_dataset_celltype.items()
    }

    residual = np.empty_like(group_mean)
    for idx, (dataset, _donor, celltype) in enumerate(keys):
        residual[idx] = group_mean[idx] - dataset_celltype_mean[(dataset, celltype)]
    residual -= residual.mean(axis=0, keepdims=True)

    celltypes = sorted({key[2] for key in keys})
    datasets = sorted({key[0] for key in keys})
    celltype_profiles = []
    retained_celltypes = []
    for celltype in celltypes:
        available = [dataset_celltype_mean[(dataset, celltype)] for dataset in datasets if (dataset, celltype) in dataset_celltype_mean]
        if available:
            celltype_profiles.append(np.stack(available).mean(axis=0))
            retained_celltypes.append(celltype)
    celltype_matrix = np.stack(celltype_profiles).astype(np.float32)
    celltype_matrix -= celltype_matrix.mean(axis=0, keepdims=True)

    n_components = min(int(donor_components), residual.shape[0] - 1, residual.shape[1] - 1)
    pca = PCA(
        n_components=n_components,
        svd_solver="arpack",
        random_state=seed,
    )
    pca.fit(residual)
    singular = pca.singular_values_
    vt = pca.components_
    donor_gene_features = (vt.T * singular[None, :]).astype(np.float32)
    features = np.concatenate([celltype_matrix.T, donor_gene_features], axis=1)
    features -= features.mean(axis=0, keepdims=True)
    feature_sd = features.std(axis=0, keepdims=True)
    features /= np.maximum(feature_sd, 1e-6)
    signal = np.linalg.norm(features, axis=1)
    normalized = features / np.maximum(signal[:, None], 1e-8)
    return normalized.astype(np.float32), signal.astype(np.float32), keys, retained_celltypes, datasets, singular.astype(np.float32)


def balanced_assignment(distance: np.ndarray) -> np.ndarray:
    """Greedily minimize distance with axis capacities differing by at most one gene."""
    num_genes, num_axes = distance.shape
    base, extra = divmod(num_genes, num_axes)
    capacity = np.full(num_axes, base, dtype=np.int64)
    capacity[:extra] += 1
    assigned = np.full(num_genes, -1, dtype=np.int32)
    used = np.zeros(num_axes, dtype=np.int64)
    assigned_count = 0
    order = np.argsort(distance, axis=None, kind="stable")
    for flat in order:
        gene_idx, axis = divmod(int(flat), num_axes)
        if assigned[gene_idx] < 0 and used[axis] < capacity[axis]:
            assigned[gene_idx] = axis
            used[axis] += 1
            assigned_count += 1
            if assigned_count == num_genes:
                break
    if bool((assigned < 0).any()):
        raise RuntimeError("Balanced gene assignment did not cover every informative gene")
    return assigned


def make_mask(features, signal, gene_mean, gene_sd, args):
    cluster = MiniBatchKMeans(
        n_clusters=args.num_axes,
        random_state=args.seed,
        n_init=int(args.kmeans_n_init),
        batch_size=2048,
        max_iter=int(args.kmeans_max_iter),
        reassignment_ratio=0.0,
    )
    num_genes = features.shape[0]
    low_fraction = float(np.clip(args.uniform_low_signal_fraction, 0.0, 1.0))
    threshold = float(np.quantile(signal, low_fraction)) if low_fraction > 0 else -np.inf
    low_pattern_signal = signal <= threshold
    low_expression = gene_mean < float(args.min_gene_mean_log_expression)
    low_variation = gene_sd < float(args.min_gene_sd)
    low = low_pattern_signal | low_expression | low_variation
    informative = np.where(~low)[0]
    cluster.fit(features[informative])
    centers = cluster.cluster_centers_.astype(np.float32)
    for _iteration in range(int(args.balanced_refinement_iterations)):
        local_distance = np.linalg.norm(features[informative, None, :] - centers[None, :, :], axis=2)
        local_labels = balanced_assignment(local_distance)
        centers = np.stack([features[informative[local_labels == axis]].mean(axis=0) for axis in range(args.num_axes)]).astype(np.float32)
        centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
    distance = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2).astype(np.float32)
    labels = np.full(num_genes, -1, dtype=np.int32)
    labels[informative] = balanced_assignment(distance[informative])

    keep = max(1, min(int(args.top_memberships), int(args.num_axes)))
    primary_weight_max = float(np.clip(args.primary_membership_weight, 1.0 / keep, 1.0))
    primary_weight_min = float(np.clip(args.primary_membership_weight_min, 1.0 / keep, primary_weight_max))
    primary_weights = np.full(num_genes, 1.0 / keep, dtype=np.float32)
    for axis in range(args.num_axes):
        members = informative[labels[informative] == axis]
        member_order = members[np.argsort(distance[members, axis], kind="stable")]
        if member_order.size == 1:
            confidence = np.ones(1, dtype=np.float32)
        else:
            confidence = np.linspace(1.0, 0.0, member_order.size, dtype=np.float32)
        primary_weights[member_order] = primary_weight_min + (primary_weight_max - primary_weight_min) * confidence
    mask = np.zeros((args.num_axes, num_genes), dtype=np.float32)
    for gene_idx in informative:
        primary = int(labels[gene_idx])
        primary_weight = float(primary_weights[gene_idx])
        secondary_weight = (1.0 - primary_weight) / max(keep - 1, 1)
        mask[primary, gene_idx] = primary_weight
        if keep > 1:
            candidates = np.argsort(distance[gene_idx])
            secondary = [int(axis) for axis in candidates if int(axis) != primary][: keep - 1]
            mask[secondary, gene_idx] = secondary_weight
    mask[:, low] = 1.0 / float(args.num_axes)
    mask /= np.maximum(mask.sum(axis=0, keepdims=True), 1e-12)
    return mask, labels, distance, low, low_pattern_signal, low_expression, low_variation, centers, threshold


def build_single(args) -> Path:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config.setdefault("data", {})["_data_run_id"] = args.data_run_id
    data_root = Path(config["output"]["root"]) / args.data_run_id
    bundle_path = data_root / "data_bundle.slim.json"
    bundle = load_data_bundle(bundle_path if bundle_path.exists() else data_root / "data_bundle.json", config)
    genes = [str(gene) for gene in bundle["gene_vocab"]]
    dataset = H5CSRDataset(bundle["splits"]["train"], bundle["file_gene_maps"], genes, config, bundle["file_metas"])
    try:
        indices = equal_dataset_indices(dataset.records, args.max_cells, args.seed)
        io_ordered_indices = np.sort(indices)
        subset = Subset(dataset, io_ordered_indices.tolist())
        sampled_group_counts = count_sample_groups(dataset.records, indices)
        selected_groups = choose_groups(sampled_group_counts, args.min_group_cells, args.max_groups)
        if not selected_groups:
            raise ValueError("No donor/cell-type groups passed the bounded scaffold selection")
        progress(
            f"bounded scaffold groups: retained {len(selected_groups)} of {len(sampled_group_counts)} sampled groups "
            f"(min_cells={args.min_group_cells}, cap={args.max_groups})"
        )
        loader = make_dataloader(subset, args.batch_size, False, args.num_workers)
        sums, counts, seen, gene_mean, gene_sd = aggregate_groups(loader, selected_groups)
    finally:
        dataset.close()

    features, signal, keys, celltypes, datasets, singular = build_features(
        sums, counts, args.min_group_cells, args.donor_components, args.seed
    )
    mask, labels, distance, low, low_pattern_signal, low_expression, low_variation, centers, threshold = make_mask(
        features, signal, gene_mean, gene_sd, args
    )
    np.savez_compressed(
        output / "module_mask_prior.npz",
        module_mask=mask,
        genes=np.asarray(genes, dtype="U"),
        gene_features=features,
        gene_signal=signal,
        sampled_gene_mean_log_expression=gene_mean,
        sampled_gene_sd=gene_sd,
        hard_cluster=labels,
        distance_to_centers=distance,
        low_signal=low,
        low_pattern_signal=low_pattern_signal,
        low_expression=low_expression,
        low_variation=low_variation,
        cluster_centers=centers,
        donor_singular_values=singular,
    )

    rows = []
    for axis in range(args.num_axes):
        order = np.argsort(mask[axis])[::-1][:50]
        rows.append({
            "axis_id": axis,
            "hard_gene_count": int(np.sum(labels == axis)),
            "mask_mass": float(mask[axis].sum()),
            "top_genes": ";".join(genes[idx] for idx in order),
        })
    with (output / "axis_top_genes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "method": "equal-dataset coarse-cell-type contrasts plus donor residual PCA after subtracting dataset-by-cell-type means",
        "num_sampled_cells": int(seen),
        "num_donor_celltype_groups": len(keys),
        "max_donor_celltype_groups": int(args.max_groups),
        "bounded_memory_design": "two-pass sample: count metadata first, then allocate gene sums only for selected groups",
        "num_genes": len(genes),
        "num_axes": int(args.num_axes),
        "celltypes": celltypes,
        "datasets": datasets,
        "gene_vocab_sha256": sha256_strings(genes),
        "sampling_seed": int(args.seed),
        "donor_decomposition": "PCA(svd_solver='arpack')",
        "kmeans_n_init": int(args.kmeans_n_init),
        "kmeans_max_iter": int(args.kmeans_max_iter),
        "balanced_refinement_iterations": int(args.balanced_refinement_iterations),
        "top_memberships_per_informative_gene": int(args.top_memberships),
        "primary_membership_weight": float(args.primary_membership_weight),
        "primary_membership_weight_min": float(args.primary_membership_weight_min),
        "uniform_low_signal_fraction": float(args.uniform_low_signal_fraction),
        "uniform_low_signal_gene_count": int(low.sum()),
        "low_pattern_signal_gene_count": int(low_pattern_signal.sum()),
        "low_expression_gene_count": int(low_expression.sum()),
        "low_variation_gene_count": int(low_variation.sum()),
        "min_gene_mean_log_expression": float(args.min_gene_mean_log_expression),
        "min_gene_sd": float(args.min_gene_sd),
        "low_signal_threshold": threshold,
        "min_hard_cluster_size": int(min(row["hard_gene_count"] for row in rows)),
        "max_hard_cluster_size": int(max(row["hard_gene_count"] for row in rows)),
        "interpretation": "Gene-responsibility prior; codebook vectors are learned independently.",
    }
    print(json.dumps(summary, indent=2))
    return output / "module_mask_prior.npz"


def normalized_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def build_consensus(inputs: list[Path], output: Path, top_genes: int = 50) -> Path:
    """Align independently sampled scaffolds and average their gene responsibilities."""
    if len(inputs) < 2:
        raise ValueError("At least two independently sampled scaffolds are required")
    output.mkdir(parents=True, exist_ok=True)
    payloads = [np.load(path, allow_pickle=False) for path in inputs]
    try:
        genes = payloads[0]["genes"].astype(str)
        masks = [payload["module_mask"].astype(np.float64) for payload in payloads]
        lows = [payload["low_signal"].astype(bool) for payload in payloads]
        for path, payload, mask in zip(inputs, payloads, masks):
            if not np.array_equal(genes, payload["genes"].astype(str)):
                raise ValueError(f"Gene order differs in {path}")
            if mask.shape != masks[0].shape:
                raise ValueError(f"Mask shape differs in {path}")

        reference_mask, reference_low = masks[0], lows[0]
        aligned_masks = [reference_mask]
        alignment_rows = []
        median_cosines = [1.0]
        for input_idx, (path, target_mask, target_low) in enumerate(
            zip(inputs[1:], masks[1:], lows[1:]), start=1
        ):
            informative = ~(reference_low | target_low)
            if int(informative.sum()) < 100:
                raise ValueError(f"Only {informative.sum()} mutually informative genes for {path}")
            similarity = normalized_rows(reference_mask[:, informative]) @ normalized_rows(
                target_mask[:, informative]
            ).T
            left_axes, right_axes = linear_sum_assignment(-similarity)
            right_by_left = right_axes[np.argsort(left_axes)]
            matched = similarity[np.arange(reference_mask.shape[0]), right_by_left]
            aligned_masks.append(target_mask[right_by_left])
            median_cosines.append(float(np.median(matched)))
            for axis, target_axis in enumerate(right_by_left):
                alignment_rows.append({
                    "input_index": input_idx,
                    "input_path": str(path),
                    "reference_axis": axis,
                    "input_axis": int(target_axis),
                    "mask_cosine_on_mutually_informative_genes": float(matched[axis]),
                    "mutually_informative_gene_count": int(informative.sum()),
                })

        low_votes = np.stack(lows).sum(axis=0)
        consensus_low = low_votes >= (len(lows) // 2 + 1)
        consensus = np.stack(aligned_masks).mean(axis=0)
        consensus[:, consensus_low] = 1.0 / consensus.shape[0]
        consensus /= np.maximum(consensus.sum(axis=0, keepdims=True), 1e-12)
        effective_axes = 1.0 / np.maximum(np.square(consensus).sum(axis=0), 1e-12)
        dominant_weight = consensus.max(axis=0)
        result = output / "module_mask_prior.npz"
        np.savez_compressed(
            result,
            module_mask=consensus.astype(np.float32),
            genes=genes,
            low_signal=consensus_low,
            low_signal_votes=low_votes.astype(np.int16),
            effective_axes=effective_axes.astype(np.float32),
            dominant_axis_weight=dominant_weight.astype(np.float32),
            source_paths=np.asarray([str(path) for path in inputs], dtype=str),
        )
        write_csv(output / "axis_alignment.csv", alignment_rows)
        top_rows = []
        for axis in range(consensus.shape[0]):
            order = np.argsort(consensus[axis])[::-1][:top_genes]
            top_rows.append({
                "axis_id": axis,
                "mask_mass": float(consensus[axis].sum()),
                "top_genes": ";".join(genes[order]),
                "top_gene_weights": ";".join(f"{consensus[axis, idx]:.6f}" for idx in order),
            })
        write_csv(output / "axis_top_genes.csv", top_rows)
        summary = {
            "num_input_scaffolds": len(inputs),
            "input_paths": [str(path) for path in inputs],
            "num_axes": int(consensus.shape[0]),
            "num_genes": int(consensus.shape[1]),
            "consensus_low_signal_gene_count": int(consensus_low.sum()),
            "median_effective_axes_assigned_genes": float(np.median(effective_axes[~consensus_low])),
            "median_dominant_weight_assigned_genes": float(np.median(dominant_weight[~consensus_low])),
            "median_axis_mask_cosine_to_reference_by_input": median_cosines,
        }
        return result
    finally:
        for payload in payloads:
            payload.close()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    inputs = []
    for seed in args.seeds:
        repeat_args = SimpleNamespace(**vars(args))
        repeat_args.seed = int(seed)
        repeat_args.output_dir = str(root / f"repeat_{seed}")
        inputs.append(build_single(repeat_args))
    consensus_path = build_consensus(inputs, root)
    print(json.dumps({"consensus_scaffold": str(consensus_path), "seeds": args.seeds}, indent=2))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
