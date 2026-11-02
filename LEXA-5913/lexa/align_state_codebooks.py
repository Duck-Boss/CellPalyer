from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-effects", required=True)
    parser.add_argument("--target-effects", required=True)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--target-name", default="target")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-genes", type=int, default=50)
    parser.add_argument(
        "--ignore-occupancy",
        action="store_true",
        help="Omit occupancy metrics when training-code occupancy is unavailable.",
    )
    return parser.parse_args()


def save_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cosine_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left / np.maximum(np.linalg.norm(left, axis=1, keepdims=True), 1e-12)
    right = right / np.maximum(np.linalg.norm(right, axis=1, keepdims=True), 1e-12)
    return left @ right.T


def axis_match(left: np.ndarray, right: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    similarity = cosine_matrix(left, right)
    li, ri = linear_sum_assignment(-similarity)
    return float(np.mean(similarity[li, ri])), ri[np.argsort(li)], similarity


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference = np.load(args.reference_effects, allow_pickle=False)
    target = np.load(args.target_effects, allow_pickle=False)
    reference_genes = reference["genes"].astype(str)
    target_genes = target["genes"].astype(str)
    common = sorted(set(reference_genes).intersection(target_genes))
    if len(common) < 100:
        raise ValueError(f"Only {len(common)} common genes")
    reference_index = {gene: idx for idx, gene in enumerate(reference_genes)}
    target_index = {gene: idx for idx, gene in enumerate(target_genes)}
    r_idx = np.asarray([reference_index[g] for g in common])
    t_idx = np.asarray([target_index[g] for g in common])
    r_effect = reference["contributions"][:, :, r_idx].astype(np.float64)
    t_effect = target["contributions"][:, :, t_idx].astype(np.float64)
    r_effect -= r_effect.mean(axis=1, keepdims=True)
    t_effect -= t_effect.mean(axis=1, keepdims=True)
    pooled_scale = np.sqrt(
        np.maximum(reference["train_gene_std"][r_idx].astype(float), 0.05)
        * np.maximum(target["train_gene_std"][t_idx].astype(float), 0.05)
    )
    r_effect /= pooled_scale[None, None, :]
    t_effect /= pooled_scale[None, None, :]

    n_axes = r_effect.shape[0]
    if t_effect.shape[:2] != r_effect.shape[:2]:
        raise ValueError(f"Shape mismatch: {r_effect.shape} versus {t_effect.shape}")
    axis_similarity = np.zeros((n_axes, n_axes), dtype=float)
    code_permutations = {}
    for left_axis in range(n_axes):
        for right_axis in range(n_axes):
            score, permutation, _ = axis_match(r_effect[left_axis], t_effect[right_axis])
            axis_similarity[left_axis, right_axis] = score
            code_permutations[(left_axis, right_axis)] = permutation
    left_axes, right_axes = linear_sum_assignment(-axis_similarity)
    right_by_left = right_axes[np.argsort(left_axes)]

    r_occ = reference["train_occupancy"].astype(float)
    t_occ = target["train_occupancy"].astype(float)
    alignment_rows = []
    code_rows = []
    aligned_similarity = []
    for left_axis, right_axis in enumerate(right_by_left):
        permutation = code_permutations[(left_axis, int(right_axis))]
        right_aligned = t_effect[right_axis, permutation]
        code_cos = np.diag(cosine_matrix(r_effect[left_axis], right_aligned))
        flattened_r = r_effect[left_axis].reshape(-1)
        flattened_t = right_aligned.reshape(-1)
        effect_cosine = float(np.dot(flattened_r, flattened_t) / max(np.linalg.norm(flattened_r) * np.linalg.norm(flattened_t), 1e-12))
        r_variation = r_effect[left_axis].std(axis=0)
        t_variation = right_aligned.std(axis=0)
        r_top = set(np.argsort(r_variation)[::-1][: args.top_genes].tolist())
        t_top = set(np.argsort(t_variation)[::-1][: args.top_genes].tolist())
        aligned_target_occupancy = t_occ[right_axis, permutation]
        occupancy_mean_absolute_difference = (
            None
            if args.ignore_occupancy
            else float(np.abs(r_occ[left_axis] - aligned_target_occupancy).mean())
        )
        alignment_rows.append({
            "reference_axis": left_axis, "target_axis": int(right_axis),
            "axis_set_similarity": float(axis_similarity[left_axis, right_axis]),
            "flattened_effect_cosine": effect_cosine,
            "top_gene_overlap_count": len(r_top & t_top),
            "top_gene_list_size_each_model": int(args.top_genes),
            "mean_absolute_code_occupancy_difference": occupancy_mean_absolute_difference,
            "mean_matched_code_cosine": float(code_cos.mean()), "min_matched_code_cosine": float(code_cos.min()),
            "target_code_permutation": ";".join(str(int(x)) for x in permutation),
        })
        aligned_similarity.append(effect_cosine)
        for reference_code, target_code in enumerate(permutation):
            code_rows.append({
                "reference_axis": left_axis, "target_axis": int(right_axis),
                "reference_code": reference_code, "target_code": int(target_code),
                "effect_cosine": float(code_cos[reference_code]),
                "reference_occupancy": None if args.ignore_occupancy else float(r_occ[left_axis, reference_code]),
                "target_occupancy": None if args.ignore_occupancy else float(t_occ[right_axis, target_code]),
            })
    save_csv(output / "axis_alignment.csv", alignment_rows)
    save_csv(output / "code_alignment.csv", code_rows)
    np.save(output / "all_pair_axis_similarity.npy", axis_similarity)

    summary = {
        "reference_name": args.reference_name, "target_name": args.target_name,
        "num_common_genes": len(common), "num_axes": n_axes,
        "median_aligned_effect_cosine": float(np.median([row["flattened_effect_cosine"] for row in alignment_rows])),
        "median_top_gene_overlap_count_out_of_each_list": float(np.median([row["top_gene_overlap_count"] for row in alignment_rows])),
        "top_gene_list_size": int(args.top_genes),
        "median_mean_absolute_code_occupancy_difference": (
            None
            if args.ignore_occupancy
            else float(np.median([row["mean_absolute_code_occupancy_difference"] for row in alignment_rows]))
        ),
        "fraction_axes_effect_cosine_ge_0_5": float(np.mean(np.asarray(aligned_similarity) >= 0.5)),
        "interpretation": (
            "Axes and states are matched by effect vectors; occupancy is unavailable."
            if args.ignore_occupancy
            else "Axes and states are matched by effect vectors; occupancy is compared after matching."
        ),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
