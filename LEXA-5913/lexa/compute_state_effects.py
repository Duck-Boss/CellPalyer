"""Compute exact decoder state effects, gene responsibilities, and code geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import entropy

from data_pipeline.clean_common import write_csv
from lexa.lexa_model import build_lexa_model
from data_pipeline.checkpoints import checkpoint_metadata, load_checkpoint_and_config
from data_pipeline.runtime import load_config, make_output_dirs, pick_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", default="best")
    parser.add_argument("--representations-name", default="representations")
    parser.add_argument(
        "--gene-scale-file",
        default=None,
        help="Optional train_gene_scale.npz from the same data split.",
    )
    parser.add_argument(
        "--allow-missing-train-codes",
        action="store_true",
        help="Use uniform occupancy when training codes are unavailable.",
    )
    parser.add_argument("--output-name", default="state_effects")
    parser.add_argument("--top-genes", type=int, default=30)
    parser.add_argument("--transition-top-genes", type=int, default=10)
    parser.add_argument(
        "--detailed-transitions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Write pair-level transition-gene tables for every axis.",
    )
    return parser.parse_args()


@torch.no_grad()
def contributions(model: torch.nn.Module) -> torch.Tensor:
    required = ["decoder_in", "decoder_out", "film_gamma", "film_beta", "module_mask", "quantization_codebook"]
    missing = [name for name in required if not hasattr(model, name)]
    if missing:
        raise TypeError(f"Exact additive effects unavailable; missing={missing}")
    num_axes_value = getattr(model, "num_axes", None)
    if num_axes_value is None:
        num_axes_value = getattr(model, "latent_tokens")
    num_axes = int(num_axes_value)
    codebook = model.quantization_codebook()
    mask = model.module_mask()
    rows = []
    for axis in range(num_axes):
        code = codebook[axis] if codebook.ndim == 3 else codebook
        h = F.gelu(model.decoder_in(code))
        h = F.gelu(h * model.film_gamma[axis].unsqueeze(0) + model.film_beta[axis].unsqueeze(0))
        raw = model.decoder_out(h)
        rows.append(raw * mask[axis].unsqueeze(0))
    return torch.stack(rows)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / max(float(np.linalg.norm(left) * np.linalg.norm(right)), 1e-12))


def effective_number(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    return float(weights.sum() ** 2 / max(float(np.square(weights).sum()), 1e-12))


def main() -> None:
    args = parse_args()
    detailed_transitions = bool(args.detailed_transitions)
    base = load_config(args.config)
    out = make_output_dirs(base, args.run_id)
    ckpt, config, ckpt_path = load_checkpoint_and_config(args.config, out["root"], args.checkpoint, return_path=True)
    genes = list(ckpt.get("gene_vocab") or [])
    device = pick_device(config["training"]["device"])
    model = build_lexa_model(len(genes), config, gene_vocab=genes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    effect = contributions(model).float().cpu().numpy()
    mask = model.module_mask().float().detach().cpu().numpy()
    num_axes, codebook_size, num_genes = effect.shape
    rep_root = out["root"] / args.representations_name
    train_codes_path = rep_root / "train_codes_and_metadata.npz"
    occupancy = np.full((num_axes, codebook_size), 1.0 / codebook_size, dtype=np.float64)
    if train_codes_path.exists():
        has_train_codes = True
        train = np.load(train_codes_path, allow_pickle=False)
        train_codes = train["codes"].astype(np.int64)
        for axis in range(num_axes):
            occupancy[axis] = np.bincount(train_codes[:, axis], minlength=codebook_size)
            occupancy[axis] /= max(occupancy[axis].sum(), 1.0)
        occupancy_source = str(train_codes_path.resolve())
    elif args.allow_missing_train_codes:
        has_train_codes = False
        occupancy_source = "uniform placeholder because train codes were not extracted"
    else:
        raise FileNotFoundError(
            f"{train_codes_path} is missing; extract representations first or pass "
            "--allow-missing-train-codes to use uniform occupancy"
        )
    uniform_center = effect - effect.mean(axis=1, keepdims=True)
    occupied_mean = np.einsum("ak,akg->ag", occupancy, effect)
    occupancy_center = effect - occupied_mean[:, None, :]
    scale_path = Path(args.gene_scale_file) if args.gene_scale_file else rep_root / "train_gene_scale.npz"
    scale = np.load(scale_path, allow_pickle=False)
    gene_mean = scale["mean"].astype(np.float64)
    gene_std = scale["std"].astype(np.float64)
    safe_std = np.maximum(gene_std, 0.05)
    reconstruction_distribution = str(config.get("model", {}).get("reconstruction_distribution", "mse_log1p")).lower()
    is_count_decoder = reconstruction_distribution in {"negative_binomial", "nb"}
    baseline_prop = None

    def expression_delta_for_standardization(delta: np.ndarray) -> np.ndarray:
        if not is_count_decoder:
            return delta
        if baseline_prop is None:
            raise RuntimeError("count baseline composition was not initialized")
        shifted = np.log(baseline_prop)[(None,) * (delta.ndim - 1) + (slice(None),)] + delta
        shifted -= shifted.max(axis=-1, keepdims=True)
        counterfactual = np.exp(shifted)
        counterfactual /= np.maximum(counterfactual.sum(axis=-1, keepdims=True), 1e-12)
        return np.log1p(counterfactual * 10000.0) - np.log1p(
            baseline_prop[(None,) * (delta.ndim - 1) + (slice(None),)] * 10000.0
        )

    if is_count_decoder:
        baseline_abundance = np.expm1(np.maximum(gene_mean, 0.0)) + 1e-8
        baseline_prop = baseline_abundance / np.maximum(baseline_abundance.sum(), 1e-12)
    expression_uniform_center = expression_delta_for_standardization(uniform_center)
    standardized = expression_uniform_center / safe_std[None, None, :]

    output = out["root"] / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "exact_code_effects.npz",
        genes=np.asarray(genes, dtype=str),
        contributions=effect.astype(np.float32),
        uniform_centered=uniform_center.astype(np.float32),
        occupancy_centered=occupancy_center.astype(np.float32),
        module_mask=mask.astype(np.float32),
        train_occupancy=occupancy.astype(np.float32),
        train_gene_std=gene_std.astype(np.float32),
    )

    code_rows: List[Dict[str, Any]] = []
    axis_rows: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    transition_gene_rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []
    for axis in range(num_axes):
        variation = uniform_center[axis].std(axis=0)
        standardized_variation = standardized[axis].std(axis=0)
        top = np.argsort(standardized_variation)[::-1][: args.top_genes]
        axis_rows.append({
            "axis_id": axis,
            "normalized_entropy": float(entropy(occupancy[axis] + 1e-12) / math.log(codebook_size)),
            "perplexity": float(np.exp(entropy(occupancy[axis] + 1e-12))),
            "active_codes": int(np.sum(occupancy[axis] > 0)),
            "effective_genes_raw": effective_number(variation),
            "effective_genes_standardized": effective_number(standardized_variation),
            "top_standardized_variable_genes": ";".join(genes[int(idx)] for idx in top),
        })
        for code in range(codebook_size):
            values = uniform_center[axis, code]
            zvalues = standardized[axis, code]
            code_rows.append({
                "axis_id": axis, "code_id": code, "occupancy": float(occupancy[axis, code]),
                "mean_abs_delta": float(np.abs(values).mean()),
                "rms_delta": float(np.sqrt(np.square(values).mean())),
                "p95_abs_delta": float(np.quantile(np.abs(values), 0.95)),
                "standardized_rms": float(np.sqrt(np.square(zvalues).mean())),
            })
            for direction, order in [
                ("up", np.argsort(zvalues)[::-1][: args.top_genes]),
                ("down", np.argsort(zvalues)[: args.top_genes]),
            ]:
                for rank, gene_idx in enumerate(order, start=1):
                    top_rows.append({
                        "axis_id": axis, "code_id": code, "direction": direction, "rank": rank,
                        "gene": genes[int(gene_idx)], "centered_delta": float(values[gene_idx]),
                        "standardized_delta": float(zvalues[gene_idx]), "train_gene_std": float(gene_std[gene_idx]),
                    })
        for left in range(codebook_size):
            for right in range(left + 1, codebook_size):
                diff = effect[axis, right] - effect[axis, left]
                expression_diff = expression_delta_for_standardization(diff)
                zdiff = expression_diff / safe_std
                pair_rows.append({
                    "axis_id": axis, "code_left": left, "code_right": right,
                    "rms_delta": float(np.sqrt(np.square(diff).mean())),
                    "mean_abs_delta": float(np.abs(diff).mean()),
                    "p95_abs_delta": float(np.quantile(np.abs(diff), 0.95)),
                    "standardized_rms": float(np.sqrt(np.square(zdiff).mean())),
                    "centered_effect_cosine": cosine(uniform_center[axis, left], uniform_center[axis, right]),
                })
                if detailed_transitions:
                    for direction, order in [
                        ("up", np.argsort(zdiff)[::-1][: args.transition_top_genes]),
                        ("down", np.argsort(zdiff)[: args.transition_top_genes]),
                    ]:
                        for rank, gene_idx in enumerate(order, start=1):
                            gate = float(mask[axis, gene_idx])
                            transition_gene_rows.append({
                                "axis_id": axis,
                                "code_left": left,
                                "code_right": right,
                                "direction_left_to_right": direction,
                                "rank": rank,
                                "gene": genes[int(gene_idx)],
                                "axis_mask_weight_fixed_across_codes": gate,
                                "masked_log_expression_delta": float(diff[gene_idx]),
                                "standardized_delta": float(zdiff[gene_idx]),
                                "unmasked_decoder_delta": float(diff[gene_idx] / max(gate, 1e-12)),
                                "train_gene_std": float(gene_std[gene_idx]),
                                "reverse_transition_identity": "right_to_left_is_negative",
                            })
    write_csv(output / "axis_summary.csv", axis_rows)
    write_csv(output / "code_summary.csv", code_rows)
    write_csv(output / "code_pair_summary.csv", pair_rows)
    write_csv(output / "top_code_genes.csv", top_rows)
    if detailed_transitions:
        write_csv(output / "top_transition_genes.csv", transition_gene_rows)

    eps = 1e-12
    gene_axis_entropy = -np.sum(mask * np.log(np.clip(mask, eps, None)), axis=0) / math.log(num_axes)
    gene_rows = [{
        "gene": gene,
        "mask_specificity": float(1.0 - gene_axis_entropy[idx]),
        "effective_axes": float(1.0 / max(float(np.square(mask[:, idx]).sum()), eps)),
        "dominant_axis": int(np.argmax(mask[:, idx])),
        "dominant_axis_weight": float(np.max(mask[:, idx])),
    } for idx, gene in enumerate(genes)]
    write_csv(output / "gene_mask_specificity.csv", gene_rows)

    codebook = model.quantization_codebook()
    rng = np.random.default_rng(int(config["seed"]) + 901)
    indices = rng.integers(0, codebook_size, size=(16, num_axes))
    if codebook.ndim == 3:
        z = torch.stack([codebook[axis, torch.as_tensor(indices[:, axis], device=device)] for axis in range(num_axes)], dim=1)
    else:
        z = codebook[torch.as_tensor(indices, device=device)]
    with torch.no_grad():
        direct = (
            model.decode_modules(z)[0]
            if is_count_decoder
            else model.decode(z)
        ).float().cpu().numpy()
    summed = np.stack([effect[axis, indices[:, axis]] for axis in range(num_axes)], axis=1).sum(axis=1)
    additivity_max_abs = float(np.max(np.abs(direct - summed)))
    additivity_rms = float(np.sqrt(np.mean(np.square(direct - summed))))

    summary = {
        "run_id": args.run_id,
        "checkpoint": checkpoint_metadata(ckpt, config, ckpt_path),
        "num_axes": num_axes,
        "codebook_size": codebook_size,
        "num_genes": num_genes,
        "mathematical_identity": (
            "biological_gene_logit[g] = sum_a contribution[a, code_a, g] before the gene softmax"
            if is_count_decoder else
            "biological_normalized_log_expression[g] = sum_a contribution[a, code_a, g]"
        ),
        "effect_units": "pre-softmax gene logits" if is_count_decoder else "normalized log1p expression",
        "gene_sd_standardization": (
            "Translate each exact gene-logit effect through the declared global baseline composition to normalized-log expression, then divide by max(training gene SD, 0.05)."
            if is_count_decoder else
            "Divide the additive normalized-log expression effect by max(training gene SD, 0.05)."
        ),
        "mask_identity": "mask[a,g] is independent of code k and sums to one over axes for every gene",
        "uniform_centering": "subtract the unweighted mean contribution over 16 codes",
        "occupancy_centering": (
            "subtract the training-occupancy-weighted mean contribution"
            if has_train_codes
            else "uniform occupancy used; training codes unavailable"
        ),
        "occupancy_source": occupancy_source,
        "gene_scale_source": str(scale_path.resolve()),
        "additivity_numerical_check": {"max_abs_residual": additivity_max_abs, "rms_residual": additivity_rms},
        "median_axis_perplexity": float(np.median([row["perplexity"] for row in axis_rows])),
        "median_effective_genes_standardized": float(np.median([row["effective_genes_standardized"] for row in axis_rows])),
        "median_gene_mask_specificity": float(np.median(1.0 - gene_axis_entropy)),
        "detailed_transition_outputs": bool(detailed_transitions),
        "num_transition_gene_rows": len(transition_gene_rows),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
