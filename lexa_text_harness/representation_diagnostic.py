#!/usr/bin/env python3
"""Evaluate whether a frozen LEXA preserves observed held-out response signals.

This diagnostic reads true treated expression and computes D(E(y)). It is NOT
an agent prediction and NOT a mathematical upper bound on every possible code
editor. Changing the dictionary, prompts, QC or protocol after inspecting it
uses the test set for development; a new untouched test split is then needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

import fit_lexa


def delta_metrics(actual_delta: np.ndarray, reconstructed_delta: np.ndarray,
                   direction_threshold: float = 0.001) -> dict[str, float | int | None]:
    """All-gene metrics in normalized-log1p expression units.

    Direction agreement is scored only where |actual delta| >= threshold.
    A reconstructed change below that same threshold has direction zero and
    does not count as agreeing with a nonzero actual change. Undefined ratios
    and correlations are returned as None, never silently replaced by zero.
    """
    actual = np.asarray(actual_delta, dtype=np.float64)
    predicted = np.asarray(reconstructed_delta, dtype=np.float64)
    if actual.ndim != 1 or actual.shape != predicted.shape or not actual.size:
        raise ValueError("Deltas must be nonempty vectors of equal shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Deltas must be finite")
    if direction_threshold <= 0:
        raise ValueError("direction_threshold must be positive")
    rmse = float(np.sqrt(np.mean(np.square(predicted - actual))))
    actual_norm = float(np.linalg.norm(actual))
    predicted_norm = float(np.linalg.norm(predicted))
    actual_centered = actual - actual.mean()
    predicted_centered = predicted - predicted.mean()
    denom = float(np.linalg.norm(actual_centered) * np.linalg.norm(predicted_centered))
    pearson = float(np.clip(np.dot(actual_centered, predicted_centered) / denom, -1, 1)) if denom > 1e-12 else None
    cosine = float(np.clip(np.dot(actual, predicted) / (actual_norm * predicted_norm), -1, 1)) if actual_norm * predicted_norm > 1e-12 else None
    active = np.abs(actual) >= direction_threshold
    actual_sign = np.where(actual >= direction_threshold, 1, np.where(actual <= -direction_threshold, -1, 0))
    predicted_sign = np.where(predicted >= direction_threshold, 1, np.where(predicted <= -direction_threshold, -1, 0))
    direction = float(np.mean(actual_sign[active] == predicted_sign[active])) if active.any() else None
    return {"n_genes": int(actual.size), "delta_rmse": rmse,
            "zero_edit_delta_rmse": float(np.sqrt(np.mean(np.square(actual)))),
            "delta_pearson": pearson, "delta_cosine": cosine,
            "direction_threshold": float(direction_threshold),
            "n_direction_genes": int(active.sum()), "direction_agreement": direction,
            "actual_delta_l2": actual_norm, "reconstructed_delta_l2": predicted_norm,
            "response_l2_ratio": predicted_norm / actual_norm if actual_norm > 1e-12 else None,
            "response_projection_slope": float(np.dot(actual, predicted)) / actual_norm ** 2 if actual_norm > 1e-12 else None}


def reconstruction_mse(observed: np.ndarray, reconstructed: np.ndarray) -> float:
    if observed.shape != reconstructed.shape or observed.ndim != 2 or observed.size == 0:
        raise ValueError("Expression matrices must be nonempty with equal shapes")
    return float(np.mean(np.square(reconstructed.astype(np.float64) - observed.astype(np.float64))))


@torch.no_grad()
def frozen_reconstruct(model, x: np.ndarray, device: torch.device,
                        batch_size: int = 64) -> np.ndarray:
    model.eval()
    output = np.empty_like(x, dtype=np.float32)
    for start in range(0, len(x), batch_size):
        values = torch.from_numpy(x[start:start + batch_size]).to(device)
        quantized = model.quantize(model.encode(values))[0]
        output[start:start + batch_size] = model.decode(quantized).float().cpu().numpy()
    return output


def id_index(values: np.ndarray, name: str) -> dict[str, int]:
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    result = {str(value): index for index, value in enumerate(values)}
    if len(result) != len(values):
        raise ValueError(f"{name} must be unique")
    return result


def matched_indices(case: dict, field: str, lookup: dict[str, int]) -> np.ndarray:
    names = [str(value) for value in case[field]]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"Case {case['case_id']} has empty or duplicate {field}")
    missing = set(names) - set(lookup)
    if missing:
        raise ValueError(f"Case {case['case_id']} has {len(missing)} unknown {field}")
    return np.asarray([lookup[name] for name in names], dtype=np.int64)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fitted", required=True, type=Path, help="Fitted directory containing best.pt, or checkpoint path")
    parser.add_argument("--prepared", required=True, type=Path, help="Prepared directory, or evaluation.npz path")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path; matching .csv is also written")
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--lexa-repo", type=Path)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:N, or auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--direction-threshold", type=float, default=0.001)
    return parser.parse_args(argv)


def run(args) -> dict[str, Any]:
    if args.batch_size < 1 or args.threads < 1 or args.direction_threshold <= 0:
        raise ValueError("batch-size, threads and direction-threshold must be positive")
    checkpoint_path = args.fitted / "best.pt" if args.fitted.is_dir() else args.fitted
    fitted_dir = checkpoint_path.parent
    evaluation_path = args.prepared / "evaluation.npz" if args.prepared.is_dir() else args.prepared
    prepared_dir = evaluation_path.parent
    cases_path = args.cases or prepared_dir / "cases.json"
    metadata_path = args.metadata or fitted_dir / "metadata.json"
    if not (fitted_dir / "COMPLETE.json").is_file():
        raise ValueError("Fitted directory has no COMPLETE.json; wait for frozen model export")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    cases_payload = json.loads(cases_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    repo = args.lexa_repo or Path(metadata.get("source_repository", str(fit_lexa.DEFAULT_REPO)))
    for filename, expected in checkpoint.get("source_sha256", {}).items():
        if fit_lexa.file_sha256(repo / "lexa" / filename) != expected:
            raise ValueError(f"Original source changed since fit: {filename}")
    prepared_path = prepared_dir / "prepared.npz"
    prepared_verified = None
    if prepared_path.is_file():
        prepared_verified = fit_lexa.file_sha256(prepared_path) == checkpoint["prepared_sha256"]
        if not prepared_verified:
            raise ValueError("Checkpoint was fitted to a different prepared.npz")
    required = {"genes", "x_test", "test_cell_ids", "test_case_ids", "control_x",
                "control_cell_ids", "control_case_ids"}
    with np.load(evaluation_path, allow_pickle=False) as archive:
        if not required <= set(archive.files):
            raise ValueError(f"Evaluation archive missing: {sorted(required - set(archive.files))}")
        genes = archive["genes"].astype(str)
        x_test = archive["x_test"].astype(np.float32)
        control_x = archive["control_x"].astype(np.float32)
        test_ids = archive["test_cell_ids"].astype(str)
        control_ids = archive["control_cell_ids"].astype(str)
        test_cases = archive["test_case_ids"].astype(str)
        control_cases = archive["control_case_ids"].astype(str)
        synthetic = bool(archive["synthetic"].item()) if "synthetic" in archive else False
    if genes.tolist() != checkpoint["gene_vocab"]:
        raise ValueError("Evaluation gene vocabulary/order differs from checkpoint")
    for values, ids, groups, name in ((x_test, test_ids, test_cases, "test"),
                                      (control_x, control_ids, control_cases, "control")):
        if values.ndim != 2 or values.shape[1] != len(genes) or values.shape[0] != len(ids) or ids.shape != groups.shape or len(ids) == 0:
            raise ValueError(f"Invalid {name} dimensions")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid {name} observed expression values")
    if synthetic != bool(checkpoint.get("synthetic", False)):
        raise ValueError("Synthetic status differs between checkpoint and evaluation")
    test_lookup = id_index(test_ids, "test_cell_ids")
    control_lookup = id_index(control_ids, "control_cell_ids")
    if set(test_lookup) & set(control_lookup):
        raise ValueError("Test-treated and control cell IDs overlap")
    torch.set_num_threads(args.threads)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    original_model, _ = fit_lexa.import_original(repo)
    model = original_model(**checkpoint["config"], module_mask_prior=checkpoint["module_mask_prior"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    model.requires_grad_(False)
    test_recon = frozen_reconstruct(model, x_test, device, args.batch_size)
    control_recon = frozen_reconstruct(model, control_x, device, args.batch_size)

    rows, skipped = [], []
    for case in cases_payload["cases"]:
        if case.get("status") != "ready":
            skipped.append({"case_id": case["case_id"], "status": case.get("status", "unknown")})
            continue
        test_ix = matched_indices(case, "treated_cell_ids", test_lookup)
        control_ix = matched_indices(case, "control_cell_ids", control_lookup)
        if not np.all(test_cases[test_ix] == case["case_id"]):
            raise ValueError(f"Test case identity mismatch: {case['case_id']}")
        if not np.all(control_cases[control_ix] == case["control_group_id"]):
            raise ValueError(f"Control group identity mismatch: {case['case_id']}")
        actual_delta = x_test[test_ix].mean(axis=0, dtype=np.float64) - control_x[control_ix].mean(axis=0, dtype=np.float64)
        decoded_delta = test_recon[test_ix].mean(axis=0, dtype=np.float64) - control_recon[control_ix].mean(axis=0, dtype=np.float64)
        metrics = delta_metrics(actual_delta, decoded_delta, args.direction_threshold)
        rows.append({"case_id": case["case_id"], "drug": case["drug"],
                     "dose_uM": case["dose_uM"], "plate": case["plate"],
                     "replicate": case["replicate"], "n_treated": len(test_ix),
                     "n_control": len(control_ix), **metrics,
                     "treated_reconstruction_mse": reconstruction_mse(x_test[test_ix], test_recon[test_ix]),
                     "control_reconstruction_mse": reconstruction_mse(control_x[control_ix], control_recon[control_ix])})

    metric_names = ("delta_rmse", "zero_edit_delta_rmse", "delta_pearson", "delta_cosine",
                    "direction_agreement", "response_l2_ratio", "response_projection_slope")
    aggregate = {}
    for name in metric_names:
        values = [float(row[name]) for row in rows if row[name] is not None]
        aggregate[name] = {"n_defined_cases": len(values),
                           "mean_over_cases": float(np.mean(values)) if values else None,
                           "median_over_cases": float(np.median(values)) if values else None}
    report = {"status": "diagnostic_complete", "uses_test_ground_truth": True,
              "not_agent_prediction": True, "not_mathematical_upper_bound": True,
              "synthetic": synthetic, "smoke_only": bool(metadata.get("smoke_only", False)),
              "warning": "Uses true held-out treated expression in D(E(y)). Inspecting this result and then changing dictionary/prompts/protocol consumes the test set for development; a fresh independent test is required.",
              "method": "For each ready case, compare mean(y_treated)-mean(x_matched_control) with mean(D(E(y_treated)))-mean(D(E(x_matched_control))) using the frozen original encoder, quantizer and decoder.",
              "units": "normalized log1p expression (CP10K)",
              "checkpoint": str(checkpoint_path.resolve()),
              "checkpoint_sha256": fit_lexa.file_sha256(checkpoint_path),
              "checkpoint_epoch": checkpoint.get("epoch"),
              "evaluation_path": str(evaluation_path.resolve()),
              "evaluation_sha256": fit_lexa.file_sha256(evaluation_path),
              "cases_path": str(cases_path.resolve()), "prepared_sha256_verified": prepared_verified,
              "protocol": cases_payload.get("protocol"), "device": str(device),
              "n_genes": len(genes), "n_test_cells": len(x_test), "n_control_cells": len(control_x),
              "n_ready_cases_scored": len(rows), "n_nonready_cases_skipped": len(skipped),
              "all_observed_test_cell_reconstruction_mse": reconstruction_mse(x_test, test_recon),
              "all_observed_control_cell_reconstruction_mse": reconstruction_mse(control_x, control_recon),
              "metric_definitions": {"delta_rmse": "RMSE over all selected genes between reconstructed and actual pseudobulk treatment deltas",
                                     "delta_pearson": "Pearson correlation across genes; null for constant vectors",
                                     "direction_agreement": f"Fraction of genes with abs(actual delta)>={args.direction_threshold} whose reconstructed delta has the same sign above the same threshold",
                                     "response_l2_ratio": "||reconstructed delta||2 / ||actual delta||2; null if actual delta is zero",
                                     "response_projection_slope": "dot(actual delta,reconstructed delta) / ||actual delta||2^2; negative values indicate reversed response",
                                     "aggregate": "Unweighted case means/medians; undefined values excluded and counted"},
              "aggregate": aggregate, "cases": rows, "skipped_cases": skipped,
              "limitations": ["Compression loss can suppress true response; high retention does not establish causal or predictive validity.",
                              "This compares encodings of observed treated cells. An editor may reach other codes, so this is not a proof of its optimal achievable error.",
                              "Case pseudobulk metrics do not establish preservation of full single-cell distributions.",
                              "Readiness follows predeclared sample-size eligibility; skipped groups remain in pooled reconstruction metrics."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    temp_csv = csv_path.with_name(csv_path.name + f".tmp.{os.getpid()}")
    columns = list(rows[0]) if rows else ["case_id", "delta_rmse", "delta_pearson", "direction_agreement", "response_l2_ratio"]
    with temp_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_csv, csv_path)
    report["case_csv"] = str(csv_path.resolve())
    fit_lexa.atomic_json(args.output, report)
    return report


def main():
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
