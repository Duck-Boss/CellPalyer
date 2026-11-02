#!/usr/bin/env python3
"""Training-only retrieval and evidence cards for the hierarchical text pilot.

No held-out expression is opened here. Metadata selects eligible cases, and
their observed zero-dose controls are legal prediction inputs. The trained
codec and 512 language labels are frozen. No API calls are made by this module.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiment import DRUG_TEXT
from tools import sha256

CANDIDATE_IDS = ["keep", "nearest", "interpolate", "lower", "upper", "half", "onehalf"]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def project_simplex(values):
    """Euclidean projection of each last-axis vector onto sum=1, nonnegative."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] == 0 or not np.isfinite(values).all():
        raise ValueError("Projection requires finite, nonempty vectors")
    shape = values.shape
    flat = values.reshape(-1, shape[-1])
    ordered = np.sort(flat, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    index = np.arange(1, flat.shape[1] + 1)
    eligible = ordered - cumulative / index > 0
    rho = eligible.sum(axis=1) - 1
    theta = cumulative[np.arange(len(flat)), rho] / (rho + 1)
    result = np.maximum(flat - theta[:, None], 0.0)
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("Simplex projection failed")
    return result.reshape(shape)


def occupancy(codes, n_states=16):
    codes = np.asarray(codes)
    if codes.ndim != 2 or not len(codes) or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("Expected nonempty integer codes[cell,axis]")
    if codes.min() < 0 or codes.max() >= n_states:
        raise ValueError("Code outside state vocabulary")
    return np.stack([np.bincount(codes[:, a], minlength=n_states) / len(codes)
                     for a in range(codes.shape[1])])


def build_training_library(codes, x, ids, metadata, n_states=16, min_cells=20):
    """Match each training well to its drug/plate/replicate zero-dose control."""
    codes, x, ids = np.asarray(codes), np.asarray(x), np.asarray(ids).astype(str)
    if len(codes) != len(x) or len(ids) != len(codes) or len(set(ids)) != len(ids):
        raise ValueError("Training rows and IDs are not aligned/unique")
    if not metadata.index.is_unique:
        raise ValueError("Metadata cell IDs must be unique")
    meta = metadata.loc[ids].copy()
    if not meta["split"].eq("train").all():
        raise ValueError("Response library may only contain training cells")
    if not meta["parse_status"].eq("ok").all():
        raise ValueError("Invalid training condition annotation")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite training expression")
    ready, excluded, array_rows = [], [], []
    for (drug, dose, plate, replicate), frame in meta[meta.dose_uM.gt(0)].groupby(
            ["drug", "dose_uM", "plate", "replicate"], sort=True):
        same = (meta.drug.eq(drug) & meta.plate.eq(plate) & meta.replicate.eq(replicate)).to_numpy()
        treatment = same & meta.dose_uM.eq(dose).to_numpy()
        control = same & meta.dose_uM.eq(0).to_numpy()
        # The original design has one physical well for each such condition.
        if meta.loc[treatment, "well"].nunique() != 1:
            raise ValueError("Drug-dose-plate-replicate maps to multiple treatment wells")
        if control.any() and meta.loc[control, "well"].nunique() != 1:
            raise ValueError("Matched control maps to multiple physical wells")
        row = {"drug": str(drug), "dose_uM": float(dose), "plate": str(plate),
               "replicate": int(replicate), "well": str(frame["well"].iloc[0]),
               "n_treated": int(treatment.sum()), "n_control": int(control.sum()),
               "treated_cell_ids": ids[treatment].tolist(), "control_cell_ids": ids[control].tolist(),
               "vehicle": str(frame["vehicle"].iloc[0])}
        if row["n_treated"] < min_cells or row["n_control"] < min_cells:
            excluded.append({**row, "status": "excluded_small_sample"})
            continue
        if not meta.loc[control, "vehicle"].eq(row["vehicle"]).all():
            raise ValueError("Matched control vehicle differs")
        p_control = occupancy(codes[control], n_states)
        p_treated = occupancy(codes[treatment], n_states)
        ready.append({**row, "status": "ready", "array_index": len(array_rows)})
        array_rows.append({"p_control": p_control, "p_treated": p_treated,
                           "delta_p": p_treated - p_control,
                           "observed_delta": x[treatment].astype(float).mean(axis=0)
                                             - x[control].astype(float).mean(axis=0)})
    if not ready:
        raise ValueError("No eligible training response wells")
    wells = {key: np.stack([r[key] for r in array_rows]) for key in array_rows[0]}
    conditions, grouped = [], []
    for drug, dose in sorted({(r["drug"], r["dose_uM"]) for r in ready}):
        indices = [i for i, row in enumerate(ready) if row["drug"] == drug and row["dose_uM"] == dose]
        conditions.append({"condition_id": f"{drug}|{dose:g}", "drug": drug, "dose_uM": dose,
                           "array_index": len(grouped), "n_wells": len(indices),
                           "n_treated": sum(ready[i]["n_treated"] for i in indices),
                           "well_indices": indices, "aggregation": "equal_weight_matched_well_deltas"})
        grouped.append({key: value[indices].mean(axis=0) for key, value in wells.items()})
    arrays = {key: np.stack([r[key] for r in grouped]) for key in grouped[0]}
    details = {"conditions": conditions, "ready_wells": ready, "excluded_wells": excluded,
               "n_train_cells": len(ids), "n_train_zero_dose_cells": int(meta.dose_uM.eq(0).sum()),
               "min_cells_per_treated_and_control": min_cells,
               "matching": "same drug, vehicle, plate and author replicate; train cells only",
               "aggregation": "Equal weight per eligible well, then retain each drug-dose condition separately",
               "limitations": ["Author replicate labels do not establish independent biological replication.",
                               "Low-sample training wells are excluded; this differs from the older nonempty-well baseline."]}
    return details, arrays, wells


def retrieve(drug, dose, library, arrays):
    if not np.isfinite(dose) or dose <= 0:
        raise ValueError("Retrieval requires a positive target dose")
    available = sorted([c for c in library["conditions"] if c["drug"] == drug], key=lambda c: c["dose_uM"])
    if not available:
        raise ValueError(f"No eligible training response for {drug}")
    nearest = min(available, key=lambda c: (abs(np.log(c["dose_uM"]) - np.log(dose)), c["dose_uM"]))
    below = [c for c in available if c["dose_uM"] <= dose]
    above = [c for c in available if c["dose_uM"] >= dose]
    lower, upper = (below[-1] if below else available[0]), (above[0] if above else available[-1])
    if lower["dose_uM"] == upper["dose_uM"]:
        weight = 0.0
    else:
        weight = float((np.log(dose) - np.log(lower["dose_uM"])) /
                       (np.log(upper["dose_uM"]) - np.log(lower["dose_uM"])))
    result = {}
    for key, values in arrays.items():
        lo, hi, near = values[lower["array_index"]], values[upper["array_index"]], values[nearest["array_index"]]
        result[key] = {"lower": lo, "upper": hi, "nearest": near,
                       "interpolate": (1 - weight) * lo + weight * hi}
    info = {"target_drug": drug, "target_dose_uM": float(dose), "available_conditions": available,
            "nearest": nearest, "lower": lower, "upper": upper, "upper_weight": weight,
            "distance": "absolute difference of natural logs of positive doses; ties select lower dose",
            "interpolation": "linear in natural log positive dose; endpoint clamp outside supported range",
            "warning": "Training examples; no target-dose treatment data is available to the predictor."}
    return info, result


def candidate_distributions(p0, retrieved):
    dp = retrieved["delta_p"]
    changes = [np.zeros_like(p0), dp["nearest"], dp["interpolate"], dp["lower"],
               dp["upper"], .5 * dp["interpolate"], 1.5 * dp["interpolate"]]
    raw = np.stack([p0 + d for d in changes], axis=1)
    q = project_simplex(raw)
    q[:, 0] = p0  # Preserve exact no-op; avoid floating-point projection perturbation.
    return q, np.abs(q - raw).sum(axis=-1), raw


def gene_rows(values, genes, symbols, count=3, direction=1):
    values = np.asarray(values)
    indices = np.argsort(-direction * values, kind="stable")
    indices = [int(i) for i in indices if direction * values[i] > 1e-8][:count]
    return [{"gene_id": str(genes[i]), "symbol": str(symbols[i]), "value": round(float(values[i]), 5)}
            for i in indices]


def make_axis_cards(effects, mask, genes, symbols, std, train_occupancy, labels):
    centered = effects - effects.mean(axis=1, keepdims=True)
    importance = np.sqrt(np.mean(centered ** 2, axis=1)) / np.maximum(std, .05)
    cards = []
    for axis in range(effects.shape[0]):
        ranking = np.argsort(-importance[axis], kind="stable")
        mask_ranking = np.argsort(-mask[axis], kind="stable")
        states = []
        for state in range(effects.shape[1]):
            label = labels.get(f"A{axis}_C{state}", {})
            states.append({"state_id": state, "name": label.get("name", "unannotated"),
                           "summary": label.get("summary", "No language annotation."),
                           "positive_genes": gene_rows(centered[axis, state], genes, symbols, 2, 1),
                           "negative_genes": gene_rows(centered[axis, state], genes, symbols, 2, -1),
                           "train_occupancy": round(float(train_occupancy[axis, state]), 5),
                           "evidence_ref": f"A{axis}_C{state}"})
        cards.append({"axis_id": axis,
                      "scope": "/".join(symbols[ranking[:4]]) + " associated expression program",
                      "scope_support": "gene_evidence_supported; biological pathway interpretation remains provisional",
                      "top_mask_genes": [{"gene_id": str(genes[g]), "symbol": str(symbols[g]),
                                           "responsibility": round(float(mask[axis, g]), 5)} for g in mask_ranking[:6]],
                      "top_varying_genes": [{"gene_id": str(genes[g]), "symbol": str(symbols[g]),
                                              "rms_over_train_sd": round(float(importance[axis, g]), 5)} for g in ranking[:6]],
                      "state_reference": "uniform mean contribution of this axis's sixteen states",
                      "annotation_method": "deterministic gene-grounded axis scope + previously frozen DeepSeek state labels; no new enrichment/API",
                      "cautions": ["The axis may mix functions; state IDs are unordered.",
                                   "High/low refers to model contribution, not measured signaling activity."],
                      "states": states})
    return cards


def train_pair_probabilities(codes, n_states=16, pseudocount=.5):
    """Smoothed pooled pair frequencies; descriptive, confounded training prior."""
    codes = np.asarray(codes)
    n_axes = codes.shape[1]
    result = np.empty((n_axes, n_axes, n_states, n_states), dtype=np.float32)
    for a in range(n_axes):
        for b in range(a, n_axes):
            counts = np.bincount(codes[:, a] * n_states + codes[:, b], minlength=n_states ** 2).reshape(n_states, n_states)
            probabilities = (counts + pseudocount) / (len(codes) + n_states ** 2 * pseudocount)
            result[a, b] = probabilities
            result[b, a] = probabilities.T
    return result


def prepare_inputs(work, output, min_cells=20):
    work, output = Path(work).resolve(), Path(output).resolve()
    fitted, prepared = work / "lexa", work / "prepared"
    inputs = {"effects": fitted / "exact_code_effects.npz", "train_codes": fitted / "train_codes.npz",
              "baseline": fitted / "baseline.npz", "prepared": prepared / "prepared.npz",
              "metadata": prepared / "metadata.csv", "cases": prepared / "cases.json",
              "labels": fitted / "labels.json"}
    provenance = {"source_sha256": {k: sha256(v) for k, v in inputs.items()},
                  "implementation_sha256": sha256(__file__), "min_cells": min_cells,
                  "work": str(work), "effects_sha256": sha256(inputs["effects"])}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["provenance"] != provenance:
            raise ValueError("Existing hierarchy inputs differ; choose a new output directory")
        for path, digest in previous["output_sha256"].items():
            if not Path(path).exists() or sha256(path) != digest:
                raise ValueError("Previously prepared hierarchy artifact changed")
        return previous
    output.mkdir(parents=True, exist_ok=True)
    with np.load(inputs["effects"], allow_pickle=False) as z:
        effects = z["contributions"].astype(float)
        mask = z["module_mask"].astype(float)
        genes, symbols = z["genes"].astype(str), z["gene_symbols"].astype(str)
        std, train_p = z["train_gene_std"].astype(float), z["train_occupancy"].astype(float)
        if bool(z["synthetic"].item()) or bool(z["smoke_only"].item()):
            raise ValueError("A fitted real-data codec is required")
    axes, states, n_genes = effects.shape
    with np.load(inputs["train_codes"], allow_pickle=False) as z:
        codes, ids = z["codes"].copy(), z["cell_ids"].astype(str)
        if bool(z["sampled"].item()) or not np.array_equal(z["genes"].astype(str), genes):
            raise ValueError("Training code provenance differs")
    with np.load(inputs["prepared"], allow_pickle=False) as z:
        # Do not load x_val or any target expression; only x_train is used.
        x, x_ids = z["x_train"].astype(float), z["train_cell_ids"].astype(str)
        if not np.array_equal(z["genes"].astype(str), genes):
            raise ValueError("Training expression gene order differs")
    if not np.array_equal(ids, x_ids):
        raise ValueError("Training code/expression cell order differs")
    metadata = pd.read_csv(inputs["metadata"], index_col="cell_id")
    library, arrays, wells = build_training_library(codes, x, ids, metadata, states, min_cells)
    case_document = json.loads(inputs["cases"].read_text())
    selected = sorted([c for c in case_document["cases"] if c["status"] == "ready"], key=lambda c: c["case_id"])
    train_meta = metadata.loc[ids]
    for case in selected:
        if (train_meta.drug.eq(case["drug"]) & train_meta.dose_uM.eq(case["dose_uM"])).any():
            raise ValueError("Target drug-dose appears in the training library")
    labels_document = json.loads(inputs["labels"].read_text())
    if labels_document["effects_sha256"] != provenance["effects_sha256"] or not labels_document.get("meta", {}).get("complete"):
        raise ValueError("Complete labels from this frozen codebook are required")
    cards = make_axis_cards(effects, mask, genes, symbols, std, train_p, labels_document["labels"])
    cards_path = output / "axis_cards.json"
    save_json(cards_path, {"axes": cards, "provenance": provenance,
                           "annotation_status": "gene-grounded scope plus frozen state labels; no new GO enrichment or biological review"})
    save_json(output / "training_library.json", {**library, "provenance": provenance})
    np.savez_compressed(output / "training_library.npz", **arrays,
                        condition_ids=np.asarray([c["condition_id"] for c in library["conditions"]]),
                        genes=genes, gene_symbols=symbols)
    np.savez_compressed(output / "training_wells.npz", **wells, genes=genes)
    np.savez_compressed(output / "train_support.npz", pair_probs=train_pair_probabilities(codes, states),
                        n_train_cells=len(codes), pseudocount=.5,
                        note="pooled training pair frequencies; shared drug/dose/cell state confounding; not a causal or realism test")
    with np.load(inputs["baseline"], allow_pickle=False) as z:
        baseline_codes, baseline_ids, baseline_x = z["codes"].copy(), z["cell_ids"].astype(str), z["observed_x"].astype(float)
        if not np.array_equal(z["genes"].astype(str), genes):
            raise ValueError("Control gene vocabulary differs")
    lookup = {value: i for i, value in enumerate(baseline_ids)}
    if len(lookup) != len(baseline_ids):
        raise ValueError("Baseline cell IDs must be unique")
    cases = []
    for case in selected:
        slug = hashlib.sha256(case["case_id"].encode()).hexdigest()[:12]
        folder = output / slug
        folder.mkdir(exist_ok=True)
        positions = np.asarray([lookup[value] for value in case["control_cell_ids"]], dtype=int)
        controls = metadata.loc[baseline_ids[positions]]
        if not (controls.drug.eq(case["drug"]) & controls.plate.eq(case["plate"])
                & controls.replicate.eq(case["replicate"]) & controls.dose_uM.eq(0)).all():
            raise ValueError("Target controls are not exactly matched")
        p0 = occupancy(baseline_codes[positions], states)
        info, retrieved = retrieve(case["drug"], case["dose_uM"], library, arrays)
        q, projection, raw_q = candidate_distributions(p0, retrieved)
        delta_genes = np.einsum("aks,asg->akg", q - p0[:, None, :], effects, optimize=True)
        context_axes = []
        for axis in range(axes):
            candidate_cards = []
            for k, name in enumerate(CANDIDATE_IDS):
                vector = delta_genes[axis, k]
                candidate_cards.append({"id": name, "q": np.round(q[axis, k], 5).tolist(),
                                        "top_up": gene_rows(vector, genes, symbols, 3, 1),
                                        "top_down": gene_rows(vector, genes, symbols, 3, -1),
                                        "tv": round(float(.5 * np.abs(q[axis, k] - p0[axis]).sum()), 5),
                                        "projection_l1": round(float(projection[axis, k]), 5),
                                        "delta_rms": round(float(np.sqrt(np.mean((vector / np.maximum(std, .05)) ** 2))), 5),
                                        "evidence_ref": f"{case['case_id']}:A{axis}:{name}"})
            context_axes.append({**cards[axis], "p0": np.round(p0[axis], 5).tolist(), "candidates": candidate_cards})
        request = {"drug": case["drug"], "dose_uM": case["dose_uM"], "exposure_hours": case["exposure_time_hours"],
                   "cell_context": "A549 human lung adenocarcinoma cell line", "vehicle": case["vehicle"],
                   "text": DRUG_TEXT.get(case["drug"], case["drug"]),
                   "task": "Predict target state occupancy using training response evidence and exact axis-state interpretation. All axes are eligible. Keeping a candidate is valid.",
                   "limits": "Population marginals do not establish paired single-cell transitions or a joint distribution. Decoder feedback is not target accuracy."}
        npz_path, context_path = folder / "case.npz", folder / "context.json"
        np.savez_compressed(npz_path, p0=p0, candidate_q=q, candidate_delta_genes=delta_genes,
                            candidate_names=np.asarray(CANDIDATE_IDS), raw_candidate_q=raw_q, projection_l1=projection,
                            nearest_observed_delta=retrieved["observed_delta"]["nearest"],
                            interpolate_observed_delta=retrieved["observed_delta"]["interpolate"],
                            reference_observed_delta=retrieved["observed_delta"]["interpolate"],
                            control_mean=baseline_x[positions].mean(axis=0), control_codes=baseline_codes[positions],
                            control_cell_ids=baseline_ids[positions], genes=genes, synthetic=False)
        save_json(context_path, {"case_id": case["case_id"], "request": request, "retrieval": info,
                                 "axis_cards": context_axes, "provenance": provenance,
                                 "numerical_values": "Text values rounded; candidate selection always uses full precision NPZ arrays."})
        cases.append({"case_id": case["case_id"], "slug": slug, "drug": case["drug"],
                      "dose_uM": case["dose_uM"], "n_control": len(positions),
                      "npz": str(npz_path), "context": str(context_path)})
    artifacts = [cards_path, output / "training_library.json", output / "training_library.npz",
                 output / "training_wells.npz", output / "train_support.npz"]
    artifacts.extend(Path(c[key]) for c in cases for key in ["npz", "context"])
    manifest = {"status": "complete", "created_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "provenance": provenance, "cases": cases, "cards_path": str(cards_path),
                "training_library_json": str(output / "training_library.json"),
                "training_library_npz": str(output / "training_library.npz"),
                "train_support_npz": str(output / "train_support.npz"),
                "n_training_conditions": len(library["conditions"]), "n_ready_training_wells": len(library["ready_wells"]),
                "n_excluded_training_wells": len(library["excluded_wells"]),
                "candidate_ids": CANDIDATE_IDS, "heldout_expression_accessed": False,
                "output_sha256": {str(path): sha256(path) for path in artifacts},
                "development_status": "Existing 5uM outcomes were previously unblinded; this is development, not an independent confirmatory test."}
    save_json(manifest_path, manifest)
    return manifest


def load_case_bundle(work, case_npz, case_context):
    work, case_npz, case_context = Path(work), Path(case_npz), Path(case_context)
    context = json.loads(case_context.read_text())
    with np.load(case_npz, allow_pickle=False) as z:
        data = {key: z[key].copy() for key in z.files}
    with np.load(work / "lexa/exact_code_effects.npz", allow_pickle=False) as z:
        effects, genes, symbols, std = z["contributions"].astype(float), z["genes"].astype(str), z["gene_symbols"].astype(str), z["train_gene_std"].astype(float)
    if not np.array_equal(genes, data["genes"].astype(str)):
        raise ValueError("Case/effects gene order mismatch")
    if sha256(work / "lexa/exact_code_effects.npz") != context["provenance"]["effects_sha256"]:
        raise ValueError("Case/effects content mismatch")
    with np.load(case_npz.parent.parent / "train_support.npz", allow_pickle=False) as z:
        pair_probs = z["pair_probs"].astype(float)
    return {"case_id": context["case_id"], "request": context["request"], "p0": data["p0"],
            "candidates": data["candidate_q"], "candidate_ids": data["candidate_names"].astype(str).tolist(),
            "effects": effects, "genes": genes, "symbols": symbols, "std": std,
            "axis_candidate_cards": [axis["candidates"] for axis in context["axis_cards"]],
            "retrieval_context": context["retrieval"], "provenance": {**context["provenance"],
                    "case_npz_sha256": sha256(case_npz), "case_context_sha256": sha256(case_context)},
            "reference_observed_delta": data["reference_observed_delta"], "train_pair_probs": pair_probs,
            "nearest_observed_delta": data["nearest_observed_delta"],
            "interpolate_observed_delta": data["interpolate_observed_delta"],
            "projection_l1": data["projection_l1"], "control_mean": data["control_mean"],
            "control_codes": data["control_codes"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", default="runs/pilot_seen_dose")
    parser.add_argument("--output", default="runs/pilot_seen_dose/experiment_v3_hierarchy/inputs")
    parser.add_argument("--min-cells", type=int, default=20)
    args = parser.parse_args()
    result = prepare_inputs(args.work, args.output, args.min_cells)
    print(json.dumps({"status": result["status"], "n_cases": len(result["cases"]),
                      "n_training_conditions": result["n_training_conditions"],
                      "n_ready_training_wells": result["n_ready_training_wells"],
                      "n_excluded_training_wells": result["n_excluded_training_wells"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
