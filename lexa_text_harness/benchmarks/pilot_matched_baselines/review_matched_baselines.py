#!/usr/bin/env python3
"""Independent audit of the existing-data perturbation baseline comparison.

The data stage opens only training/validation expression and case annotations.
Held-out expression is reserved for the post-seal score stage. This module does
not import any model, adapter, training code, or shared scoring implementation.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


SOURCE_NAMES = ("prepared.npz", "cases.json", "metadata.csv", "manifest.json")
FORBIDDEN_INPUT_KEYS = {
    "x_test", "y_test", "test_expression", "observed_delta", "true_delta",
    "true_treated_mean", "observed_treated_mean", "treated_cell_ids",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def nested_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from nested_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_keys(item)


def source_data_review(prepared):
    prepared = Path(prepared).resolve()
    with np.load(prepared / "prepared.npz", allow_pickle=False) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    require(not bool(data["synthetic"].item()), "Synthetic source cannot be benchmarked")
    require(data["x_train"].shape == (5285, 2000), "Training shape changed")
    require(data["x_val"].shape == (2742, 2000), "Validation shape changed")
    require(data["control_x"].shape == (1409, 2000), "Conditioning control shape changed")
    genes = data["genes"].astype(str)
    require(len(genes) == 2000 and len(set(genes)) == 2000, "Gene vocabulary changed")
    train_ids = data["train_cell_ids"].astype(str)
    val_ids = data["val_cell_ids"].astype(str)
    control_ids = data["control_cell_ids"].astype(str)
    for name, values in (("train", train_ids), ("val", val_ids), ("controls", control_ids)):
        require(len(set(values)) == len(values), "Duplicate " + name + " cell IDs")
    require(not set(train_ids) & set(val_ids), "Train/validation cell overlap")
    metadata = pd.read_csv(prepared / "metadata.csv").set_index("cell_id", verify_integrity=True)
    subsets = {"train": metadata.loc[train_ids], "val": metadata.loc[val_ids]}
    for split, rows in subsets.items():
        require(rows["split"].eq(split).all(), "Expression order/metadata split differs")
        require(not rows["dose_uM"].eq(5).any(), "5uM target appeared in fitting data")
        require(rows["pass_qc"].all(), "Failed-QC row appeared in fitting data")
    require(subsets["train"]["replicate"].isin((1, 2)).all(), "Training replicate rule differs")
    require(subsets["val"]["replicate"].eq(3).all(), "Validation replicate rule differs")
    control_metadata = metadata.loc[control_ids]
    require(control_metadata["is_control"].all(), "Noncontrol in control array")
    require(control_metadata["dose_uM"].eq(0).all(), "Nonzero dose in control array")
    control_lookup = {cell: i for i, cell in enumerate(control_ids)}
    for split, ids, expression in (("train", train_ids, data["x_train"]), ("val", val_ids, data["x_val"])):
        mask = subsets[split]["is_control"].to_numpy(dtype=bool)
        np.testing.assert_array_equal(expression[mask], data["control_x"][[control_lookup[cell] for cell in ids[mask]]])
    cases = [case for case in read(prepared / "cases.json")["cases"] if case["status"] == "ready"]
    require(len(cases) == 16, "Ready-case selection changed")
    require(len({case["case_id"] for case in cases}) == len(cases), "Duplicate case IDs")
    ready_controls = set()
    ready_treated = set()
    drug_counts = {}
    for case in cases:
        require(float(case["dose_uM"]) == 5, "Non-5uM case selected")
        require(case["cell_line"] == "A549" and case["exposure_time_hours"] == 24, "Case context changed")
        controls = set(case["control_cell_ids"])
        treated = set(case["treated_cell_ids"])
        require(len(controls) == case["n_control"] and len(treated) == case["n_treated"], "Case counts differ")
        require(not treated & ready_treated, "A target cell belongs to multiple cases")
        require(controls <= set(control_ids), "Case controls absent from available conditioning data")
        require(not treated & (set(train_ids) | set(val_ids) | set(control_ids)), "Target expression cell leaked into fit/control pool")
        for ids, dose in ((controls, 0), (treated, 5)):
            rows = metadata.loc[sorted(ids)]
            require(rows["drug"].eq(case["drug"]).all(), "Case drug mismatch")
            require(rows["plate"].eq(case["plate"]).all(), "Case plate mismatch")
            require(rows["replicate"].eq(case["replicate"]).all(), "Case replicate mismatch")
            require(rows["dose_uM"].eq(dose).all(), "Case dose mismatch")
        expected_vehicle = "Ethanol" if case["drug"] == "Dex" else "DMSO"
        require(metadata.loc[sorted(controls), "vehicle"].eq(expected_vehicle).all(), "Matched control vehicle mismatch")
        ready_controls |= controls
        ready_treated |= treated
        row = drug_counts.setdefault(case["drug"], {"wells": 0, "treated": 0, "control_ids": set()})
        row["wells"] += 1
        row["treated"] += len(treated)
        row["control_ids"] |= controls
    require(len(ready_controls) == 951 and len(ready_treated) == 786, "Scored population differs")
    for row in drug_counts.values():
        require(row["wells"] == 4, "Per-drug well weighting changed")
        row["unique_controls"] = len(row.pop("control_ids"))
    return {
        "status": "passed", "n_train": 5285, "n_val": 2742, "n_genes": 2000,
        "n_ready_cases": 16, "n_ready_treated": 786, "n_unique_ready_controls": 951,
        "n_all_available_controls": 1409, "by_drug": drug_counts,
        "all_control_splits": control_metadata["split"].value_counts().to_dict(),
        "ready_control_splits": metadata.loc[sorted(ready_controls), "split"].value_counts().to_dict(),
        "source_hashes": {name: sha(prepared / name) for name in SOURCE_NAMES},
        "case_ids": [case["case_id"] for case in cases],
        "genes_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
        "target_expression_opened": False,
    }, data, cases


def review_prediction_cases(path, source_cases, genes):
    value = read(path)
    forbidden = set(nested_keys(value)) & FORBIDDEN_INPUT_KEYS
    require(not forbidden, "Outcome fields in model input: " + ", ".join(sorted(forbidden)))
    exported = value["cases"] if isinstance(value, dict) else value
    require(isinstance(exported, list) and len(exported) == 16, "Expected 16 exported prediction cases")
    require(len({row["case_id"] for row in exported}) == 16, "Duplicate exported cases")
    source = {row["case_id"]: row for row in source_cases}
    require(set(source) == {row["case_id"] for row in exported}, "Exported prediction case population differs")
    for row in exported:
        original = source[row["case_id"]]
        require(set(row["control_cell_ids"]) == set(original["control_cell_ids"]), "Exported control identities differ")
        for key in ("drug", "dose_uM", "plate", "replicate"):
            if key in row:
                require(row[key] == original[key], "Exported case metadata differs: " + key)
    if isinstance(value, dict) and "genes" in value:
        require(np.array_equal(value["genes"], genes.astype(str)), "Exported gene order differs")
    if isinstance(value, dict) and "genes_sha256" in value:
        require(value["genes_sha256"] == hashlib.sha256("\n".join(genes.astype(str)).encode()).hexdigest(), "Exported vocabulary checksum differs")
    return {"status": "passed", "sha256": sha(path), "n_cases": 16, "forbidden_outcome_fields": []}


def data_stage(args):
    report, data, cases = source_data_review(args.prepared)
    if args.prediction_cases is not None:
        report["prediction_cases"] = review_prediction_cases(args.prediction_cases, cases, data["genes"])
    if args.protocol is not None:
        protocol = read(args.protocol)
        report["protocol_sha256"] = sha(args.protocol)
        # Source mappings vary by orchestration; verify every recognized original
        # source digest but do not mistake a mere recorded digest for a match.
        checks = 0
        for name, record in protocol.get("inputs", {}).items():
            if Path(name).name in SOURCE_NAMES:
                require(isinstance(record, dict), "Protocol source record must contain path/digest")
                require(Path(record["path"]).resolve() == (Path(args.prepared) / Path(name).name).resolve(), "Protocol source path differs")
                require(sha(record["path"]) == record["sha256"], "Protocol input digest differs: " + name)
                checks += 1
        for key in ("source_hashes", "source_sha256", "input_hashes", "inputs_sha256", "files_sha256"):
            mapping = protocol.get(key)
            if not isinstance(mapping, dict):
                continue
            for name, digest in mapping.items():
                basename = Path(name).name
                if basename in SOURCE_NAMES and isinstance(digest, str):
                    require(sha(Path(args.prepared) / basename) == digest, "Protocol source hash differs: " + name)
                    checks += 1
        report["protocol_original_source_hashes_verified"] = checks
        require(checks >= 4, "Protocol does not bind all four original source files")
        if args.prediction_cases is not None:
            require(protocol["prediction_cases_sha256"] == sha(args.prediction_cases), "Protocol prediction input digest differs")
        require(protocol["models"] == ["cpa", "codex"], "Planned model population differs")
        require(protocol["training_seeds"] == [0, 1, 2], "Planned independent seeds differ")
        require(protocol["n_train"] == 5285 and protocol["n_val"] == 2742, "Protocol fitting population differs")
        require(protocol["n_cases"] == 16 and protocol["n_genes"] == 2000, "Protocol case/feature population differs")
        require(set(protocol["prespecified_prediction_views"]) >= {"native", "calibrated"}, "Both prespecified anchors must be retained")
    report["checked_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    report["information_conditions"] = "Identical dataset/test/metrics; CPA/CODEX explicitly supervise numerical drug-dose responses during training, while pure-mechanism DeepSeek uses no response retrieval at inference."
    save(args.output, report)
    return report


def verify_global_seal(experiment, required=True):
    path = Path(experiment) / "prediction_seal.json"
    if not path.exists():
        require(not required, "Global prediction seal is absent; outcomes remain unopened")
        return None
    seal = read(path)
    require(seal.get("status") == "all_six_model_predictions_sealed", "Global seal has an unexpected status")
    require(seal.get("n_fitted_models") == 6 and seal.get("n_prediction_cases_per_model") == 16, "Global seal model/case population differs")
    require(seal.get("score_exists_at_seal", False) is False, "Seal was made after scoring")
    files = seal.get("files")
    require(isinstance(files, dict) and len(files) > 0, "Global seal does not bind prediction artifacts")
    for mapping in (files, seal.get("source_files", {})):
        for name, digest in mapping.items():
            candidate = Path(name)
            if not candidate.is_absolute():
                candidate = Path(experiment) / name
            require(sha(candidate) == digest, "Sealed file changed: " + name)
    return {"sha256": sha(path), "n_files": len(files)}


def models_stage(args):
    experiment = Path(args.experiment).resolve()
    protocol = read(experiment / "protocol.json")
    report, source, cases = source_data_review(args.prepared)
    exported = review_prediction_cases(experiment / "data/prediction_cases.json", cases, source["genes"])
    planned_order = [row["case_id"] for row in read(experiment / "data/prediction_cases.json")["cases"]]
    require(protocol["prediction_cases_sha256"] == exported["sha256"], "Protocol/input manifest binding differs")
    artifacts = []
    required_prediction_keys = {"case_ids", "genes", "pred_treated_mean", "pred_control_mean"}
    for model in protocol["models"]:
        for seed in protocol["training_seeds"]:
            folder = experiment / "models" / model / ("seed_" + str(seed))
            prediction_path = folder / "predictions.npz"
            with np.load(prediction_path, allow_pickle=False) as archive:
                require(set(archive.files) == required_prediction_keys, "Prediction schema contains missing/extra fields")
                require(archive["case_ids"].astype(str).tolist() == planned_order, "Prediction case order differs")
                require(np.array_equal(archive["genes"].astype(str), source["genes"].astype(str)), "Prediction gene order differs")
                for key in ("pred_treated_mean", "pred_control_mean"):
                    require(archive[key].shape == (16, 2000), "Prediction matrix shape differs")
                    require(np.isfinite(archive[key]).all(), "Prediction contains nonfinite values")
            config_path = folder / "config.json"
            config = read(config_path)
            require(config["seed"] == seed and config["device"] == "cpu", "Seed/device differs from protocol")
            require(config["n_train"] == 5285 and config["n_val"] == 2742 and config["n_genes"] == 2000, "Fitted population differs")
            require(not config.get("test_target_access", False), "Config admits test target access")
            require(config.get("no_test_targets_loaded", True) is True, "Config admits test target loading")
            checkpoint = folder / "best.pt"
            require(checkpoint.is_file() and checkpoint.stat().st_size > 0, "Selected checkpoint missing")
            if model == "cpa":
                history_path = folder / "history.jsonl"
                history = pd.DataFrame([json.loads(line) for line in history_path.read_text().splitlines() if line.strip()])
                summary = read(folder / "summary.json")
            else:
                history_path = folder / "training.csv"
                history = pd.read_csv(history_path)
                summary = read(folder / "training_summary.json")
            require(len(history) > 2, "Formal result unexpectedly has only smoke-test epochs")
            require(history["epoch"].is_unique and history["epoch"].is_monotonic_increasing, "Epoch history is not a unique increasing sequence")
            require(np.isfinite(history["val_rmse"]).all(), "Validation history is nonfinite")
            require(not set(str(k).lower() for k in history.columns) & FORBIDDEN_INPUT_KEYS, "History includes outcome fields")
            chosen = history[history["epoch"] == int(summary["best_epoch"])]
            require(len(chosen) == 1, "Selected epoch missing from history")
            best = chosen.iloc[0]
            require(float(best["val_rmse"]) <= float(history["val_rmse"].min()) + 1.0001e-8, "Selected checkpoint did not minimize validation RMSE")
            require(abs(float(summary["best_validation_rmse"]) - float(best["val_rmse"])) < 1e-8, "Selected validation value differs")
            if (folder / "prediction_seal.json").exists():
                local_seal = read(folder / "prediction_seal.json")
                require(local_seal.get("test_target_access") is False, "Local seal admits target access")
                for name, digest in local_seal["files"].items():
                    require(sha(folder / name) == digest, "Local sealed artifact changed")
            artifacts.append({"model": model, "seed": seed, "n_epochs": len(history), "best_epoch": int(best["epoch"]),
                              "best_val_rmse": float(best["val_rmse"]),
                              "sha256": {name: sha(folder / name) for name in ("predictions.npz", "best.pt", "config.json", history_path.name)}})
    seal = verify_global_seal(experiment, required=args.require_seal)
    result = {"status": "passed", "n_models": len(artifacts), "artifacts": artifacts, "global_seal": seal,
              "target_expression_opened": False, "source_data_review": report}
    save(args.output, result)
    return result


REFERENCE_MODELS = ("no_edit", "literal_rule", "lexa_pure_mechanism", "expression_nearest", "expression_interpolate", "lexa_response_retrieval")
METRICS = ("delta_rmse", "delta_pearson", "absolute_rmse", "absolute_r2", "response_l2_ratio", "top100_delta_rmse", "top100_direction_accuracy")


def historical_reference_review(experiment, project_root, cases, genes):
    experiment = Path(experiment)
    provenance = read(experiment / "references/provenance.json")
    for name, digest in provenance["sources"].items():
        require(sha(name) == digest, "Historical source changed: " + name)
    previous = Path(project_root) / "runs/pilot_seen_dose"
    v4 = previous / "experiment_v4_pure_mechanism_fixed"
    v3 = previous / "experiment_v3_hierarchy_fixed"
    maps = {base: {row["case_id"]: row for row in read(base / "inputs/manifest.json")["cases"]} for base in (v3, v4)}
    seals = {base: read(base / "prediction_seal.json") for base in (v3, v4)}
    expected = {name: np.zeros((16, 2000), dtype=float) for name in REFERENCE_MODELS}
    checked = set()

    def load(base, path):
        path = Path(path).resolve()
        relative = str(path.relative_to(base.resolve()))
        require(seals[base]["files"].get(relative) == sha(path), "Historical prediction differs from original seal: " + relative)
        checked.add(str(path))
        with np.load(path, allow_pickle=False) as archive:
            require(np.array_equal(archive["genes"].astype(str), genes), "Historical vocabulary differs")
            return {key: archive[key].copy() for key in archive.files}

    for i, case in enumerate(cases):
        cid = case["case_id"]
        row4, row3 = maps[v4][cid], maps[v3][cid]
        literal = load(v4, v4 / "baselines" / row4["slug"] / "literal_keyword.npz")
        full = load(v4, v4 / "predictions/full" / row4["slug"] / "joint.npz")
        for name, archive in (("literal_rule", literal), ("lexa_pure_mechanism", full)):
            index = np.flatnonzero(np.isclose(archive["alphas"], 0.3))
            require(len(index) == 1, "Historical main alpha differs")
            expected[name][i] = archive["delta_by_alpha"][index[0]]
        inputs = load(v3, row3["npz"])
        expected["expression_nearest"][i] = inputs["nearest_observed_delta"]
        expected["expression_interpolate"][i] = inputs["interpolate_observed_delta"]
        retrieval = load(v3, v3 / "predictions" / row3["slug"] / "joint.npz")
        expected["lexa_response_retrieval"][i] = retrieval["delta_mean"]
    with np.load(experiment / "references/predictions.npz", allow_pickle=False) as archive:
        require(archive["case_ids"].astype(str).tolist() == [case["case_id"] for case in cases], "Packed reference case order differs")
        require(np.array_equal(archive["genes"].astype(str), genes), "Packed reference vocabulary differs")
        names = archive["model_names"].astype(str).tolist()
        require(tuple(names) == REFERENCE_MODELS, "Packed historical model population differs")
        require(archive["delta"].shape == (6, 16, 2000), "Packed reference dimensions differ")
        for i, name in enumerate(names):
            np.testing.assert_array_equal(archive["delta"][i], expected[name], err_msg="Reference delta differs: " + name)
    return {"status": "passed", "n_models": 6, "n_original_sealed_numeric_files_checked": len(checked),
            "packed_sha256": sha(experiment / "references/predictions.npz")}, expected


def references_stage(args):
    protocol = read(Path(args.experiment) / "protocol.json")
    cases = read(Path(args.experiment) / "data/prediction_cases.json")["cases"]
    with np.load(Path(args.prepared) / "prepared.npz", allow_pickle=False) as archive:
        genes = archive["genes"].astype(str)
    project_root = Path(args.prepared).resolve().parents[2]
    review, _ = historical_reference_review(args.experiment, project_root, cases, genes)
    require(protocol["prediction_cases_sha256"] == sha(Path(args.experiment) / "data/prediction_cases.json"), "Input manifest differs")
    review["target_expression_opened"] = False
    save(args.output, review)
    return review


def fresh_metrics(delta, actual_treatment, actual_control):
    delta = np.asarray(delta, dtype=np.float64)
    treatment = np.asarray(actual_treatment, dtype=np.float64)
    control = np.asarray(actual_control, dtype=np.float64)
    truth = treatment - control
    residual = delta - truth
    numerator = float(np.dot(residual, residual))
    pcentered = delta - delta.mean()
    ycentered = truth - truth.mean()
    pstd = float(np.sqrt(np.dot(pcentered, pcentered) / len(delta)))
    ystd = float(np.sqrt(np.dot(ycentered, ycentered) / len(delta)))
    corr = float(np.dot(pcentered, ycentered) / np.sqrt(np.dot(pcentered, pcentered) * np.dot(ycentered, ycentered))) if pstd > 1e-12 and ystd > 1e-12 else np.nan
    tcentered = treatment - treatment.mean()
    denominator = float(np.dot(tcentered, tcentered))
    top = np.argsort(-np.abs(truth))[:100]
    truth_norm_squared = float(np.dot(truth, truth))
    return {
        "delta_rmse": float(np.sqrt(numerator / len(delta))),
        "delta_pearson": corr,
        "absolute_rmse": float(np.sqrt(numerator / len(delta))),
        "absolute_r2": float(1 - numerator / denominator) if denominator > 1e-15 else np.nan,
        "response_l2_ratio": float(np.sqrt(np.dot(delta, delta) / truth_norm_squared)) if truth_norm_squared > 1e-24 else np.nan,
        "top100_delta_rmse": float(np.sqrt(np.dot(residual[top], residual[top]) / len(top))),
        "top100_direction_accuracy": float(np.count_nonzero(np.sign(delta[top]) == np.sign(truth[top])) / len(top)),
    }


def compare_table(path, expected, keys, columns):
    actual = pd.read_csv(path)
    require(not actual.duplicated(keys).any(), "Duplicate table keys: " + str(path))
    require(len(actual) == len(expected), "Table row count differs: " + str(path))
    left = actual.set_index(keys).sort_index()
    right = expected.set_index(keys).sort_index()
    require(left.index.equals(right.index), "Table identity population differs: " + str(path))
    errors = {}
    for column in columns:
        x = left[column].to_numpy(dtype=float)
        y = right[column].to_numpy(dtype=float)
        np.testing.assert_allclose(x, y, atol=2e-10, rtol=2e-9, equal_nan=True, err_msg=path.name + ": " + column)
        finite = np.isfinite(x) & np.isfinite(y)
        errors[column] = float(np.max(np.abs(x[finite] - y[finite]))) if finite.any() else 0.0
    return errors


def score_stage(args):
    experiment = Path(args.experiment).resolve()
    # No biological target expression is opened before both pre-score sealing
    # and completed scoring are independently verified.
    seal = verify_global_seal(experiment, required=True)
    manifest = read(experiment / "score_manifest.json")
    require(manifest["status"] == "complete", "Main scoring is incomplete")
    require(manifest["prediction_seal_sha256"] == seal["sha256"], "Scoring references a different prediction seal")
    for name, digest in manifest["outputs_sha256"].items():
        require(sha(experiment / name) == digest, "Scored output changed: " + name)
    protocol = read(experiment / "protocol.json")
    cases = read(experiment / "data/prediction_cases.json")["cases"]
    _, prepared, original_cases = source_data_review(args.prepared)
    genes = prepared["genes"].astype(str)
    project_root = Path(args.prepared).resolve().parents[2]
    reference_review, references = historical_reference_review(experiment, project_root, cases, genes)
    actual_evaluation = Path(args.prepared) / "evaluation.npz"
    require(protocol["evaluation_archive_sha256"] == sha(actual_evaluation) == manifest["evaluation_sha256"], "Evaluation source changed")
    # First target-array access by this score-stage invocation occurs here.
    with np.load(actual_evaluation, allow_pickle=False) as archive:
        evaluation = {key: archive[key].copy() for key in archive.files}
    require(np.array_equal(evaluation["genes"].astype(str), genes), "Evaluation gene order differs")
    originals = {case["case_id"]: case for case in original_cases}
    cache = {}
    for model in protocol["models"]:
        for seed in protocol["training_seeds"]:
            with np.load(experiment / "models" / model / ("seed_" + str(seed)) / "predictions.npz", allow_pickle=False) as archive:
                cache[model, seed] = (archive["pred_treated_mean"].astype(float), archive["pred_control_mean"].astype(float))
    rows = []
    used_treated, used_controls = set(), set()
    for i, case in enumerate(cases):
        cid = case["case_id"]
        treatment_mask = evaluation["test_case_ids"].astype(str) == cid
        control_mask = np.isin(evaluation["control_cell_ids"].astype(str), case["control_cell_ids"])
        treated_ids = evaluation["test_cell_ids"][treatment_mask].astype(str)
        controls = evaluation["control_cell_ids"][control_mask].astype(str)
        require(set(treated_ids) == set(originals[cid]["treated_cell_ids"]), "Target cell identities changed")
        require(set(controls) == set(case["control_cell_ids"]), "Conditioning cell identities changed")
        require(not set(treated_ids) & used_treated, "Target cell reused across cases")
        used_treated.update(treated_ids)
        used_controls.update(controls)
        target = evaluation["x_test"][treatment_mask].astype(float).mean(0)
        control = evaluation["control_x"][control_mask].astype(float).mean(0)
        candidates = []
        for (model, seed), (treated_mean, zero_mean) in cache.items():
            candidates.append((model, seed, "native", treated_mean[i] - control))
            candidates.append((model, seed, "calibrated", treated_mean[i] - zero_mean[i]))
        candidates.extend((name, -1, "existing", values[i]) for name, values in references.items())
        for model, seed, view, delta in candidates:
            rows.append({"model": model, "seed": seed, "view": view, "case_id": cid, "drug": case["drug"],
                         "n_treated": len(treated_ids), "n_control": len(controls), **fresh_metrics(delta, target, control)})
    require(len(used_treated) == 786 and len(used_controls) == 951, "Final scored population differs")
    wells = pd.DataFrame(rows)
    require(len(wells) == 288, "Expected192 new-model and96 historical well rows")
    by_drug = wells.groupby(["model", "seed", "view", "drug"], as_index=False)[list(METRICS)].mean()
    by_seed = by_drug.groupby(["model", "seed", "view"], as_index=False)[list(METRICS)].mean()
    coverage = wells.groupby(["model", "seed", "view"])["delta_pearson"].count().rename("delta_pearson_defined_wells").reset_index()
    by_seed = by_seed.merge(coverage, on=["model", "seed", "view"])
    by_seed["n_wells"] = 16
    summary = by_seed.groupby(["model", "view"])[list(METRICS)].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(column).rstrip("_") if isinstance(column, tuple) else column for column in summary.columns]
    counts = by_seed.groupby(["model", "view"]).agg(n_seeds=("seed", "size"),
        delta_pearson_defined_wells_min=("delta_pearson_defined_wells", "min"),
        delta_pearson_defined_wells_max=("delta_pearson_defined_wells", "max")).reset_index()
    summary = summary.merge(counts, on=["model", "view"])
    errors = {
        "scores_by_well.csv": compare_table(experiment / "scores_by_well.csv", wells, ["model", "seed", "view", "case_id"], list(METRICS) + ["n_treated", "n_control"]),
        "scores_by_drug.csv": compare_table(experiment / "scores_by_drug.csv", by_drug, ["model", "seed", "view", "drug"], METRICS),
        "scores_by_seed.csv": compare_table(experiment / "scores_by_seed.csv", by_seed, ["model", "seed", "view"], list(METRICS) + ["delta_pearson_defined_wells", "n_wells"]),
        "scores_summary.csv": compare_table(experiment / "scores_summary.csv", summary, ["model", "view"], [column for column in summary.columns if column not in ("model", "view")]),
    }
    result = {"status": "passed", "prediction_seal_sha256": seal["sha256"], "n_well_rows": 288, "n_drug_rows": len(by_drug),
              "n_seed_rows": len(by_seed), "n_summary_rows": len(summary), "n_treated": len(used_treated),
              "n_unique_controls": len(used_controls), "reference_review": reference_review, "max_abs_error_by_table": errors,
              "correlation_defined_wells": by_seed[["model", "seed", "view", "delta_pearson_defined_wells"]].to_dict(orient="records"),
              "target_expression_opened": True, "checked_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}
    save(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("data", "models", "references", "score"))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--prediction-cases", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--require-seal", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage in ("models", "references", "score"):
        require(args.experiment is not None, "--experiment is required for model audit")
        result = {"models": models_stage, "references": references_stage, "score": score_stage}[args.stage](args)
    else:
        result = data_stage(args)
    print(json.dumps({"status": result["status"], "n_cases": result.get("n_ready_cases", 16), "target_expression_opened": result["target_expression_opened"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
