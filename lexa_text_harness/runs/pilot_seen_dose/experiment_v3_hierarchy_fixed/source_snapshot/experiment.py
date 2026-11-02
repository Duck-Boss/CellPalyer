"""Build blinded text requests, run agent ablations, and score held-out groups."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from agent import ChatCompletionsClient, SYSTEM_PROMPT, run_agent
from tools import Workspace, sha256


DRUG_TEXT = {
    "Dex": "Dexamethasone is a glucocorticoid receptor agonist. Responses depend on the starting cell state, dose and exposure time.",
    "Nutlin": "Nutlin-3a antagonizes the p53-MDM2 interaction. Downstream response depends on the functional p53 pathway and cellular context.",
    "BMS": "BMS-345541 inhibits IKK/NF-kappaB-dependent signaling. Predict the transcriptional response in the supplied cell context.",
    "SAHA": "Vorinostat (SAHA) is a histone deacetylase inhibitor. Transcriptional and stress responses depend on dose and cellular context.",
}


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _input_provenance(prepared, fitted):
    prepared, fitted = Path(prepared), Path(fitted)
    return {"effects_sha256": sha256(fitted / "exact_code_effects.npz"),
            "prepared_manifest_sha256": sha256(prepared / "manifest.json"),
            "prepared_cases_sha256": sha256(prepared / "cases.json"),
            "fitted_baseline_sha256": sha256(fitted / "baseline.npz"),
            "prepared_source": str(prepared.resolve()), "fitted_source": str(fitted.resolve())}


def _check_unchanged_provenance(previous, proposed):
    for key, value in proposed.items():
        # Older manifests did not record every field; existing fields remain binding.
        if key in previous and previous[key] != value:
            raise ValueError(f"Existing run has different {key}; choose a new run output")


def feedback_statistics(trace):
    """Distinguish feedback-based revisions from same-response proposals."""
    rounds = [event["round"] for event in trace.get("events", [])
              if event.get("tool") == "evaluate_edits" and event.get("success")]
    first_round = min(rounds) if rounds else None
    return {"successful_evaluation_api_rounds": rounds,
            "evaluations_after_prior_round_feedback": sum(r > first_round for r in rounds) if rounds else 0,
            "distinct_evaluation_api_rounds": len(set(rounds)),
            "same_round_extra_evaluations": len(rounds) - len(set(rounds))}


def selected_cases(prepared, limit):
    value = json.loads((Path(prepared) / "cases.json").read_text())
    cases = [c for c in value["cases"] if c["status"] == "ready"]
    # Round-robin drugs so a four-case pilot includes all four mechanisms.
    groups = {drug: sorted([c for c in cases if c["drug"] == drug], key=lambda c: c["case_id"])
              for drug in sorted({c["drug"] for c in cases})}
    result = []
    for i in range(max((len(v) for v in groups.values()), default=0)):
        result.extend(group[i] for group in groups.values() if i < len(group))
    return result[:limit] if limit else result


def request_for_case(case, labels_path=None):
    value = {
        "mode": "perturbation_prediction",
        "text": DRUG_TEXT.get(case["drug"], case["drug"]),
        "cell_context": "A549 human lung adenocarcinoma cell line",
        "drug": case["drug"], "dose": {"value": case["dose_uM"], "unit": "uM"},
        "exposure": {"value": 24, "unit": "hours"},
        "solvent": "ethanol" if case["drug"] == "Dex" else "DMSO",
        "task": "Predict a small, mechanistically plausible change in code-state occupancy relative to the matched zero-dose control. You may abstain with no edits if evidence is inadequate.",
        "important": "There is no measured treated expression in your tools. Decoder feedback checks the hypothesis, not treatment accuracy. State fraction is a hypothesis, not a calibrated dose model.",
    }
    if labels_path:
        value["state_labels_file"] = str(Path(labels_path).resolve())
    return value


def build_requests(prepared, fitted, output, labels_path=None, limit=4):
    output, fitted = Path(output), Path(fitted)
    provenance = _input_provenance(prepared, fitted)
    selected = selected_cases(prepared, limit)
    if not selected:
        raise ValueError("No held-out cases meet the minimum sample requirement")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        _check_unchanged_provenance(previous, provenance)
        previous_ids = {case["case_id"] for case in previous["cases"]}
        if not previous_ids.issubset({case["case_id"] for case in selected}):
            raise ValueError("Existing case population would shrink or change; choose a new run output")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(fitted / "baseline.npz", allow_pickle=False) as payload:
        baseline = {k: payload[k].copy() for k in payload.files}
    ids = baseline["cell_ids"].astype(str)
    lookup = {value: i for i, value in enumerate(ids)}
    manifest = []
    for case in selected:
        slug = hashlib.sha256(case["case_id"].encode()).hexdigest()[:12]
        folder = output / slug
        folder.mkdir(exist_ok=True)
        positions = np.array([lookup[value] for value in case["control_cell_ids"]], dtype=int)
        # All legitimate matched control cells are used; there is no cell-level pairing.
        np.savez_compressed(folder / "baseline.npz", codes=baseline["codes"][positions],
                            genes=baseline["genes"], cell_ids=ids[positions], synthetic=False)
        request = request_for_case(case, labels_path)
        save_json(folder / "request.json", request)
        manifest.append({"case_id": case["case_id"], "slug": slug, "drug": case["drug"],
                         "dose_uM": case["dose_uM"], "n_control": len(positions),
                         "baseline": str((folder / "baseline.npz").resolve()),
                         "request": str((folder / "request.json").resolve())})
    save_json(output / "manifest.json", {**provenance,
                                         "cases": manifest, "case_limit": limit,
                                         "note": "This manifest contains inputs only, no observed treatment expression."})
    return manifest


def shuffled_labels(source, destination):
    value = json.loads(Path(source).read_text())
    keys = sorted(value["labels"])
    rng = np.random.default_rng(20261026)
    original = value["labels"]
    swapped = {key: dict(label) for key, label in original.items()}
    for axis in sorted({key.split('_C')[0] for key in keys}):
        local = [key for key in keys if key.startswith(axis + '_C')]
        order = rng.permutation(len(local))
        for i, key in enumerate(local):
            for field in ("name", "summary"):
                swapped[key][field] = original[local[int(order[i])]][field]
    value["labels"] = swapped
    value["negative_control"] = "Only name/summary are shuffled within each axis; identifiers, gene references and exact numerical evidence remain attached to their original states."
    save_json(destination, value)
    return destination


def run_methods(prepared, fitted, output, methods, client, limit=4, max_rounds=8):
    fitted, output = Path(fitted), Path(output)
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("Methods must be a nonempty list without duplicates")
    plan = {"methods": methods, "case_limit": limit, "max_rounds": max_rounds,
            **_input_provenance(prepared, fitted)}
    plan_path = output / "run_plan.json"
    if plan_path.exists():
        previous = json.loads(plan_path.read_text())
        _check_unchanged_provenance(previous, _input_provenance(prepared, fitted))
        if previous["methods"] != methods or previous["max_rounds"] != max_rounds:
            raise ValueError("Existing methods or API budget changed; choose a new run output")
    labels = fitted / "labels.json"
    if any(method in methods for method in ("one_shot", "agent", "shuffled_labels")):
        if not labels.exists() or not json.loads(labels.read_text()).get("meta", {}).get("complete"):
            raise ValueError("Complete language labels are required; run annotate first or choose agent_no_labels")
    cases = build_requests(prepared, fitted, output / "inputs", labels if labels.exists() else None, limit)
    if "shuffled_labels" in methods:
        shuffled_labels(labels, output / "shuffled_labels.json")
    save_json(plan_path, plan)
    completed = []
    for method in methods:
        for case in cases:
            folder = output / method / case["slug"]
            folder.mkdir(parents=True, exist_ok=True)
            request = json.loads(Path(case["request"]).read_text())
            if method == "agent_no_labels":
                request.pop("state_labels_file", None)
            elif method == "shuffled_labels":
                request["state_labels_file"] = str((output / "shuffled_labels.json").resolve())
            request_path = folder / "request.json"
            save_json(request_path, request)
            result_path = folder / "agent_result.json"
            fingerprint = {"effects_sha256": sha256(fitted / "exact_code_effects.npz"),
                           "baseline_sha256": sha256(case["baseline"]),
                           "request_sha256": sha256(request_path), "method": method,
                           "state_labels_sha256": sha256(request["state_labels_file"]) if request.get("state_labels_file") else None,
                           "model": getattr(client, "model", None),
                           "endpoint": getattr(client, "url", None), "max_rounds": max_rounds,
                           "max_evaluations": 1 if method == "one_shot" else 3,
                           "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                           "implementation_sha256": {name: sha256(Path(__file__).parent / name) for name in ["agent.py", "tools.py", "experiment.py"]}}
            if result_path.exists():
                old = json.loads(result_path.read_text())
                if old.get("fingerprint") != fingerprint:
                    raise ValueError("Existing result belongs to different inputs/model/budget; choose a new run output")
                if old.get("decoder_evaluated") and old.get("status") != "transport_error" and old.get("evaluation") and Path(old["evaluation"].get("artifact", "")).is_file():
                    completed.append(str(result_path))
                    continue
            # A new attempt keeps orphaned evaluations and failed API traces reviewable.
            attempt_number = len(list((folder / "attempts").glob("*"))) + 1 if (folder / "attempts").exists() else 1
            attempt = folder / "attempts" / f"{attempt_number:03d}"
            attempt.mkdir(parents=True, exist_ok=False)
            workspace = Workspace(str(fitted / "exact_code_effects.npz"), case["baseline"],
                                  str(request_path), "mse_log1p", str(attempt))
            result = run_agent(workspace, client, max_rounds=max_rounds,
                               max_evaluations=1 if method == "one_shot" else 3)
            trace = result.pop("trace")
            result.update(case_id=case["case_id"], method=method, fingerprint=fingerprint,
                          attempt_dir=str(attempt.resolve()), feedback_statistics=feedback_statistics(trace))
            if result.get("evaluation"):
                result["evaluation"]["artifact_sha256"] = sha256(result["evaluation"]["artifact"])
            save_json(attempt / "agent_trace.json", trace)
            save_json(attempt / "agent_result.json", result)
            save_json(folder / "agent_trace.json", trace)
            save_json(result_path, result)
            completed.append(str(result_path))
            print(json.dumps({"method": method, "case": case["case_id"],
                              "status": result["status"], "evaluated": result["decoder_evaluated"]}), flush=True)
            if result["status"] == "transport_error":
                raise RuntimeError("API transport failed; finished results were preserved")
            if not result["decoder_evaluated"]:
                raise RuntimeError("No proposal was evaluated; this case is incomplete. Review the trace before retrying.")
    return completed


def safe_correlation(a, b):
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def metrics(predicted, observed):
    take = min(100, len(observed))
    truth_idx = np.argsort(-np.abs(observed))[:take]
    predicted_idx = set(np.argsort(-np.abs(predicted))[:take]) if np.any(np.abs(predicted) > 1e-8) else set()
    active = np.abs(observed[truth_idx]) > 1e-8
    return {"delta_rmse": float(np.sqrt(np.mean(np.square(predicted - observed)))),
            "delta_pearson": safe_correlation(predicted, observed),
            "top100_delta_rmse": float(np.sqrt(np.mean(np.square(predicted[truth_idx] - observed[truth_idx])))),
            "top100_direction_accuracy": float(np.mean(np.sign(predicted[truth_idx][active]) == np.sign(observed[truth_idx][active]))) if active.any() else None,
            "top100_abs_delta_overlap": len(predicted_idx.intersection(truth_idx.tolist())) / take,
            "note": "Top100 denotes largest absolute observed mean shifts, not statistical DE calls."}


def training_deltas(prepared):
    with np.load(Path(prepared) / "prepared.npz", allow_pickle=False) as payload:
        x = payload["x_train"]
        ids = payload["train_cell_ids"].astype(str)
    metadata = pd.read_csv(Path(prepared) / "metadata.csv", index_col="cell_id").loc[ids]
    responses = []
    for (drug, dose), rows in metadata.groupby(["drug", "dose_uM"]):
        if dose <= 0:
            continue
        well_deltas = []
        for (plate, replicate), well in rows.groupby(["plate", "replicate"]):
            shared = (metadata["drug"].to_numpy() == drug) & (metadata["plate"].to_numpy() == plate) & (metadata["replicate"].to_numpy() == replicate)
            treatment = shared & (metadata["dose_uM"].to_numpy() == dose)
            control = shared & (metadata["dose_uM"].to_numpy() == 0)
            if treatment.any() and control.any():
                well_deltas.append(x[treatment].mean(axis=0) - x[control].mean(axis=0))
        if well_deltas:
            responses.append((drug, float(dose), np.stack(well_deltas).mean(axis=0)))
    return responses


def _score_preflight(prepared, output):
    """Validate the entire frozen prediction population before opening targets."""
    manifest = json.loads((output / "inputs/manifest.json").read_text())
    plan = json.loads((output / "run_plan.json").read_text())
    for key in ("effects_sha256", "prepared_manifest_sha256"):
        if plan.get(key) != manifest.get(key):
            raise ValueError("Prediction plan provenance differs from input manifest")
    current = {"prepared_manifest_sha256": sha256(prepared / "manifest.json"),
               "prepared_cases_sha256": sha256(prepared / "cases.json"),
               "prepared_source": str(prepared.resolve())}
    if manifest.get("fitted_source"):
        current = _input_provenance(prepared, manifest["fitted_source"])
    _check_unchanged_provenance(manifest, current)
    _check_unchanged_provenance(plan, current)
    _check_unchanged_provenance(plan, {key: manifest[key] for key in current if key in manifest})
    if not plan.get("methods") or len(plan["methods"]) != len(set(plan["methods"])):
        raise ValueError("Prediction plan must contain distinct methods")
    case_ids = [case["case_id"] for case in manifest["cases"]]
    planned_ids = [case["case_id"] for case in selected_cases(prepared, plan["case_limit"])]
    if not case_ids or len(case_ids) != len(set(case_ids)) or case_ids != planned_ids:
        raise ValueError("Prediction population differs from the frozen selection rule")
    predictions, diagnostics = {}, []
    for case in manifest["cases"]:
        predictions[case["case_id"]] = {}
        for method in plan["methods"]:
            result_path = output / method / case["slug"] / "agent_result.json"
            if not result_path.is_file():
                raise ValueError("A planned method/case has no result; refusing to open held-out targets")
            result = json.loads(result_path.read_text())
            fingerprint = result.get("fingerprint", {})
            if result.get("case_id") != case["case_id"] or result.get("method") != method or fingerprint.get("method") != method:
                raise ValueError("Prediction case or method provenance differs")
            expected = {"effects_sha256": manifest["effects_sha256"],
                        "baseline_sha256": sha256(case["baseline"]),
                        "request_sha256": sha256(result_path.parent / "request.json")}
            if any(fingerprint.get(key) != value for key, value in expected.items()):
                raise ValueError("Prediction effects, baseline or request content changed")
            request = json.loads((result_path.parent / "request.json").read_text())
            labels_hash = sha256(request["state_labels_file"]) if request.get("state_labels_file") else None
            if fingerprint.get("state_labels_sha256") != labels_hash:
                raise ValueError("Prediction language labels changed")
            evaluation = result.get("evaluation")
            if not result.get("decoder_evaluated") or result.get("status") == "transport_error" or not evaluation:
                raise ValueError("A planned result was not numerically evaluated; refusing to open held-out targets")
            if any(evaluation.get("provenance", {}).get(key) != value for key, value in expected.items()):
                raise ValueError("Numerical evaluation provenance differs from its prediction")
            artifact = Path(evaluation.get("artifact", ""))
            if not artifact.is_file() or evaluation.get("artifact_sha256") != sha256(artifact):
                raise ValueError("Prediction artifact is missing or its checksum differs")
            with np.load(artifact, allow_pickle=False) as payload:
                genes, delta = payload["genes"].astype(str), payload["delta_mean"].copy()
                if delta.ndim != 1 or delta.shape != genes.shape or not len(delta) or not np.isfinite(delta).all():
                    raise ValueError("Prediction artifact contains an invalid numerical delta")
                if "synthetic" in payload and bool(payload["synthetic"].item()):
                    raise ValueError("Synthetic predictions cannot be scored as real experiments")
            predictions[case["case_id"]][method] = {"genes": genes, "delta": delta}
            feedback = result.get("feedback_statistics", {})
            diagnostics.append({"case_id": case["case_id"], "drug": case["drug"], "method": method,
                                "status": result.get("status"), "api_requests": result.get("api_requests"),
                                "evaluations": result.get("evaluations"), "tool_calls": result.get("tool_calls"),
                                **{key: feedback.get(key) for key in ["evaluations_after_prior_round_feedback",
                                   "distinct_evaluation_api_rounds", "same_round_extra_evaluations"]}})
    return manifest, plan, predictions, diagnostics


def score(prepared, run_output):
    prepared, output = Path(prepared), Path(run_output)
    manifest, plan, cached_predictions, diagnostics = _score_preflight(prepared, output)
    # This is the first held-out expression access, after every planned prediction passed preflight.
    with np.load(prepared / "evaluation.npz", allow_pickle=False) as payload:
        data = {key: payload[key].copy() for key in payload.files}
    all_cases = {case["case_id"]: case for case in json.loads((prepared / "cases.json").read_text())["cases"]}
    responses = training_deltas(prepared)
    rows = []
    for case in manifest["cases"]:
        detail = all_cases[case["case_id"]]
        treatment = data["x_test"][data["test_case_ids"].astype(str) == case["case_id"]]
        control = data["control_x"][np.isin(data["control_cell_ids"].astype(str), detail["control_cell_ids"])]
        if not len(treatment) or not len(control):
            raise ValueError("Missing held-out treatment or matched controls")
        observed = treatment.mean(axis=0) - control.mean(axis=0)
        predictions = {"no_edit": np.zeros_like(observed)}
        if responses:
            predictions["training_mean"] = np.stack([value for _, _, value in responses]).mean(axis=0)
            available = [(dose, value) for drug, dose, value in responses if drug == case["drug"]]
            if available:
                nearest = min(available, key=lambda pair: abs(np.log1p(pair[0]) - np.log1p(case["dose_uM"])))
                predictions["nearest_training_dose"] = nearest[1]
        for method in plan["methods"]:
            prediction = cached_predictions[case["case_id"]][method]
            if not np.array_equal(prediction["genes"], data["genes"].astype(str)):
                raise ValueError("Prediction and evaluator gene orders differ")
            predictions[method] = prediction["delta"]
        for method, prediction in predictions.items():
            rows.append({"case_id": case["case_id"], "drug": case["drug"], "dose_uM": case["dose_uM"],
                         "method": method, "n_treated": len(treatment), "n_control": len(control),
                         **metrics(prediction, observed)})
    table = pd.DataFrame(rows)
    table.to_csv(output / "scores_by_well.csv", index=False)
    columns = ["delta_rmse", "delta_pearson", "top100_delta_rmse", "top100_direction_accuracy", "top100_abs_delta_overlap"]
    condition = table.groupby(["method", "drug", "dose_uM"], as_index=False)[columns].mean()
    condition.to_csv(output / "scores_by_condition.csv", index=False)
    summary = condition.groupby("method", as_index=False)[columns].mean()
    summary.to_csv(output / "scores_summary.csv", index=False)
    diagnostic_table = pd.DataFrame(diagnostics)
    diagnostic_table.to_csv(output / "run_diagnostics.csv", index=False)
    diagnostic_columns = ["api_requests", "evaluations", "tool_calls", "evaluations_after_prior_round_feedback",
                          "distinct_evaluation_api_rounds", "same_round_extra_evaluations"]
    diagnostic_table.groupby("method", as_index=False)[diagnostic_columns].mean().to_csv(
        output / "run_diagnostics_summary.csv", index=False)
    save_json(output / "evaluation_scope.json", {
        "selected_cases": len(manifest["cases"]), "reported_methods": sorted(table["method"].unique()),
        "evaluated_gene_count": len(data["genes"]),
        "gene_population": "Training-selected feature vocabulary only; not all genes in the raw assay.",
        "evaluation_sha256": sha256(prepared / "evaluation.npz"),
        "aggregation": "Equal well means within drug-dose condition, then equal condition means.",
        "limitation": "Pilot point estimates; correlated culture replicates and selected subset do not establish general biological efficacy. No hypothesis tests or cell-level independent confidence intervals.",
    })
    return summary.to_dict(orient="records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["requests", "agents", "score"])
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--fitted", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-limit", type=int, default=4)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"))
    parser.add_argument("--base-url", default=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    parser.add_argument("--api-key-file")
    parser.add_argument("--methods", default="one_shot,agent")
    parser.add_argument("--max-rounds", type=int, default=8)
    args = parser.parse_args()
    if args.case_limit < 0:
        parser.error("case-limit must be nonnegative; zero means all ready cases")
    if args.stage == "requests":
        labels = Path(args.fitted) / "labels.json"
        build_requests(args.prepared, args.fitted, Path(args.output) / "inputs", labels if labels.exists() else None, args.case_limit)
    elif args.stage == "score":
        print(json.dumps(score(args.prepared, args.output), ensure_ascii=False))
    else:
        key = Path(args.api_key_file).read_text().strip() if args.api_key_file else os.environ.get("DEEPSEEK_API_KEY", "")
        methods = args.methods.split(",")
        if set(methods) - {"one_shot", "agent", "agent_no_labels", "shuffled_labels"}:
            parser.error("Unsupported method")
        client = ChatCompletionsClient(args.model, key, args.base_url)
        run_methods(args.prepared, args.fitted, args.output, methods, client, args.case_limit, args.max_rounds)


if __name__ == "__main__":
    main()
