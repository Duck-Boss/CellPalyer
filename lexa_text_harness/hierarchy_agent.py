"""Hierarchical text-only occupancy candidate selection with a joint review.

Prediction inputs contain training response priors and matched untreated cells.
This module never reads treatment evaluation targets. Numeric feedback describes
the frozen decoder and training priors, not prediction accuracy or causality.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np

from agent import TransportError
from annotate_cards import atomic_write_json, file_sha256


PROMPT_VERSION = "lexa-hierarchy-occupancy-v1"
LOCAL_BATCH_SIZE = 8
SYSTEM_PROMPT = """You predict a population response using a frozen LEXA codec.
Use the supplied hierarchy: axis functional scope, signed state descriptions,
then exact candidate changes relative to the matched control population. Axis
and state annotations are evidence summaries, not causal truth. State IDs are
unordered categories. Every candidate is a 16-state population occupancy
distribution, not a paired transition of the same biological cell.
The seven candidates are keep, nearest, interpolate, lower, upper, half, onehalf.
They come from training-only matched responses and fixed dose interpolation;
some may coincide. The supplied interpolation is a useful starting hypothesis,
not a ground-truth answer. Choose another candidate only with a stated reason
grounded in the displayed training, gene-effect or context evidence. Drug
mechanism text alone does not calibrate dose response. Uncertainty does not
automatically imply zero response; keep is available when evidence supports it.
All 32 axes can change. Do not impose a three-axis limit. Shared genes may
reinforce or cancel across axes; additive decoding does not imply biological or
statistical independence. Joint-support diagnostics are only heuristics.
Do not invent states, probabilities, effects, evidence references or unseen
treatment outcomes. Select exactly one displayed candidate for every requested
axis, with no omitted, duplicated or extra axes. Use at least one supplied
evidence_ref per choice. Brief reasons may state uncertainty or retain a prior.
Return only JSON, with exactly these fields:
{"selections":[{"axis_id":0,"candidate_id":"interpolate","reason":"brief evidence-based reason","evidence_refs":["A0/interpolate"]}],"summary":"brief result","limitations":["brief limitation"]}
Keep each reason under 80 characters and prefer one evidence_ref per selection,
summary under 400 characters and at most three short limitations. Return no
hidden reasoning or external text. All requested axes must fit the response.
"""


class SelectionError(ValueError):
    """Local, credential-free output validation failure."""


class StageFailure(RuntimeError):
    """An API stage exhausted its predeclared attempt budget."""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _array_digest(value: Any) -> str:
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(_json(list(a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _candidate_ids(bundle: dict) -> list[str]:
    values = [str(v) for v in bundle["candidate_ids"]]
    if not values or len(set(values)) != len(values) or "keep" not in values:
        raise ValueError("Candidate IDs must be unique and contain keep")
    return values


def evidence_refs(axis: int, candidates: list[str], states: int = 16) -> set[str]:
    return {f"A{axis}", *(f"A{axis}/{name}" for name in candidates),
            *(f"A{axis}/C{s}" for s in range(states))}


def parse_selection(content: str, expected_axes: list[int], candidate_ids: list[str]) -> dict:
    if not isinstance(content, str) or not content.strip():
        raise SelectionError("Response must contain a JSON object")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"} or lines[-1].strip() != "```":
            raise SelectionError("Invalid JSON code fence")
        text = "\n".join(lines[1:-1])
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        raise SelectionError("Response must be one valid JSON object without surrounding prose") from None
    if not isinstance(value, dict) or set(value) != {"selections", "summary", "limitations"}:
        raise SelectionError("Required object fields: selections, summary, limitations only")
    selections = value["selections"]
    if not isinstance(selections, list) or len(selections) != len(expected_axes):
        raise SelectionError(f"Return exactly {len(expected_axes)} selections for axes {expected_axes}")
    if not isinstance(value["summary"], str) or not value["summary"].strip() or len(value["summary"]) > 1000:
        raise SelectionError("summary must be a nonempty string of at most 1000 characters")
    if not isinstance(value["limitations"], list) or not 1 <= len(value["limitations"]) <= 6 or any(not isinstance(v, str) or not v.strip() or len(v) > 500 for v in value["limitations"]):
        raise SelectionError("limitations must contain one to six short nonempty strings")
    seen = set()
    for row in selections:
        if not isinstance(row, dict) or set(row) != {"axis_id", "candidate_id", "reason", "evidence_refs"}:
            raise SelectionError("Each selection requires axis_id, candidate_id, reason, evidence_refs only")
        axis = row["axis_id"]
        if type(axis) is not int or axis not in expected_axes:
            raise SelectionError(f"axis_id must be an integer in {expected_axes}")
        if axis in seen:
            raise SelectionError(f"Duplicate axis_id {axis}; each requested axis must occur once")
        seen.add(axis)
        if not isinstance(row["candidate_id"], str) or row["candidate_id"] not in candidate_ids:
            raise SelectionError(f"Invalid candidate on axis {axis}; choose from {candidate_ids}")
        if not isinstance(row["reason"], str) or not row["reason"].strip() or len(row["reason"]) > 500:
            raise SelectionError(f"Axis {axis} reason must be a nonempty short string")
        refs = row["evidence_refs"]
        allowed = evidence_refs(axis, candidate_ids)
        if not isinstance(refs, list) or not 1 <= len(refs) <= 6 or any(not isinstance(ref, str) for ref in refs) or len(set(refs)) != len(refs) or not set(refs).issubset(allowed):
            raise SelectionError(f"Axis {axis} requires one to six unique supplied evidence_refs from that axis")
    if seen != set(expected_axes):
        raise SelectionError(f"Missing required axes {sorted(set(expected_axes) - seen)}")
    value["selections"] = sorted(selections, key=lambda r: r["axis_id"])
    return value


def _choices(value: dict | list) -> dict[int, str]:
    if isinstance(value, dict) and "selections" in value:
        value = value["selections"]
    if isinstance(value, list):
        result = {}
        for row in value:
            axis = row["axis_id"]
            if type(axis) is not int or axis in result:
                raise ValueError("Selection axes must be unique integers")
            result[axis] = row["candidate_id"]
        return result
    if not isinstance(value, dict) or any(type(a) is not int for a in value):
        raise ValueError("Selections must map integer axes to candidate IDs")
    return dict(value)


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / den) if den > 1e-12 else None


def _top_genes(bundle: dict, delta: np.ndarray, count: int = 10) -> dict:
    genes = np.asarray(bundle["genes"]).astype(str)
    symbols = np.asarray(bundle.get("symbols", genes)).astype(str)
    std = np.maximum(np.asarray(bundle["std"], dtype=float), .05)
    scaled = delta / std
    def rows(indices):
        return [{"gene": genes[i], "symbol": symbols[i], "delta": round(float(delta[i]), 6),
                 "scaled_delta": round(float(scaled[i]), 6)} for i in indices]
    up = np.flatnonzero(scaled > 0)
    down = np.flatnonzero(scaled < 0)
    return {"up": rows(up[np.argsort(-scaled[up], kind="stable")][:count]),
            "down": rows(down[np.argsort(scaled[down], kind="stable")][:count])}


def evaluate_selection(bundle: dict, selections: dict | list) -> dict:
    """Exactly decode a full candidate selection; no target treatment data."""
    choices = _choices(selections)
    p0 = np.asarray(bundle["p0"], dtype=float)
    candidates = np.asarray(bundle["candidates"], dtype=float)
    effects = np.asarray(bundle["effects"], dtype=float)
    names = _candidate_ids(bundle)
    axes, states = p0.shape
    if set(choices) != set(range(axes)) or any(v not in names for v in choices.values()):
        raise ValueError("Exactly one available candidate is required for each axis")
    if candidates.shape != (axes, len(names), states) or effects.shape[:2] != p0.shape:
        raise ValueError("Inconsistent candidate and decoder dimensions")
    index = np.array([names.index(choices[a]) for a in range(axes)])
    q = candidates[np.arange(axes), index].copy()
    for value in (p0, q):
        if not np.isfinite(value).all() or (value < -1e-10).any() or not np.allclose(value.sum(axis=1), 1, atol=1e-8):
            raise ValueError("Invalid occupancy probabilities")
    axis_delta = np.einsum("as,asg->ag", q - p0, effects)
    delta = axis_delta.sum(axis=0)
    std = np.maximum(np.asarray(bundle["std"], dtype=float), .05)
    tv = np.abs(q - p0).sum(axis=1) / 2
    feedback = {
        "definition": "Exact frozen-decoder delta relative to current control. Prior agreement is not treatment accuracy.",
        "delta_l2": float(np.linalg.norm(delta)),
        "standardized_delta_rms": float(np.sqrt(np.mean((delta / std) ** 2))),
        "mean_axis_tv_from_control": float(tv.mean()),
        "axes_tv_gt_0_01": int((tv > .01).sum()),
        "axes_tv_gt_0_1": int((tv > .1).sum()),
        "axis_tv": tv.tolist(),
        "top_gene_changes": _top_genes(bundle, delta),
        "candidate_counts": {name: sum(v == name for v in choices.values()) for name in names},
        "prior_comparisons": {},
    }
    for prior in ("nearest", "interpolate"):
        if prior in names:
            prior_q = candidates[:, names.index(prior)]
            prior_delta = np.einsum("as,asg->g", prior_q - p0, effects)
            norm = float(np.linalg.norm(prior_delta))
            feedback["prior_comparisons"][prior] = {
                "mean_occupancy_tv": float(np.abs(q - prior_q).sum(axis=1).mean() / 2),
                "gene_delta_cosine": _cosine(delta, prior_delta),
                "delta_l2_ratio": float(np.linalg.norm(delta) / norm) if norm > 1e-12 else None,
            }
    reference = bundle.get("reference_observed_delta")
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        norm = float(np.linalg.norm(reference))
        feedback["interpolated_training_expression_prior"] = {
            "definition": "Agreement with training-only interpolated response, not a measured response at the requested dose",
            "gene_delta_cosine": _cosine(delta, reference),
            "delta_l2_ratio": float(np.linalg.norm(delta) / norm) if norm > 1e-12 else None,
            "top_gene_changes": _top_genes(bundle, reference, 8),
        }
    # Cancellation is a property of summed effects, not a pathology by itself.
    signed_l1 = np.abs(axis_delta).sum(axis=0)
    cancellation = signed_l1 - np.abs(delta)
    idx = np.argsort(-(cancellation / std), kind="stable")[:8]
    feedback["largest_cross_axis_cancellation"] = [
        {"gene": str(bundle["genes"][g]), "symbol": str(bundle.get("symbols", bundle["genes"])[g]),
         "summed_delta": float(delta[g]), "absolute_axis_effect_sum": float(signed_l1[g])}
        for g in idx if cancellation[g] > 1e-8]
    if "projection_l1" in bundle:
        projection = np.asarray(bundle["projection_l1"], dtype=float)
        feedback["selected_projection_l1_mean"] = float(projection[np.arange(axes), index].mean())
    if "train_pair_probs" in bundle:
        pair = np.asarray(bundle["train_pair_probs"], dtype=float)
        if pair.shape != (axes, axes, states, states) or not np.isfinite(pair).all() or (pair < 0).any():
            raise ValueError("Invalid train pair support array")
        def surprise(dist):
            values = [-float(np.einsum("s,t,st->", dist[a], dist[b], np.log(np.maximum(pair[a, b], 1e-8))))
                      for a in range(axes) for b in range(a + 1, axes)]
            return float(np.mean(values)) if values else None
        feedback["train_pair_support_diagnostic"] = {
            "definition": "Mean negative log training pair frequency under independently sampled candidate marginals; does not identify a joint cell distribution or prove realism",
            "candidate": surprise(q), "control": surprise(p0),
            "interpolate": surprise(candidates[:, names.index("interpolate")]) if "interpolate" in names else None,
        }
    return {"q": q, "delta_mean": delta, "axis_delta_genes": axis_delta, "feedback": feedback}


def _normalize_cards(cards: Any, bundle: dict) -> list[dict]:
    if isinstance(cards, dict):
        for key in ("axis_cards", "axes", "cards"):
            if key in cards:
                cards = cards[key]
                break
    if cards is None:
        cards = bundle.get("axis_cards")
    if not isinstance(cards, list):
        raise ValueError("cards must provide a list of axis cards")
    by_axis = {card["axis_id"]: card for card in cards}
    expected = set(range(len(bundle["p0"])))
    if len(by_axis) != len(cards) or set(by_axis) != expected:
        raise ValueError("Axis cards must exactly cover all axes")
    return [copy.deepcopy(by_axis[a]) for a in range(len(cards))]


def _overview(card: dict) -> dict:
    out = {"axis_id": card["axis_id"], "evidence_ref": f"A{card['axis_id']}"}
    for key in ("scope", "scope_support", "language_annotation", "top_varying_genes"):
        if key in card:
            value = card[key]
            out[key] = value[:5] if key == "top_varying_genes" and isinstance(value, list) else value
    return out


def _compact_genes(rows: list[dict]) -> list[list]:
    """Keep unique gene identity while removing repeated JSON field names."""
    return [[row.get("gene_id", row.get("gene")), row.get("symbol"),
             row.get("value", row.get("delta"))] for row in rows]


def _axis_context(bundle: dict, cards: list[dict], axis: int, compact: bool = False) -> dict:
    card = cards[axis]
    result = _overview(card)
    if "state_reference" in card:
        result["state_reference"] = card["state_reference"]
    states = []
    for state in card.get("states", []):
        row = {k: copy.deepcopy(state[k]) for k in ("state_id", "name", "summary", "train_occupancy") if k in state}
        row["evidence_ref"] = f"A{axis}/C{state['state_id']}"
        if not compact:
            for key in ("positive_genes", "negative_genes", "up_genes", "down_genes"):
                if key in state:
                    row[key] = _compact_genes(state[key][:2])
        elif "summary" in row:
            del row["summary"]
        states.append(row)
    result["states"] = states
    result["control_occupancy"] = np.round(bundle["p0"][axis], 5).tolist()
    names = _candidate_ids(bundle)
    raw = bundle.get("axis_candidate_cards")
    if raw is not None:
        raw = raw[axis]
        if isinstance(raw, dict):
            raw = raw.get("candidates", [])
    raw_lookup = {str(v.get("id", v.get("candidate_id"))): v for v in (raw or [])}
    result["candidates"] = []
    p0 = np.asarray(bundle["p0"])[axis]
    for i, name in enumerate(names):
        q = np.asarray(bundle["candidates"])[axis, i]
        delta = (q - p0) @ np.asarray(bundle["effects"])[axis]
        row = {"id": name, "evidence_ref": f"A{axis}/{name}",
               "q": np.round(q, 5).tolist(), "tv_from_control": round(float(np.abs(q - p0).sum() / 2), 5)}
        top = _top_genes(bundle, delta, 2 if compact else 3)
        row["top_up"], row["top_down"] = _compact_genes(top["up"]), _compact_genes(top["down"])
        for key in ("projection_l1", "delta_rms"):
            if key in raw_lookup.get(name, {}):
                row[key] = raw_lookup[name][key]
        result["candidates"].append(row)
    return _plain(result)


def _request_document(bundle: dict, cards: list[dict], axes: list[int], local: dict | None = None) -> dict:
    value = {"prompt_version": PROMPT_VERSION, "case_id": bundle["case_id"],
             "request": _plain(bundle["request"]),
             "training_retrieval": _plain(bundle.get("retrieval_context", {})),
             "requested_axes": axes,
             "compact_gene_record_fields": ["unique_gene_id", "display_symbol", "signed_model_contribution_delta"],
             "gene_effect_references": "State gene values are centered on the uniform within-axis state mean. Candidate gene values are expected changes from the current control occupancy.",
             "task": "Select one candidate per requested axis, using shared context and the signed state/transition evidence.",
             "axis_overview": [_overview(c) for c in cards],
             "axes": [_axis_context(bundle, cards, a, compact=local is not None) for a in axes]}
    if local is not None:
        value["task"] = "Review the assembled local proposal jointly. Keep or revise each of all axes using exact aggregate decoder feedback and training priors. This is the one allowed joint revision; no target response is available."
        value["local_proposal"] = local["response"]
        value["local_occupancy"] = local["q"].tolist()
        value["local_decoder_feedback"] = local["feedback"]
    return _plain(value)


def _completion_content(completion: dict) -> str:
    try:
        message = completion["choices"][0]["message"]
        if message.get("tool_calls"):
            raise SelectionError("Return JSON text rather than tool calls")
        return message["content"]
    except (KeyError, IndexError, TypeError):
        raise SelectionError("Completion is missing assistant text") from None


def _run_stage(stage_dir: Path, payload: dict, axes: list[int], candidate_ids: list[str],
               client: Callable, fingerprint: str, max_retries: int) -> dict:
    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_fingerprint = _digest({"run": fingerprint, "payload": payload, "axes": axes})
    contract = {"fingerprint": stage_fingerprint, "required_axes": axes,
                "candidate_ids": candidate_ids, "max_attempts": max_retries + 1}
    contract_path = stage_dir / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Stage resume fingerprint or attempt budget changed")
    else:
        atomic_write_json(contract_path, contract)
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _json(payload)}]
    for attempt in range(1, max_retries + 2):
        stem = f"attempt_{attempt:02d}"
        request_path = stage_dir / (stem + ".request.json")
        response_path = stage_dir / (stem + ".response.json")
        status_path = stage_dir / (stem + ".status.json")
        request = {"stage_fingerprint": stage_fingerprint, "attempt": attempt,
                   "messages": copy.deepcopy(messages), "tools": []}
        reused = request_path.exists()
        if reused:
            if json.loads(request_path.read_text()) != request:
                raise ValueError("Stored API request differs from deterministic resume")
        else:
            atomic_write_json(request_path, request)
        completion = None
        error = None
        if response_path.exists():
            completion = json.loads(response_path.read_text())
            if status_path.exists():
                status = json.loads(status_path.read_text())
                if status.get("response_sha256") != file_sha256(response_path):
                    raise ValueError("Stored API response checksum changed")
        elif reused:
            # Unknown network outcome still consumes one attempt; never resend
            # the same attempt invisibly after a process interruption.
            if status_path.exists():
                status = json.loads(status_path.read_text())
                error = status.get("error", "Previous API attempt has no saved response")
            else:
                error = "Interrupted API attempt has an unknown outcome; counted against budget"
        else:
            try:
                completion = client(messages=copy.deepcopy(messages), tools=[])
                atomic_write_json(response_path, completion)
            except TransportError as exc:
                error = str(exc)
            except Exception:
                error = "API call failed; exception details omitted"
        parsed = None
        if completion is not None:
            try:
                parsed = parse_selection(_completion_content(completion), axes, candidate_ids)
            except SelectionError as exc:
                error = str(exc)
        status = {"fingerprint": stage_fingerprint, "attempt": attempt,
                  "status": "completed" if parsed is not None else "failed",
                  "error": error, "response_sha256": file_sha256(response_path) if response_path.exists() else None}
        atomic_write_json(status_path, status)
        if parsed is not None:
            return {"response": parsed, "attempts_used": attempt,
                    "stage_fingerprint": stage_fingerprint, "response_sha256": status["response_sha256"]}
        if completion is not None:
            try:
                content = _completion_content(completion)
            except SelectionError:
                content = "Invalid completion without usable text"
            messages.append({"role": "assistant", "content": content if isinstance(content, str) else "Invalid response"})
        messages.append({"role": "user", "content": "Local validation failed: " + str(error) +
                         ". Return a corrected complete JSON object. Numerical prediction quality has not been scored."})
    raise StageFailure(f"Stage {stage_dir.name} exhausted {max_retries + 1} attempts; all attempts retained")


def _atomic_npz(path: Path, **values) -> None:
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **values)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def _save_prediction(output: Path, name: str, bundle: dict, response: dict,
                     fingerprint: str, stages: list[dict]) -> dict:
    evaluated = evaluate_selection(bundle, response)
    result_path, array_path = output / (name + ".json"), output / (name + ".npz")
    document = {"status": "completed", "method": "hierarchical_" + name,
                "case_id": bundle["case_id"], "fingerprint": fingerprint,
                "response": response, "selections": response["selections"],
                "feedback": evaluated["feedback"], "stages": stages,
                "attempts_used": sum(s["attempts_used"] for s in stages),
                "arrays": str(array_path.resolve()),
                "biological_effect_validated": False}
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get("fingerprint") != fingerprint or previous.get("response") != response:
            raise ValueError("Prediction resume changed inputs or selections")
        if not array_path.exists() or previous.get("arrays_sha256") != file_sha256(array_path):
            raise ValueError("Stored prediction arrays are missing or changed")
        with np.load(array_path, allow_pickle=False) as saved:
            for key, value in (("q", evaluated["q"]), ("p0", bundle["p0"]), ("delta_mean", evaluated["delta_mean"])):
                if not np.array_equal(saved[key], value):
                    raise ValueError("Stored prediction is inconsistent with selected candidates")
        if previous.get("feedback") != evaluated["feedback"]:
            raise ValueError("Stored prediction feedback changed")
        return {**previous, "q": evaluated["q"], "delta_mean": evaluated["delta_mean"]}
    _atomic_npz(array_path, q=evaluated["q"], p0=bundle["p0"], delta_mean=evaluated["delta_mean"],
                genes=np.asarray(bundle["genes"]).astype(str))
    document["arrays_sha256"] = file_sha256(array_path)
    atomic_write_json(result_path, document)
    return {**document, "q": evaluated["q"], "delta_mean": evaluated["delta_mean"]}


def run_case(bundle: dict, cards: Any, client: Callable, output: str | Path,
             max_retries: int = 2) -> dict:
    """Four local batches then one successful joint review; bounded repairs.

    Both local and joint predictions are persisted before any external scoring.
    No hidden selection based on target or numerical prior agreement occurs.
    """
    if type(max_retries) is not int or not 0 <= max_retries <= 2:
        raise ValueError("max_retries must be 0, 1 or 2")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cards = _normalize_cards(cards, bundle)
    names = _candidate_ids(bundle)
    # Only prediction-defining source files enter the fingerprint. New reports
    # and independent audits do not invalidate already paid-for API attempts.
    own_dir = Path(__file__).resolve().parent
    sources = {name: file_sha256(own_dir / name) for name in
               ("hierarchy_agent.py", "agent.py", "annotate_cards.py", "hierarchy_data.py") if (own_dir / name).exists()}
    arrays = {name: _array_digest(bundle[name]) for name in
              ("p0", "candidates", "effects", "genes", "symbols", "std", "reference_observed_delta",
               "projection_l1", "train_pair_probs") if name in bundle}
    identity = {"prompt_version": PROMPT_VERSION, "system_prompt": SYSTEM_PROMPT,
                "source_sha256": sources, "array_sha256": arrays,
                "cards_sha256": _digest(_plain(cards)),
                "request_sha256": _digest(_plain(bundle["request"])),
                "retrieval_sha256": _digest(_plain(bundle.get("retrieval_context", {}))),
                "candidate_ids": names, "case_id": bundle["case_id"],
                "model": getattr(client, "model", "injected-test-client"),
                "endpoint": getattr(client, "url", "injected-test-client"),
                "disable_thinking": getattr(client, "_disable_thinking", None),
                "max_retries": max_retries, "local_batch_size": LOCAL_BATCH_SIZE,
                "successful_joint_reviews": 1}
    fingerprint = _digest(identity)
    manifest = {"fingerprint": fingerprint, "identity": identity, "provenance": _plain(bundle.get("provenance", {}))}
    manifest_path = output / "run_contract.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Case resume fingerprint changed; choose a new output directory")
    else:
        atomic_write_json(manifest_path, manifest)
    axes = list(range(len(bundle["p0"])))
    stages, selections, summaries, limitations = [], [], [], []
    for first in range(0, len(axes), LOCAL_BATCH_SIZE):
        batch = axes[first:first + LOCAL_BATCH_SIZE]
        stage = _run_stage(output / "requests" / f"local_{first // LOCAL_BATCH_SIZE:02d}",
                           _request_document(bundle, cards, batch), batch, names, client, fingerprint, max_retries)
        stages.append({k: stage[k] for k in ("attempts_used", "stage_fingerprint", "response_sha256")})
        selections.extend(stage["response"]["selections"])
        summaries.append(stage["response"]["summary"])
        limitations.extend(stage["response"]["limitations"])
    local_response = {"selections": selections, "summary": " | ".join(summaries),
                      "limitations": list(dict.fromkeys(limitations))}
    local = _save_prediction(output, "local", bundle, local_response, fingerprint, stages)
    joint_stage = _run_stage(output / "requests" / "joint_00",
                             _request_document(bundle, cards, axes, local=local), axes, names,
                             client, fingerprint, max_retries)
    joint_stages = stages + [{k: joint_stage[k] for k in ("attempts_used", "stage_fingerprint", "response_sha256")}]
    joint = _save_prediction(output, "joint", bundle, joint_stage["response"], fingerprint, joint_stages)
    complete = {"status": "completed", "fingerprint": fingerprint,
                "case_id": bundle["case_id"], "local": str((output / "local.json").resolve()),
                "joint": str((output / "joint.json").resolve()),
                "local_sha256": file_sha256(output / "local.json"),
                "joint_sha256": file_sha256(output / "joint.json"),
                "api_attempts": joint["attempts_used"]}
    atomic_write_json(output / "COMPLETE.json", complete)
    return {"local": local, "joint": joint, "complete": complete}


def run_cases(inputs: str | Path, output: str | Path, client: Callable,
              max_retries: int = 2, work: str | Path | None = None) -> list[dict]:
    """Convenience sequential runner; orchestration may instead call run_case."""
    from hierarchy_data import load_case_bundle
    inputs, output = Path(inputs), Path(output)
    manifest = json.loads((inputs / "manifest.json").read_text())
    work = Path(work) if work is not None else Path(manifest["provenance"]["work"])
    cards = json.loads(Path(manifest["cards_path"]).read_text())
    results = []
    for case in manifest["cases"]:
        bundle = load_case_bundle(work, case["npz"], case["context"])
        results.append(run_case(bundle, cards, client, output / case["slug"], max_retries=max_retries))
    return results
