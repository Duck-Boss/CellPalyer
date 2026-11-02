"""Annotate frozen LEXA state cards with plain-language labels, one axis per call.

Numerical effects are read only. Labels are hypotheses grounded in the shown
genes, not evidence for a causal effect or a quantitative drug response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import numpy as np

from agent import ChatCompletionsClient


PROMPT_VERSION = "lexa-state-labels-v1"
MAX_REQUESTS_PER_RUN = 32
NAME_MAX_LENGTH = 80
SUMMARY_MAX_LENGTH = 400

SYSTEM_PROMPT = f"""You annotate frozen LEXA state cards using plain language.
Prompt version: {PROMPT_VERSION}.
Every state belongs to the same axis. Its signed effects are centered relative
to the uniform average over the states of this axis. These are decoder-derived
associations, not causal regulation, drug efficacy, or a learned dose response.
Use both positive and negative gene evidence. 'gene' is the unique gene ID;
'symbol' is only an optional human-readable name and may be duplicated.
Use gene IDs, not symbols, in evidence_genes. Each state's evidence_genes must
be a subset of the gene IDs displayed on that specific state's card.
Do not change or return numerical effect values, occupancy, or gene identities.
Do not force states into different biological functions or an ordered scale.
If evidence is weak, use an 'unclear' name and describe that uncertainty.
Avoid confident mechanistic claims based on a few marker genes.
Return only one JSON object with exactly this structure:
{{"labels":[{{"state_id":0,"name":"short label","summary":"evidence-based description with uncertainty","evidence_genes":["unique_gene_id"]}}]}}
Include every state ID shown in the input exactly once, no other states.
Names must be nonempty strings of at most {NAME_MAX_LENGTH} characters and
summaries at most {SUMMARY_MAX_LENGTH} characters. evidence_genes may be empty.
Keep the full response compact: use a short name, one brief summary sentence,
and at most two evidence gene IDs per state so that every state fits the budget.
Do not include reasoning traces, tool calls, or text outside the JSON object.
"""


class AnnotationError(ValueError):
    """Invalid input or model annotation; messages contain no API response body."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_effects(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"contributions", "genes", "train_gene_std", "train_occupancy"}
        if not required.issubset(archive.files):
            raise AnnotationError("Effects file is missing required numeric card arrays.")
        effects = np.asarray(archive["contributions"], dtype=np.float64)
        genes = np.asarray(archive["genes"])
        std = np.asarray(archive["train_gene_std"], dtype=np.float64)
        occupancy = np.asarray(archive["train_occupancy"], dtype=np.float64)
        symbols = np.asarray(archive["gene_symbols"]) if "gene_symbols" in archive.files else genes.copy()
    if effects.ndim != 3 or not 1 <= effects.shape[0] <= 32 or effects.shape[1] != 16:
        raise AnnotationError("Expected one to 32 axes and exactly 16 states per axis.")
    axes, states, gene_count = effects.shape
    if gene_count == 0 or genes.shape != (gene_count,) or symbols.shape != (gene_count,) or std.shape != (gene_count,) or occupancy.shape != (axes, states):
        raise AnnotationError("Effects, gene IDs, symbols, standard deviations, and occupancy have incompatible dimensions.")
    if genes.dtype.kind not in {"U", "S"} or symbols.dtype.kind not in {"U", "S"}:
        raise AnnotationError("Gene IDs and symbols must be string arrays.")
    gene_ids = genes.astype(str).tolist()
    gene_symbols = symbols.astype(str).tolist()
    if any(not gene.strip() for gene in gene_ids) or len(set(gene_ids)) != len(gene_ids):
        raise AnnotationError("Gene IDs must be nonempty and unique; duplicate symbols are allowed separately.")
    if not np.isfinite(effects).all() or not np.isfinite(std).all() or not np.isfinite(occupancy).all():
        raise AnnotationError("Numeric card arrays must contain only finite values.")
    if (std < 0).any() or (occupancy < 0).any() or (occupancy > 1 + 1e-6).any():
        raise AnnotationError("Standard deviations and occupancy must be nonnegative probabilities/scales.")
    if not np.allclose(occupancy.sum(axis=1), 1, rtol=1e-5, atol=1e-6):
        raise AnnotationError("Each axis's train_occupancy must sum to one.")
    return {"contributions": effects, "genes": gene_ids, "gene_symbols": gene_symbols, "std": std, "occupancy": occupancy}


def build_axis_cards(effects: dict[str, Any], axis_id: int, top_genes: int = 6) -> dict:
    values = effects["contributions"][axis_id]
    centered = values - values.mean(axis=0, keepdims=True)
    safe_std = np.maximum(effects["std"], 0.05)
    scaled = centered / safe_std[None, :]
    states = []
    for state_id in range(values.shape[0]):
        positive = np.flatnonzero(scaled[state_id] > 0)
        negative = np.flatnonzero(scaled[state_id] < 0)
        # Stable sorting also makes tied cards and resumes reproducible.
        up = positive[np.argsort(-scaled[state_id, positive], kind="stable")][:top_genes]
        down = negative[np.argsort(scaled[state_id, negative], kind="stable")][:top_genes]

        def rows(indices: np.ndarray) -> list[dict]:
            return [
                {
                    "gene": effects["genes"][int(index)],
                    "symbol": effects["gene_symbols"][int(index)],
                    "delta": float(centered[state_id, index]),
                    "scaled_delta": float(scaled[state_id, index]),
                    "gene_std": float(effects["std"][index]),
                }
                for index in indices
            ]

        states.append({"state_id": state_id, "occupancy": float(effects["occupancy"][axis_id, state_id]), "up_genes": rows(up), "down_genes": rows(down)})
    return {
        "axis_id": axis_id,
        "effect_definition": "decoder contribution minus the uniform within-axis state mean; units follow the source effects file",
        "scaling_definition": "scaled_delta = delta / max(training_gene_std, 0.05); this is numeric scaling, not a causal effect; for count-decoder logits it is not standardized expression",
        "states": states,
    }


def parse_label_response(content: str, cards: dict) -> dict[str, dict]:
    if not isinstance(content, str) or not content.strip():
        raise AnnotationError("The model did not return textual JSON labels.")
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in {"```", "```json"} or lines[-1].strip() != "```":
            raise AnnotationError("Invalid JSON code fence.")
        text = "\n".join(lines[1:-1])
    try:
        response = json.loads(text)
    except (ValueError, TypeError):
        raise AnnotationError("The model response is not a valid JSON label object.") from None
    if not isinstance(response, dict) or set(response) != {"labels"} or not isinstance(response["labels"], list):
        raise AnnotationError("Expected an object containing only a labels array.")
    expected = {state["state_id"]: state for state in cards["states"]}
    if len(response["labels"]) != len(expected):
        raise AnnotationError("Labels must cover the entire current axis exactly once.")
    labels = {}
    seen = set()
    for label in response["labels"]:
        if not isinstance(label, dict) or set(label) != {"state_id", "name", "summary", "evidence_genes"}:
            raise AnnotationError("Each state label must contain only the four declared fields.")
        state_id = label["state_id"]
        if type(state_id) is not int or state_id not in expected or state_id in seen:
            raise AnnotationError("State IDs are duplicated or outside the current axis.")
        seen.add(state_id)
        for key, limit in (("name", NAME_MAX_LENGTH), ("summary", SUMMARY_MAX_LENGTH)):
            if not isinstance(label[key], str) or not label[key].strip() or len(label[key]) > limit:
                raise AnnotationError("Label name/summary must be nonempty short strings.")
        state = expected[state_id]
        allowed_genes = {row["gene"] for row in state["up_genes"] + state["down_genes"]}
        evidence = label["evidence_genes"]
        if not isinstance(evidence, list) or any(not isinstance(gene, str) for gene in evidence) or len(evidence) != len(set(evidence)) or not set(evidence).issubset(allowed_genes):
            raise AnnotationError("Evidence must use unique gene IDs displayed on the corresponding state card.")
        labels[f"A{cards['axis_id']}_C{state_id}"] = {
            "axis_id": cards["axis_id"],
            "state_id": state_id,
            "name": label["name"].strip(),
            "summary": label["summary"].strip(),
            "evidence_genes": list(evidence),
        }
    return labels


def atomic_write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(document, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verify_resume(document: dict, digest: str, model: str, effects: dict) -> None:
    if not isinstance(document, dict) or not isinstance(document.get("meta"), dict) or not isinstance(document.get("labels"), dict):
        raise AnnotationError("Invalid existing labels document.")
    meta = document["meta"]
    if document.get("effects_sha256") != digest or meta.get("effects_sha256") != digest or meta.get("model") != model or meta.get("prompt_version") != PROMPT_VERSION:
        raise AnnotationError("Resume requires identical effects hash, model, and prompt version.")
    axes, states, _ = effects["contributions"].shape
    completed = meta.get("complete_axes")
    if not isinstance(completed, list) or any(type(axis) is not int or not 0 <= axis < axes for axis in completed) or len(completed) != len(set(completed)):
        raise AnnotationError("Existing complete_axes metadata is invalid.")
    expected_keys = {f"A{axis}_C{state}" for axis in completed for state in range(states)}
    if set(document["labels"]) != expected_keys or bool(meta.get("complete")) != (len(completed) == axes):
        raise AnnotationError("Existing completion metadata and stored labels disagree.")
    for axis in completed:
        cards = build_axis_cards(effects, axis)
        candidate = []
        for state in range(states):
            record = document["labels"][f"A{axis}_C{state}"]
            if not isinstance(record, dict) or record.get("axis_id") != axis or record.get("state_id") != state:
                raise AnnotationError("Existing state label identity is invalid.")
            candidate.append({key: record.get(key) for key in ("state_id", "name", "summary", "evidence_genes")})
        parse_label_response(json.dumps({"labels": candidate}), cards)


def run_annotation(
    effects_path: str | Path,
    output_path: str | Path,
    model: str,
    send_callable: Callable[..., dict],
    max_axes: int = MAX_REQUESTS_PER_RUN,
    resume: bool = False,
) -> dict:
    """Annotate at most 32 pending axes, one request each and no implicit retry.

    A failed axis stops this run without creating labels for it. Successful axes
    are atomically saved; --resume skips them and validates input provenance.
    """
    if not isinstance(model, str) or not model.strip():
        raise AnnotationError("An explicit model name is required.")
    model = model.strip()
    if type(max_axes) is not int or not 1 <= max_axes <= MAX_REQUESTS_PER_RUN:
        raise AnnotationError("max_axes must be an integer from 1 to 32.")
    effects_path, output_path = Path(effects_path), Path(output_path)
    digest = file_sha256(effects_path)
    effects = load_effects(effects_path)
    axes, states, _ = effects["contributions"].shape
    if output_path.exists():
        if not resume:
            raise AnnotationError("Output already exists; use --resume to preserve completed axes.")
        try:
            document = json.loads(output_path.read_text(encoding="utf-8"))
        except (ValueError, OSError, UnicodeError):
            raise AnnotationError("Unable to read the existing labels document.") from None
        _verify_resume(document, digest, model, effects)
    else:
        document = {
            "effects_sha256": digest,
            "labels": {},
            "meta": {
                "effects_sha256": digest,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "complete_axes": [],
                "complete": False,
                "num_axes": axes,
                "states_per_axis": states,
                "api_requests_total": 0,
            },
        }
    document["meta"]["requests_this_run"] = 0
    document["meta"]["status"] = "in_progress"
    document["meta"].pop("last_error", None)
    atomic_write_json(output_path, document)
    pending = [axis for axis in range(axes) if axis not in document["meta"]["complete_axes"]]
    for axis in pending[:max_axes]:
        cards = build_axis_cards(effects, axis)
        document["meta"]["requests_this_run"] += 1
        document["meta"]["api_requests_total"] += 1
        # Count attempts before the request, including interrupted/failed calls.
        atomic_write_json(output_path, document)
        try:
            response = send_callable(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": json.dumps(cards, ensure_ascii=False, allow_nan=False)}],
                tools=[],
            )
            message = response["choices"][0]["message"]
            if message.get("tool_calls"):
                raise AnnotationError("Annotation responses must contain labels, not tool calls.")
            content = message["content"]
            validated = parse_label_response(content, cards)
        except AnnotationError:
            document["meta"]["status"] = "failed"
            document["meta"]["last_error"] = {"axis_id": axis, "kind": "invalid_labels", "details": "Response failed schema or evidence validation; response body omitted."}
            atomic_write_json(output_path, document)
            return document
        except Exception:
            # API keys, headers, raw HTTP error bodies, and CoT never enter output.
            document["meta"]["status"] = "failed"
            document["meta"]["last_error"] = {"axis_id": axis, "kind": "request_failed", "details": "Request or completion failed; exception and response details omitted."}
            atomic_write_json(output_path, document)
            return document
        document["labels"].update(validated)
        document["meta"]["complete_axes"].append(axis)
        document["meta"]["complete_axes"].sort()
        document["meta"]["complete"] = len(document["meta"]["complete_axes"]) == axes
        atomic_write_json(output_path, document)
    document["meta"]["status"] = "complete" if document["meta"]["complete"] else "partial"
    atomic_write_json(output_path, document)
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL"))
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-file")
    parser.add_argument("--max-axes", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.model or not args.model.strip():
        parser.error("Set --model or DEEPSEEK_MODEL explicitly.")
    if not 1 <= args.max_axes <= 32:
        parser.error("--max-axes must be from 1 to 32.")
    try:
        api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip() if args.api_key_file else os.environ.get("DEEPSEEK_API_KEY", "").strip()
    except (OSError, UnicodeError):
        parser.error("Unable to read --api-key-file; file details omitted.")
    if not api_key:
        parser.error("Set DEEPSEEK_API_KEY or provide --api-key-file.")
    try:
        client = ChatCompletionsClient(model=args.model, api_key=api_key, base_url=args.base_url, timeout=45.0)
        document = run_annotation(args.effects, args.output, args.model, client, args.max_axes, args.resume)
    except (AnnotationError, OSError, ValueError):
        parser.exit(2, "Annotation setup failed. Check the effects arrays, output/resume provenance, model, and base URL; details omitted.\n")
    print(json.dumps({"status": document["meta"]["status"], "complete": document["meta"]["complete"], "complete_axes": document["meta"]["complete_axes"], "requests_this_run": document["meta"]["requests_this_run"], "output": str(args.output)}, ensure_ascii=False))
    return 1 if document["meta"]["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
