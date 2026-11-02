"""Text-only, budgeted LEXA editing agent; no model or LEXA training.

The model proposes edits.  Only the frozen decoder's tool results establish
that an edit was evaluated; neither tool execution nor text establishes drug
efficacy or a causal biological effect.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Callable
import urllib.error
import urllib.parse
import urllib.request


SYSTEM_PROMPT = """You edit a frozen LEXA representation using plain text and tools.
Do not train models, invent vocabulary tokens, or execute code.
First call get_context, then inspect_states for every axis you consider editing.
State IDs are categorical, not an ordered dose scale. Use evidence in the state
cards and the actual source-to-target changes, not merely appealing state names.
An edit sets one fixed target state in a selected fraction of a source-state
group. Optional from_code restricts eligible cells to that source state; without
it, all source states are eligible. Optional fraction is in [0, 1], defaults to
1, and uses the workspace's deterministic selection. This is a population edit,
not a paired-cell intervention or a calibrated dose-response function.
Propose at most three distinct axes with integer, zero-based axis_id and to_code;
from_code and fraction are optional. Inspect source-state effects before editing.
Call evaluate_edits on your proposal before discussing its numerical effect.
If you abstain, call evaluate_edits with edits: [] before returning your final answer.
You may inspect other states and revise after reading actual decoder results.
Drug/cytokine text, dose, and time are context for a hypothesis, not a learned
quantitative drug-response mapping. Missing support information means unknown.
Numerical validity and agreement with a functional text goal do not validate
drug efficacy, causal regulation, or an observed biological response.
Your final answer must be JSON with edits, summary, and limitations. Reference
only the most recently successfully evaluated edits; if none were evaluated,
return edits: [] and explicitly state that no proposal was evaluated.
"""


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_context",
            "description": "Read the fixed baseline, functional goal, dimensions, units, and provenance.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_states",
            "description": "Read source states, candidate state cards, and transition evidence for one axis.",
            "parameters": {
                "type": "object",
                "properties": {"axis_id": {"type": "integer", "minimum": 0}},
                "required": ["axis_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_edits",
            "description": "Numerically evaluate a proposal with the frozen decoder; this does not validate drug efficacy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "edits": {
                        "type": "array",
                        "minItems": 0,
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "properties": {
                                "axis_id": {"type": "integer", "minimum": 0},
                                "to_code": {"type": "integer", "minimum": 0},
                                "from_code": {"type": "integer", "minimum": 0, "description": "Optional eligible source state; absent means all source states."},
                                "fraction": {"type": "number", "minimum": 0, "maximum": 1, "description": "Deterministic fraction of eligible source cells to edit; defaults to one."},
                            },
                            "required": ["axis_id", "to_code"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["edits"],
                "additionalProperties": False,
            },
        },
    },
]


class TransportError(RuntimeError):
    """A deliberately sanitized transport error, without response bodies."""


class ChatCompletionsClient:
    """Small OpenAI-compatible transport. Credentials never enter messages/logs."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        timeout: float = 45.0,
        disable_thinking: bool = True,
    ) -> None:
        if not model or not model.strip():
            raise ValueError("An explicit model name is required.")
        if not api_key or not api_key.strip():
            raise ValueError("A nonempty API key is required.")
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an HTTP(S) API base URL.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain credentials, a query, or a fragment.")
        if not 0 < timeout <= 45:
            raise ValueError("timeout must be greater than zero and at most 45 seconds.")
        self.model = model.strip()
        self._api_key = api_key.strip()
        root = base_url.rstrip("/")
        self.url = root if root.endswith("/chat/completions") else root + "/chat/completions"
        self.timeout = timeout
        # Provider extensions must not leak into requests to other endpoints.
        self._disable_thinking = bool(disable_thinking and parsed.hostname == "api.deepseek.com")

    def __call__(self, *, messages: list[dict], tools: list[dict]) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.0,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        if self._disable_thinking:
            payload["thinking"] = {"type": "disabled"}
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
            headers={
                "Authorization": "Bearer " + self._api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read(4 * 1024 * 1024 + 1)
        except urllib.error.HTTPError as exc:
            status = exc.code
            exc.close()
            raise TransportError(f"API HTTP error {status}; response body omitted.") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise TransportError("API connection failed or timed out; details omitted.") from None
        if len(body) > 4 * 1024 * 1024:
            raise TransportError("API response exceeded the size limit.")
        try:
            result = json.loads(body)
        except (ValueError, UnicodeError):
            raise TransportError("API returned invalid JSON; response body omitted.") from None
        if not isinstance(result, dict):
            raise TransportError("API returned an invalid completion object.")
        return result


class EditValidationError(ValueError):
    """Locally generated, credential-free feedback for a repairable proposal."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _validate_edits(value: Any) -> list[dict]:
    if not isinstance(value, list) or not 0 <= len(value) <= 3:
        raise ValueError("edits must contain zero to three items.")
    seen = set()
    for edit in value:
        if not isinstance(edit, dict) or not {"axis_id", "to_code"}.issubset(edit) or set(edit) - {"axis_id", "to_code", "from_code", "fraction"}:
            raise ValueError("Each edit requires axis_id/to_code; only from_code/fraction are optional.")
        if any(type(edit[key]) is not int or edit[key] < 0 for key in ("axis_id", "to_code")):
            raise ValueError("Edit identifiers must be nonnegative integers.")
        if "from_code" in edit and (type(edit["from_code"]) is not int or edit["from_code"] < 0):
            raise ValueError("from_code must be a nonnegative integer.")
        if "fraction" in edit and (type(edit["fraction"]) not in {int, float} or not math.isfinite(edit["fraction"]) or not 0 <= edit["fraction"] <= 1):
            raise ValueError("fraction must be a finite number from zero to one.")
        if edit["axis_id"] in seen:
            raise EditValidationError(
                "duplicate_axis",
                f"axis_id {edit['axis_id']} occurs more than once. Each axis may appear only once. "
                "Choose one from_code-to_code transition for that axis, or omit from_code "
                "to apply a single target to all source states. Return a corrected proposal; "
                "repeating the same rejected edits will fail again.",
            )
        seen.add(edit["axis_id"])
    return copy.deepcopy(value)


def _canonical_edits(edits: list[dict]) -> str:
    return json.dumps(sorted(edits, key=lambda item: item["axis_id"]), sort_keys=True)


def run_agent(workspace: Any, send_callable: Callable[..., dict], max_rounds: int = 4,
              max_evaluations: int | None = None) -> dict:
    """Run at most ``max_rounds`` API requests, with no implicit retries.

    ``send_callable(messages=..., tools=...)`` returns a chat-completions object.
    The result's authoritative edits always come from the last successful tool
    evaluation. ``trace`` is separate from the final CLI result artifact.
    """
    if type(max_rounds) is not int or not 1 <= max_rounds <= 20:
        raise ValueError("max_rounds must be an integer from 1 to 20.")
    if max_evaluations is not None and (type(max_evaluations) is not int or not 1 <= max_evaluations <= 3):
        raise ValueError("max_evaluations must be from one to three")
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Read get_context for the request and control population; inspect evidence and evaluate a small edit proposal."},
    ]
    context_loaded = False
    inspected_axes: set[int] = set()
    last_edits: list[dict] | None = None
    last_evaluation: dict | None = None
    model_final_matches: bool | None = None
    events: list[dict] = []
    requests_made = 0
    tool_calls_made = 0
    evaluations_made = 0
    status = "budget_exhausted"

    for round_idx in range(max_rounds):
        requests_made += 1
        try:
            completion = send_callable(messages=copy.deepcopy(messages), tools=copy.deepcopy(TOOL_DEFINITIONS))
            raw_message = completion["choices"][0]["message"]
            if not isinstance(raw_message, dict):
                raise ValueError("Invalid assistant message.")
            content = raw_message.get("content")
            calls = raw_message.get("tool_calls") or []
            if content is not None and not isinstance(content, str):
                raise ValueError("Invalid assistant content.")
            if not isinstance(calls, list) or len(calls) > 16:
                raise ValueError("Invalid or excessive tool calls.")
            if any(
                not isinstance(call, dict)
                or not isinstance(call.get("id"), str)
                or not isinstance(call.get("function"), dict)
                for call in calls
            ):
                raise ValueError("Invalid tool call format.")
        except Exception:
            # Never serialize exception text, headers, or an HTTP error body.
            events.append({"round": round_idx + 1, "error": "transport_or_completion_error"})
            status = "transport_error"
            break

        assistant: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            assistant["tool_calls"] = calls
        messages.append(assistant)
        if not calls:
            status = "completed"
            try:
                final_claim = json.loads(content or "")
                claimed_edits = _validate_edits(final_claim.get("edits"))
                model_final_matches = last_edits is not None and _canonical_edits(claimed_edits) == _canonical_edits(last_edits)
            except (ValueError, TypeError, AttributeError):
                model_final_matches = False
            break

        for call in calls:
            tool_calls_made += 1
            function = call["function"]
            name = function.get("name")
            result: dict
            try:
                argument_text = function.get("arguments")
                if not isinstance(argument_text, str):
                    raise ValueError("Tool arguments must be JSON text.")
                arguments = json.loads(argument_text)
                if not isinstance(arguments, dict):
                    raise ValueError("Tool arguments must be an object.")
                if name == "get_context":
                    if arguments:
                        raise ValueError("get_context accepts no arguments.")
                    result = workspace.get_context()
                elif name == "inspect_states":
                    if set(arguments) != {"axis_id"} or type(arguments["axis_id"]) is not int or arguments["axis_id"] < 0:
                        raise ValueError("inspect_states requires a nonnegative integer axis_id.")
                    if not context_loaded:
                        raise ValueError("Call get_context first.")
                    result = workspace.inspect_states(arguments["axis_id"])
                elif name == "evaluate_edits":
                    if set(arguments) != {"edits"}:
                        raise ValueError("evaluate_edits requires only edits.")
                    edits = _validate_edits(arguments["edits"])
                    if not context_loaded or any(edit["axis_id"] not in inspected_axes for edit in edits):
                        raise ValueError("Read context and inspect every edited axis before evaluation.")
                    result = workspace.evaluate_edits(edits)
                else:
                    raise ValueError("Unknown tool name.")
                if not isinstance(result, dict):
                    raise TypeError("Tools must return an object.")
                # Also ensure that the exact result can be replayed to the API.
                json.dumps(result, ensure_ascii=False, allow_nan=False)
                successful = "error" not in result and result.get("ok") is not False
                if successful:
                    if name == "get_context":
                        context_loaded = True
                    elif name == "inspect_states":
                        inspected_axes.add(arguments["axis_id"])
                    elif name == "evaluate_edits":
                        last_edits = copy.deepcopy(edits)
                        last_evaluation = copy.deepcopy(result)
                        evaluations_made += 1
            except EditValidationError as exc:
                result = {"error": "invalid_tool_request", "code": exc.code,
                          "message": str(exc), "edit_applied": False}
            except (ValueError, TypeError, KeyError):
                result = {
                    "error": "invalid_tool_request",
                    "message": "Use the declared JSON schema, read context first, inspect each edited axis, and check axis/code ranges. No edit was recorded by this failed call.",
                }
            except Exception:
                result = {"error": "workspace_error", "message": "The workspace tool failed; exception details omitted. No successful edit was recorded."}
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result, ensure_ascii=False, allow_nan=False)})
            events.append({"round": round_idx + 1, "tool": name, "success": "error" not in result and result.get("ok") is not False})
            if max_evaluations is not None and evaluations_made >= max_evaluations:
                break
        if max_evaluations is not None and evaluations_made >= max_evaluations:
            status = "evaluation_budget_complete"
            break

    return {
        "mode": getattr(workspace, "request", {}).get("mode", "functional_edit"),
        "status": status,
        "decoder_evaluated": last_edits is not None,
        "last_evaluated_proposal": {"edits": last_edits} if last_edits is not None else None,
        "evaluation": last_evaluation,
        "biological_effect_validated": False,
        "note": "Frozen-decoder numerical evaluation only; drug efficacy and causal biological effects were not validated.",
        "model_final_matches_last_evaluation": model_final_matches,
        "api_requests": requests_made,
        "evaluations": evaluations_made,
        "tool_calls": tool_calls_made,
        "trace": {"messages": messages, "events": events},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effects", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--distribution", choices=("mse_log1p", "nb"), default="mse_log1p")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL"))
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--api-key-file")
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if not args.model or not args.model.strip():
        parser.error("Set --model or DEEPSEEK_MODEL explicitly.")
    if not 1 <= args.max_rounds <= 20:
        parser.error("--max-rounds must be from 1 to 20.")
    try:
        api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip() if args.api_key_file else os.environ.get("DEEPSEEK_API_KEY", "").strip()
    except OSError:
        parser.error("Unable to read --api-key-file; file error details omitted.")
    if not api_key:
        parser.error("Set DEEPSEEK_API_KEY or provide --api-key-file.")
    try:
        client = ChatCompletionsClient(model=args.model, api_key=api_key, base_url=args.base_url)
    except ValueError as exc:
        parser.error(str(exc))
    from tools import Workspace

    workspace = Workspace(
        effects_path=args.effects,
        baseline_path=args.baseline,
        request_path=args.request,
        distribution=args.distribution,
        output_dir=args.output,
        synthetic=args.synthetic,
    )
    result = run_agent(workspace, client, max_rounds=args.max_rounds)
    trace = result.pop("trace")
    result["synthetic"] = args.synthetic
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "agent_trace.json").write_text(json.dumps(trace, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "agent_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "decoder_evaluated": result["decoder_evaluated"], "biological_effect_validated": False, "synthetic": args.synthetic, "result": str(output / "agent_result.json")}, ensure_ascii=False))
    return 1 if result["status"] == "transport_error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
