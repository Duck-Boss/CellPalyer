"""Targeted checks for probability decoding, protocol validation and recovery."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hierarchy_agent import (SelectionError, StageFailure, evaluate_selection,
                             parse_selection, run_case)


NAMES = ["keep", "nearest", "interpolate", "lower", "upper", "half", "onehalf"]


def bundle_and_cards(axes=2):
    p0 = np.zeros((axes, 16)); p0[:, 0] = 1
    q = np.repeat(p0[:, None, :], len(NAMES), axis=1)
    for i in range(1, len(NAMES)):
        q[:, i, 0] = 1 - i / 8
        q[:, i, 1] = i / 8
    rng = np.random.default_rng(781)
    effects = rng.normal(size=(axes, 16, 5))
    bundle = {"case_id": "test-case", "request": {"drug": "test", "dose_uM": 5},
              "p0": p0, "candidates": q, "candidate_ids": NAMES,
              "effects": effects, "genes": np.array([f"G{i}" for i in range(5)]),
              "symbols": np.array([f"S{i}" for i in range(5)]), "std": np.ones(5),
              "reference_observed_delta": np.ones(5), "retrieval_context": {"train_only": True},
              "projection_l1": np.zeros((axes, len(NAMES))),
              "train_pair_probs": np.ones((axes, axes, 16, 16)) / 256,
              "provenance": {"test": True}}
    cards = [{"axis_id": a, "scope": "test genes", "scope_support": "unknown",
              "states": [{"state_id": s, "name": f"state{s}", "summary": "signed contribution",
                          "positive_genes": [], "negative_genes": [], "train_occupancy": 1 / 16}
                         for s in range(16)]} for a in range(axes)]
    return bundle, cards


def selection(axes, name="interpolate"):
    return {"selections": [{"axis_id": a, "candidate_id": name,
                            "reason": "Retain training prior", "evidence_refs": [f"A{a}/{name}"]}
                           for a in axes], "summary": "Candidate prediction", "limitations": ["Not a causal result"]}


class Client:
    model = "unit-test-model"
    url = "https://unit-test.invalid/chat/completions"
    _disable_thinking = True

    def __init__(self, invalid_first=False, always_invalid=False, joint_keep=False):
        self.calls = 0
        self.invalid_first = invalid_first
        self.always_invalid = always_invalid
        self.joint_keep = joint_keep

    def __call__(self, *, messages, tools):
        self.calls += 1
        data = json.loads(messages[1]["content"])
        if self.always_invalid or (self.invalid_first and self.calls == 1):
            content = json.dumps({"selections": []})
        else:
            name = "keep" if self.joint_keep and "local_proposal" in data else "interpolate"
            content = json.dumps(selection(data["requested_axes"], name))
        return {"choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10}}


class ValidationTests(unittest.TestCase):
    def test_valid_fenced_response(self):
        value = parse_selection("```json\n" + json.dumps(selection([0, 1])) + "\n```", [0, 1], NAMES)
        self.assertEqual(len(value["selections"]), 2)

    def test_duplicate_missing_bool_unknown_and_external_evidence(self):
        mutations = []
        duplicate = selection([0, 0]); mutations.append(duplicate)
        missing = selection([0]); mutations.append(missing)
        boolean = selection([0, 1]); boolean["selections"][0]["axis_id"] = False; mutations.append(boolean)
        unknown = selection([0, 1]); unknown["selections"][0]["candidate_id"] = "imaginary"; mutations.append(unknown)
        evidence = selection([0, 1]); evidence["selections"][0]["evidence_refs"] = ["A1/interpolate"]; mutations.append(evidence)
        for value in mutations:
            with self.subTest(value=value), self.assertRaises(SelectionError):
                parse_selection(json.dumps(value), [0, 1], NAMES)

    def test_decoding_matches_explicit_state_weight_sum(self):
        bundle, _ = bundle_and_cards()
        out = evaluate_selection(bundle, {0: "nearest", 1: "upper"})
        expected = sum((out["q"][a, s] - bundle["p0"][a, s]) * bundle["effects"][a, s]
                       for a in range(2) for s in range(16))
        np.testing.assert_allclose(out["delta_mean"], expected, atol=1e-12)
        np.testing.assert_array_equal(evaluate_selection(bundle, {0: "keep", 1: "keep"})["delta_mean"], np.zeros(5))
        self.assertIn("train_pair_support_diagnostic", out["feedback"])

    def test_invalid_probability_is_rejected(self):
        bundle, _ = bundle_and_cards()
        bundle["candidates"][0, 2, 1] = 2
        with self.assertRaises(ValueError):
            evaluate_selection(bundle, {0: "interpolate", 1: "keep"})


class ExecutionTests(unittest.TestCase):
    def test_repair_then_resume_preserves_calls_and_joint_result(self):
        bundle, cards = bundle_and_cards()
        client = Client(invalid_first=True, joint_keep=True)
        with tempfile.TemporaryDirectory() as temp:
            first = run_case(bundle, cards, client, temp, max_retries=2)
            self.assertEqual(client.calls, 3)
            self.assertEqual(first["complete"]["api_attempts"], 3)
            self.assertGreater(np.linalg.norm(first["local"]["delta_mean"]), 0)
            np.testing.assert_array_equal(first["joint"]["delta_mean"], np.zeros(5))
            resumed = run_case(bundle, cards, client, temp, max_retries=2)
            self.assertEqual(client.calls, 3)
            self.assertEqual(resumed["complete"], first["complete"])
            requests = list(Path(temp).glob("requests/*/*.request.json"))
            self.assertEqual(len(requests), 3)
            failed = json.loads((Path(temp) / "requests/local_00/attempt_01.status.json").read_text())
            self.assertEqual(failed["status"], "failed")
            repaired_request = json.loads((Path(temp) / "requests/local_00/attempt_02.request.json").read_text())
            self.assertIn("validation failed", repaired_request["messages"][-1]["content"])

    def test_attempt_budget_survives_resume(self):
        bundle, cards = bundle_and_cards()
        client = Client(always_invalid=True)
        with tempfile.TemporaryDirectory() as temp:
            for _ in range(2):
                with self.assertRaises(StageFailure):
                    run_case(bundle, cards, client, temp, max_retries=2)
            self.assertEqual(client.calls, 3)
            self.assertFalse((Path(temp) / "local.json").exists())

    def test_changed_cards_model_or_arrays_refuse_resume(self):
        bundle, cards = bundle_and_cards()
        with tempfile.TemporaryDirectory() as temp:
            client = Client()
            run_case(bundle, cards, client, temp)
            edited = copy.deepcopy(cards); edited[0]["scope"] = "different"
            with self.assertRaises(ValueError):
                run_case(bundle, edited, client, temp)
            bundle["effects"][0, 0, 0] += .1
            with self.assertRaises(ValueError):
                run_case(bundle, cards, client, temp)
            self.assertEqual(client.calls, 2)

    def test_modified_response_or_prediction_detected(self):
        bundle, cards = bundle_and_cards()
        for kind in ("response", "prediction"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                client = Client()
                run_case(bundle, cards, client, temp)
                if kind == "response":
                    path = Path(temp) / "requests/local_00/attempt_01.response.json"
                    doc = json.loads(path.read_text()); doc["usage"]["prompt_tokens"] = 200
                    path.write_text(json.dumps(doc))
                else:
                    path = Path(temp) / "local.npz"
                    with path.open("ab") as stream:
                        stream.write(b"altered")
                with self.assertRaises(ValueError):
                    run_case(bundle, cards, client, temp)
                self.assertEqual(client.calls, 2)

    def test_all32_axes_batched_then_joint(self):
        bundle, cards = bundle_and_cards(axes=32)
        with tempfile.TemporaryDirectory() as temp:
            client = Client()
            result = run_case(bundle, cards, client, temp)
            self.assertEqual(client.calls, 5)
            self.assertEqual(len(result["local"]["selections"]), 32)
            joint_request = json.loads((Path(temp) / "requests/joint_00/attempt_01.request.json").read_text())
            payload = json.loads(joint_request["messages"][1]["content"])
            self.assertEqual(len(payload["local_occupancy"]), 32)
            self.assertIn("local_decoder_feedback", payload)


if __name__ == "__main__":
    unittest.main()
