"""Offline validation for plain-text state annotation and safe resumption."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from annotate_cards import AnnotationError, PROMPT_VERSION, build_axis_cards, load_effects, main, parse_label_response, run_annotation


class CardTransport:
    def __init__(self, break_axis=None, mutation=None):
        self.calls = 0
        self.break_axis = break_axis
        self.mutation = mutation
        self.cards = []

    def __call__(self, *, messages, tools):
        self.calls += 1
        self.assert_no_credentials = "DEEPSEEK_API_KEY" not in json.dumps(messages)
        cards = json.loads(messages[-1]["content"])
        self.cards.append(cards)
        if cards["axis_id"] == self.break_axis:
            raise RuntimeError("credential-and-HTTP-body-must-not-be-saved")
        labels = []
        for state in cards["states"]:
            rows = state["up_genes"] + state["down_genes"]
            labels.append({"state_id": state["state_id"], "name": "unclear programme", "summary": "Decoder-derived signature with uncertain biological interpretation.", "evidence_genes": [rows[0]["gene"]] if rows else []})
        if self.mutation:
            self.mutation(labels)
        return {"choices": [{"message": {"content": "```json\n" + json.dumps({"labels": labels}) + "\n```"}}]}


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.effects = self.root / "effects.npz"
        self.output = self.root / "labels.json"
        self.write_effects()

    def tearDown(self):
        self.temporary.cleanup()

    def write_effects(self, shift=0.0, axes=2):
        values = np.zeros((axes, 16, 4), dtype=np.float32)
        for axis in range(axes):
            for state in range(16):
                values[axis, state] = [state + shift, -state, (state % 3) + axis, 0]
        np.savez_compressed(self.effects, contributions=values, genes=np.array(["ENSG1", "ENSG2", "ENSG3", "ENSG4"]), gene_symbols=np.array(["SHARED", "SHARED", "THREE", "FOUR"]), train_gene_std=np.array([1.0, 2.0, 0.01, 0.0]), train_occupancy=np.full((axes, 16), 1 / 16))

    def test_cards_preserve_unique_ids_duplicate_symbols_and_scaled_effects(self):
        cards = build_axis_cards(load_effects(self.effects), 0)
        self.assertEqual(len(cards["states"]), 16)
        rows = cards["states"][0]["up_genes"] + cards["states"][0]["down_genes"]
        first = next(row for row in rows if row["gene"] == "ENSG1")
        self.assertEqual(first["symbol"], "SHARED")
        self.assertAlmostEqual(first["delta"], -7.5)
        self.assertAlmostEqual(first["scaled_delta"], -7.5)
        third = next(row for row in rows if row["gene"] == "ENSG3")
        self.assertAlmostEqual(third["scaled_delta"], third["delta"] / 0.05)

    def test_full_axis_labels_and_completion_are_saved(self):
        transport = CardTransport()
        document = run_annotation(self.effects, self.output, "explicit-model", transport)
        self.assertTrue(document["meta"]["complete"])
        self.assertEqual(document["meta"]["complete_axes"], [0, 1])
        self.assertEqual(len(document["labels"]), 32)
        self.assertEqual(transport.calls, 2)
        self.assertTrue(transport.assert_no_credentials)
        self.assertEqual(json.loads(self.output.read_text()), document)
        self.assertEqual(document["meta"]["prompt_version"], PROMPT_VERSION)

    def test_missing_or_duplicate_or_out_of_range_states_rejected(self):
        def missing(labels):
            labels.pop()

        def duplicate(labels):
            labels[-1]["state_id"] = 0

        def out_of_range(labels):
            labels[-1]["state_id"] = 16

        for mutation in (missing, duplicate, out_of_range):
            with self.subTest(mutation=mutation.__name__):
                output = self.root / (mutation.__name__ + ".json")
                document = run_annotation(self.effects, output, "explicit-model", CardTransport(mutation=mutation))
                self.assertEqual(document["meta"]["status"], "failed")
                self.assertEqual(document["labels"], {})
                self.assertEqual(document["meta"]["complete_axes"], [])

    def test_invented_gene_and_symbol_instead_of_id_rejected(self):
        for gene in ("invented", "SHARED"):
            with self.subTest(gene=gene):
                transport = CardTransport(mutation=lambda labels: labels[0].update(evidence_genes=[gene]))
                document = run_annotation(self.effects, self.root / (gene + ".json"), "explicit-model", transport)
                self.assertEqual(document["meta"]["last_error"]["kind"], "invalid_labels")
                self.assertFalse(document["meta"]["complete"])

    def test_resume_same_hash_skips_finished_axis(self):
        first = CardTransport()
        document = run_annotation(self.effects, self.output, "explicit-model", first, max_axes=1)
        self.assertFalse(document["meta"]["complete"])
        second = CardTransport()
        document = run_annotation(self.effects, self.output, "explicit-model", second, resume=True)
        self.assertTrue(document["meta"]["complete"])
        self.assertEqual([card["axis_id"] for card in second.cards], [1])
        self.assertEqual(document["meta"]["api_requests_total"], 2)

    def test_resume_refuses_different_hash_model_or_prompt(self):
        run_annotation(self.effects, self.output, "explicit-model", CardTransport(), max_axes=1)
        with self.assertRaises(AnnotationError):
            run_annotation(self.effects, self.output, "other-model", CardTransport(), resume=True)
        self.write_effects(shift=1)
        with self.assertRaises(AnnotationError):
            run_annotation(self.effects, self.output, "explicit-model", CardTransport(), resume=True)
        self.write_effects()
        saved = json.loads(self.output.read_text())
        saved["meta"]["prompt_version"] = "other-version"
        self.output.write_text(json.dumps(saved))
        with self.assertRaises(AnnotationError):
            run_annotation(self.effects, self.output, "explicit-model", CardTransport(), resume=True)

    def test_failure_preserves_prior_axis_and_omits_exception_body(self):
        transport = CardTransport(break_axis=1)
        document = run_annotation(self.effects, self.output, "explicit-model", transport)
        self.assertEqual(document["meta"]["status"], "failed")
        self.assertEqual(document["meta"]["complete_axes"], [0])
        self.assertEqual(len(document["labels"]), 16)
        self.assertNotIn("credential-and-HTTP-body", self.output.read_text())
        resumed = run_annotation(self.effects, self.output, "explicit-model", CardTransport(), resume=True)
        self.assertTrue(resumed["meta"]["complete"])

    def test_max_axes_budget_is_enforced(self):
        for maximum in (0, 33, True):
            with self.subTest(maximum=maximum):
                with self.assertRaises(AnnotationError):
                    run_annotation(self.effects, self.output, "explicit-model", CardTransport(), max_axes=maximum)

    def test_cli_missing_credentials_makes_no_request(self):
        argv = ["annotate_cards.py", "--effects", str(self.effects), "--output", str(self.output), "--model", "explicit-model"]
        with patch.dict(os.environ, {}, clear=True), patch("sys.argv", argv), patch("annotate_cards.ChatCompletionsClient") as client, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                main()
        self.assertEqual(caught.exception.code, 2)
        client.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_empty_names_long_summaries_and_wrong_card_genes_rejected(self):
        cards = build_axis_cards(load_effects(self.effects), 0)
        transport = CardTransport()
        completion = transport(messages=[{"content": json.dumps(cards)}], tools=[])
        content = completion["choices"][0]["message"]["content"]
        valid = parse_label_response(content, cards)
        labels = [{key: value for key, value in label.items() if key != "axis_id"} for label in valid.values()]
        labels[0]["name"] = " "
        with self.assertRaises(AnnotationError):
            parse_label_response(json.dumps({"labels": labels}), cards)
        labels[0]["name"] = "unclear"
        labels[0]["summary"] = "x" * 401
        with self.assertRaises(AnnotationError):
            parse_label_response(json.dumps({"labels": labels}), cards)
        labels[0]["summary"] = "unclear"
        labels[0]["evidence_genes"] = ["ENSG4"]
        with self.assertRaises(AnnotationError):
            parse_label_response(json.dumps({"labels": labels}), cards)


if __name__ == "__main__":
    unittest.main()
