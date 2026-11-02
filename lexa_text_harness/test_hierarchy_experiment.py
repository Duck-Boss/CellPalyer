import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import hierarchy_experiment as hx


def axis_card(axis):
    return {"axis_id": axis, "top_mask_genes": [{"gene_id": f"G{axis}"}],
            "states": [{"state_id": 0, "name": "mixed", "summary": "signed evidence"}]}


def annotation(axis):
    return {"axis_id": axis, "name": "provisional expression program", "summary": "Gene-grounded scope.",
            "functions": ["unknown"], "evidence_genes": [f"G{axis}"], "uncertainty": "Not a causal pathway label."}


class FakeAnnotationClient:
    model = "offline-fixture"
    url = "offline://no-network"

    def __init__(self):
        self.calls = 0

    def __call__(self, *, messages, tools):
        self.calls += 1
        batch = json.loads(messages[1]["content"])["axes"]
        return {"choices": [{"message": {"content": json.dumps({"axes": [annotation(c["axis_id"]) for c in batch]})}}]}


class HierarchyExperimentTests(unittest.TestCase):
    def test_annotation_requires_displayed_axis_specific_gene(self):
        batch = [axis_card(0), axis_card(1)]
        rows = [annotation(0), annotation(1)]
        self.assertEqual(set(hx.parse_annotations(json.dumps({"axes": rows}), batch)), {0, 1})
        rows[0]["evidence_genes"] = ["G1"]
        with self.assertRaisesRegex(ValueError, "displayed gene"):
            hx.parse_annotations(json.dumps({"axes": rows}), batch)

    def test_annotation_rejects_duplicate_axis_gene_and_boolean_id(self):
        batch = [axis_card(0), axis_card(1)]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            hx.parse_annotations(json.dumps({"axes": [annotation(0), annotation(0)]}), batch)
        rows = [annotation(0), annotation(1)]
        rows[0]["evidence_genes"] = ["G0", "G0"]
        with self.assertRaisesRegex(ValueError, "unique displayed"):
            hx.parse_annotations(json.dumps({"axes": rows}), batch)
        rows = [annotation(0), annotation(1)]
        rows[0]["axis_id"] = False
        with self.assertRaisesRegex(ValueError, "axis ID"):
            hx.parse_annotations(json.dumps({"axes": rows}), batch)

    def test_annotation_unknown_can_abstain_from_genes(self):
        row = annotation(0)
        row["evidence_genes"] = []
        self.assertEqual(hx.parse_annotations(json.dumps({"axes": [row]}), [axis_card(0)])[0]["functions"], ["unknown"])

    def test_full_annotation_batches_cached_and_input_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            cards = {"axes": [axis_card(a) for a in range(32)]}
            client = FakeAnnotationClient()
            result = hx.annotate_axes(cards, client, tmp)
            self.assertEqual(client.calls, 4)
            self.assertEqual(len(result["axes"]), 32)
            self.assertEqual(hx.annotate_axes(cards, client, tmp), result)
            self.assertEqual(client.calls, 4)
            changed = copy.deepcopy(cards)
            changed["axes"][0]["scope"] = "different"
            with self.assertRaisesRegex(ValueError, "inputs changed"):
                hx.annotate_axes(changed, client, tmp)

    def test_score_missing_seal_never_opens_expression(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(hx.np, "load") as load:
            with self.assertRaises(FileNotFoundError):
                hx.score(Path(tmp) / "work", Path(tmp) / "output")
            load.assert_not_called()

    def test_mutated_sealed_input_blocks_score_before_expression(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            file = out / "sealed.txt"
            file.write_text("original")
            seal = {"status": "all_predictions_complete", "files": {"sealed.txt": hx.sha256(file)},
                    "source_hashes": hx.source_hashes()}
            hx.save(out / "prediction_seal.json", seal)
            file.write_text("changed")
            with patch.object(hx.np, "load") as load:
                with self.assertRaisesRegex(ValueError, "Sealed file changed"):
                    hx.score(out / "work", out)
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
