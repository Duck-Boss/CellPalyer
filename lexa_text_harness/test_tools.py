import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import Workspace


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        effect = np.zeros((2, 3, 4), np.float32)
        effect[0, 1] = [1, -2, 0, 0]
        effect[0, 2] = [2, 0, 0, 0]
        effect[1, 1] = [0, 0, 3, -1]
        self.effect = effect
        np.savez(self.root / 'effects.npz', contributions=effect,
                 genes=np.array(['G1', 'G2', 'G3', 'G4']),
                 train_gene_std=np.ones(4), train_occupancy=np.ones((2, 3)) / 3,
                 synthetic=True)
        np.savez(self.root / 'baseline.npz', codes=np.array([[0, 0], [0, 0], [1, 0], [1, 0]]), synthetic=True)
        (self.root / 'request.json').write_text(json.dumps({'mode': 'functional_edit', 'text': 'test'}))

    def tearDown(self):
        self.tmp.cleanup()

    def workspace(self, distribution='mse_log1p', synthetic=True):
        return Workspace(str(self.root / 'effects.npz'), str(self.root / 'baseline.npz'),
                         str(self.root / 'request.json'), distribution, str(self.root / 'out'), synthetic)

    def test_signed_transition_and_baseline_immutable(self):
        w = self.workspace()
        result = w.evaluate_edits([{'axis_id': 0, 'from_code': 1, 'to_code': 2}])
        data = np.load(result['artifact'])
        np.testing.assert_allclose(data['delta_mean'], [0.5, 1, 0, 0])
        np.testing.assert_array_equal(w.codes[:, 0], [0, 0, 1, 1])

    def test_fraction_edits_only_eligible_cells(self):
        w = self.workspace()
        edited, description = w.apply_edits([{'axis_id': 0, 'from_code': 0, 'to_code': 2, 'fraction': 0.5}])
        self.assertEqual(description[0]['changed_cells'], 1)
        np.testing.assert_array_equal(edited[2:], w.codes[2:])
        again, _ = w.apply_edits([{'axis_id': 0, 'from_code': 0, 'to_code': 2, 'fraction': 0.5}])
        np.testing.assert_array_equal(again, edited)

    def test_combined_edit_is_sum_in_mse(self):
        w = self.workspace()
        original = w.decode(w.codes)
        both, _ = w.apply_edits([{'axis_id': 0, 'to_code': 2}, {'axis_id': 1, 'to_code': 1}])
        a, _ = w.apply_edits([{'axis_id': 0, 'to_code': 2}])
        b, _ = w.apply_edits([{'axis_id': 1, 'to_code': 1}])
        np.testing.assert_allclose(w.decode(both) - original, w.decode(a) + w.decode(b) - 2 * original)

    def test_nb_decodes_before_averaging(self):
        w = self.workspace('nb')
        decoded = w.decode(w.codes)
        np.testing.assert_allclose(np.expm1(decoded).sum(axis=1), 10000, rtol=1e-5)
        effect_mean = np.zeros((1, 4), np.float32)
        for axis in range(2):
            effect_mean += self.effect[axis, w.codes[:, axis]].mean(axis=0)
        wrong = np.log1p(np.exp(effect_mean - effect_mean.max()) / np.exp(effect_mean - effect_mean.max()).sum() * 10000)
        self.assertFalse(np.allclose(decoded.mean(axis=0), wrong[0]))

    def test_invalid_edits_rejected(self):
        w = self.workspace()
        for edit in ([{'axis_id': True, 'to_code': 0}],
                     [{'axis_id': 0, 'to_code': 3}],
                     [{'axis_id': 0, 'to_code': 0, 'fraction': float('nan')}],
                     [{'axis_id': 0, 'to_code': 0}, {'axis_id': 0, 'to_code': 1}],
                     [{'axis_id': 0, 'to_code': 0, 'command': 'anything'}]):
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                w.apply_edits(edit)

    def test_synthetic_requires_explicit_flag(self):
        with self.assertRaises(ValueError):
            self.workspace(synthetic=False)

    def test_noop_and_no_target_access(self):
        w = self.workspace()
        result = w.evaluate_edits([])
        self.assertEqual(result['changed_cells'], 0)
        self.assertEqual(result['standardized_delta_rms'], 0)
        self.assertIn('no observed treatment data', result['evaluation_scope'].lower())


if __name__ == '__main__':
    unittest.main()
