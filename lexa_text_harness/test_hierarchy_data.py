import unittest

import numpy as np
import pandas as pd

from hierarchy_data import (project_simplex, occupancy, build_training_library,
                            retrieve, candidate_distributions, CANDIDATE_IDS)


class HierarchyDataTests(unittest.TestCase):
    def test_euclidean_projection_not_clip_normalize(self):
        q = project_simplex(np.array([[.8, .8, -.6], [.2, .3, .5], [3., 1., 0.]]))
        np.testing.assert_allclose(q, [[.5, .5, 0], [.2, .3, .5], [1., 0, 0]], atol=1e-12)
        with self.assertRaises(ValueError):
            project_simplex([np.nan, 1])

    def test_code_ids_are_categorical(self):
        result = occupancy(np.array([[0, 2], [2, 2], [2, 0]]), 3)
        np.testing.assert_allclose(result, [[1/3, 0, 2/3], [1/3, 0, 2/3]])
        with self.assertRaises(ValueError):
            occupancy(np.array([[.1, 2.]]), 3)

    def fixture(self):
        rows, codes, values = [], [], []
        # Unequal treatment cell counts must not weight the well responses.
        for drug, rep, dose, n, code, expression in [
                ("D", 1, 0, 2, 0, 1), ("D", 1, 1, 2, 1, 3),
                ("D", 2, 0, 2, 0, 10), ("D", 2, 1, 6, 0, 16),
                ("X", 1, 0, 4, 1, 1000), ("D", 3, 1, 2, 1, 99)]:
            for i in range(n):
                cid = f"{drug}_{rep}_{dose}_{i}"
                rows.append({"cell_id": cid, "drug": drug, "replicate": rep, "dose_uM": dose,
                             "plate": "A", "well": f"{drug}_{rep}_{dose}", "vehicle": "DMSO",
                             "split": "train", "parse_status": "ok"})
                codes.append([code]); values.append([expression])
        metadata = pd.DataFrame(rows).set_index("cell_id")
        return np.array(codes), np.array(values), metadata.index.to_numpy(), metadata

    def test_equal_well_mean_and_strict_drug_replicate_controls(self):
        library, arrays, wells = build_training_library(*self.fixture(), n_states=2, min_cells=2)
        self.assertEqual(len(library["ready_wells"]), 2)
        self.assertEqual(len(library["excluded_wells"]), 1)
        self.assertEqual(library["excluded_wells"][0]["n_control"], 0)
        np.testing.assert_allclose(arrays["observed_delta"], [[4.]])
        np.testing.assert_allclose(arrays["delta_p"], [[[-.5, .5]]])
        self.assertEqual(library["conditions"][0]["n_wells"], 2)

    def test_reject_nontraining_ids(self):
        codes, x, ids, metadata = self.fixture()
        metadata.loc[ids[0], "split"] = "test"
        with self.assertRaisesRegex(ValueError, "only contain training"):
            build_training_library(codes, x, ids, metadata, n_states=2, min_cells=2)

    def test_log_dose_interpolation_and_no_extrapolation(self):
        library = {"conditions": [{"drug": "D", "dose_uM": 1., "array_index": 0},
                                   {"drug": "D", "dose_uM": 100., "array_index": 1}]}
        arrays = {"delta_p": np.array([[[-.1, .1]], [[-.5, .5]]]), "observed_delta": np.array([[2.], [10.]])}
        info, result = retrieve("D", 10., library, arrays)
        self.assertAlmostEqual(info["upper_weight"], .5)
        self.assertEqual(info["nearest"]["dose_uM"], 1.)
        np.testing.assert_allclose(result["observed_delta"]["interpolate"], [6.])
        _, result = retrieve("D", .01, library, arrays)
        np.testing.assert_allclose(result["observed_delta"]["interpolate"], [2.])
        with self.assertRaises(ValueError):
            retrieve("unknown", 5, library, arrays)

    def test_candidates_keep_exact_and_record_projection(self):
        p0 = np.array([[.1, .9]])
        retrieved = {"delta_p": {k: np.array([[-.4, .4]]) for k in ["nearest", "interpolate", "lower", "upper"]}}
        q, correction, raw = candidate_distributions(p0, retrieved)
        self.assertEqual(q.shape, (1, len(CANDIDATE_IDS), 2))
        np.testing.assert_array_equal(q[:, 0], p0)
        np.testing.assert_allclose(q[:, 2], [[0, 1]])
        self.assertAlmostEqual(correction[0, 2], .6)
        np.testing.assert_allclose(q.sum(axis=-1), 1)


if __name__ == "__main__":
    unittest.main()
