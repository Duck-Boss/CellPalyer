"""Pure metric tests for representation retention; no model fit or real targets."""

import unittest

import numpy as np

from representation_diagnostic import delta_metrics, reconstruction_mse


class RepresentationDiagnosticTest(unittest.TestCase):
    def test_exact_response_retains_sign_and_amplitude(self):
        actual = np.asarray([1.0, -1.0, 0.0])
        metrics = delta_metrics(actual, actual)
        self.assertEqual(metrics["delta_rmse"], 0)
        self.assertAlmostEqual(metrics["delta_pearson"], 1)
        self.assertEqual(metrics["direction_agreement"], 1)
        self.assertEqual(metrics["response_l2_ratio"], 1)
        self.assertAlmostEqual(metrics["response_projection_slope"], 1)
        self.assertEqual(metrics["n_direction_genes"], 2)

    def test_amplitude_alone_does_not_hide_reversed_direction(self):
        actual = np.asarray([1.0, -1.0, 0.0])
        metrics = delta_metrics(actual, -actual)
        self.assertAlmostEqual(metrics["delta_rmse"], 2 * np.sqrt(2 / 3))
        self.assertAlmostEqual(metrics["delta_pearson"], -1)
        self.assertEqual(metrics["direction_agreement"], 0)
        self.assertEqual(metrics["response_l2_ratio"], 1)
        self.assertAlmostEqual(metrics["response_projection_slope"], -1)

    def test_collapsed_response_is_undefined_correlation_and_disagrees(self):
        metrics = delta_metrics(np.asarray([1.0, -1.0, 0.0]), np.zeros(3))
        self.assertIsNone(metrics["delta_pearson"])
        self.assertEqual(metrics["direction_agreement"], 0)
        self.assertEqual(metrics["response_l2_ratio"], 0)
        no_effect = delta_metrics(np.zeros(3), np.zeros(3))
        self.assertIsNone(no_effect["response_l2_ratio"])
        self.assertIsNone(no_effect["direction_agreement"])
        self.assertIsNone(no_effect["response_projection_slope"])

    def test_invalid_vectors_rejected_and_mse_per_cell_gene(self):
        with self.assertRaises(ValueError):
            delta_metrics(np.ones(2), np.ones(3))
        with self.assertRaises(ValueError):
            delta_metrics(np.asarray([np.nan]), np.ones(1))
        self.assertEqual(reconstruction_mse(np.zeros((2, 3)), np.ones((2, 3))), 1)


if __name__ == "__main__":
    unittest.main()
