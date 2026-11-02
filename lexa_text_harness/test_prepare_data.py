"""Small synthetic checks of condition parsing, QC, splits and train-only HVGs."""

from pathlib import Path
import gzip
import tempfile
import unittest

import numpy as np
import pandas as pd
from scipy import sparse

import prepare_data as prep


def synthetic_metadata():
    rows = []
    # The actual study uses a more interleaved plate layout. This mock keeps
    # the same author replicate-column rule and three doses per drug.
    for drug, row in zip(prep.DRUGS, "ACEG"):
        for dose_idx, dose in enumerate([0, 0.1, 1, 5]):
            for replicate in (1, 2, 3):
                column = dose_idx * 3 + replicate
                oligo = f"{drug}_{dose}_A{row}{column:02}"
                rows.append({"cell_id": f"cell_{len(rows)}", "top_oligo": oligo,
                             "pass_qc": True, **prep.parse_condition(oligo)})
    return pd.DataFrame(rows)


class PrepareDataTest(unittest.TestCase):
    def test_parser_preserves_invalid_well_and_missing_is_not_control(self):
        parsed = prep.parse_condition("Dex_0.1_BC07")
        self.assertEqual((parsed["drug"], parsed["dose_uM"], parsed["plate"], parsed["replicate"]),
                         ("Dex", 0.1, "B", 1))
        malformed = prep.parse_condition("SAHA_100_E09")
        self.assertEqual(malformed["well"], "E09")
        self.assertEqual(malformed["parse_status"], "invalid_well")
        self.assertIsNone(malformed["plate"])
        self.assertIsNone(malformed["replicate"])
        self.assertFalse(prep.parse_condition(np.nan)["is_control"])
        self.assertEqual(prep.parse_condition("Dex_0_AC08")["vehicle"], "Ethanol")
        self.assertEqual(prep.parse_condition("BMS_0_AG08")["vehicle"], "DMSO")

    def test_qc_strict_hash_thresholds_and_count_derived_expression(self):
        # Last gene mitochondrial; row 0 is 10% exactly and should pass.
        counts = sparse.csr_matrix([[4, 5, 1], [4, 5, 1], [4, 5, 1],
                                     [4, 5, 1], [4, 4, 2], [1, 0, 0]])
        metadata = pd.DataFrame({"top_to_second_best_ratio": [11, 10, 11, 11, 11, 11],
                                 "qval": [0.009, 0.009, 0.01, 0.009, 0.009, 0.009],
                                 "hash_umis": [31, 31, 31, 30, 31, 31],
                                 "n.umi": [999999] * 6})
        qc = prep.calculate_qc(counts, np.asarray(["A", "B", "MT-CO1"]), metadata,
                               min_umi=10, min_genes=3, max_mito_pct=10)
        self.assertEqual(qc["pass_qc"].tolist(), [True, False, False, False, False, False])
        self.assertEqual(qc["total_umi"].tolist(), [10, 10, 10, 10, 10, 1])

    def test_seen_drug_holdout_never_splits_a_well_and_keeps_controls(self):
        metadata = synthetic_metadata()
        result, doses = prep.assign_splits(metadata)
        self.assertEqual(doses, {drug: 1.0 for drug in prep.DRUGS})
        self.assertTrue(result.loc[result["dose_uM"].eq(1), "split"].eq("test").all())
        self.assertTrue(result.loc[result["is_control"], "split"].isin(["train", "val"]).all())
        self.assertEqual(int(result.groupby("well")["split"].nunique().max()), 1)
        self.assertTrue(result.loc[result["split"].eq("val"), "replicate"].eq(3).all())

    def test_leave_drug_out_and_exact_drug_control_matching(self):
        metadata = synthetic_metadata()
        result, _ = prep.assign_splits(metadata, "leave_drug_out", "Dex")
        test = result[result["split"].eq("test")]
        self.assertTrue(test["drug"].eq("Dex").all())
        self.assertTrue(test["dose_uM"].gt(0).all())
        self.assertFalse((result["drug"].eq("Dex") & result["dose_uM"].gt(0)
                          & result["split"].isin(["train", "val"])).any())
        cases = prep.build_cases(result, min_case_cells=1)
        self.assertEqual(len(cases), 9)
        for case in cases:
            self.assertEqual(case["status"], "ready")
            control = result[result["cell_id"].isin(case["control_cell_ids"])]
            self.assertTrue(control["is_control"].all())
            self.assertTrue(control["drug"].eq(case["drug"]).all())
            self.assertTrue(control["replicate"].eq(case["replicate"]).all())

    def test_hvg_ignores_test_expression_and_normalizes_before_selection(self):
        # Gene 0 varies in train. Gene 2 varies enormously only in held-out
        # cells, and must not become an HVG due to that held-out variation.
        counts = sparse.csr_matrix([[1, 9, 0], [5, 5, 0], [1, 9, 1000000], [1, 9, 0]])
        logx = prep.normalize_log1p(counts)
        np.testing.assert_allclose(np.expm1(logx[:2].toarray()).sum(axis=1), 10000, rtol=2e-6)
        mask = np.asarray([True, True, False, False])
        selected, mean, variance, detected = prep.select_training_hvg(logx, counts, mask, n_genes=2, min_gene_cells=1)
        self.assertEqual(set(selected), {0, 1})
        self.assertEqual(detected[2], 0)
        self.assertEqual(mean[2], 0)
        self.assertEqual(variance[2], 0)

    def test_headerless_gene_cell_triplets_and_duplicate_coordinates(self):
        with tempfile.TemporaryDirectory(prefix="lexa_triplets_") as directory:
            path = Path(directory) / "counts.gz"
            with gzip.open(path, "wt") as handle:
                handle.write("1\t1\t2\n1\t1\t3\n3\t2\t7\n")
            matrix = prep.read_triplet_counts(path, n_cells=2, n_genes=3, chunk_size=2)
            np.testing.assert_array_equal(matrix.toarray(), [[5, 0, 0], [0, 0, 7]])


if __name__ == "__main__":
    unittest.main()
