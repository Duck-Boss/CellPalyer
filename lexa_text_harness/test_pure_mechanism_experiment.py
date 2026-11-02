import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pure_mechanism_experiment as e


class ExperimentTest(unittest.TestCase):
    def test_score_refuses_unsealed_before_target_load(self):
        with tempfile.TemporaryDirectory() as d, patch.object(e.np,'load',side_effect=AssertionError('Target access')):
            with self.assertRaises(FileNotFoundError):e.score(Path(d),Path(d))

    def test_seal_checks_artifacts_and_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'prediction.txt').write_text('fixed')
            e.save(root/'prediction_seal.json',{'status':'all_predictions_complete','score_exists_at_seal':False,
                'files':{'prediction.txt':e.sha(root/'prediction.txt')},'source_hashes':e.source_hashes()})
            e.verify_seal(root)
            (root/'prediction.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Sealed file changed'):e.verify_seal(root)

    def test_noedit_direction_and_magnitude(self):
        m=e.metrics(np.zeros(3),np.array([1.,-2.,3.]))
        self.assertAlmostEqual(m['delta_rmse'],np.sqrt(14/3))
        self.assertTrue(np.isnan(m['delta_pearson']))
        self.assertEqual(m['response_l2_ratio'],0)
        self.assertEqual(m['top100_direction_accuracy'],0)
        perfect=e.metrics(np.array([1.,-2.,3.]),np.array([1.,-2.,3.]))
        self.assertEqual(perfect['delta_rmse'],0)
        self.assertAlmostEqual(perfect['delta_cosine'],1)


if __name__=='__main__':unittest.main()
