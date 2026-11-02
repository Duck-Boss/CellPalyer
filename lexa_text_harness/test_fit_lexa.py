"""CPU-only synthetic checks; no public data download or API request."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import fit_lexa


class FitLexaTest(unittest.TestCase):
    def write_synthetic(self, path):
        rng = np.random.default_rng(71)
        # Positive log1p expression with correlated groups, never real cells.
        counts = rng.poisson(3, (64, 64)).astype(np.float32)
        expression = np.log1p(counts / np.maximum(counts.sum(axis=1, keepdims=True), 1) * 10000)
        np.savez_compressed(path, x_train=expression[:40], x_val=expression[40:52],
                            control_x=expression[52:], genes=np.asarray([f"SYNTHETIC_{i}" for i in range(64)]),
                            train_cell_ids=np.asarray([f"train_{i}" for i in range(40)]),
                            val_cell_ids=np.asarray([f"val_{i}" for i in range(12)]),
                            control_cell_ids=np.asarray([f"control_{i}" for i in range(12)]),
                            synthetic=np.asarray(True),
                            provenance_json=np.asarray(json.dumps({"dataset": "synthetic unittest only"})))

    def test_train_export_and_exact_code_edit(self):
        with tempfile.TemporaryDirectory(prefix="lexa_fit_synthetic_") as directory:
            root = Path(directory)
            prepared = root / "synthetic_prepared.npz"
            self.write_synthetic(prepared)
            output = root / "model"
            args = fit_lexa.parse_args(["--prepared", str(prepared), "--output", str(output),
                                        "--device", "cpu", "--epochs", "2", "--batch-size", "8",
                                        "--smoke", "2", "--threads", "2", "--seed", "71"])
            meta = fit_lexa.run(args)
            self.assertTrue(meta["synthetic"])
            self.assertEqual(meta["status"], "smoke_complete")
            self.assertEqual(meta["training"]["updates"], 2)
            self.assertEqual(meta["training"]["cell_updates"], 16)
            self.assertLess(meta["additivity_max_abs_error"], 2e-5)
            complete = json.loads((output / "COMPLETE.json").read_text())
            self.assertEqual(complete["status"], "smoke_complete")
            with np.load(output / "exact_code_effects.npz", allow_pickle=False) as archive:
                effects = archive["contributions"]
                self.assertEqual(effects.shape, (32, 16, 64))
                np.testing.assert_allclose(archive["train_occupancy"].sum(axis=1), 1)
                self.assertTrue(bool(archive["synthetic"]))
            with np.load(output / "baseline.npz", allow_pickle=False) as archive:
                codes = archive["codes"].copy()
                base_recon = archive["reconstructed_x"].copy()
                self.assertEqual(archive["cell_ids"][0], "control_0")
            ckpt = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
            original_model, _ = fit_lexa.import_original(args.lexa_repo)
            model = original_model(**ckpt["config"], module_mask_prior=ckpt["module_mask_prior"])
            model.load_state_dict(ckpt["model_state"])
            model.eval()
            old = int(codes[0, 3])
            new = (old + 1) % 16
            edited = codes[:1].copy()
            edited[0, 3] = new
            with torch.no_grad():
                cb = model.quantization_codebook()
                z = torch.stack([cb[a, torch.as_tensor(edited[:, a], dtype=torch.long)] for a in range(32)], dim=1)
                decoded = model.decode(z).numpy()[0]
            # An actual discrete edit must induce the exact declared gene delta.
            expected_delta = effects[3, new] - effects[3, old]
            np.testing.assert_allclose(decoded - base_recon[0], expected_delta, atol=2e-6, rtol=1e-4)

    def test_overlapping_train_val_ids_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lexa_fit_overlap_") as directory:
            path = Path(directory) / "prepared.npz"
            self.write_synthetic(path)
            with np.load(path) as archive:
                payload = {key: archive[key] for key in archive.files}
            payload["val_cell_ids"] = payload["val_cell_ids"].astype("U32")
            payload["val_cell_ids"][0] = payload["train_cell_ids"][0]
            np.savez(path, **payload)
            with self.assertRaisesRegex(ValueError, "overlap"):
                fit_lexa.load_prepared(path)


if __name__ == "__main__":
    unittest.main()
