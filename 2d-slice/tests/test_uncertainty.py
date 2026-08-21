from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PlaneUncertaintyTests(unittest.TestCase):
    def test_head_is_positive_and_backpropagates_only_through_head(self) -> None:
        from slice_decoder.uncertainty import (
            PlaneUncertaintyConfig,
            PlaneUncertaintyHead,
            gaussian_scale_nll,
        )

        model = PlaneUncertaintyHead(
            PlaneUncertaintyConfig(
                feature_channels=8,
                hidden_channels=4,
                initial_scale=0.2,
            )
        )
        features = torch.randn(2, 8, 12, 12)
        prediction = torch.randn(2, 1, 12, 12)
        target = torch.randn_like(prediction)
        scale = model(features, prediction)
        self.assertEqual(scale.shape, prediction.shape)
        self.assertTrue(bool(torch.all(scale > 0.0)))
        torch.testing.assert_close(scale, torch.full_like(scale, 0.2))
        gaussian_scale_nll(target - prediction, scale).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertIsNone(features.grad)
        self.assertIsNone(prediction.grad)

    def test_conformal_quantile_uses_finite_sample_rank(self) -> None:
        from slice_decoder.uncertainty import conformal_scale_quantile

        scores = np.arange(1.0, 10.0)
        self.assertEqual(conformal_scale_quantile(scores, 0.8), 8.0)
        self.assertEqual(conformal_scale_quantile(scores, 0.9), 9.0)

    def test_checkpoint_round_trip(self) -> None:
        from slice_decoder.uncertainty import (
            PlaneUncertaintyConfig,
            PlaneUncertaintyHead,
            load_uncertainty_checkpoint,
            save_uncertainty_checkpoint,
        )

        torch.manual_seed(7)
        model = PlaneUncertaintyHead(
            PlaneUncertaintyConfig(feature_channels=3, hidden_channels=5)
        )
        features = torch.randn(1, 3, 8, 8)
        prediction = torch.randn(1, 1, 8, 8)
        expected = model(features, prediction)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "uncertainty.pt"
            save_uncertainty_checkpoint(
                path,
                model,
                calibration={"q95": 2.1},
                metadata={"decoder": "plane.pt"},
            )
            restored, calibration, metadata = load_uncertainty_checkpoint(path)
            actual = restored(features, prediction)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(calibration, {"q95": 2.1})
        self.assertEqual(metadata, {"decoder": "plane.pt"})


if __name__ == "__main__":
    unittest.main()
