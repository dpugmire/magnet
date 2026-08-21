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
class PointDecoderTests(unittest.TestCase):
    def test_latent_sampling_is_exact_for_affine_features(self) -> None:
        from slice_decoder.point_decoder import sample_latent_features

        depth, height, width = 3, 4, 5
        z, y, x = np.meshgrid(
            np.linspace(-1.0, 1.0, depth),
            np.linspace(-1.0, 1.0, height),
            np.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        values = 2.0 * x - 3.0 * y + 5.0 * z + 7.0
        latent = torch.from_numpy(values.astype(np.float32))[None, None]
        points = torch.tensor(
            [
                [[-0.7, -0.5, -0.25], [0.1, 0.2, 0.3]],
                [[0.8, -0.4, 0.6], [-0.2, 0.7, -0.8]],
            ],
            dtype=torch.float32,
        )
        sampled = sample_latent_features(latent, points)
        expected = (
            2.0 * points[..., 0]
            - 3.0 * points[..., 1]
            + 5.0 * points[..., 2]
            + 7.0
        )
        torch.testing.assert_close(sampled[..., 0], expected)

    def test_decoder_shapes_and_backpropagates(self) -> None:
        from slice_decoder.point_decoder import PointDecoderConfig, PointQueryDecoder

        config = PointDecoderConfig(
            latent_channels=6,
            hidden_dimension=16,
            hidden_layers=2,
            positional_frequencies=3,
        )
        model = PointQueryDecoder(config)
        latent = torch.randn(1, 6, 2, 4, 4)
        points = torch.rand(3, 20, 3) * 2.0 - 1.0
        output = model(latent, points)
        self.assertEqual(output.shape, (3, 20, 1))
        output.square().mean().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters())
        )

    def test_checkpoint_round_trip(self) -> None:
        from slice_decoder.point_decoder import (
            PointDecoderConfig,
            PointQueryDecoder,
            load_point_decoder_checkpoint,
            save_point_decoder_checkpoint,
        )

        torch.manual_seed(3)
        model = PointQueryDecoder(PointDecoderConfig(4, 12, 2, 1))
        latent = torch.randn(1, 4, 2, 3, 3)
        points = torch.rand(1, 7, 3) * 2.0 - 1.0
        expected = model(latent, points)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "point.pt"
            save_point_decoder_checkpoint(path, model, metadata={"block": 2})
            restored, metadata = load_point_decoder_checkpoint(path)
            actual = restored(latent, points)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(metadata["block"], 2)


if __name__ == "__main__":
    unittest.main()
