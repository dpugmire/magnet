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
class PlaneDecoderTests(unittest.TestCase):
    def test_plane_sampling_is_exact_for_affine_features(self) -> None:
        from slice_decoder.plane_decoder import sample_latent_plane_features

        depth, height, width = 3, 4, 5
        z, y, x = np.meshgrid(
            np.linspace(-1.0, 1.0, depth),
            np.linspace(-1.0, 1.0, height),
            np.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        values = 2.0 * x - 3.0 * y + 5.0 * z + 7.0
        latent = torch.from_numpy(values.astype(np.float32))[None, None]
        plane_x, plane_y = torch.meshgrid(
            torch.linspace(-0.8, 0.8, 7),
            torch.linspace(-0.7, 0.7, 9),
            indexing="ij",
        )
        plane_z = 0.25 + 0.1 * plane_x
        points = torch.stack((plane_y, plane_x, plane_z), dim=-1).unsqueeze(0)
        sampled = sample_latent_plane_features(latent, points)
        expected = (
            2.0 * points[..., 0] - 3.0 * points[..., 1] + 5.0 * points[..., 2] + 7.0
        )
        torch.testing.assert_close(sampled[:, 0], expected)

    def test_slab_sampling_uses_latent_cell_offsets(self) -> None:
        from slice_decoder.plane_decoder import (
            sample_latent_plane_features,
            sample_latent_plane_slab_features,
        )

        z = torch.linspace(-1.0, 1.0, 3).reshape(1, 1, 3, 1, 1)
        latent = z.expand(1, 1, 3, 4, 5).contiguous()
        axis = torch.linspace(-0.75, 0.75, 6)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        points = torch.stack((x, y, torch.zeros_like(x)), dim=-1).unsqueeze(0)
        normals = torch.tensor([[0.0, 0.0, 1.0]])

        slab = sample_latent_plane_slab_features(
            latent,
            points,
            normals,
            sample_count=3,
            radius_cells=1.0,
        )
        self.assertEqual(slab.shape, (1, 3, 6, 6))
        torch.testing.assert_close(slab[:, 0], torch.full((1, 6, 6), -1.0))
        torch.testing.assert_close(slab[:, 1], torch.zeros(1, 6, 6))
        torch.testing.assert_close(slab[:, 2], torch.full((1, 6, 6), 1.0))

        center_only = sample_latent_plane_slab_features(
            latent,
            points,
            normals,
            sample_count=1,
            radius_cells=1.0,
        )
        torch.testing.assert_close(
            center_only,
            sample_latent_plane_features(latent, points),
        )

    def test_decoder_shapes_and_backpropagates(self) -> None:
        from slice_decoder.plane_decoder import (
            PlaneConvolutionalDecoder,
            PlaneDecoderConfig,
        )

        config = PlaneDecoderConfig(
            latent_channels=6,
            coarse_resolution=4,
            output_resolution=16,
            hidden_channels=12,
            minimum_channels=4,
            coarse_blocks=1,
            positional_frequencies=2,
            slab_samples=5,
            slab_radius_cells=1.0,
        )
        model = PlaneConvolutionalDecoder(config)
        latent = torch.randn(1, 6, 2, 4, 4)
        axis = torch.linspace(-1.0, 1.0, 16)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        points = torch.stack((x, y, torch.zeros_like(x)), dim=-1)
        output = model(latent, points.repeat(3, 1, 1, 1))
        self.assertEqual(output.shape, (3, 1, 16, 16))
        output.square().mean().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in model.parameters())
        )

    def test_checkpoint_round_trip(self) -> None:
        from slice_decoder.plane_decoder import (
            PlaneConvolutionalDecoder,
            PlaneDecoderConfig,
            load_plane_decoder_checkpoint,
            save_plane_decoder_checkpoint,
        )

        torch.manual_seed(3)
        config = PlaneDecoderConfig(
            latent_channels=4,
            coarse_resolution=4,
            output_resolution=8,
            hidden_channels=8,
            minimum_channels=4,
            coarse_blocks=1,
            positional_frequencies=1,
            slab_samples=3,
            slab_radius_cells=0.5,
        )
        model = PlaneConvolutionalDecoder(config)
        latent = torch.randn(1, 4, 2, 4, 4)
        axis = torch.linspace(-1.0, 1.0, 8)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        points = torch.stack((x, y, torch.zeros_like(x)), dim=-1).unsqueeze(0)
        expected = model(latent, points)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "plane.pt"
            save_plane_decoder_checkpoint(path, model, metadata={"block": 2})
            restored, metadata = load_plane_decoder_checkpoint(path)
            actual = restored(latent, points)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(metadata["block"], 2)
        self.assertEqual(restored.config.slab_samples, 3)

    def test_loads_pre_slab_checkpoint_as_single_plane(self) -> None:
        from slice_decoder.plane_decoder import (
            PlaneConvolutionalDecoder,
            PlaneDecoderConfig,
            load_plane_decoder_checkpoint,
        )

        config = PlaneDecoderConfig(
            latent_channels=4,
            coarse_resolution=4,
            output_resolution=8,
            hidden_channels=8,
            minimum_channels=4,
            coarse_blocks=1,
            positional_frequencies=1,
        )
        model = PlaneConvolutionalDecoder(config)
        legacy_config = config.to_dict()
        legacy_config.pop("slab_samples")
        legacy_config.pop("slab_radius_cells")
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy-plane.pt"
            torch.save(
                {
                    "formatVersion": 1,
                    "modelConfig": legacy_config,
                    "modelState": model.state_dict(),
                    "metadata": {},
                },
                path,
            )
            restored, _ = load_plane_decoder_checkpoint(path)
        self.assertEqual(restored.config.slab_samples, 1)
        self.assertEqual(restored.config.slab_radius_cells, 1.0)

    def test_config_requires_power_of_two_scale(self) -> None:
        from slice_decoder.plane_decoder import PlaneDecoderConfig

        with self.assertRaisesRegex(ValueError, "power of two"):
            PlaneDecoderConfig(
                latent_channels=4,
                coarse_resolution=6,
                output_resolution=16,
            )
        with self.assertRaisesRegex(ValueError, "must be odd"):
            PlaneDecoderConfig(
                latent_channels=4,
                coarse_resolution=4,
                output_resolution=16,
                slab_samples=4,
            )


if __name__ == "__main__":
    unittest.main()
