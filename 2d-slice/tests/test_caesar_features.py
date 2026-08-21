from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


if nn is not None:

    class _ChannelProjection3d(nn.Module):
        def __init__(self, input_channels: int, output_channels: int) -> None:
            super().__init__()
            self.projection = nn.Conv3d(input_channels, output_channels, kernel_size=1)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.projection(values)

    class _Double3d(nn.Module):
        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.interpolate(
                values,
                scale_factor=2.0,
                mode="trilinear",
                align_corners=True,
            )

    class _FakeEntropyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dec = nn.ModuleList(
                [
                    nn.ModuleList([_ChannelProjection3d(64, 48), _Double3d()]),
                    nn.ModuleList([_ChannelProjection3d(48, 32), _Double3d()]),
                    nn.ModuleList([nn.Identity(), nn.Identity()]),
                ]
            )

    class _FakeSuperResolution(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Conv2d(32, 1, kernel_size=1)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            values = self.projection(values)
            return torch.nn.functional.interpolate(
                values,
                scale_factor=4.0,
                mode="bilinear",
                align_corners=True,
            )

    class _FakeCaesarModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.entropy_model = _FakeEntropyModel()
            self.sr_model = _FakeSuperResolution()


@unittest.skipIf(torch is None, "PyTorch is not installed")
class CaesarFeatureTests(unittest.TestCase):
    def test_tap_metadata_records_expected_shapes_and_sizes(self) -> None:
        from slice_decoder.caesar_features import caesar_feature_tap_metadata

        early = caesar_feature_tap_metadata("early")
        late = caesar_feature_tap_metadata("late")
        self.assertEqual(early.shape, (48, 4, 32, 32))
        self.assertEqual(early.float32_bytes, 786_432)
        self.assertEqual(late.shape, (32, 8, 64, 64))
        self.assertEqual(late.float32_bytes, 4_194_304)

    def test_extract_and_continue_match_complete_fake_decoder(self) -> None:
        from slice_decoder import caesar_features

        fake_model = _FakeCaesarModel()
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.pt"
            model_path.write_bytes(b"fake")
            with mock.patch.object(
                caesar_features,
                "_load_caesar_v_model",
                return_value=fake_model,
            ):
                decoder = caesar_features.FrozenCaesarVFeatureDecoder(model_path)

        latent = torch.randn(2, 64, 2, 16, 16)
        early = decoder.extract(latent, "early")
        late = decoder.extract(latent, "late")
        self.assertEqual(early.shape, (2, 48, 4, 32, 32))
        self.assertEqual(late.shape, (2, 32, 8, 64, 64))
        early_output = decoder.continue_decode(early, "early")
        late_output = decoder.continue_decode(late, "late")
        self.assertEqual(early_output.shape, (2, 1, 8, 256, 256))
        torch.testing.assert_close(early_output, late_output)
        torch.testing.assert_close(decoder.decode_latent(latent), late_output)
        selected_indices = (6, 1, 4)
        selected_output = decoder.decode_late_frames(late, selected_indices)
        self.assertEqual(selected_output.shape, (2, 1, 3, 256, 256))
        torch.testing.assert_close(
            selected_output,
            late_output[:, :, selected_indices],
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in decoder.caesar_model.parameters()
            )
        )

    def test_rejects_incorrect_latent_and_feature_shapes(self) -> None:
        from slice_decoder import caesar_features

        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.pt"
            model_path.write_bytes(b"fake")
            with mock.patch.object(
                caesar_features,
                "_load_caesar_v_model",
                return_value=_FakeCaesarModel(),
            ):
                decoder = caesar_features.FrozenCaesarVFeatureDecoder(model_path)

        with self.assertRaisesRegex(ValueError, "latent must"):
            decoder.extract(torch.zeros(1, 32, 2, 16, 16), "early")
        with self.assertRaisesRegex(ValueError, "late features"):
            decoder.continue_decode(torch.zeros(1, 48, 4, 32, 32), "late")
        valid_late = torch.zeros(1, 32, 8, 64, 64)
        with self.assertRaisesRegex(ValueError, "at least one"):
            decoder.decode_late_frames(valid_late, ())
        with self.assertRaisesRegex(ValueError, "unique"):
            decoder.decode_late_frames(valid_late, (2, 2))
        with self.assertRaises(IndexError):
            decoder.decode_late_frames(valid_late, (8,))

    def test_copies_trainable_downstream_modules(self) -> None:
        from slice_decoder import caesar_features

        fake_model = _FakeCaesarModel()
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_path = Path(temporary_directory) / "model.pt"
            model_path.write_bytes(b"fake")
            with mock.patch.object(
                caesar_features,
                "_load_caesar_v_model",
                return_value=fake_model,
            ):
                decoder = caesar_features.FrozenCaesarVFeatureDecoder(model_path)

        stage, super_resolution = decoder.copy_downstream_2d_decoder()
        self.assertTrue(
            all(
                parameter.requires_grad
                for module in (stage, super_resolution)
                for parameter in module.parameters()
            )
        )
        source_parameter = next(decoder.caesar_model.sr_model.parameters())
        copied_parameter = next(super_resolution.parameters())
        self.assertNotEqual(source_parameter.data_ptr(), copied_parameter.data_ptr())
        with torch.no_grad():
            copied_parameter.add_(1.0)
        self.assertFalse(torch.equal(source_parameter, copied_parameter))


if __name__ == "__main__":
    unittest.main()
