"""Frozen intermediate feature taps from the pretrained CAESAR-V decoder."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
from torch import nn


CaesarFeatureTap = Literal["early", "late"]


@dataclass(frozen=True)
class CaesarFeatureTapMetadata:
    name: CaesarFeatureTap
    channels: int
    depth: int
    height: int
    width: int

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.channels, self.depth, self.height, self.width

    @property
    def float32_bytes(self) -> int:
        return self.channels * self.depth * self.height * self.width * 4


_TAP_METADATA = {
    "early": CaesarFeatureTapMetadata("early", 48, 4, 32, 32),
    "late": CaesarFeatureTapMetadata("late", 32, 8, 64, 64),
}


def caesar_feature_tap_metadata(tap: CaesarFeatureTap) -> CaesarFeatureTapMetadata:
    try:
        return _TAP_METADATA[tap]
    except KeyError as error:
        raise ValueError(f"unsupported CAESAR feature tap {tap!r}") from error


def _load_caesar_v_model(model_path: Path, device: torch.device) -> nn.Module:
    try:
        from CAESAR.models.compress_modules3d_mid_SR import CompressorMix
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "loading CAESAR feature taps requires external/pyCAESAR on PYTHONPATH"
        ) from error

    model = CompressorMix(
        dim=16,
        dim_mults=[1, 2, 3, 4],
        reverse_dim_mults=[4, 3, 2],
        hyper_dims_mults=[4, 4, 4],
        channels=1,
        out_channels=1,
        d3=True,
        sr_dim=16,
    )
    state_dict: dict[str, Any] = torch.load(model_path, map_location=device)
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class FrozenCaesarVFeatureDecoder(nn.Module):
    """Expose and continue CAESAR-V's pretrained 3D decoder stages."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.device = torch.device(device)
        self.caesar_model = _load_caesar_v_model(self.model_path, self.device)
        self.caesar_model.eval()
        for parameter in self.caesar_model.parameters():
            parameter.requires_grad_(False)

    @property
    def decoder_stages(self) -> nn.ModuleList:
        return self.caesar_model.entropy_model.dec

    @staticmethod
    def _validate_latent(latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        if latent.ndim != 5:
            raise ValueError(f"latent must be [B,C,D,H,W], got {latent.shape}")
        expected_input = (64, 2, 16, 16)
        if latent.shape[1:] != expected_input:
            raise ValueError(
                f"CAESAR-V latent must be [B,{expected_input}], got {latent.shape}"
            )
        return latent

    @staticmethod
    def _validate_features(
        features: torch.Tensor,
        tap: CaesarFeatureTap,
    ) -> torch.Tensor:
        if features.ndim == 4:
            features = features.unsqueeze(0)
        if features.ndim != 5:
            raise ValueError(f"features must be [B,C,D,H,W], got {features.shape}")
        expected = caesar_feature_tap_metadata(tap).shape
        if features.shape[1:] != expected:
            raise ValueError(
                f"{tap} features must be [B,{expected}], got {features.shape}"
            )
        return features

    @staticmethod
    def _apply_stage(stage: nn.ModuleList, values: torch.Tensor) -> torch.Tensor:
        residual, upsample = stage
        return upsample(residual(values))

    def extract(
        self,
        latent: torch.Tensor,
        tap: CaesarFeatureTap,
    ) -> torch.Tensor:
        """Run the frozen decoder through the selected contextual feature tap."""

        early, late = self.extract_taps(latent, include_late=tap == "late")
        return early if tap == "early" else late

    def extract_taps(
        self,
        latent: torch.Tensor,
        *,
        include_late: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Return early and optionally late features from one decoder pass."""

        values = self._validate_latent(latent).to(self.device)
        with torch.no_grad():
            early = self._apply_stage(self.decoder_stages[0], values)
            late = (
                self._apply_stage(self.decoder_stages[1], early)
                if include_late
                else None
            )
        return early, late

    def copy_downstream_2d_decoder(self) -> tuple[nn.Module, nn.Module]:
        """Copy CAESAR's trainable late-feature-to-scalar 2D modules."""

        residual, upsample = self.decoder_stages[2]
        stage = nn.Sequential(copy.deepcopy(residual), copy.deepcopy(upsample))
        super_resolution = copy.deepcopy(self.caesar_model.sr_model)
        for module in (stage, super_resolution):
            module.train()
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        return stage, super_resolution

    def continue_decode(
        self,
        features: torch.Tensor,
        tap: CaesarFeatureTap,
    ) -> torch.Tensor:
        """Continue a tapped feature tensor to normalized CAESAR base output."""

        values = self._validate_features(features, tap).to(self.device)
        with torch.no_grad():
            if tap == "early":
                values = self._apply_stage(self.decoder_stages[1], values)
        return self.decode_late_frames(values, range(values.shape[2]))

    def decode_late_frames(
        self,
        features: torch.Tensor,
        frame_indices: Sequence[int],
    ) -> torch.Tensor:
        """Decode selected normalized base frames from one late feature volume.

        The remaining CAESAR-V layers process depth frames independently. The
        returned depth dimension follows ``frame_indices`` exactly.
        """

        values = self._validate_features(features, "late").to(self.device)
        indices = tuple(int(index) for index in frame_indices)
        if not indices:
            raise ValueError("at least one late frame must be selected")
        if len(set(indices)) != len(indices):
            raise ValueError("late frame indices must be unique")
        depth = values.shape[2]
        if any(index < 0 or index >= depth for index in indices):
            raise IndexError(
                f"late frame indices must be within [0,{depth - 1}], got {indices}"
            )

        selected_indices = torch.tensor(indices, device=values.device)
        values = torch.index_select(values, 2, selected_indices)
        batch_size = values.shape[0]
        selected_depth = values.shape[2]
        with torch.no_grad():
            values = values.permute(0, 2, 1, 3, 4).reshape(
                -1,
                values.shape[1],
                values.shape[3],
                values.shape[4],
            )
            values = self._apply_stage(self.decoder_stages[2], values)
            values = self.caesar_model.sr_model(values)
            values = values.reshape(
                batch_size,
                selected_depth,
                values.shape[1],
                values.shape[2],
                values.shape[3],
            ).permute(0, 2, 1, 3, 4)
        return values

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode canonical q_latent through all neural CAESAR-V stages."""

        early = self.extract(latent, "early")
        return self.continue_decode(early, "early")
