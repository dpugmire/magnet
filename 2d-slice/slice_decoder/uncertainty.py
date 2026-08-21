"""Calibrated per-pixel uncertainty for frozen plane decoders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional


@dataclass(frozen=True)
class PlaneUncertaintyConfig:
    feature_channels: int
    hidden_channels: int = 32
    minimum_scale: float = 1.0e-4
    initial_scale: float = 0.1

    def __post_init__(self) -> None:
        if self.feature_channels <= 0:
            raise ValueError("feature_channels must be positive")
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if not math.isfinite(self.minimum_scale) or self.minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive and finite")
        if not math.isfinite(self.initial_scale):
            raise ValueError("initial_scale must be finite")
        if self.initial_scale <= self.minimum_scale:
            raise ValueError("initial_scale must exceed minimum_scale")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class PlaneUncertaintyHead(nn.Module):
    """Predict a positive normalized error scale from frozen decoder features."""

    def __init__(self, config: PlaneUncertaintyConfig) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Conv2d(
                config.feature_channels + 1,
                config.hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(
                config.hidden_channels,
                config.hidden_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(config.hidden_channels, 1, kernel_size=1),
        )
        final_layer = self.network[-1]
        if not isinstance(final_layer, nn.Conv2d):
            raise RuntimeError("unexpected uncertainty network")
        initial_raw_scale = math.log(
            math.expm1(config.initial_scale - config.minimum_scale)
        )
        with torch.no_grad():
            final_layer.weight.zero_()
            final_layer.bias.fill_(initial_raw_scale)

    def forward(
        self,
        features: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(f"features must be [B,C,H,W], got {features.shape}")
        if prediction.ndim != 4 or prediction.shape[1] != 1:
            raise ValueError(
                f"prediction must be [B,1,H,W], got {prediction.shape}"
            )
        if features.shape[0] != prediction.shape[0] or features.shape[2:] != (
            prediction.shape[2:]
        ):
            raise ValueError("features and prediction must have matching grids")
        if features.shape[1] != self.config.feature_channels:
            raise ValueError(
                f"features provide {features.shape[1]} channels, expected "
                f"{self.config.feature_channels}"
            )
        raw_scale = self.network(torch.cat((features, prediction), dim=1))
        return functional.softplus(raw_scale) + self.config.minimum_scale


def gaussian_scale_nll(
    residual: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log likelihood without the constant term."""

    if residual.shape != scale.shape:
        raise ValueError(
            f"residual and scale shapes differ: {residual.shape} vs {scale.shape}"
        )
    if torch.any(scale <= 0.0):
        raise ValueError("scale must be positive")
    return torch.mean(0.5 * (residual / scale).square() + torch.log(scale))


def conformal_scale_quantile(
    scores: np.ndarray,
    coverage: float,
) -> float:
    """Return the finite-sample split-conformal scale quantile."""

    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be in (0,1)")
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("scores contain no finite values")
    rank = min(int(math.ceil((values.size + 1) * coverage)), values.size)
    return float(np.partition(values, rank - 1)[rank - 1])


def save_uncertainty_checkpoint(
    path: str | Path,
    model: PlaneUncertaintyHead,
    *,
    calibration: dict[str, float],
    metadata: dict[str, Any],
) -> Path:
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "formatVersion": 1,
            "modelConfig": model.config.to_dict(),
            "modelState": model.state_dict(),
            "calibration": calibration,
            "metadata": metadata,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_uncertainty_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[PlaneUncertaintyHead, dict[str, float], dict[str, Any]]:
    checkpoint = torch.load(
        Path(path).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("formatVersion", 0)) != 1:
        raise ValueError("unsupported uncertainty checkpoint format")
    model = PlaneUncertaintyHead(
        PlaneUncertaintyConfig(**checkpoint["modelConfig"])
    ).to(device)
    model.load_state_dict(checkpoint["modelState"])
    calibration = {
        str(name): float(value)
        for name, value in dict(checkpoint.get("calibration", {})).items()
    }
    metadata = dict(checkpoint.get("metadata", {}))
    return model, calibration, metadata
