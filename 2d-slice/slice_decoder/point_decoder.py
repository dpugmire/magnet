"""Point-query decoder for direct CAESAR latent-to-field evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class PointDecoderConfig:
    latent_channels: int
    hidden_dimension: int = 128
    hidden_layers: int = 4
    positional_frequencies: int = 4

    def __post_init__(self) -> None:
        if self.latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if self.hidden_dimension <= 0:
            raise ValueError("hidden_dimension must be positive")
        if self.hidden_layers <= 0:
            raise ValueError("hidden_layers must be positive")
        if self.positional_frequencies < 0:
            raise ValueError("positional_frequencies must be nonnegative")

    @property
    def coordinate_channels(self) -> int:
        return 3 + 6 * self.positional_frequencies

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def positional_encoding(points: torch.Tensor, frequencies: int) -> torch.Tensor:
    """Encode normalized ``[...,3]`` coordinates with Fourier features."""

    if points.shape[-1] != 3:
        raise ValueError(f"points must end in three coordinates, got {points.shape}")
    if frequencies < 0:
        raise ValueError("frequencies must be nonnegative")
    if frequencies == 0:
        return points
    bands = torch.pow(
        points.new_tensor(2.0),
        torch.arange(frequencies, device=points.device, dtype=points.dtype),
    )
    angles = torch.pi * points.unsqueeze(-1) * bands
    encoded = torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)
    return torch.cat((points, encoded.flatten(start_dim=-2)), dim=-1)


def sample_latent_features(latent: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Trilinearly sample ``[B,C,D,H,W]`` latents at ``[B,N,3]`` points."""

    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if latent.ndim != 5:
        raise ValueError(f"latent must be [B,C,D,H,W], got {latent.shape}")
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be [B,N,3], got {points.shape}")
    if latent.shape[0] == 1 and points.shape[0] != 1:
        latent = latent.expand(points.shape[0], -1, -1, -1, -1)
    elif latent.shape[0] != points.shape[0]:
        raise ValueError(
            f"latent batch {latent.shape[0]} does not match points batch {points.shape[0]}"
        )

    # PyTorch calls this mode "bilinear" for both 4D bilinear and 5D trilinear
    # inputs. The coordinate order is x, y, z and matches geometry.py.
    grid = points.reshape(points.shape[0], points.shape[1], 1, 1, 3)
    sampled = functional.grid_sample(
        latent,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled[:, :, :, 0, 0].transpose(1, 2)


class PointQueryDecoder(nn.Module):
    """Map local latent features and coordinates directly to scalar values."""

    def __init__(self, config: PointDecoderConfig) -> None:
        super().__init__()
        self.config = config
        input_dimension = config.latent_channels + config.coordinate_channels
        layers: list[nn.Module] = []
        current_dimension = input_dimension
        for _ in range(config.hidden_layers):
            layers.append(nn.Linear(current_dimension, config.hidden_dimension))
            layers.append(nn.SiLU())
            current_dimension = config.hidden_dimension
        layers.append(nn.Linear(current_dimension, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        features = sample_latent_features(latent, points)
        coordinates = positional_encoding(points, self.config.positional_frequencies)
        return self.network(torch.cat((features, coordinates), dim=-1))


def save_point_decoder_checkpoint(
    path: str | Path,
    model: PointQueryDecoder,
    *,
    metadata: dict[str, Any],
) -> Path:
    checkpoint_path = Path(path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "formatVersion": 1,
            "modelConfig": model.config.to_dict(),
            "modelState": model.state_dict(),
            "metadata": metadata,
        },
        checkpoint_path,
    )
    return checkpoint_path


def load_point_decoder_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[PointQueryDecoder, dict[str, Any]]:
    checkpoint = torch.load(
        Path(path).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("formatVersion", 0)) != 1:
        raise ValueError("unsupported point-decoder checkpoint format")
    config = PointDecoderConfig(**checkpoint["modelConfig"])
    model = PointQueryDecoder(config).to(device)
    model.load_state_dict(checkpoint["modelState"])
    return model, dict(checkpoint.get("metadata", {}))
