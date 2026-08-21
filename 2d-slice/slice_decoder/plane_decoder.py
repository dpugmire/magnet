"""Plane-aligned convolutional decoder for CAESAR latent volumes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional

from .point_decoder import positional_encoding


@dataclass(frozen=True)
class PlaneDecoderConfig:
    latent_channels: int
    coarse_resolution: int = 16
    output_resolution: int = 128
    hidden_channels: int = 64
    minimum_channels: int = 16
    coarse_blocks: int = 2
    positional_frequencies: int = 4
    slab_samples: int = 1
    slab_radius_cells: float = 1.0

    def __post_init__(self) -> None:
        positive_fields = (
            "latent_channels",
            "coarse_resolution",
            "output_resolution",
            "hidden_channels",
            "minimum_channels",
            "coarse_blocks",
            "slab_samples",
        )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.positional_frequencies < 0:
            raise ValueError("positional_frequencies must be nonnegative")
        if self.minimum_channels > self.hidden_channels:
            raise ValueError("minimum_channels must not exceed hidden_channels")
        if self.coarse_resolution < 2:
            raise ValueError("coarse_resolution must be at least two")
        if self.slab_samples % 2 == 0:
            raise ValueError(
                "slab_samples must be odd so the requested plane is sampled"
            )
        if not math.isfinite(self.slab_radius_cells) or self.slab_radius_cells < 0.0:
            raise ValueError("slab_radius_cells must be nonnegative and finite")
        if self.slab_samples > 1 and self.slab_radius_cells == 0.0:
            raise ValueError("multiple slab samples require a positive slab radius")
        ratio = self.output_resolution / self.coarse_resolution
        stages = math.log2(ratio) if ratio > 0.0 else -1.0
        if ratio < 1.0 or not stages.is_integer():
            raise ValueError(
                "output_resolution must be coarse_resolution times a power of two"
            )

    @property
    def coordinate_channels(self) -> int:
        return 3 + 6 * self.positional_frequencies

    @property
    def upsampling_stages(self) -> int:
        return int(math.log2(self.output_resolution // self.coarse_resolution))

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _prepare_latent_and_points(
    latent: torch.Tensor,
    points: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if latent.ndim == 4:
        latent = latent.unsqueeze(0)
    if points.ndim == 3:
        points = points.unsqueeze(0)
    if latent.ndim != 5:
        raise ValueError(f"latent must be [B,C,D,H,W], got {latent.shape}")
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"points must be [B,H,W,3], got {points.shape}")
    if latent.shape[0] == 1 and points.shape[0] != 1:
        latent = latent.expand(points.shape[0], -1, -1, -1, -1)
    elif latent.shape[0] != points.shape[0]:
        raise ValueError(
            f"latent batch {latent.shape[0]} does not match points batch "
            f"{points.shape[0]}"
        )

    if latent.device != points.device:
        raise ValueError(
            f"latent device {latent.device} does not match points device {points.device}"
        )
    return latent, points


def _sample_latent_grid(latent: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    output_device = latent.device

    # PyTorch does not currently implement 5D grid_sample on MPS. The CAESAR
    # latent is fixed during this training, so sampling the small coarse plane
    # on CPU preserves model gradients while the learned CNN remains on MPS.
    if latent.device.type == "mps":
        latent = latent.cpu()
        grid = grid.cpu()
    sampled = functional.grid_sample(
        latent,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.to(output_device)


def sample_latent_plane_features(
    latent: torch.Tensor,
    points: torch.Tensor,
) -> torch.Tensor:
    """Trilinearly sample a plane as ``[B,C,H,W]`` from a latent volume."""

    latent, points = _prepare_latent_and_points(latent, points)
    return _sample_latent_grid(latent, points.unsqueeze(1))[:, :, 0]


def sample_latent_plane_slab_features(
    latent: torch.Tensor,
    points: torch.Tensor,
    normals: torch.Tensor,
    *,
    sample_count: int,
    radius_cells: float,
) -> torch.Tensor:
    """Sample parallel planes and concatenate them as ``[B,S*C,H,W]``.

    The slab radius is measured in latent-grid cells along the plane normal.
    Coordinates outside the latent volume use border padding.
    """

    if sample_count <= 0 or sample_count % 2 == 0:
        raise ValueError("sample_count must be a positive odd integer")
    if not math.isfinite(radius_cells) or radius_cells < 0.0:
        raise ValueError("radius_cells must be nonnegative and finite")
    if sample_count > 1 and radius_cells == 0.0:
        raise ValueError("multiple slab samples require a positive radius")

    latent, points = _prepare_latent_and_points(latent, points)
    if normals.ndim == 1:
        normals = normals.unsqueeze(0)
    if normals.ndim != 2 or normals.shape[-1] != 3:
        raise ValueError(f"normals must be [B,3], got {normals.shape}")
    if normals.shape[0] == 1 and points.shape[0] != 1:
        normals = normals.expand(points.shape[0], -1)
    elif normals.shape[0] != points.shape[0]:
        raise ValueError(
            f"normal batch {normals.shape[0]} does not match points batch "
            f"{points.shape[0]}"
        )
    if normals.device != points.device:
        raise ValueError(
            f"normal device {normals.device} does not match points device "
            f"{points.device}"
        )
    normal_lengths = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
    if torch.any(normal_lengths <= 1.0e-12):
        raise ValueError("plane normals must be nonzero")
    normals = normals / normal_lengths

    depth, height, width = latent.shape[2:]
    index_scales = points.new_tensor(
        [(width - 1) / 2.0, (height - 1) / 2.0, (depth - 1) / 2.0]
    )
    normal_index_lengths = torch.linalg.vector_norm(
        normals * index_scales,
        dim=-1,
    )
    if torch.any(normal_index_lengths <= 1.0e-12):
        raise ValueError("latent grid has no extent along the plane normal")
    normalized_cell_steps = 1.0 / normal_index_lengths
    if sample_count == 1:
        offsets = points.new_zeros(1)
    else:
        offsets = torch.linspace(
            -radius_cells,
            radius_cells,
            sample_count,
            device=points.device,
            dtype=points.dtype,
        )
    displacements = (
        normalized_cell_steps[:, None, None, None, None]
        * offsets[None, :, None, None, None]
        * normals[:, None, None, None, :]
    )
    grid = points[:, None] + displacements
    sampled = _sample_latent_grid(latent, grid)
    return sampled.permute(0, 2, 1, 3, 4).reshape(
        sampled.shape[0],
        sample_count * sampled.shape[1],
        sampled.shape[3],
        sampled.shape[4],
    )


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.activation(values + self.layers(values))


class _UpsampleBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(input_channels, output_channels, kernel_size=1)
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        )
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        upsampled = functional.interpolate(
            values,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=True,
        )
        return self.activation(self.projection(upsampled) + self.layers(upsampled))


class PlaneConvolutionalDecoder(nn.Module):
    """Decode an entire plane jointly from a coarse latent feature map."""

    def __init__(self, config: PlaneDecoderConfig) -> None:
        super().__init__()
        self.config = config
        input_channels = (
            config.latent_channels * config.slab_samples + config.coordinate_channels
        )
        coarse_layers: list[nn.Module] = [
            nn.Conv2d(input_channels, config.hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
        ]
        coarse_layers.extend(
            _ResidualBlock(config.hidden_channels) for _ in range(config.coarse_blocks)
        )
        self.coarse_network = nn.Sequential(*coarse_layers)

        upsample_blocks: list[nn.Module] = []
        current_channels = config.hidden_channels
        for stage in range(config.upsampling_stages):
            output_channels = max(
                config.minimum_channels,
                config.hidden_channels // (2 ** (stage + 1)),
            )
            upsample_blocks.append(_UpsampleBlock(current_channels, output_channels))
            current_channels = output_channels
        self.upsample_network = nn.Sequential(*upsample_blocks)
        self.output_network = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(current_channels, 1, kernel_size=1),
        )

    def forward(self, latent: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
        if points.ndim == 3:
            points = points.unsqueeze(0)
        if points.ndim != 4 or points.shape[-1] != 3:
            raise ValueError(f"points must be [B,H,W,3], got {points.shape}")
        expected_shape = (
            self.config.output_resolution,
            self.config.output_resolution,
        )
        if points.shape[1:3] != expected_shape:
            raise ValueError(
                f"point grid must be {expected_shape}, got {points.shape[1:3]}"
            )

        point_channels = points.permute(0, 3, 1, 2)
        coarse_point_channels = functional.interpolate(
            point_channels,
            size=(self.config.coarse_resolution, self.config.coarse_resolution),
            mode="bilinear",
            align_corners=True,
        )
        coarse_points = coarse_point_channels.permute(0, 2, 3, 1)
        axis_u = coarse_points[:, 0, -1] - coarse_points[:, 0, 0]
        axis_v = coarse_points[:, -1, 0] - coarse_points[:, 0, 0]
        normals = torch.linalg.cross(axis_u, axis_v, dim=-1)
        latent_features = sample_latent_plane_slab_features(
            latent,
            coarse_points,
            normals,
            sample_count=self.config.slab_samples,
            radius_cells=self.config.slab_radius_cells,
        )
        coordinate_features = positional_encoding(
            coarse_points,
            self.config.positional_frequencies,
        ).permute(0, 3, 1, 2)
        values = self.coarse_network(
            torch.cat((latent_features, coordinate_features), dim=1)
        )
        values = self.upsample_network(values)
        return self.output_network(values)


def save_plane_decoder_checkpoint(
    path: str | Path,
    model: PlaneConvolutionalDecoder,
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


def load_plane_decoder_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[PlaneConvolutionalDecoder, dict[str, Any]]:
    checkpoint = torch.load(
        Path(path).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    if int(checkpoint.get("formatVersion", 0)) != 1:
        raise ValueError("unsupported plane-decoder checkpoint format")
    config = PlaneDecoderConfig(**checkpoint["modelConfig"])
    model = PlaneConvolutionalDecoder(config).to(device)
    model.load_state_dict(checkpoint["modelState"])
    return model, dict(checkpoint.get("metadata", {}))
