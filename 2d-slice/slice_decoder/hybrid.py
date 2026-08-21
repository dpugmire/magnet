"""Utilities for uncertainty-guided exact CAESAR fallback experiments."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def select_largest_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    """Select an exact fraction of the largest finite values."""

    scores = np.asarray(values)
    if scores.size == 0:
        raise ValueError("values must not be empty")
    if not np.all(np.isfinite(scores)):
        raise ValueError("values must be finite")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0,1]")
    selected_count = int(round(fraction * scores.size))
    mask = np.zeros(scores.size, dtype=bool)
    if selected_count == 0:
        return mask.reshape(scores.shape)
    if selected_count == scores.size:
        return np.ones(scores.shape, dtype=bool)
    order = np.argsort(-scores.reshape(-1), kind="stable")
    mask[order[:selected_count]] = True
    return mask.reshape(scores.shape)


def select_random_mask(
    shape: Sequence[int],
    fraction: float,
    generator: np.random.Generator,
) -> np.ndarray:
    """Select an exact fraction of pixels without replacement."""

    if not shape or any(int(size) <= 0 for size in shape):
        raise ValueError("shape must contain positive dimensions")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0,1]")
    pixel_count = math.prod(int(size) for size in shape)
    selected_count = int(round(fraction * pixel_count))
    mask = np.zeros(pixel_count, dtype=bool)
    if selected_count:
        mask[generator.choice(pixel_count, selected_count, replace=False)] = True
    return mask.reshape(tuple(int(size) for size in shape))


def select_largest_tiles(
    values: np.ndarray,
    fraction: float,
    *,
    tile_size: int,
) -> tuple[np.ndarray, int]:
    """Select whole tiles by maximum value until the pixel budget is met."""

    scores = np.asarray(values)
    if scores.ndim != 2 or scores.size == 0:
        raise ValueError("values must be a nonempty 2D array")
    if not np.all(np.isfinite(scores)):
        raise ValueError("values must be finite")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0,1]")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    target_count = int(round(fraction * scores.size))
    mask = np.zeros(scores.shape, dtype=bool)
    if target_count == 0:
        return mask, 0

    tiles: list[tuple[float, int, int, int, int]] = []
    for row_start in range(0, scores.shape[0], tile_size):
        row_end = min(row_start + tile_size, scores.shape[0])
        for column_start in range(0, scores.shape[1], tile_size):
            column_end = min(column_start + tile_size, scores.shape[1])
            tile_score = float(
                np.max(scores[row_start:row_end, column_start:column_end])
            )
            tiles.append(
                (tile_score, row_start, row_end, column_start, column_end)
            )
    tiles.sort(key=lambda tile: tile[0], reverse=True)
    selected_tiles = 0
    for _, row_start, row_end, column_start, column_end in tiles:
        mask[row_start:row_end, column_start:column_end] = True
        selected_tiles += 1
        if int(np.count_nonzero(mask)) >= target_count:
            break
    return mask, selected_tiles


def connected_component_count(mask: np.ndarray) -> int:
    """Count four-connected components in a two-dimensional mask."""

    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    visited = np.zeros(selected.shape, dtype=bool)
    component_count = 0
    height, width = selected.shape
    for start_row, start_column in np.argwhere(selected):
        if visited[start_row, start_column]:
            continue
        component_count += 1
        visited[start_row, start_column] = True
        stack = [(int(start_row), int(start_column))]
        while stack:
            row, column = stack.pop()
            for next_row, next_column in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if (
                    0 <= next_row < height
                    and 0 <= next_column < width
                    and selected[next_row, next_column]
                    and not visited[next_row, next_column]
                ):
                    visited[next_row, next_column] = True
                    stack.append((next_row, next_column))
    return component_count


def required_depth_frames(
    points: np.ndarray,
    mask: np.ndarray,
    *,
    depth: int,
) -> tuple[int, ...]:
    """Return depth frames needed to interpolate the selected plane pixels."""

    coordinates = np.asarray(points, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    if coordinates.shape[:-1] != selected.shape or coordinates.shape[-1] != 3:
        raise ValueError("points and mask shapes do not match")
    if depth <= 0:
        raise ValueError("depth must be positive")
    if not np.any(selected):
        return ()
    z = np.clip(coordinates[..., 2][selected], -1.0, 1.0)
    z_index = (z + 1.0) * 0.5 * max(depth - 1, 0)
    lower = np.floor(z_index).astype(np.int64)
    upper = np.minimum(lower + 1, depth - 1)
    interpolated_upper = upper[(z_index - lower) > 1.0e-12]
    return tuple(
        int(index)
        for index in np.unique(np.concatenate((lower, interpolated_upper)))
    )


def sample_decoded_frames(
    frames: np.ndarray,
    frame_indices: Sequence[int],
    points: np.ndarray,
    *,
    volume_depth: int,
) -> np.ndarray:
    """Trilinearly sample points using a selected set of complete depth frames."""

    values = np.asarray(frames)
    coordinates = np.asarray(points, dtype=np.float64)
    indices = tuple(int(index) for index in frame_indices)
    if values.ndim != 3:
        raise ValueError("frames must be [F,H,W]")
    if values.shape[0] != len(indices):
        raise ValueError("frame count does not match frame_indices")
    if coordinates.ndim < 2 or coordinates.shape[-1] != 3:
        raise ValueError("points must be [...,3]")
    if volume_depth <= 0:
        raise ValueError("volume_depth must be positive")
    if len(set(indices)) != len(indices):
        raise ValueError("frame_indices must be unique")
    if any(index < 0 or index >= volume_depth for index in indices):
        raise IndexError("frame index is outside the source volume")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("points must be finite")

    height, width = values.shape[1:]
    flat = np.clip(coordinates.reshape(-1, 3), -1.0, 1.0)
    x = (flat[:, 0] + 1.0) * 0.5 * max(width - 1, 0)
    y = (flat[:, 1] + 1.0) * 0.5 * max(height - 1, 0)
    z = (flat[:, 2] + 1.0) * 0.5 * max(volume_depth - 1, 0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    z1 = np.minimum(z0 + 1, volume_depth - 1)

    lookup = np.full(volume_depth, -1, dtype=np.int64)
    lookup[np.asarray(indices, dtype=np.int64)] = np.arange(len(indices))
    local_z0 = lookup[z0]
    local_z1 = lookup[z1]
    local_z1[(local_z1 < 0) & np.isclose(z - z0, 0.0, atol=1.0e-12)] = local_z0[
        (local_z1 < 0) & np.isclose(z - z0, 0.0, atol=1.0e-12)
    ]
    if np.any(local_z0 < 0) or np.any(local_z1 < 0):
        missing = np.unique(
            np.concatenate((z0[local_z0 < 0], z1[local_z1 < 0]))
        )
        raise ValueError(f"required frames were not decoded: {missing.tolist()}")

    wx = x - x0
    wy = y - y0
    wz = z - z0

    def bilinear(local_z: np.ndarray) -> np.ndarray:
        c00 = values[local_z, y0, x0]
        c01 = values[local_z, y0, x1]
        c10 = values[local_z, y1, x0]
        c11 = values[local_z, y1, x1]
        lower = c00 * (1.0 - wx) + c01 * wx
        upper = c10 * (1.0 - wx) + c11 * wx
        return lower * (1.0 - wy) + upper * wy

    sampled_lower = bilinear(local_z0)
    sampled_upper = bilinear(local_z1)
    sampled = sampled_lower * (1.0 - wz) + sampled_upper * wz
    return sampled.reshape(coordinates.shape[:-1])
