"""Plane geometry and reference sampling for dense scalar volumes.

Coordinates use the same normalized convention as PyTorch ``grid_sample`` with
``align_corners=True``: ``x``, ``y``, and ``z`` each span ``[-1, 1]`` at the
first and last voxel centers. Dense volumes are stored as ``[D, H, W]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


_VECTOR_SHAPE = (3,)


def _as_vector(name: str, value: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != _VECTOR_SHAPE:
        raise ValueError(f"{name} must have shape (3,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.copy()


def _as_bounds(name: str, value: Sequence[float]) -> tuple[float, float]:
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower, upper = float(value[0]), float(value[1])
    if not np.isfinite(lower) or not np.isfinite(upper):
        raise ValueError(f"{name} must contain only finite values")
    if lower >= upper:
        raise ValueError(f"{name} must be strictly increasing, got {value}")
    return lower, upper


@dataclass(frozen=True)
class Plane:
    """An oriented plane parameterized by ``origin + u*axis_u + v*axis_v``."""

    origin: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    u_bounds: tuple[float, float] = (-1.0, 1.0)
    v_bounds: tuple[float, float] = (-1.0, 1.0)
    tolerance: float = 1.0e-6

    def __post_init__(self) -> None:
        origin = _as_vector("origin", self.origin)
        axis_u = _as_vector("axis_u", self.axis_u)
        axis_v = _as_vector("axis_v", self.axis_v)
        u_bounds = _as_bounds("u_bounds", self.u_bounds)
        v_bounds = _as_bounds("v_bounds", self.v_bounds)

        tolerance = float(self.tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be a positive finite number")

        norm_u = float(np.linalg.norm(axis_u))
        norm_v = float(np.linalg.norm(axis_v))
        if not np.isclose(norm_u, 1.0, atol=tolerance, rtol=0.0):
            raise ValueError(f"axis_u must be unit length, got {norm_u}")
        if not np.isclose(norm_v, 1.0, atol=tolerance, rtol=0.0):
            raise ValueError(f"axis_v must be unit length, got {norm_v}")

        dot = float(np.dot(axis_u, axis_v))
        if not np.isclose(dot, 0.0, atol=tolerance, rtol=0.0):
            raise ValueError(f"plane axes must be orthogonal, dot={dot}")

        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "axis_u", axis_u)
        object.__setattr__(self, "axis_v", axis_v)
        object.__setattr__(self, "u_bounds", u_bounds)
        object.__setattr__(self, "v_bounds", v_bounds)
        object.__setattr__(self, "tolerance", tolerance)

    @property
    def normal(self) -> np.ndarray:
        """Return the right-handed unit normal ``axis_u x axis_v``."""

        return np.cross(self.axis_u, self.axis_v)

    def points(self, height: int, width: int) -> np.ndarray:
        """Return an ``[height, width, 3]`` normalized coordinate grid."""

        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")

        u = np.linspace(*self.u_bounds, num=width, dtype=np.float64)
        v = np.linspace(*self.v_bounds, num=height, dtype=np.float64)
        grid_u, grid_v = np.meshgrid(u, v, indexing="xy")
        return (
            self.origin
            + grid_u[..., None] * self.axis_u
            + grid_v[..., None] * self.axis_v
        )


def index_to_normalized(index: float, size: int) -> float:
    """Map a voxel-center index to ``[-1, 1]`` with aligned corners."""

    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if not np.isfinite(index):
        raise ValueError("index must be finite")
    if index < 0.0 or index > size - 1:
        raise ValueError(f"index {index} is outside [0, {size - 1}]")
    if size == 1:
        return 0.0
    return 2.0 * float(index) / float(size - 1) - 1.0


def axis_aligned_plane(
    axis: str,
    index: int,
    volume_shape: Sequence[int],
) -> Plane:
    """Construct an axis-aligned plane through a ``[D,H,W]`` volume.

    Output columns vary along ``axis_u`` and rows vary along ``axis_v``:

    - ``axis='z'`` produces an ``[H,W]`` plane.
    - ``axis='y'`` produces a ``[D,W]`` plane.
    - ``axis='x'`` produces a ``[D,H]`` plane.
    """

    if len(volume_shape) != 3:
        raise ValueError("volume_shape must be [D, H, W]")
    depth, height, width = (int(value) for value in volume_shape)
    if depth <= 0 or height <= 0 or width <= 0:
        raise ValueError(f"volume dimensions must be positive, got {volume_shape}")

    axis = axis.lower()
    if axis == "z":
        fixed = index_to_normalized(index, depth)
        return Plane(
            origin=np.array([0.0, 0.0, fixed]),
            axis_u=np.array([1.0, 0.0, 0.0]),
            axis_v=np.array([0.0, 1.0, 0.0]),
        )
    if axis == "y":
        fixed = index_to_normalized(index, height)
        return Plane(
            origin=np.array([0.0, fixed, 0.0]),
            axis_u=np.array([1.0, 0.0, 0.0]),
            axis_v=np.array([0.0, 0.0, 1.0]),
        )
    if axis == "x":
        fixed = index_to_normalized(index, width)
        return Plane(
            origin=np.array([fixed, 0.0, 0.0]),
            axis_u=np.array([0.0, 1.0, 0.0]),
            axis_v=np.array([0.0, 0.0, 1.0]),
        )
    raise ValueError(f"axis must be 'x', 'y', or 'z', got {axis!r}")


def sample_volume(
    volume: np.ndarray,
    points: np.ndarray,
    *,
    boundary: str = "error",
    tolerance: float = 1.0e-7,
) -> np.ndarray:
    """Trilinearly sample a ``[D,H,W]`` volume at normalized ``[...,3]`` points.

    Point components are ordered ``(x, y, z)``. ``boundary='error'`` rejects
    coordinates outside the volume, while ``boundary='border'`` clamps them to
    the nearest border voxel.
    """

    values = np.asarray(volume)
    if values.ndim != 3:
        raise ValueError(f"volume must have shape [D,H,W], got {values.shape}")
    if any(size <= 0 for size in values.shape):
        raise ValueError("volume dimensions must be positive")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"volume must be numeric, got {values.dtype}")

    coordinates = np.asarray(points, dtype=np.float64)
    if coordinates.ndim < 1 or coordinates.shape[-1] != 3:
        raise ValueError(f"points must have shape [...,3], got {coordinates.shape}")
    if coordinates.size == 0:
        raise ValueError("points must not be empty")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("points must contain only finite values")
    if boundary not in {"error", "border"}:
        raise ValueError("boundary must be 'error' or 'border'")

    minimum = float(np.min(coordinates))
    maximum = float(np.max(coordinates))
    if boundary == "error" and (minimum < -1.0 - tolerance or maximum > 1.0 + tolerance):
        raise ValueError(
            f"sample coordinates are outside [-1,1]: min={minimum}, max={maximum}"
        )
    coordinates = np.clip(coordinates, -1.0, 1.0)

    depth, height, width = values.shape
    flat = coordinates.reshape(-1, 3)
    x = (flat[:, 0] + 1.0) * 0.5 * max(width - 1, 0)
    y = (flat[:, 1] + 1.0) * 0.5 * max(height - 1, 0)
    z = (flat[:, 2] + 1.0) * 0.5 * max(depth - 1, 0)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    z1 = np.minimum(z0 + 1, depth - 1)

    wx = x - x0
    wy = y - y0
    wz = z - z0

    c000 = values[z0, y0, x0]
    c001 = values[z0, y0, x1]
    c010 = values[z0, y1, x0]
    c011 = values[z0, y1, x1]
    c100 = values[z1, y0, x0]
    c101 = values[z1, y0, x1]
    c110 = values[z1, y1, x0]
    c111 = values[z1, y1, x1]

    c00 = c000 * (1.0 - wx) + c001 * wx
    c01 = c010 * (1.0 - wx) + c011 * wx
    c10 = c100 * (1.0 - wx) + c101 * wx
    c11 = c110 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c01 * wy
    c1 = c10 * (1.0 - wy) + c11 * wy
    result = c0 * (1.0 - wz) + c1 * wz
    return result.reshape(coordinates.shape[:-1])


def sample_plane(
    volume: np.ndarray,
    plane: Plane,
    height: int,
    width: int,
    *,
    boundary: str = "error",
) -> np.ndarray:
    """Sample an oriented plane from a dense ``[D,H,W]`` volume."""

    return sample_volume(volume, plane.points(height, width), boundary=boundary)
