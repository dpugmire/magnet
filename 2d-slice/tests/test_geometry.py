from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

from slice_decoder.geometry import (  # noqa: E402
    Plane,
    axis_aligned_plane,
    sample_plane,
    sample_volume,
)


class GeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        depth, height, width = 5, 6, 7
        z, y, x = np.meshgrid(
            np.linspace(-1.0, 1.0, depth),
            np.linspace(-1.0, 1.0, height),
            np.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.volume = 2.0 * x - 3.0 * y + 5.0 * z + 7.0

    def _sample_axis_plane(self, axis: str, index: int) -> np.ndarray:
        plane = axis_aligned_plane(axis, index, self.volume.shape)
        if axis == "z":
            resolution = self.volume.shape[1:]
        elif axis == "y":
            resolution = (self.volume.shape[0], self.volume.shape[2])
        else:
            resolution = self.volume.shape[:2]
        return sample_plane(self.volume, plane, *resolution)

    def test_axis_aligned_plane_matches_numpy_slice(self) -> None:
        for axis, index, expected in (
            ("z", 2, self.volume[2, :, :]),
            ("y", 3, self.volume[:, 3, :]),
            ("x", 4, self.volume[:, :, 4]),
        ):
            with self.subTest(axis=axis):
                actual = self._sample_axis_plane(axis, index)
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)

    def test_arbitrary_plane_is_exact_for_affine_field(self) -> None:
        inverse_root_two = 1.0 / np.sqrt(2.0)
        plane = Plane(
            origin=(0.1, -0.2, 0.0),
            axis_u=(1.0, 0.0, 0.0),
            axis_v=(0.0, inverse_root_two, inverse_root_two),
            u_bounds=(-0.5, 0.5),
            v_bounds=(-0.4, 0.4),
        )
        points = plane.points(9, 11)
        expected = (
            2.0 * points[..., 0]
            - 3.0 * points[..., 1]
            + 5.0 * points[..., 2]
            + 7.0
        )
        actual = sample_volume(self.volume, points)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-12)

    def test_reversed_basis_flips_slice(self) -> None:
        plane = axis_aligned_plane("z", 1, self.volume.shape)
        flipped = Plane(
            origin=plane.origin,
            axis_u=-plane.axis_u,
            axis_v=plane.axis_v,
            u_bounds=plane.u_bounds,
            v_bounds=plane.v_bounds,
        )
        height, width = self.volume.shape[1:]
        np.testing.assert_allclose(
            sample_plane(self.volume, flipped, height, width),
            np.flip(sample_plane(self.volume, plane, height, width), axis=1),
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_boundary_modes(self) -> None:
        points = np.array([[[1.2, 0.0, 0.0]]])
        with self.assertRaisesRegex(ValueError, "outside"):
            sample_volume(self.volume, points, boundary="error")
        clipped = sample_volume(self.volume, points, boundary="border")
        expected = sample_volume(self.volume, np.array([[[1.0, 0.0, 0.0]]]))
        np.testing.assert_allclose(clipped, expected)

    def test_plane_rejects_nonorthogonal_basis(self) -> None:
        inverse_root_two = 1.0 / np.sqrt(2.0)
        with self.assertRaisesRegex(ValueError, "orthogonal"):
            Plane(
                origin=(0.0, 0.0, 0.0),
                axis_u=(1.0, 0.0, 0.0),
                axis_v=(inverse_root_two, inverse_root_two, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
