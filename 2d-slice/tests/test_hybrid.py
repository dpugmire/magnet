from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))


class HybridTests(unittest.TestCase):
    def test_pixel_and_tile_selection(self) -> None:
        from slice_decoder.hybrid import (
            select_largest_mask,
            select_largest_tiles,
            select_random_mask,
        )

        values = np.arange(64).reshape(8, 8)
        pixel_mask = select_largest_mask(values, 0.25)
        self.assertEqual(np.count_nonzero(pixel_mask), 16)
        self.assertTrue(np.all(values[pixel_mask] >= 48))

        tile_mask, tile_count = select_largest_tiles(
            values,
            0.25,
            tile_size=4,
        )
        self.assertEqual(tile_count, 1)
        self.assertEqual(np.count_nonzero(tile_mask), 16)
        self.assertTrue(np.all(tile_mask[4:, 4:]))

        random_mask = select_random_mask(
            values.shape,
            0.25,
            np.random.default_rng(3),
        )
        self.assertEqual(np.count_nonzero(random_mask), 16)

    def test_connected_components(self) -> None:
        from slice_decoder.hybrid import connected_component_count

        mask = np.zeros((5, 6), dtype=bool)
        mask[0:2, 0:2] = True
        mask[3, 3:5] = True
        mask[4, 4] = True
        self.assertEqual(connected_component_count(mask), 2)

    def test_selected_frame_sampling_matches_full_volume(self) -> None:
        from slice_decoder.geometry import sample_volume
        from slice_decoder.hybrid import (
            required_depth_frames,
            sample_decoded_frames,
        )

        depth, height, width = 5, 7, 9
        z, y, x = np.meshgrid(
            np.linspace(-1.0, 1.0, depth),
            np.linspace(-1.0, 1.0, height),
            np.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        volume = 2.0 * x - 3.0 * y + 5.0 * z + 7.0
        points = np.array(
            [
                [-0.8, 0.2, -0.7],
                [0.4, -0.5, 0.1],
                [0.7, 0.8, 0.9],
            ]
        )
        mask = np.ones(points.shape[:-1], dtype=bool)
        frame_indices = required_depth_frames(points, mask, depth=depth)
        selected_frames = volume[np.asarray(frame_indices)]
        actual = sample_decoded_frames(
            selected_frames,
            frame_indices,
            points,
            volume_depth=depth,
        )
        expected = sample_volume(volume, points)
        np.testing.assert_allclose(actual, expected)

        with self.assertRaisesRegex(ValueError, "required frames"):
            sample_decoded_frames(
                selected_frames[:-1],
                frame_indices[:-1],
                points,
                volume_depth=depth,
            )


if __name__ == "__main__":
    unittest.main()
