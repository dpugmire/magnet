from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

from slice_decoder.datasets import (  # noqa: E402
    RandomPlaneDataset,
    load_reference_block,
    random_contained_plane,
)
from slice_decoder.geometry import sample_volume  # noqa: E402
from slice_decoder.metrics import error_decomposition, field_error_metrics  # noqa: E402


def _write_artifact(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    depth, height, width = 8, 6, 7
    z, y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, depth),
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    raw = 2.0 * x - y + 0.5 * z
    caesar = raw + 0.125
    latent = np.zeros((4, 2, 3, 3), dtype=np.float32)
    source_path = directory / "source.npz"
    np.savez(source_path, data=raw[None, None], variable_name=np.array(["field"]))
    np.save(directory / "original.npy", raw.astype(np.float32))
    np.save(directory / "caesar_reference.npy", caesar.astype(np.float32))
    np.save(directory / "q_latent.npy", latent)
    manifest = {
        "frameStart": 0,
        "frameEnd": 8,
        "variableIndex": 0,
        "sectionIndex": 0,
        "axisSemantic": "spatial_z",
        "sourceMetadata": {
            "archive": str(source_path),
            "selection": {
                "variableIndex": 0,
                "sectionIndex": 0,
                "frameStart": 0,
                "frameEnd": 8,
            },
        },
        "blocks": [
            {
                "variableIndex": 0,
                "sectionIndex": 0,
                "startIndex": 0,
                "endIndex": 8,
                "latentShape": [4, 2, 3, 3],
                "scale": [2.0],
                "offset": [0.25],
            }
        ],
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return raw, caesar


class ReferenceDatasetTests(unittest.TestCase):
    def test_load_reference_block_and_generate_deterministic_pair(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            _, caesar = _write_artifact(directory)
            block = load_reference_block(directory)
            self.assertEqual(block.latent.shape, (4, 2, 3, 3))
            self.assertEqual(block.raw_volume.shape, (8, 6, 7))
            self.assertEqual(block.raw_volume.dtype, np.float64)
            self.assertEqual(block.raw_source, (directory / "source.npz").resolve())
            self.assertEqual(block.scale, 2.0)
            self.assertEqual(block.offset, 0.25)

            dataset = RandomPlaneDataset(
                block,
                sample_count=3,
                height=9,
                width=11,
                seed=42,
                orientation="random",
            )
            first = dataset[1]
            repeated = dataset[1]
            np.testing.assert_array_equal(first["points"], repeated["points"])
            np.testing.assert_array_equal(
                first["target_normalized"], repeated["target_normalized"]
            )
            self.assertLessEqual(float(np.max(np.abs(first["points"]))), 1.0)

            expected = sample_volume(caesar, first["points"])
            np.testing.assert_allclose(
                first["target_normalized"][:, 0],
                (expected - 0.25) / 2.0,
                rtol=1.0e-6,
                atol=1.0e-6,
            )

    def test_random_planes_are_contained_and_orthonormal(self) -> None:
        generator = np.random.default_rng(7)
        for orientation in ("axis-aligned", "random", "mixed"):
            for _ in range(20):
                plane = random_contained_plane(
                    generator,
                    orientation=orientation,
                    maximum_offset=0.75,
                )
                points = plane.points(17, 19)
                self.assertLessEqual(float(np.max(np.abs(points))), 1.0 + 1.0e-12)
                self.assertAlmostEqual(float(np.linalg.norm(plane.axis_u)), 1.0)
                self.assertAlmostEqual(float(np.linalg.norm(plane.axis_v)), 1.0)
                self.assertAlmostEqual(float(np.dot(plane.axis_u, plane.axis_v)), 0.0)


class MetricTests(unittest.TestCase):
    def test_error_metrics_and_decomposition(self) -> None:
        raw = np.array([0.0, 1.0, 2.0, 3.0])
        caesar = raw + 0.1
        direct = caesar - 0.025
        metrics = field_error_metrics(raw, caesar)
        self.assertAlmostEqual(metrics["rmse"], 0.1)
        self.assertAlmostEqual(metrics["meanAbsoluteError"], 0.1)
        self.assertAlmostEqual(metrics["rangeNormalizedRmse"], 0.1 / 3.0)

        decomposition = error_decomposition(raw, caesar, direct)
        self.assertAlmostEqual(decomposition["compression"]["rmse"], 0.1)
        self.assertAlmostEqual(decomposition["sliceDecoder"]["rmse"], 0.025)
        self.assertAlmostEqual(decomposition["endToEnd"]["rmse"], 0.075)


if __name__ == "__main__":
    unittest.main()
