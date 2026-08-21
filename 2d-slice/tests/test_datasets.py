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
    MultiBlockPlaneDataset,
    RandomPlaneDataset,
    discover_reference_artifacts,
    load_reference_collection,
    load_reference_block,
    load_reference_blocks,
    parse_index_specification,
    random_contained_plane,
    split_reference_blocks,
)
from slice_decoder.geometry import sample_volume  # noqa: E402
from slice_decoder.metrics import (  # noqa: E402
    error_decomposition,
    field_error_metrics,
    plane_decoder_error_decomposition,
    staged_error_decomposition,
)


def _write_artifact(directory: Path) -> tuple[np.ndarray, np.ndarray]:
    depth, height, width = 8, 6, 7
    z, y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, depth),
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    raw = 2.0 * x - y + 0.5 * z
    base = raw + 0.5
    caesar = raw + 0.125
    latent = np.zeros((4, 2, 3, 3), dtype=np.float32)
    source_path = directory / "source.npz"
    np.savez(source_path, data=raw[None, None], variable_name=np.array(["field"]))
    np.save(directory / "original.npy", raw.astype(np.float32))
    np.save(directory / "caesar_base.npy", base.astype(np.float32))
    np.save(directory / "caesar_reference.npy", caesar.astype(np.float32))
    np.save(directory / "caesar_residual.npy", (caesar - base).astype(np.float32))
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
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return raw, caesar


def _write_multiblock_artifact(
    directory: Path,
    source_path: Path,
    source_data: np.ndarray,
    *,
    section_index: int,
) -> None:
    directory.mkdir()
    raw = source_data[0, section_index]
    caesar = raw + 0.05 * (section_index + 1)
    latent = np.empty((4, 4, 3, 3), dtype=np.float32)
    latent[:, :2] = float(section_index * 10)
    latent[:, 2:] = float(section_index * 10 + 1)
    np.save(directory / "original.npy", raw.astype(np.float32))
    np.save(directory / "caesar_reference.npy", caesar.astype(np.float32))
    np.save(directory / "q_latent.npy", latent)
    blocks = []
    for block_index in range(2):
        blocks.append(
            {
                "variableIndex": 0,
                "sectionIndex": 0,
                "startIndex": block_index * 8,
                "endIndex": (block_index + 1) * 8,
                "latentShape": [4, 2, 3, 3],
                "scale": [2.0 + section_index],
                "offset": [0.1 * block_index],
            }
        )
    manifest = {
        "frameStart": 0,
        "frameEnd": 16,
        "variableIndex": 0,
        "sectionIndex": 0,
        "axisSemantic": "spatial_z",
        "blocks": blocks,
        "sourceMetadata": {
            "archive": str(source_path),
            "selection": {
                "variableIndex": 0,
                "sectionIndex": section_index,
                "frameStart": 0,
                "frameEnd": 16,
            },
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


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
            self.assertIsNotNone(block.base_volume)
            self.assertIsNotNone(block.residual_volume)
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
            np.testing.assert_allclose(
                first["base_values"][:, 0],
                sample_volume(block.base_volume, first["points"]),
                rtol=1.0e-6,
                atol=1.0e-6,
            )
            np.testing.assert_allclose(
                first["residual_values"][:, 0],
                sample_volume(block.residual_volume, first["points"]),
                rtol=1.0e-6,
                atol=1.0e-6,
            )

            base_dataset = RandomPlaneDataset(
                block,
                sample_count=1,
                height=9,
                width=11,
                seed=42,
                orientation="random",
                target="base",
                include_reference_fields=False,
            )
            base_sample = base_dataset[0]
            expected_base = sample_volume(block.base_volume, base_sample["points"])
            np.testing.assert_allclose(
                base_sample["target_normalized"][:, 0],
                (expected_base - 0.25) / 2.0,
                rtol=1.0e-6,
                atol=1.0e-6,
            )
            self.assertNotIn("raw_values", base_sample)
            self.assertNotIn("caesar_values", base_sample)

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

    def test_collection_loading_and_source_section_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_data = np.arange(1 * 3 * 16 * 6 * 7, dtype=np.float64).reshape(
                1, 3, 16, 6, 7
            )
            source_path = root / "source.npz"
            np.savez(source_path, data=source_data)
            artifact_directories = []
            for section in range(3):
                directory = root / f"section-{section:02d}"
                _write_multiblock_artifact(
                    directory,
                    source_path,
                    source_data,
                    section_index=section,
                )
                artifact_directories.append(directory)

            discovered = discover_reference_artifacts(artifact_roots=(root,))
            self.assertEqual(
                discovered,
                tuple(directory.resolve() for directory in artifact_directories),
            )
            self.assertEqual(len(load_reference_blocks(artifact_directories[0])), 2)
            blocks = load_reference_collection(discovered)
            self.assertEqual(len(blocks), 6)
            self.assertEqual(blocks[1].source_frame_start, 8)
            self.assertEqual(blocks[1].source_frame_end, 16)
            self.assertEqual(blocks[2].source_section_index, 1)
            np.testing.assert_array_equal(blocks[2].raw_volume, source_data[0, 1, 0:8])
            self.assertTrue(np.all(blocks[3].latent == 11.0))

            split = split_reference_blocks(
                blocks,
                train_sections=parse_index_specification("0"),
                validation_sections=parse_index_specification("1"),
                test_sections=parse_index_specification("2"),
            )
            self.assertEqual(len(split.train), 2)
            self.assertEqual(len(split.validation), 2)
            self.assertEqual(len(split.test), 2)
            with self.assertRaisesRegex(ValueError, "disjoint"):
                split_reference_blocks(
                    blocks,
                    train_sections=(0, 1),
                    validation_sections=(1,),
                    test_sections=(2,),
                )

            dataset = MultiBlockPlaneDataset(
                split.train,
                sample_count=2,
                height=5,
                width=5,
                seed=9,
            )
            self.assertEqual(int(dataset[0]["block_position"]), 0)
            self.assertEqual(int(dataset[1]["block_position"]), 1)
            self.assertTrue(np.all(dataset[0]["latent"] == 0.0))
            self.assertTrue(np.all(dataset[1]["latent"] == 1.0))


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

        base = raw + 0.4
        staged = staged_error_decomposition(raw, base, caesar, direct)
        self.assertAlmostEqual(staged["baseCompression"]["rmse"], 0.4)
        self.assertAlmostEqual(staged["residualCorrection"]["rmse"], 0.3)
        self.assertAlmostEqual(staged["pointDecoderVsBase"]["rmse"], 0.325)
        self.assertAlmostEqual(staged["pointDecoderVsFinal"]["rmse"], 0.025)

        plane = base - 0.05
        plane_decomposition = plane_decoder_error_decomposition(
            raw, base, caesar, plane
        )
        self.assertAlmostEqual(plane_decomposition["planeDecoderVsBase"]["rmse"], 0.05)
        self.assertAlmostEqual(plane_decomposition["planeDecoderVsFinal"]["rmse"], 0.25)


if __name__ == "__main__":
    unittest.main()
