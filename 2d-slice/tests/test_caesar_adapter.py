from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIRECTORY))

from slice_decoder.caesar_adapter import (  # noqa: E402
    CaesarReference,
    LatentBlock,
    decode_caesar_base_reconstruction,
    extract_latent_blocks,
    inspect_npz,
    open_stored_npz_array,
    prepare_caesar_subset,
    save_caesar_reference,
    stack_latent_depth,
)


class ArchiveTests(unittest.TestCase):
    def test_inspect_memmap_and_subset(self) -> None:
        source = np.arange(2 * 3 * 8 * 4 * 5, dtype=np.float64).reshape(2, 3, 8, 4, 5)
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source_path = directory / "source.npz"
            subset_path = directory / "subset.npz"
            np.savez(
                source_path,
                data=source,
                variable_name=np.array(["density", "pressure"]),
            )

            metadata = inspect_npz(source_path)
            self.assertEqual(metadata.data.shape, source.shape)
            self.assertEqual(metadata.data.dtype, "<f8")
            self.assertEqual(metadata.variable_names, ("density", "pressure"))

            mapped = open_stored_npz_array(source_path, "data")
            self.assertIsInstance(mapped, np.memmap)
            np.testing.assert_array_equal(mapped[1, 2, 3:7], source[1, 2, 3:7])

            prepared_path = prepare_caesar_subset(
                source_path,
                subset_path,
                variable_index=1,
                section_index=2,
                frame_start=3,
                frame_end=7,
            )
            self.assertEqual(prepared_path, subset_path.resolve())
            with np.load(subset_path) as subset:
                self.assertEqual(subset["data"].dtype, np.float32)
                self.assertEqual(subset["data"].shape, (1, 1, 4, 4, 5))
                np.testing.assert_array_equal(
                    subset["data"][0, 0], source[1, 2, 3:7].astype(np.float32)
                )
                np.testing.assert_array_equal(
                    subset["variable_name"], np.array(["pressure"])
                )

    def test_compressed_member_is_rejected_for_memmap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "compressed.npz"
            np.savez_compressed(source_path, data=np.zeros((1, 1, 2, 2, 2)))
            with self.assertRaisesRegex(ValueError, "memory-mapped"):
                open_stored_npz_array(source_path, "data")


class LatentTests(unittest.TestCase):
    def test_decode_base_reconstruction_skips_residual_postprocessing(self) -> None:
        latent_batches = [{"compressed": "payload"}]
        shape = (1, 1, 8, 4, 4)
        decoded = np.full(shape, 2.0, dtype=np.float32)

        class FakeCaesar:
            use_diffusion = False

            def decompress_caesar_v(self, latent, requested_shape, filtered):
                self.arguments = (latent, requested_shape, filtered)
                return decoded

            def transform_shape(self, values):
                return values + 1.0

        class FakeDataset:
            def recons_data(self, values):
                return values[:, :, :6]

        caesar = FakeCaesar()
        result = decode_caesar_base_reconstruction(
            caesar,
            {
                "latent": latent_batches,
                "shape": shape,
                "filtered_blocks": [(0, 0.0)],
            },
            FakeDataset(),
        )
        self.assertEqual(caesar.arguments, (latent_batches, shape, [(0, 0.0)]))
        np.testing.assert_array_equal(result, np.full((1, 1, 6, 4, 4), 3.0))

    def test_extract_flattened_caesar_v_latents_and_stack(self) -> None:
        q_latent = np.arange(4 * 3 * 2 * 2, dtype=np.float32).reshape(4, 3, 2, 2)
        compressed = {
            "latent": [
                {
                    "q_latent": q_latent,
                    "index": [
                        np.array([0, 0]),
                        np.array([4, 4]),
                        np.array([0, 8]),
                        np.array([8, 16]),
                    ],
                    "scale": np.array([2.0, 3.0]),
                    "offset": np.array([-1.0, 5.0]),
                }
            ]
        }
        blocks = extract_latent_blocks(compressed)

        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0].latent.shape, (3, 2, 2, 2))
        np.testing.assert_array_equal(
            blocks[0].latent, np.transpose(q_latent[:2], (1, 0, 2, 3))
        )
        np.testing.assert_array_equal(blocks[1].scale, np.array(3.0))
        np.testing.assert_array_equal(blocks[1].offset, np.array(5.0))

        stacked = stack_latent_depth(blocks, variable_index=0, section_index=4)
        self.assertEqual(stacked.latent.shape, (3, 4, 2, 2))
        np.testing.assert_array_equal(
            stacked.latent, np.transpose(q_latent, (1, 0, 2, 3))
        )
        self.assertEqual(stacked.start_index, 0)
        self.assertEqual(stacked.end_index, 16)

    def test_extract_accepts_native_five_dimensional_latents(self) -> None:
        q_latent = np.zeros((2, 3, 2, 4, 4), dtype=np.float32)
        blocks = extract_latent_blocks(
            {
                "latent": [
                    {
                        "q_latent": q_latent,
                        "index": [
                            np.array([0, 0]),
                            np.array([0, 0]),
                            np.array([0, 8]),
                            np.array([8, 16]),
                        ],
                        "scale": np.ones((2, 1, 1, 1)),
                        "offset": np.zeros((2, 1, 1, 1)),
                    }
                ]
            }
        )
        self.assertEqual(blocks[0].latent.shape, (3, 2, 4, 4))

    def test_stack_rejects_noncontiguous_blocks(self) -> None:
        latent = np.zeros((1, 2, 2, 2), dtype=np.float32)
        blocks = [
            LatentBlock(0, 0, 0, 8, latent, np.array(1.0), np.array(0.0)),
            LatentBlock(0, 0, 9, 17, latent, np.array(1.0), np.array(0.0)),
        ]
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            stack_latent_depth(blocks, variable_index=0, section_index=0)

    def test_extract_requires_q_latent(self) -> None:
        with self.assertRaisesRegex(KeyError, "q_latent"):
            extract_latent_blocks(
                {
                    "latent": [
                        {
                            "index": [
                                np.array([0]),
                                np.array([0]),
                                np.array([0]),
                                np.array([8]),
                            ]
                        }
                    ]
                }
            )

    def test_reference_manifest_records_shapes_and_source_metadata(self) -> None:
        latent = np.zeros((2, 2, 3, 3), dtype=np.float32)
        block = LatentBlock(
            variable_index=3,
            section_index=2,
            start_index=0,
            end_index=8,
            latent=latent,
            scale=np.array(2.0),
            offset=np.array(-1.0),
        )
        volume = np.zeros((8, 12, 12), dtype=np.float32)
        base = volume + 0.25
        reconstructed = volume + 0.125
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            data_path = directory / "input.npz"
            model_path = directory / "model.pt"
            np.savez(data_path, data=volume[None, None])
            model_path.write_bytes(b"test checkpoint")
            reference = CaesarReference(
                original=volume,
                base_reconstructed=base,
                reconstructed=reconstructed,
                latent_blocks=(block,),
                compressed_bytes=123.0,
                source_data=data_path,
                model_path=model_path,
                variable_index=3,
                section_index=2,
                frame_start=0,
                frame_end=8,
                n_frame=8,
                error_bound=0.01,
            )

            manifest_path = save_caesar_reference(
                reference,
                directory / "artifacts",
                axis_semantic="spatial_z",
                source_metadata={"archive": "source.npz"},
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["formatVersion"], 2)
            self.assertEqual(manifest["latentShape"], [2, 2, 3, 3])
            self.assertEqual(manifest["baseReconstructedShape"], [8, 12, 12])
            self.assertAlmostEqual(manifest["stageMetrics"]["baseVsRaw"]["rmse"], 0.25)
            self.assertEqual(manifest["sourceMetadata"]["archive"], "source.npz")
            np.testing.assert_array_equal(
                np.load(directory / "artifacts" / "caesar_base.npy"), base
            )
            np.testing.assert_array_equal(
                np.load(directory / "artifacts" / "caesar_residual.npy"),
                reconstructed - base,
            )


if __name__ == "__main__":
    unittest.main()
