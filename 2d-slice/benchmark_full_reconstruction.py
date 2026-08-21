#!/usr/bin/env python3
"""Benchmark full CAESAR reconstruction followed by arbitrary plane sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch

from slice_decoder.caesar_features import FrozenCaesarVFeatureDecoder
from slice_decoder.datasets import (
    MultiBlockPlaneDataset,
    ReferenceBlock,
    discover_reference_artifacts,
    load_reference_collection,
    parse_index_specification,
    split_reference_blocks,
)
from slice_decoder.metrics import field_error_metrics
from slice_decoder.plane_decoder import sample_latent_plane_features


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark full CAESAR decode followed by plane sampling"
    )
    parser.add_argument("--artifact-dir", type=Path, action="append", default=[])
    parser.add_argument("--artifact-root", type=Path, action="append", default=[])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-sections", default=None)
    parser.add_argument("--validation-sections", default=None)
    parser.add_argument("--test-sections", default=None)
    parser.add_argument(
        "--expected-axis-semantic",
        choices=("spatial_z", "time", "unconfirmed"),
        default=None,
    )
    parser.add_argument("--resolution", type=_positive_integer, default=128)
    parser.add_argument("--eval-planes", type=_positive_integer, default=8)
    parser.add_argument(
        "--orientation",
        choices=("axis-aligned", "random", "mixed"),
        default="mixed",
    )
    parser.add_argument("--maximum-offset", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cpu")
    return parser


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _block_summary(blocks: Sequence[ReferenceBlock]) -> dict[str, object]:
    return {
        "blockCount": len(blocks),
        "sourceSections": sorted({block.source_section_index for block in blocks}),
        "sourceFrameRange": [
            min(block.source_frame_start for block in blocks),
            max(block.source_frame_end for block in blocks),
        ],
    }


def _resolve_blocks(
    args: argparse.Namespace,
) -> tuple[
    tuple[Path, ...],
    tuple[ReferenceBlock, ...],
    tuple[ReferenceBlock, ...],
    tuple[ReferenceBlock, ...],
    str,
]:
    artifact_directories = discover_reference_artifacts(
        artifact_directories=args.artifact_dir,
        artifact_roots=args.artifact_root,
    )
    blocks = load_reference_collection(artifact_directories)
    semantics = {block.axis_semantic for block in blocks}
    if len(semantics) != 1:
        raise ValueError(f"reference artifacts have mixed axis semantics: {semantics}")
    if args.expected_axis_semantic is not None and semantics != {
        args.expected_axis_semantic
    }:
        raise ValueError(
            f"expected axis semantic {args.expected_axis_semantic!r}, "
            f"found {next(iter(semantics))!r}"
        )
    if any(block.base_volume is None for block in blocks):
        raise ValueError("full-reconstruction comparison requires CAESAR base data")

    split_arguments = (
        args.train_sections,
        args.validation_sections,
        args.test_sections,
    )
    if any(value is not None for value in split_arguments):
        if any(value is None for value in split_arguments):
            raise ValueError(
                "train, validation, and test sections must be specified together"
            )
        split = split_reference_blocks(
            blocks,
            train_sections=parse_index_specification(args.train_sections),
            validation_sections=parse_index_specification(args.validation_sections),
            test_sections=parse_index_specification(args.test_sections),
        )
        return (
            artifact_directories,
            split.train,
            split.validation,
            split.test,
            "source-section",
        )
    return artifact_directories, blocks, blocks, blocks, "same-block"


def _evaluate(
    decoder: FrozenCaesarVFeatureDecoder,
    blocks: Sequence[ReferenceBlock],
    *,
    output_directory: Path,
    split_name: str,
    resolution: int,
    eval_planes: int,
    seed: int,
    orientation: str,
    maximum_offset: float,
    device: torch.device,
) -> dict[str, object]:
    dataset = MultiBlockPlaneDataset(
        blocks,
        sample_count=eval_planes * len(blocks),
        height=resolution,
        width=resolution,
        seed=seed,
        orientation=orientation,
        maximum_offset=maximum_offset,
        target="base",
    )
    fields: dict[str, list[np.ndarray]] = {
        "raw": [],
        "base": [],
        "caesar": [],
        "full": [],
    }
    by_section: dict[int, dict[str, list[np.ndarray]]] = {}
    reconstruction_latencies: list[float] = []
    sampling_latencies: list[float] = []
    decoded_bytes = 0

    warm_latent = torch.from_numpy(np.array(blocks[0].latent, copy=True))
    decoder.decode_latent(warm_latent.unsqueeze(0).to(device))
    _synchronize(device)

    with torch.inference_mode():
        for block_position, block in enumerate(blocks):
            latent = torch.from_numpy(np.array(block.latent, copy=True))
            start_time = time.perf_counter()
            decoded = decoder.decode_latent(latent.unsqueeze(0).to(device))
            _synchronize(device)
            reconstruction_latencies.append((time.perf_counter() - start_time) * 1000.0)
            decoded_bytes = decoded.numel() * decoded.element_size()

            for index in range(block_position, len(dataset), len(blocks)):
                sample = dataset[index]
                points = torch.from_numpy(sample["points"]).reshape(
                    1,
                    resolution,
                    resolution,
                    3,
                )
                points = points.to(device)
                if not sampling_latencies:
                    sample_latent_plane_features(decoded, points)
                    _synchronize(device)
                start_time = time.perf_counter()
                prediction_normalized = sample_latent_plane_features(decoded, points)
                _synchronize(device)
                sampling_latencies.append((time.perf_counter() - start_time) * 1000.0)

                prediction = prediction_normalized[0, 0].cpu().numpy()
                prediction = prediction * float(sample["scale"]) + float(
                    sample["offset"]
                )
                current = {
                    "raw": sample["raw_values"][:, 0].reshape(resolution, resolution),
                    "base": sample["base_values"][:, 0].reshape(resolution, resolution),
                    "caesar": sample["caesar_values"][:, 0].reshape(
                        resolution, resolution
                    ),
                    "full": prediction,
                }
                for name, values in current.items():
                    fields[name].append(values)
                section = int(sample["source_section_index"])
                section_fields = by_section.setdefault(
                    section,
                    {name: [] for name in fields},
                )
                for name, values in current.items():
                    section_fields[name].append(values)

                if index == 0:
                    np.savez(
                        output_directory
                        / f"{split_name}_full_reconstruction_example.npz",
                        **current,
                        points=sample["points"].reshape(resolution, resolution, 3),
                        plane_origin=sample["plane_origin"],
                        plane_axis_u=sample["plane_axis_u"],
                        plane_axis_v=sample["plane_axis_v"],
                        plane_bounds=sample["plane_bounds"],
                        source_section_index=sample["source_section_index"],
                        source_frame_start=sample["source_frame_start"],
                    )

    def decompose(values: dict[str, list[np.ndarray]]) -> dict[str, object]:
        flattened = {
            name: np.concatenate([array.reshape(-1) for array in arrays])
            for name, arrays in values.items()
        }
        return {
            "baseCompression": field_error_metrics(flattened["raw"], flattened["base"]),
            "finalCompression": field_error_metrics(
                flattened["raw"], flattened["caesar"]
            ),
            "fullReconstructionVsBase": field_error_metrics(
                flattened["base"], flattened["full"]
            ),
            "fullReconstructionVsFinal": field_error_metrics(
                flattened["caesar"], flattened["full"]
            ),
            "endToEnd": field_error_metrics(flattened["raw"], flattened["full"]),
        }

    reconstruction_mean = float(np.mean(reconstruction_latencies))
    sampling_mean = float(np.mean(sampling_latencies))
    return {
        "aggregate": decompose(fields),
        "bySection": {
            str(section): decompose(values)
            for section, values in sorted(by_section.items())
        },
        "inference": {
            "meanMillisecondsPerReconstruction": reconstruction_mean,
            "meanMillisecondsPerPlaneSample": sampling_mean,
            "estimatedColdMillisecondsPerSlice": reconstruction_mean + sampling_mean,
            "amortizedMillisecondsPerSlice": reconstruction_mean / eval_planes
            + sampling_mean,
            "planesPerReconstruction": eval_planes,
            "decodedBytesPerBlock": decoded_bytes,
            "outputShape": [resolution, resolution],
            "planeCount": len(dataset),
            "blockCount": len(blocks),
            "device": str(device),
            "scope": (
                "frozen CAESAR neural base reconstruction plus dense-volume "
                "plane sampling; excludes CAESAR compression and disk I/O"
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not 0.0 <= args.maximum_offset < 1.0:
        raise ValueError("maximum offset must be in [0,1)")
    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    model_path = args.model.expanduser().resolve()
    device = torch.device(args.device)
    (
        artifact_directories,
        train_blocks,
        validation_blocks,
        test_blocks,
        split_mode,
    ) = _resolve_blocks(args)
    decoder = FrozenCaesarVFeatureDecoder(model_path, device=device)

    validation = _evaluate(
        decoder,
        validation_blocks,
        output_directory=output_directory,
        split_name="validation",
        resolution=args.resolution,
        eval_planes=args.eval_planes,
        seed=args.seed + 1_000_000,
        orientation=args.orientation,
        maximum_offset=args.maximum_offset,
        device=device,
    )
    test = _evaluate(
        decoder,
        test_blocks,
        output_directory=output_directory,
        split_name="test",
        resolution=args.resolution,
        eval_planes=args.eval_planes,
        seed=args.seed + 2_000_000,
        orientation=args.orientation,
        maximum_offset=args.maximum_offset,
        device=device,
    )
    results = {
        "formatVersion": 1,
        "method": "full-caesar-reconstruction-then-plane-sampling",
        "splitMode": split_mode,
        "validation": validation,
        "test": test,
        "data": {
            "artifactDirectories": [str(path) for path in artifact_directories],
            "train": _block_summary(train_blocks),
            "validation": _block_summary(validation_blocks),
            "test": _block_summary(test_blocks),
            "caesarModel": str(model_path),
            "axisSemantic": train_blocks[0].axis_semantic,
        },
        "sampling": {
            "resolution": args.resolution,
            "planesPerBlock": args.eval_planes,
            "orientation": args.orientation,
            "maximumOffset": args.maximum_offset,
            "seed": args.seed,
        },
    }
    metrics_path = output_directory / "full_reconstruction_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
