#!/usr/bin/env python3
"""Benchmark uncertainty-guided exact CAESAR frame fallback."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
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
)
from slice_decoder.hybrid import (
    connected_component_count,
    required_depth_frames,
    sample_decoded_frames,
    select_largest_mask,
    select_largest_tiles,
)
from slice_decoder.plane_decoder import (
    PlaneConvolutionalDecoder,
    load_plane_decoder_checkpoint,
)
from slice_decoder.uncertainty import load_uncertainty_checkpoint


_POLICIES = (
    "uncertaintyPixels",
    "uncertaintyTiles",
    "randomPixels",
    "errorOraclePixels",
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("fraction must be in [0,1]")
    return parsed


def _fractions(value: str) -> tuple[float, ...]:
    parsed = tuple(_fraction(token.strip()) for token in value.split(","))
    if not parsed:
        raise argparse.ArgumentTypeError("at least one fraction is required")
    if len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("fractions must be unique")
    return tuple(sorted(parsed))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, action="append", default=[])
    parser.add_argument("--artifact-root", type=Path, action="append", default=[])
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--uncertainty-checkpoint", type=Path, required=True)
    parser.add_argument("--caesar-model", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-sections", default="14-15")
    parser.add_argument(
        "--expected-axis-semantic",
        choices=("spatial_z", "time", "unconfirmed"),
        default=None,
    )
    parser.add_argument("--eval-planes", type=_positive_integer, default=8)
    parser.add_argument(
        "--fractions",
        type=_fractions,
        default=_fractions("0,0.01,0.05,0.10,0.20,0.50"),
    )
    parser.add_argument("--tile-size", type=_positive_integer, default=16)
    parser.add_argument(
        "--exact-fallback-fraction",
        type=_fraction,
        default=0.10,
    )
    parser.add_argument(
        "--exact-plane-limit",
        type=_nonnegative_integer,
        default=0,
        help="Limit actual CAESAR fallback planes; zero evaluates every plane",
    )
    parser.add_argument(
        "--orientation",
        choices=("axis-aligned", "random", "mixed"),
        default="mixed",
    )
    parser.add_argument("--maximum-offset", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--context-device", default=None)
    return parser


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _block_summary(blocks: Sequence[ReferenceBlock]) -> dict[str, object]:
    return {
        "blockCount": len(blocks),
        "sourceSections": sorted({block.source_section_index for block in blocks}),
        "sourceFrameRange": [
            min(block.source_frame_start for block in blocks),
            max(block.source_frame_end for block in blocks),
        ],
    }


@dataclass
class _RefinementAccumulator:
    squared_error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    pixel_count: int = 0
    selected_pixel_count: int = 0
    maximum_absolute_error: float = 0.0
    plane_p95_sum: float = 0.0
    plane_count: int = 0
    component_count_sum: int = 0
    required_frame_count_sum: int = 0
    tile_count_sum: int = 0

    def add(
        self,
        signed_error: np.ndarray,
        mask: np.ndarray,
        points: np.ndarray,
        *,
        depth: int,
        tile_count: int = 0,
    ) -> None:
        error = np.asarray(signed_error, dtype=np.float64).copy()
        selected = np.asarray(mask, dtype=bool)
        if error.shape != selected.shape:
            raise ValueError("error and refinement mask shapes differ")
        error[selected] = 0.0
        absolute_error = np.abs(error)
        self.squared_error_sum += float(np.sum(error * error, dtype=np.float64))
        self.absolute_error_sum += float(np.sum(absolute_error, dtype=np.float64))
        self.pixel_count += error.size
        self.selected_pixel_count += int(np.count_nonzero(selected))
        self.maximum_absolute_error = max(
            self.maximum_absolute_error,
            float(np.max(absolute_error)),
        )
        self.plane_p95_sum += float(np.percentile(absolute_error, 95.0))
        self.plane_count += 1
        self.component_count_sum += connected_component_count(selected)
        self.required_frame_count_sum += len(
            required_depth_frames(points, selected, depth=depth)
        )
        self.tile_count_sum += tile_count

    def result(self, *, depth: int) -> dict[str, float | int]:
        if self.pixel_count == 0 or self.plane_count == 0:
            raise RuntimeError("refinement accumulator is empty")
        return {
            "planeCount": self.plane_count,
            "pixelCount": self.pixel_count,
            "selectedPixelCount": self.selected_pixel_count,
            "selectedPixelFraction": self.selected_pixel_count / self.pixel_count,
            "rmseVsCaesarBase": float(
                np.sqrt(self.squared_error_sum / self.pixel_count)
            ),
            "maeVsCaesarBase": self.absolute_error_sum / self.pixel_count,
            "maximumAbsoluteError": self.maximum_absolute_error,
            "meanPlaneP95AbsoluteError": self.plane_p95_sum / self.plane_count,
            "meanConnectedComponentsPerPlane": (
                self.component_count_sum / self.plane_count
            ),
            "meanRequiredDepthFramesPerPlane": (
                self.required_frame_count_sum / self.plane_count
            ),
            "meanDecodedDepthFraction": (
                self.required_frame_count_sum / (self.plane_count * depth)
            ),
            "meanSelectedTilesPerPlane": self.tile_count_sum / self.plane_count,
        }


@dataclass
class _ExactAccumulator:
    squared_error_sum: float = 0.0
    absolute_error_sum: float = 0.0
    pixel_count: int = 0
    selected_pixel_count: int = 0
    plane_count: int = 0
    frame_count: int = 0
    maximum_reference_difference: float = 0.0
    decode_milliseconds: float = 0.0
    sampling_milliseconds: float = 0.0

    def add(
        self,
        signed_error: np.ndarray,
        *,
        selected_pixel_count: int,
        frame_count: int,
        maximum_reference_difference: float,
        decode_milliseconds: float,
        sampling_milliseconds: float,
    ) -> None:
        error = np.asarray(signed_error, dtype=np.float64)
        self.squared_error_sum += float(np.sum(error * error, dtype=np.float64))
        self.absolute_error_sum += float(np.sum(np.abs(error), dtype=np.float64))
        self.pixel_count += error.size
        self.selected_pixel_count += selected_pixel_count
        self.plane_count += 1
        self.frame_count += frame_count
        self.maximum_reference_difference = max(
            self.maximum_reference_difference,
            maximum_reference_difference,
        )
        self.decode_milliseconds += decode_milliseconds
        self.sampling_milliseconds += sampling_milliseconds

    def result(self, *, depth: int) -> dict[str, object]:
        if self.pixel_count == 0 or self.plane_count == 0:
            raise RuntimeError("exact fallback accumulator is empty")
        return {
            "planeCount": self.plane_count,
            "pixelCount": self.pixel_count,
            "selectedPixelCount": self.selected_pixel_count,
            "selectedPixelFraction": self.selected_pixel_count / self.pixel_count,
            "rmseVsCaesarBase": float(
                np.sqrt(self.squared_error_sum / self.pixel_count)
            ),
            "maeVsCaesarBase": self.absolute_error_sum / self.pixel_count,
            "maximumDecodedVsStoredBaseDifference": (
                self.maximum_reference_difference
            ),
            "meanRequiredDepthFramesPerPlane": self.frame_count / self.plane_count,
            "meanDecodedDepthFraction": self.frame_count
            / (self.plane_count * depth),
            "additionalLatency": {
                "meanFrameDecodeMillisecondsPerSlice": (
                    self.decode_milliseconds / self.plane_count
                ),
                "meanSelectedSamplingMillisecondsPerSlice": (
                    self.sampling_milliseconds / self.plane_count
                ),
                "meanTotalMillisecondsPerSlice": (
                    (self.decode_milliseconds + self.sampling_milliseconds)
                    / self.plane_count
                ),
                "scope": (
                    "remaining CAESAR frame decode and selected-point sampling; "
                    "late feature extraction is shared with the direct decoder"
                ),
            },
        }


def _select_blocks(
    blocks: Sequence[ReferenceBlock],
    specification: str,
) -> tuple[ReferenceBlock, ...]:
    sections = parse_index_specification(specification)
    selected = tuple(
        block for block in blocks if block.source_section_index in sections
    )
    if not selected:
        available = sorted({block.source_section_index for block in blocks})
        raise ValueError(
            f"test split is empty; requested {sorted(sections)}, "
            f"available sections are {available}"
        )
    return selected


def _prediction_and_uncertainty(
    decoder: PlaneConvolutionalDecoder,
    uncertainty: torch.nn.Module,
    feature_decoder: FrozenCaesarVFeatureDecoder,
    latent: torch.Tensor,
    points: torch.Tensor,
    *,
    q95: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    late = feature_decoder.extract(latent, "late")
    features = decoder.forward_features(late, points)
    prediction = decoder.output_network[-1](features)
    predicted_scale = uncertainty(features, prediction)
    return late, prediction, q95 * predicted_scale


def _masks(
    uncertainty: np.ndarray,
    absolute_error: np.ndarray,
    random_scores: np.ndarray,
    fraction: float,
    *,
    tile_size: int,
) -> dict[str, tuple[np.ndarray, int]]:
    tile_mask, tile_count = select_largest_tiles(
        uncertainty,
        fraction,
        tile_size=tile_size,
    )
    return {
        "uncertaintyPixels": (select_largest_mask(uncertainty, fraction), 0),
        "uncertaintyTiles": (tile_mask, tile_count),
        "randomPixels": (select_largest_mask(random_scores, fraction), 0),
        "errorOraclePixels": (select_largest_mask(absolute_error, fraction), 0),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not 0.0 <= args.maximum_offset < 1.0:
        raise ValueError("maximum offset must be in [0,1)")
    if 0.0 not in args.fractions:
        raise ValueError("--fractions must include zero for the baseline")
    device = torch.device(args.device)
    if args.context_device is not None:
        context_device = torch.device(args.context_device)
    elif device.type == "mps":
        context_device = torch.device("cpu")
    else:
        context_device = device
    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    decoder_path = args.decoder_checkpoint.expanduser().resolve()
    uncertainty_path = args.uncertainty_checkpoint.expanduser().resolve()
    decoder, decoder_metadata = load_plane_decoder_checkpoint(
        decoder_path,
        device=device,
    )
    if not isinstance(decoder, PlaneConvolutionalDecoder):
        raise ValueError("hybrid benchmark requires a convolutional plane decoder")
    if decoder_metadata.get("featureTap") != "late":
        raise ValueError("hybrid benchmark currently requires a late-context decoder")
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    uncertainty, calibration, uncertainty_metadata = load_uncertainty_checkpoint(
        uncertainty_path,
        device=device,
    )
    uncertainty.eval()
    for parameter in uncertainty.parameters():
        parameter.requires_grad_(False)
    if uncertainty.config.feature_channels != decoder.feature_channels:
        raise ValueError("uncertainty head and plane decoder feature counts differ")
    recorded_decoder_hash = uncertainty_metadata.get("decoderCheckpointSha256")
    if recorded_decoder_hash and recorded_decoder_hash != _sha256(decoder_path):
        raise ValueError("uncertainty checkpoint belongs to another plane decoder")
    if "q95" not in calibration:
        raise ValueError("uncertainty checkpoint has no q95 calibration factor")
    q95 = float(calibration["q95"])

    recorded_model = uncertainty_metadata.get("caesarModel") or decoder_metadata.get(
        "caesarModel"
    )
    model_path = args.caesar_model or (
        None if recorded_model is None else Path(recorded_model)
    )
    if model_path is None:
        raise ValueError("no CAESAR model was specified or recorded")
    model_path = model_path.expanduser().resolve()
    feature_decoder = FrozenCaesarVFeatureDecoder(
        model_path,
        device=context_device,
    )

    artifact_directories = args.artifact_dir
    if not artifact_directories and not args.artifact_root:
        artifact_directories = [
            Path(path)
            for path in uncertainty_metadata.get("artifactDirectories", [])
        ]
    resolved_artifacts = discover_reference_artifacts(
        artifact_directories=artifact_directories,
        artifact_roots=args.artifact_root,
    )
    all_blocks = load_reference_collection(resolved_artifacts)
    if any(block.base_volume is None for block in all_blocks):
        raise ValueError("hybrid benchmark requires staged CAESAR base artifacts")
    semantics = {block.axis_semantic for block in all_blocks}
    if len(semantics) != 1:
        raise ValueError(f"reference artifacts have mixed semantics: {semantics}")
    axis_semantic = next(iter(semantics))
    if (
        args.expected_axis_semantic is not None
        and axis_semantic != args.expected_axis_semantic
    ):
        raise ValueError(
            f"expected {args.expected_axis_semantic!r}, found {axis_semantic!r}"
        )
    test_blocks = _select_blocks(all_blocks, args.test_sections)
    resolution = decoder.config.output_resolution
    dataset = MultiBlockPlaneDataset(
        test_blocks,
        sample_count=args.eval_planes * len(test_blocks),
        height=resolution,
        width=resolution,
        seed=args.seed + 2_000_000,
        orientation=args.orientation,
        maximum_offset=args.maximum_offset,
        target="base",
        include_reference_fields=True,
    )
    depth = int(test_blocks[0].base_volume.shape[0])
    if depth != 8:
        raise ValueError(f"late CAESAR feature fallback expects depth 8, got {depth}")
    if any(block.base_volume.shape[0] != depth for block in test_blocks):
        raise ValueError("test blocks have inconsistent depth")

    accumulators = {
        (policy, fraction): _RefinementAccumulator()
        for policy in _POLICIES
        for fraction in args.fractions
    }
    exact = {
        "uncertaintyPixels": _ExactAccumulator(),
        "uncertaintyTiles": _ExactAccumulator(),
    }
    direct_latencies: list[float] = []
    exact_limit = len(dataset) if args.exact_plane_limit == 0 else min(
        args.exact_plane_limit, len(dataset)
    )
    frame_decoder_warmed = False

    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            latent = torch.from_numpy(np.array(sample["latent"], copy=True)).unsqueeze(
                0
            )
            points_numpy = sample["points"].reshape(resolution, resolution, 3)
            points = torch.from_numpy(points_numpy).unsqueeze(0).to(device)
            if index == 0:
                _prediction_and_uncertainty(
                    decoder,
                    uncertainty,
                    feature_decoder,
                    latent,
                    points,
                    q95=q95,
                )
                _synchronize(device)
                _synchronize(context_device)
            start = time.perf_counter()
            late, prediction_normalized, half_width_normalized = (
                _prediction_and_uncertainty(
                    decoder,
                    uncertainty,
                    feature_decoder,
                    latent,
                    points,
                    q95=q95,
                )
            )
            _synchronize(device)
            _synchronize(context_device)
            direct_latencies.append((time.perf_counter() - start) * 1000.0)

            block_scale = float(sample["scale"])
            block_offset = float(sample["offset"])
            prediction = (
                prediction_normalized[0, 0].cpu().numpy() * block_scale
                + block_offset
            )
            half_width = (
                half_width_normalized[0, 0].cpu().numpy() * block_scale
            )
            base = sample["base_values"][:, 0].reshape(resolution, resolution)
            raw = sample["raw_values"][:, 0].reshape(resolution, resolution)
            signed_error = prediction - base
            absolute_error = np.abs(signed_error)
            random_scores = np.random.default_rng(
                np.random.SeedSequence([args.seed + 3_000_000, index])
            ).random(signed_error.shape)

            for fraction in args.fractions:
                masks = _masks(
                    half_width,
                    absolute_error,
                    random_scores,
                    fraction,
                    tile_size=args.tile_size,
                )
                for policy, (mask, tile_count) in masks.items():
                    accumulators[(policy, fraction)].add(
                        signed_error,
                        mask,
                        points_numpy,
                        depth=depth,
                        tile_count=tile_count,
                    )

            if index < exact_limit:
                pixel_mask = select_largest_mask(
                    half_width,
                    args.exact_fallback_fraction,
                )
                tile_mask, _ = select_largest_tiles(
                    half_width,
                    args.exact_fallback_fraction,
                    tile_size=args.tile_size,
                )
                exact_outputs: dict[
                    str, tuple[np.ndarray, tuple[int, ...], np.ndarray]
                ] = {}
                for policy, selected in (
                    ("uncertaintyPixels", pixel_mask),
                    ("uncertaintyTiles", tile_mask),
                ):
                    frame_indices = required_depth_frames(
                        points_numpy,
                        selected,
                        depth=depth,
                    )
                    hybrid = prediction.copy()
                    maximum_difference = 0.0
                    decode_milliseconds = 0.0
                    sampling_milliseconds = 0.0
                    if frame_indices:
                        if not frame_decoder_warmed:
                            feature_decoder.decode_late_frames(late, frame_indices)
                            _synchronize(context_device)
                            frame_decoder_warmed = True
                        decode_start = time.perf_counter()
                        decoded = feature_decoder.decode_late_frames(
                            late, frame_indices
                        )
                        _synchronize(context_device)
                        decode_milliseconds = (
                            time.perf_counter() - decode_start
                        ) * 1000.0
                        decoded_frames = decoded[0, 0].cpu().numpy()
                        sampling_start = time.perf_counter()
                        selected_values = sample_decoded_frames(
                            decoded_frames,
                            frame_indices,
                            points_numpy[selected],
                            volume_depth=depth,
                        )
                        selected_values = selected_values * block_scale + block_offset
                        sampling_milliseconds = (
                            time.perf_counter() - sampling_start
                        ) * 1000.0
                        maximum_difference = float(
                            np.max(np.abs(selected_values - base[selected]))
                        )
                        hybrid[selected] = selected_values
                    exact[policy].add(
                        hybrid - base,
                        selected_pixel_count=int(np.count_nonzero(selected)),
                        frame_count=len(frame_indices),
                        maximum_reference_difference=maximum_difference,
                        decode_milliseconds=decode_milliseconds,
                        sampling_milliseconds=sampling_milliseconds,
                    )
                    exact_outputs[policy] = (selected, frame_indices, hybrid)

                if index == 0:
                    pixel_selected, pixel_frames, pixel_hybrid = exact_outputs[
                        "uncertaintyPixels"
                    ]
                    tile_selected, tile_frames, tile_hybrid = exact_outputs[
                        "uncertaintyTiles"
                    ]
                    np.savez(
                        output_directory / "hybrid_refinement_example.npz",
                        raw=raw,
                        base=base,
                        plane=prediction,
                        predicted_95_half_width=half_width,
                        pixel_selected_mask=pixel_selected,
                        pixel_required_depth_frames=np.asarray(pixel_frames),
                        pixel_hybrid=pixel_hybrid,
                        tile_selected_mask=tile_selected,
                        tile_required_depth_frames=np.asarray(tile_frames),
                        tile_hybrid=tile_hybrid,
                        original_signed_error=signed_error,
                        pixel_hybrid_signed_error=pixel_hybrid - base,
                        tile_hybrid_signed_error=tile_hybrid - base,
                        points=points_numpy,
                        plane_origin=sample["plane_origin"],
                        plane_axis_u=sample["plane_axis_u"],
                        plane_axis_v=sample["plane_axis_v"],
                        plane_bounds=sample["plane_bounds"],
                        source_section_index=sample["source_section_index"],
                        source_frame_start=sample["source_frame_start"],
                    )
            if index == 0 or (index + 1) % 64 == 0 or index + 1 == len(dataset):
                print(f"plane {index + 1:4d}/{len(dataset)}")

    simulation = {
        policy: [
            {
                "requestedFraction": fraction,
                **accumulators[(policy, fraction)].result(depth=depth),
            }
            for fraction in args.fractions
        ]
        for policy in _POLICIES
    }
    direct_latency = {
        "meanMillisecondsPerSlice": float(np.mean(direct_latencies)),
        "medianMillisecondsPerSlice": float(np.median(direct_latencies)),
        "p95MillisecondsPerSlice": float(np.percentile(direct_latencies, 95.0)),
        "scope": (
            "late feature extraction, learned plane decoder, and uncertainty "
            "head; excludes CAESAR compression and disk I/O"
        ),
    }
    baseline_rmse = simulation["uncertaintyPixels"][0]["rmseVsCaesarBase"]
    exact_results: dict[str, dict[str, object]] = {}
    for policy, accumulator in exact.items():
        result = accumulator.result(depth=depth)
        additional_latency = result["additionalLatency"]
        result["combinedMeanMillisecondsPerSlice"] = (
            direct_latency["meanMillisecondsPerSlice"]
            + additional_latency["meanTotalMillisecondsPerSlice"]
        )
        result["relativeRmseReduction"] = (
            baseline_rmse - result["rmseVsCaesarBase"]
        ) / baseline_rmse
        exact_results[policy] = result
    metrics = {
        "formatVersion": 1,
        "target": "caesarBase",
        "decoderCheckpoint": str(decoder_path),
        "uncertaintyCheckpoint": str(uncertainty_path),
        "caesarModel": str(model_path),
        "q95": q95,
        "simulation": {
            "fractions": list(args.fractions),
            "tileSize": args.tile_size,
            "policies": simulation,
            "interpretation": (
                "selected pixels are replaced by stored CAESAR-base values to "
                "simulate an exact fallback"
            ),
        },
        "exactOnDemandFrameFallback": {
            "requestedFraction": args.exact_fallback_fraction,
            "tileSize": args.tile_size,
            "policies": exact_results,
        },
        "directSliceLatency": direct_latency,
        "data": {
            "artifactDirectories": [str(path) for path in resolved_artifacts],
            "test": _block_summary(test_blocks),
            "testSections": sorted(parse_index_specification(args.test_sections)),
            "axisSemantic": axis_semantic,
            "planeCount": len(dataset),
            "planeShape": [resolution, resolution],
            "orientation": args.orientation,
            "maximumOffset": args.maximum_offset,
            "seed": args.seed,
        },
        "exactness": {
            "smallestExactDecodeUnit": "one complete depth frame",
            "reason": (
                "CAESAR BCRN uses whole-frame contrast and average statistics; "
                "spatial crops do not preserve exact full-frame output"
            ),
        },
        "devices": {
            "planeDecoder": str(device),
            "caesarFeaturesAndFallback": str(context_device),
        },
    }
    metrics_path = output_directory / "hybrid_refinement_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
