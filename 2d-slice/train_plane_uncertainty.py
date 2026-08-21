#!/usr/bin/env python3
"""Train and calibrate per-pixel uncertainty for a frozen plane decoder."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Callable, Sequence

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
from slice_decoder.plane_decoder import (
    PlaneConvolutionalDecoder,
    PlaneFeatureInput,
    load_plane_decoder_checkpoint,
)
from slice_decoder.uncertainty import (
    PlaneUncertaintyConfig,
    PlaneUncertaintyHead,
    conformal_scale_quantile,
    gaussian_scale_nll,
    save_uncertainty_checkpoint,
)


_COVERAGES = (0.90, 0.95, 0.99)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        action="append",
        default=[],
        help="Staged reference artifact directory; may be repeated",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        action="append",
        default=[],
        help="Recursively discover staged artifacts; may be repeated",
    )
    parser.add_argument("--decoder-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--uncertainty-train-sections", default="12")
    parser.add_argument("--calibration-sections", default="13")
    parser.add_argument("--test-sections", default="14-15")
    parser.add_argument(
        "--expected-axis-semantic",
        choices=("spatial_z", "time", "unconfirmed"),
        default=None,
    )
    parser.add_argument("--steps", type=_positive_integer, default=3000)
    parser.add_argument("--batch-size", type=_positive_integer, default=2)
    parser.add_argument("--eval-planes", type=_positive_integer, default=8)
    parser.add_argument("--hidden-channels", type=_positive_integer, default=32)
    parser.add_argument("--minimum-scale", type=float, default=1.0e-4)
    parser.add_argument("--initial-scale", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument(
        "--orientation",
        choices=("axis-aligned", "random", "mixed"),
        default="mixed",
    )
    parser.add_argument("--maximum-offset", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--context-device",
        default=None,
        help="Frozen CAESAR device; defaults to CPU for MPS, otherwise --device",
    )
    parser.add_argument(
        "--caesar-model",
        type=Path,
        default=None,
        help="Override the CAESAR checkpoint recorded by a contextual decoder",
    )
    parser.add_argument("--log-interval", type=_positive_integer, default=100)
    return parser


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _synchronize_devices(*devices: torch.device) -> None:
    seen: set[str] = set()
    for device in devices:
        if str(device) not in seen:
            _synchronize(device)
            seen.add(str(device))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_blocks(
    blocks: Sequence[ReferenceBlock],
    specification: str,
    split_name: str,
) -> tuple[ReferenceBlock, ...]:
    sections = parse_index_specification(specification)
    selected = tuple(
        block for block in blocks if block.source_section_index in sections
    )
    if not selected:
        available = sorted({block.source_section_index for block in blocks})
        raise ValueError(
            f"{split_name} is empty; requested {sorted(sections)}, "
            f"available sections are {available}"
        )
    return selected


def _block_summary(blocks: Sequence[ReferenceBlock]) -> dict[str, object]:
    return {
        "blockCount": len(blocks),
        "sourceSections": sorted({block.source_section_index for block in blocks}),
        "sourceFrameRange": [
            min(block.source_frame_start for block in blocks),
            max(block.source_frame_end for block in blocks),
        ],
    }


def _batch(
    dataset: MultiBlockPlaneDataset,
    start: int,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = [dataset[start + offset] for offset in range(count)]
    latent = torch.from_numpy(
        np.stack([sample["latent"] for sample in samples])
    )
    points = torch.from_numpy(
        np.stack([sample["points"] for sample in samples])
    )
    targets = torch.from_numpy(
        np.stack([sample["target_normalized"] for sample in samples])
    )
    return latent, points, targets


def _reshape_points(
    points: torch.Tensor,
    batch_size: int,
    resolution: int,
    device: torch.device,
) -> torch.Tensor:
    return points.reshape(batch_size, resolution, resolution, 3).to(device)


def _reshape_targets(
    targets: torch.Tensor,
    batch_size: int,
    resolution: int,
    device: torch.device,
) -> torch.Tensor:
    return (
        targets.reshape(batch_size, resolution, resolution, 1)
        .permute(0, 3, 1, 2)
        .to(device)
    )


def _predict(
    decoder: PlaneConvolutionalDecoder,
    uncertainty: PlaneUncertaintyHead,
    latent: torch.Tensor,
    points: torch.Tensor,
    prepare_input: Callable[[torch.Tensor], PlaneFeatureInput],
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        features = decoder.forward_features(prepare_input(latent), points)
        prediction = decoder.output_network[-1](features)
    return prediction, uncertainty(features, prediction)


def _pearson(first: np.ndarray, second: np.ndarray) -> float | None:
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size < 2 or np.std(first) == 0.0 or np.std(second) == 0.0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def _orientation(sample: dict[str, np.ndarray]) -> tuple[str, np.ndarray, float]:
    axis_u = np.asarray(sample["plane_axis_u"], dtype=np.float64)
    axis_v = np.asarray(sample["plane_axis_v"], dtype=np.float64)
    normal = np.cross(axis_u, axis_v)
    normal /= np.linalg.norm(normal)
    axis_aligned = bool(
        np.isclose(np.max(np.abs(normal)), 1.0, atol=1.0e-6)
        and np.count_nonzero(np.abs(normal) > 1.0e-6) == 1
    )
    origin = np.asarray(sample["plane_origin"], dtype=np.float64)
    return ("axis-aligned" if axis_aligned else "random"), normal, float(
        np.dot(origin, normal)
    )


def _calibration_key(coverage: float) -> str:
    return f"q{int(round(coverage * 100.0))}"


def _collect_calibration(
    decoder: PlaneConvolutionalDecoder,
    uncertainty: PlaneUncertaintyHead,
    dataset: MultiBlockPlaneDataset,
    *,
    device: torch.device,
    context_device: torch.device,
    prepare_input: Callable[[torch.Tensor], PlaneFeatureInput],
) -> tuple[dict[str, float], dict[str, object]]:
    scores: list[np.ndarray] = []
    normalized_errors: list[np.ndarray] = []
    normalized_scales: list[np.ndarray] = []
    decoder.eval()
    uncertainty.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            latent = torch.from_numpy(np.array(sample["latent"], copy=True)).unsqueeze(
                0
            )
            points = torch.from_numpy(sample["points"])
            points = _reshape_points(points, 1, dataset.height, device)
            features = decoder.forward_features(prepare_input(latent), points)
            prediction = decoder.output_network[-1](features)
            scale = uncertainty(features, prediction)
            target = torch.from_numpy(sample["target_normalized"])
            target = _reshape_targets(target, 1, dataset.height, device)
            absolute_error = torch.abs(target - prediction)
            scores.append((absolute_error / scale).cpu().numpy().reshape(-1))
            normalized_errors.append(absolute_error.cpu().numpy().reshape(-1))
            normalized_scales.append(scale.cpu().numpy().reshape(-1))
    all_scores = np.concatenate(scores)
    all_errors = np.concatenate(normalized_errors)
    all_scales = np.concatenate(normalized_scales)
    quantiles = {
        _calibration_key(coverage): conformal_scale_quantile(all_scores, coverage)
        for coverage in _COVERAGES
    }
    empirical = {
        str(int(round(coverage * 100.0))): float(
            np.mean(all_errors <= quantiles[_calibration_key(coverage)] * all_scales)
        )
        for coverage in _COVERAGES
    }
    return quantiles, {
        "pixelCount": int(all_scores.size),
        "planeCount": len(dataset),
        "scoreMedian": float(np.median(all_scores)),
        "scoreP95": float(np.percentile(all_scores, 95.0)),
        "quantiles": quantiles,
        "empiricalCoverage": empirical,
    }


def _risk_refinement_curve(
    absolute_error: np.ndarray,
    predicted_half_width: np.ndarray,
    *,
    seed: int,
) -> list[dict[str, float | int]]:
    error = np.asarray(absolute_error, dtype=np.float64).reshape(-1)
    width = np.asarray(predicted_half_width, dtype=np.float64).reshape(-1)
    if error.shape != width.shape:
        raise ValueError("error and predicted width arrays differ")
    order = np.argsort(width)[::-1]
    random_order = np.random.default_rng(seed).permutation(error.size)
    squared_error = error * error
    fractions = (0.0, 0.10, 0.20, 0.25, 0.50, 0.75, 1.0)
    result: list[dict[str, float | int]] = []
    for fraction in fractions:
        refined_count = min(int(round(fraction * error.size)), error.size)
        retained_count = error.size - refined_count

        def risks(indices: np.ndarray) -> tuple[float, float]:
            retained = squared_error[indices[refined_count:]]
            selective = 0.0 if retained.size == 0 else float(np.sqrt(np.mean(retained)))
            overall = float(
                np.sqrt(np.sum(retained, dtype=np.float64) / squared_error.size)
            )
            return overall, selective

        uncertainty_overall, uncertainty_selective = risks(order)
        random_overall, random_selective = risks(random_order)
        result.append(
            {
                "refinedFraction": fraction,
                "refinedPixelCount": refined_count,
                "retainedPixelCount": retained_count,
                "uncertaintyRankedOverallRmse": uncertainty_overall,
                "uncertaintyRankedRetainedRmse": uncertainty_selective,
                "randomOverallRmse": random_overall,
                "randomRetainedRmse": random_selective,
            }
        )
    return result


def _evaluate(
    decoder: PlaneConvolutionalDecoder,
    uncertainty: PlaneUncertaintyHead,
    dataset: MultiBlockPlaneDataset,
    *,
    calibration: dict[str, float],
    device: torch.device,
    context_device: torch.device,
    prepare_input: Callable[[torch.Tensor], PlaneFeatureInput],
    output_directory: Path,
    seed: int,
) -> dict[str, object]:
    absolute_errors: list[np.ndarray] = []
    signed_errors: list[np.ndarray] = []
    predicted_scales: list[np.ndarray] = []
    half_widths: dict[float, list[np.ndarray]] = {
        coverage: [] for coverage in _COVERAGES
    }
    plane_records: list[dict[str, object]] = []
    latencies: list[float] = []
    decoder.eval()
    uncertainty.eval()

    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            latent = torch.from_numpy(np.array(sample["latent"], copy=True)).unsqueeze(
                0
            )
            points = torch.from_numpy(sample["points"])
            points = _reshape_points(points, 1, dataset.height, device)
            if index == 0:
                _predict(decoder, uncertainty, latent, points, prepare_input)
                _synchronize_devices(device, context_device)
            start = time.perf_counter()
            prediction_normalized, scale_normalized = _predict(
                decoder,
                uncertainty,
                latent,
                points,
                prepare_input,
            )
            _synchronize_devices(device, context_device)
            elapsed_milliseconds = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_milliseconds)

            block_scale = float(sample["scale"])
            block_offset = float(sample["offset"])
            prediction = (
                prediction_normalized[0, 0].cpu().numpy() * block_scale
                + block_offset
            )
            predicted_scale = scale_normalized[0, 0].cpu().numpy() * block_scale
            raw = sample["raw_values"][:, 0].reshape(dataset.height, dataset.width)
            base = sample["base_values"][:, 0].reshape(dataset.height, dataset.width)
            caesar = sample["caesar_values"][:, 0].reshape(
                dataset.height, dataset.width
            )
            signed_error = prediction - base
            absolute_error = np.abs(signed_error)
            absolute_errors.append(absolute_error.astype(np.float32))
            signed_errors.append(signed_error.astype(np.float32))
            predicted_scales.append(predicted_scale.astype(np.float32))

            coverage_values: dict[str, float] = {}
            for coverage in _COVERAGES:
                width = calibration[_calibration_key(coverage)] * predicted_scale
                half_widths[coverage].append(width.astype(np.float32))
                coverage_values[str(int(round(coverage * 100.0)))] = float(
                    np.mean(absolute_error <= width)
                )

            orientation, normal, signed_offset = _orientation(sample)
            q95_width = half_widths[0.95][-1]
            plane_records.append(
                {
                    "planeIndex": index,
                    "sourceSectionIndex": int(sample["source_section_index"]),
                    "sourceFrameStart": int(sample["source_frame_start"]),
                    "blockPosition": int(sample["block_position"]),
                    "orientation": orientation,
                    "origin": np.asarray(sample["plane_origin"]).tolist(),
                    "axisU": np.asarray(sample["plane_axis_u"]).tolist(),
                    "axisV": np.asarray(sample["plane_axis_v"]).tolist(),
                    "normal": normal.tolist(),
                    "signedOffset": signed_offset,
                    "bounds": np.asarray(sample["plane_bounds"]).tolist(),
                    "rmse": float(np.sqrt(np.mean(signed_error * signed_error))),
                    "mae": float(np.mean(absolute_error)),
                    "p95AbsoluteError": float(np.percentile(absolute_error, 95.0)),
                    "maximumAbsoluteError": float(np.max(absolute_error)),
                    "meanPredictedScale": float(np.mean(predicted_scale)),
                    "meanPredicted95HalfWidth": float(np.mean(q95_width)),
                    "absoluteErrorVs95HalfWidthCorrelation": _pearson(
                        absolute_error, q95_width
                    ),
                    "coverage": coverage_values,
                    "inferenceMilliseconds": elapsed_milliseconds,
                }
            )

            if index == 0:
                q95_width = half_widths[0.95][-1]
                np.savez(
                    output_directory / "test_plane_uncertainty_example.npz",
                    raw=raw,
                    base=base,
                    caesar=caesar,
                    plane=prediction,
                    signed_error=signed_error,
                    absolute_error=absolute_error,
                    predicted_scale=predicted_scale,
                    predicted_90_half_width=half_widths[0.90][-1],
                    predicted_95_half_width=q95_width,
                    predicted_99_half_width=half_widths[0.99][-1],
                    covered_95=(absolute_error <= q95_width),
                    points=sample["points"].reshape(
                        dataset.height, dataset.width, 3
                    ),
                    plane_origin=sample["plane_origin"],
                    plane_axis_u=sample["plane_axis_u"],
                    plane_axis_v=sample["plane_axis_v"],
                    plane_bounds=sample["plane_bounds"],
                    source_section_index=sample["source_section_index"],
                    source_frame_start=sample["source_frame_start"],
                )

    plane_metrics_path = output_directory / "test_plane_uncertainty_metrics.jsonl"
    with plane_metrics_path.open("w", encoding="utf-8") as stream:
        for record in plane_records:
            stream.write(json.dumps(record) + "\n")

    error = np.concatenate([values.reshape(-1) for values in signed_errors])
    absolute_error = np.concatenate(
        [values.reshape(-1) for values in absolute_errors]
    )
    predicted_scale = np.concatenate(
        [values.reshape(-1) for values in predicted_scales]
    )
    concatenated_widths = {
        coverage: np.concatenate(
            [values.reshape(-1) for values in half_widths[coverage]]
        )
        for coverage in _COVERAGES
    }
    q95_width = concatenated_widths[0.95]

    by_orientation: dict[str, dict[str, object]] = {}
    for orientation in ("axis-aligned", "random"):
        records = [
            record for record in plane_records if record["orientation"] == orientation
        ]
        if records:
            by_orientation[orientation] = {
                "planeCount": len(records),
                "meanRmse": float(np.mean([record["rmse"] for record in records])),
                "meanCoverage95": float(
                    np.mean([record["coverage"]["95"] for record in records])
                ),
                "meanPredicted95HalfWidth": float(
                    np.mean(
                        [record["meanPredicted95HalfWidth"] for record in records]
                    )
                ),
            }

    return {
        "planeCount": len(dataset),
        "pixelCount": int(error.size),
        "rmseVsCaesarBase": float(np.sqrt(np.mean(error * error))),
        "maeVsCaesarBase": float(np.mean(absolute_error)),
        "p95AbsoluteError": float(np.percentile(absolute_error, 95.0)),
        "maximumAbsoluteError": float(np.max(absolute_error)),
        "absoluteErrorVsPredictedScaleCorrelation": _pearson(
            absolute_error, predicted_scale
        ),
        "absoluteErrorVs95HalfWidthCorrelation": _pearson(
            absolute_error, q95_width
        ),
        "coverage": {
            str(int(round(coverage * 100.0))): float(
                np.mean(absolute_error <= concatenated_widths[coverage])
            )
            for coverage in _COVERAGES
        },
        "predictedHalfWidth": {
            str(int(round(coverage * 100.0))): {
                "mean": float(np.mean(concatenated_widths[coverage])),
                "median": float(np.median(concatenated_widths[coverage])),
                "p95": float(np.percentile(concatenated_widths[coverage], 95.0)),
            }
            for coverage in _COVERAGES
        },
        "byOrientation": by_orientation,
        "riskRefinementCurve": _risk_refinement_curve(
            absolute_error,
            q95_width,
            seed=seed + 3_000_000,
        ),
        "inference": {
            "meanMillisecondsPerSlice": float(np.mean(latencies)),
            "medianMillisecondsPerSlice": float(np.median(latencies)),
            "p95MillisecondsPerSlice": float(np.percentile(latencies, 95.0)),
            "device": str(device),
            "contextDevice": str(context_device),
            "scope": (
                "frozen feature preparation, scalar decoder, and uncertainty head; "
                "excludes CAESAR compression and disk I/O"
            ),
        },
        "planeMetrics": str(plane_metrics_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("weight decay must be nonnegative")
    if not 0.0 <= args.maximum_offset < 1.0:
        raise ValueError("maximum offset must be in [0,1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    decoder_path = args.decoder_checkpoint.expanduser().resolve()
    decoder, decoder_metadata = load_plane_decoder_checkpoint(
        decoder_path,
        device=device,
    )
    if not isinstance(decoder, PlaneConvolutionalDecoder):
        raise ValueError(
            "uncertainty currently supports convolutional plane decoders only"
        )
    decoder.eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)

    artifact_directories = args.artifact_dir
    if not artifact_directories and not args.artifact_root:
        artifact_directories = [
            Path(path) for path in decoder_metadata.get("artifactDirectories", [])
        ]
    resolved_artifacts = discover_reference_artifacts(
        artifact_directories=artifact_directories,
        artifact_roots=args.artifact_root,
    )
    all_blocks = load_reference_collection(resolved_artifacts)
    if any(block.base_volume is None for block in all_blocks):
        raise ValueError("uncertainty requires staged CAESAR base artifacts")
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

    section_sets = (
        parse_index_specification(args.uncertainty_train_sections),
        parse_index_specification(args.calibration_sections),
        parse_index_specification(args.test_sections),
    )
    if (
        section_sets[0] & section_sets[1]
        or section_sets[0] & section_sets[2]
        or section_sets[1] & section_sets[2]
    ):
        raise ValueError("uncertainty train, calibration, and test sections overlap")
    training_blocks = _select_blocks(
        all_blocks,
        args.uncertainty_train_sections,
        "uncertainty training split",
    )
    calibration_blocks = _select_blocks(
        all_blocks,
        args.calibration_sections,
        "calibration split",
    )
    test_blocks = _select_blocks(all_blocks, args.test_sections, "test split")

    feature_tap = decoder_metadata.get("featureTap")
    contextual = feature_tap in {"early", "late"}
    if feature_tap not in {None, "early", "late"}:
        raise ValueError(f"unsupported decoder feature tap {feature_tap!r}")
    if args.context_device is not None:
        context_device = torch.device(args.context_device)
    elif contextual and device.type == "mps":
        context_device = torch.device("cpu")
    else:
        context_device = device

    feature_decoder: FrozenCaesarVFeatureDecoder | None = None
    caesar_model_path: Path | None = None
    if contextual:
        recorded_path = decoder_metadata.get("caesarModel")
        selected_path = args.caesar_model or (
            None if recorded_path is None else Path(recorded_path)
        )
        if selected_path is None:
            raise ValueError("contextual decoder metadata has no CAESAR checkpoint")
        caesar_model_path = selected_path.expanduser().resolve()
        feature_decoder = FrozenCaesarVFeatureDecoder(
            caesar_model_path,
            device=context_device,
        )

    def prepare_input(latent: torch.Tensor) -> PlaneFeatureInput:
        if feature_decoder is None:
            return latent.to(device)
        if feature_tap not in {"early", "late"}:
            raise RuntimeError("contextual feature tap was not resolved")
        return feature_decoder.extract(latent.to(context_device), feature_tap)

    uncertainty = PlaneUncertaintyHead(
        PlaneUncertaintyConfig(
            feature_channels=decoder.feature_channels,
            hidden_channels=args.hidden_channels,
            minimum_scale=args.minimum_scale,
            initial_scale=args.initial_scale,
        )
    ).to(device)
    optimizer = torch.optim.Adam(
        uncertainty.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    resolution = decoder.config.output_resolution
    training_dataset = MultiBlockPlaneDataset(
        training_blocks,
        sample_count=args.steps * args.batch_size,
        height=resolution,
        width=resolution,
        seed=args.seed,
        orientation=args.orientation,
        maximum_offset=args.maximum_offset,
        target="base",
        include_reference_fields=False,
    )

    first_loss: float | None = None
    final_loss = float("nan")
    training_start = time.perf_counter()
    uncertainty.train()
    for step in range(args.steps):
        latent, points, targets = _batch(
            training_dataset,
            step * args.batch_size,
            args.batch_size,
        )
        points = _reshape_points(points, args.batch_size, resolution, device)
        target = _reshape_targets(targets, args.batch_size, resolution, device)
        prediction, predicted_scale = _predict(
            decoder,
            uncertainty,
            latent,
            points,
            prepare_input,
        )
        loss = gaussian_scale_nll(target - prediction, predicted_scale)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite training loss at step {step + 1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        final_loss = float(loss.detach().cpu())
        if first_loss is None:
            first_loss = final_loss
        if step == 0 or (step + 1) % args.log_interval == 0 or step + 1 == args.steps:
            print(
                f"step {step + 1:6d}/{args.steps}: "
                f"normalized Gaussian NLL={final_loss:.8e}"
            )
    _synchronize_devices(device, context_device)
    training_seconds = time.perf_counter() - training_start

    calibration_dataset = MultiBlockPlaneDataset(
        calibration_blocks,
        sample_count=args.eval_planes * len(calibration_blocks),
        height=resolution,
        width=resolution,
        seed=args.seed + 1_000_000,
        orientation=args.orientation,
        maximum_offset=args.maximum_offset,
        target="base",
        include_reference_fields=False,
    )
    calibration, calibration_metrics = _collect_calibration(
        decoder,
        uncertainty,
        calibration_dataset,
        device=device,
        context_device=context_device,
        prepare_input=prepare_input,
    )

    checkpoint_path = save_uncertainty_checkpoint(
        output_directory / "plane_uncertainty.pt",
        uncertainty,
        calibration=calibration,
        metadata={
            "decoderCheckpoint": str(decoder_path),
            "decoderCheckpointSha256": _sha256(decoder_path),
            "decoderInputRepresentation": decoder_metadata.get(
                "inputRepresentation", "q_latent"
            ),
            "featureTap": feature_tap,
            "caesarModel": (
                None if caesar_model_path is None else str(caesar_model_path)
            ),
            "artifactDirectories": [str(path) for path in resolved_artifacts],
            "uncertaintyTrain": _block_summary(training_blocks),
            "calibration": _block_summary(calibration_blocks),
            "test": _block_summary(test_blocks),
            "target": "caesarBase",
            "axisSemantic": axis_semantic,
            "seed": args.seed,
            "orientation": args.orientation,
        },
    )

    test_dataset = MultiBlockPlaneDataset(
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
    test_metrics = _evaluate(
        decoder,
        uncertainty,
        test_dataset,
        calibration=calibration,
        device=device,
        context_device=context_device,
        prepare_input=prepare_input,
        output_directory=output_directory,
        seed=args.seed,
    )
    metrics = {
        "formatVersion": 1,
        "target": "caesarBase",
        "method": decoder_metadata.get("inputRepresentation", "q_latent"),
        "featureTap": feature_tap,
        "decoderCheckpoint": str(decoder_path),
        "uncertaintyCheckpoint": str(checkpoint_path),
        "training": {
            "sections": sorted(section_sets[0]),
            "blocks": _block_summary(training_blocks),
            "steps": args.steps,
            "batchSize": args.batch_size,
            "firstNormalizedGaussianNll": first_loss,
            "finalNormalizedGaussianNll": final_loss,
            "seconds": training_seconds,
            "learningRate": args.learning_rate,
            "weightDecay": args.weight_decay,
        },
        "calibration": {
            "sections": sorted(section_sets[1]),
            "blocks": _block_summary(calibration_blocks),
            **calibration_metrics,
        },
        "test": {
            "sections": sorted(section_sets[2]),
            "blocks": _block_summary(test_blocks),
            **test_metrics,
        },
        "configuration": {
            **uncertainty.config.to_dict(),
            "resolution": resolution,
            "evalPlanesPerBlock": args.eval_planes,
            "orientation": args.orientation,
            "maximumOffset": args.maximum_offset,
            "seed": args.seed,
            "device": str(device),
            "contextDevice": str(context_device),
        },
        "data": {
            "artifactDirectories": [str(path) for path in resolved_artifacts],
            "axisSemantic": axis_semantic,
            "caesarModel": (
                None if caesar_model_path is None else str(caesar_model_path)
            ),
        },
    }
    metrics_path = output_directory / "plane_uncertainty_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
