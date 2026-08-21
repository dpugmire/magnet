#!/usr/bin/env python3
"""Train and evaluate a plane-aligned convolutional CAESAR decoder."""

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

from slice_decoder.caesar_features import (
    FrozenCaesarVFeatureDecoder,
    caesar_feature_tap_metadata,
)
from slice_decoder.datasets import (
    MultiBlockPlaneDataset,
    ReferenceBlock,
    discover_reference_artifacts,
    load_reference_collection,
    parse_index_specification,
    split_reference_blocks,
)
from slice_decoder.metrics import plane_decoder_error_decomposition
from slice_decoder.plane_decoder import (
    CaesarInitializedPlaneDecoder,
    PlaneFeatureInput,
    PlaneConvolutionalDecoder,
    PlaneDecoderConfig,
    save_plane_decoder_checkpoint,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a plane-aligned convolutional decoder on CAESAR latents"
    )
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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--block-index",
        type=_nonnegative_integer,
        default=None,
        help="One-artifact smoke mode: restrict training to one block",
    )
    parser.add_argument("--train-sections", default=None)
    parser.add_argument("--validation-sections", default=None)
    parser.add_argument("--test-sections", default=None)
    parser.add_argument(
        "--expected-axis-semantic",
        choices=("spatial_z", "time", "unconfirmed"),
        default=None,
    )
    parser.add_argument("--steps", type=_positive_integer, default=10_000)
    parser.add_argument("--batch-size", type=_positive_integer, default=2)
    parser.add_argument("--resolution", type=_positive_integer, default=128)
    parser.add_argument(
        "--coarse-resolution",
        type=_positive_integer,
        default=None,
        help="Defaults to 16 for q_latent or the selected feature-tap resolution",
    )
    parser.add_argument("--eval-planes", type=_positive_integer, default=8)
    parser.add_argument("--hidden-channels", type=_positive_integer, default=64)
    parser.add_argument("--minimum-channels", type=_positive_integer, default=16)
    parser.add_argument("--coarse-blocks", type=_positive_integer, default=2)
    parser.add_argument(
        "--positional-frequencies", type=_nonnegative_integer, default=4
    )
    parser.add_argument(
        "--slab-samples",
        type=_positive_integer,
        default=None,
        help="Defaults to five for q_latent or one for contextual features",
    )
    parser.add_argument(
        "--slab-radius-cells",
        type=float,
        default=1.0,
        help="Half-width of the sampled slab in latent-grid cells",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-6)
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=None,
        help="Optional maximum L2 gradient norm",
    )
    parser.add_argument(
        "--orientation",
        choices=("axis-aligned", "random", "mixed"),
        default="mixed",
    )
    parser.add_argument("--maximum-offset", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--caesar-model",
        type=Path,
        default=None,
        help="CAESAR-V checkpoint used for a frozen contextual feature tap",
    )
    parser.add_argument(
        "--feature-tap",
        choices=("early", "late", "early-late"),
        default=None,
    )
    parser.add_argument(
        "--head-initialization",
        choices=("random", "caesar"),
        default="random",
        help="Use the custom random head or initialize from CAESAR's final 2D decoder",
    )
    parser.add_argument(
        "--context-device",
        default=None,
        help="Frozen CAESAR device; defaults to CPU for MPS, otherwise --device",
    )
    parser.add_argument("--log-interval", type=_positive_integer, default=100)
    return parser


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _synchronize_devices(*devices: torch.device) -> None:
    synchronized: set[str] = set()
    for device in devices:
        if str(device) not in synchronized:
            _synchronize(device)
            synchronized.add(str(device))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _numpy_batch(
    dataset: MultiBlockPlaneDataset,
    start: int,
    count: int,
) -> dict[str, np.ndarray]:
    samples = [dataset[start + item] for item in range(count)]
    return {
        "latent": np.stack([sample["latent"] for sample in samples]),
        "points": np.stack([sample["points"] for sample in samples]),
        "target_normalized": np.stack(
            [sample["target_normalized"] for sample in samples]
        ),
    }


def _resolve_block_sets(
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
    all_blocks = load_reference_collection(artifact_directories)
    if args.block_index is not None:
        if len(artifact_directories) != 1:
            raise ValueError("--block-index requires exactly one artifact directory")
        if not 0 <= args.block_index < len(all_blocks):
            raise IndexError(
                f"block index {args.block_index} is outside [0,{len(all_blocks) - 1}]"
            )
        all_blocks = (all_blocks[args.block_index],)

    semantics = {block.axis_semantic for block in all_blocks}
    if len(semantics) != 1:
        raise ValueError(f"reference artifacts have mixed axis semantics: {semantics}")
    if args.expected_axis_semantic is not None and semantics != {
        args.expected_axis_semantic
    }:
        raise ValueError(
            f"expected axis semantic {args.expected_axis_semantic!r}, "
            f"found {next(iter(semantics))!r}"
        )
    if any(block.base_volume is None for block in all_blocks):
        raise ValueError(
            "plane-decoder training requires staged artifacts containing "
            "caesar_base.npy"
        )

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
            all_blocks,
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

    return artifact_directories, all_blocks, all_blocks, all_blocks, "same-block"


def _block_summary(blocks: Sequence[ReferenceBlock]) -> dict[str, object]:
    return {
        "blockCount": len(blocks),
        "sourceSections": sorted({block.source_section_index for block in blocks}),
        "sourceFrameRange": [
            min(block.source_frame_start for block in blocks),
            max(block.source_frame_end for block in blocks),
        ],
    }


def evaluate_plane_decoder(
    model: PlaneConvolutionalDecoder | CaesarInitializedPlaneDecoder,
    dataset: MultiBlockPlaneDataset,
    *,
    device: torch.device,
    output_directory: Path,
    split_name: str,
    prepare_latent: Callable[[torch.Tensor], PlaneFeatureInput] | None = None,
    context_device: torch.device | None = None,
    inference_scope: str = "plane decoder only",
) -> dict[str, object]:
    model.eval()
    raw_fields: list[np.ndarray] = []
    base_fields: list[np.ndarray] = []
    caesar_fields: list[np.ndarray] = []
    plane_fields: list[np.ndarray] = []
    latencies: list[float] = []
    by_section: dict[int, dict[str, list[np.ndarray]]] = {}

    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            latent = torch.from_numpy(np.array(sample["latent"], copy=True))
            latent = latent.unsqueeze(0)
            points = torch.from_numpy(sample["points"])
            points = points.reshape(1, dataset.height, dataset.width, 3).to(device)
            if index == 0:
                warm_input = (
                    latent.to(device)
                    if prepare_latent is None
                    else prepare_latent(latent)
                )
                model(warm_input, points)
                _synchronize_devices(device, context_device or device)
            start_time = time.perf_counter()
            model_input = (
                latent.to(device) if prepare_latent is None else prepare_latent(latent)
            )
            prediction_normalized = model(model_input, points)
            _synchronize_devices(device, context_device or device)
            latencies.append((time.perf_counter() - start_time) * 1000.0)

            scale = float(sample["scale"])
            offset = float(sample["offset"])
            prediction = prediction_normalized[0, 0].detach().cpu().numpy()
            prediction = prediction * scale + offset
            raw = sample["raw_values"][:, 0].reshape(dataset.height, dataset.width)
            base = sample["base_values"][:, 0].reshape(dataset.height, dataset.width)
            caesar = sample["caesar_values"][:, 0].reshape(
                dataset.height, dataset.width
            )
            raw_fields.append(raw)
            base_fields.append(base)
            caesar_fields.append(caesar)
            plane_fields.append(prediction)

            section = int(sample["source_section_index"])
            fields = by_section.setdefault(
                section,
                {"raw": [], "base": [], "caesar": [], "plane": []},
            )
            fields["raw"].append(raw)
            fields["base"].append(base)
            fields["caesar"].append(caesar)
            fields["plane"].append(prediction)

            if index == 0:
                np.savez(
                    output_directory / f"{split_name}_plane_decoder_example.npz",
                    raw=raw,
                    base=base,
                    caesar=caesar,
                    plane=prediction,
                    residual=sample["residual_values"][:, 0].reshape(
                        dataset.height, dataset.width
                    ),
                    points=sample["points"].reshape(dataset.height, dataset.width, 3),
                    plane_origin=sample["plane_origin"],
                    plane_axis_u=sample["plane_axis_u"],
                    plane_axis_v=sample["plane_axis_v"],
                    plane_bounds=sample["plane_bounds"],
                    source_section_index=sample["source_section_index"],
                    source_frame_start=sample["source_frame_start"],
                )

    def decompose(fields: dict[str, list[np.ndarray]]) -> dict[str, object]:
        return plane_decoder_error_decomposition(
            np.concatenate([values.reshape(-1) for values in fields["raw"]]),
            np.concatenate([values.reshape(-1) for values in fields["base"]]),
            np.concatenate([values.reshape(-1) for values in fields["caesar"]]),
            np.concatenate([values.reshape(-1) for values in fields["plane"]]),
        )

    aggregate_fields = {
        "raw": raw_fields,
        "base": base_fields,
        "caesar": caesar_fields,
        "plane": plane_fields,
    }
    return {
        "aggregate": decompose(aggregate_fields),
        "bySection": {
            str(section): decompose(fields)
            for section, fields in sorted(by_section.items())
        },
        "planeInference": {
            "meanMillisecondsPerSlice": float(np.mean(latencies)),
            "medianMillisecondsPerSlice": float(np.median(latencies)),
            "p95MillisecondsPerSlice": float(np.percentile(latencies, 95.0)),
            "outputShape": [dataset.height, dataset.width],
            "planeCount": len(dataset),
            "device": str(device),
            "contextDevice": str(context_device or device),
            "scope": f"{inference_scope}; excludes CAESAR compression and disk I/O",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    if args.weight_decay < 0.0:
        raise ValueError("weight decay must be nonnegative")
    if args.gradient_clip_norm is not None and args.gradient_clip_norm <= 0.0:
        raise ValueError("gradient clip norm must be positive")
    if not 0.0 <= args.maximum_offset < 1.0:
        raise ValueError("maximum offset must be in [0,1)")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    (
        artifact_directories,
        train_blocks,
        validation_blocks,
        test_blocks,
        split_mode,
    ) = _resolve_block_sets(args)
    block = train_blocks[0]
    contextual = args.caesar_model is not None or args.feature_tap is not None
    if contextual and (args.caesar_model is None or args.feature_tap is None):
        raise ValueError("--caesar-model and --feature-tap must be specified together")
    if args.context_device is not None:
        context_device = torch.device(args.context_device)
    elif contextual and device.type == "mps":
        context_device = torch.device("cpu")
    else:
        context_device = device

    feature_decoder: FrozenCaesarVFeatureDecoder | None = None
    caesar_model_path: Path | None = None
    feature_tap: str | None = None
    if contextual:
        caesar_model_path = args.caesar_model.expanduser().resolve()
        feature_tap = args.feature_tap
        feature_decoder = FrozenCaesarVFeatureDecoder(
            caesar_model_path,
            device=context_device,
        )
        if feature_tap == "early-late":
            early_metadata = caesar_feature_tap_metadata("early")
            late_metadata = caesar_feature_tap_metadata("late")
            latent_channels = early_metadata.channels + late_metadata.channels
            default_coarse_resolution = late_metadata.width
        else:
            tap_metadata = caesar_feature_tap_metadata(feature_tap)
            latent_channels = tap_metadata.channels
            default_coarse_resolution = tap_metadata.width
        coarse_resolution = args.coarse_resolution or default_coarse_resolution
        slab_samples = args.slab_samples or 1
        input_representation = f"caesar-{feature_tap}-features"
        inference_scope = (
            f"frozen CAESAR {feature_tap} feature extraction plus "
            f"{args.head_initialization}-initialized plane head"
        )
    else:
        latent_channels = int(block.latent.shape[0])
        coarse_resolution = args.coarse_resolution or 16
        slab_samples = args.slab_samples or 5
        input_representation = "q_latent"
        inference_scope = "plane decoder only"

    if args.head_initialization == "caesar" and (
        not contextual or feature_tap != "late"
    ):
        raise ValueError(
            "--head-initialization caesar requires --feature-tap late and "
            "--caesar-model"
        )

    def prepare_latent(latent: torch.Tensor) -> PlaneFeatureInput:
        if feature_decoder is None or feature_tap is None:
            return latent.to(device)
        if feature_tap == "early-late":
            early, late = feature_decoder.extract_taps(latent.to(context_device))
            if late is None:
                raise RuntimeError("late CAESAR features were not extracted")
            return early, late
        return feature_decoder.extract(latent.to(context_device), feature_tap)

    config = PlaneDecoderConfig(
        latent_channels=latent_channels,
        coarse_resolution=coarse_resolution,
        output_resolution=args.resolution,
        hidden_channels=args.hidden_channels,
        minimum_channels=args.minimum_channels,
        coarse_blocks=args.coarse_blocks,
        positional_frequencies=args.positional_frequencies,
        slab_samples=slab_samples,
        slab_radius_cells=args.slab_radius_cells,
    )
    if args.head_initialization == "caesar":
        if feature_decoder is None:
            raise RuntimeError("CAESAR feature decoder was not initialized")
        caesar_stage, super_resolution = feature_decoder.copy_downstream_2d_decoder()
        model: PlaneConvolutionalDecoder | CaesarInitializedPlaneDecoder
        model = CaesarInitializedPlaneDecoder(
            config,
            caesar_stage,
            super_resolution,
        ).to(device)
    else:
        model = PlaneConvolutionalDecoder(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_dataset = MultiBlockPlaneDataset(
        train_blocks,
        sample_count=args.steps * args.batch_size,
        height=args.resolution,
        width=args.resolution,
        seed=args.seed,
        orientation=args.orientation,
        maximum_offset=args.maximum_offset,
        target="base",
        include_reference_fields=False,
    )
    first_loss: float | None = None
    final_loss = float("nan")
    start_training = time.perf_counter()
    model.train()
    for step in range(args.steps):
        batch = _numpy_batch(train_dataset, step * args.batch_size, args.batch_size)
        latent = prepare_latent(torch.from_numpy(batch["latent"]))
        points = torch.from_numpy(batch["points"])
        points = points.reshape(
            args.batch_size,
            args.resolution,
            args.resolution,
            3,
        ).to(device)
        target = torch.from_numpy(batch["target_normalized"])
        target = (
            target.reshape(
                args.batch_size,
                args.resolution,
                args.resolution,
                1,
            )
            .permute(0, 3, 1, 2)
            .to(device)
        )
        prediction = model(latent, points)
        loss = torch.mean((prediction - target) ** 2)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite training loss at step {step + 1}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=args.gradient_clip_norm,
                error_if_nonfinite=True,
            )
        optimizer.step()

        final_loss = float(loss.detach().cpu())
        if first_loss is None:
            first_loss = final_loss
        if step == 0 or (step + 1) % args.log_interval == 0 or step + 1 == args.steps:
            print(f"step {step + 1:6d}/{args.steps}: normalized MSE={final_loss:.8e}")
    _synchronize_devices(device, context_device)
    training_seconds = time.perf_counter() - start_training

    checkpoint_path = save_plane_decoder_checkpoint(
        output_directory / "plane_decoder.pt",
        model,
        metadata={
            "artifactDirectories": [str(path) for path in artifact_directories],
            "splitMode": split_mode,
            "train": _block_summary(train_blocks),
            "validation": _block_summary(validation_blocks),
            "test": _block_summary(test_blocks),
            "axisSemantic": block.axis_semantic,
            "target": "caesarBase",
            "seed": args.seed,
            "orientation": args.orientation,
            "inputRepresentation": input_representation,
            "featureTap": feature_tap,
            "headInitialization": args.head_initialization,
            "caesarModel": (
                None if caesar_model_path is None else str(caesar_model_path)
            ),
            "caesarModelSha256": (
                None if caesar_model_path is None else _sha256(caesar_model_path)
            ),
        },
    )

    validation_dataset = MultiBlockPlaneDataset(
        validation_blocks,
        sample_count=args.eval_planes * len(validation_blocks),
        height=args.resolution,
        width=args.resolution,
        seed=args.seed + 1_000_000,
        orientation=args.orientation,
        maximum_offset=args.maximum_offset,
        target="base",
    )
    test_dataset = MultiBlockPlaneDataset(
        test_blocks,
        sample_count=args.eval_planes * len(test_blocks),
        height=args.resolution,
        width=args.resolution,
        seed=args.seed + 2_000_000,
        orientation=args.orientation,
        maximum_offset=args.maximum_offset,
        target="base",
    )
    validation_results = evaluate_plane_decoder(
        model,
        validation_dataset,
        device=device,
        output_directory=output_directory,
        split_name="validation",
        prepare_latent=prepare_latent,
        context_device=context_device,
        inference_scope=inference_scope,
    )
    test_results = evaluate_plane_decoder(
        model,
        test_dataset,
        device=device,
        output_directory=output_directory,
        split_name="test",
        prepare_latent=prepare_latent,
        context_device=context_device,
        inference_scope=inference_scope,
    )
    results = {
        "formatVersion": 2 if contextual else 1,
        "target": "caesarBase",
        "inputRepresentation": input_representation,
        "headInitialization": args.head_initialization,
        "splitMode": split_mode,
        "validation": validation_results,
        "test": test_results,
        "training": {
            "steps": args.steps,
            "batchSize": args.batch_size,
            "planeShape": [args.resolution, args.resolution],
            "firstNormalizedMse": first_loss,
            "finalNormalizedMse": final_loss,
            "seconds": training_seconds,
            "orientation": args.orientation,
            "maximumOffset": args.maximum_offset,
            "seed": args.seed,
            "learningRate": args.learning_rate,
            "weightDecay": args.weight_decay,
            "gradientClipNorm": args.gradient_clip_norm,
        },
        "model": {
            **config.to_dict(),
            "parameterCount": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "checkpoint": str(checkpoint_path),
            "frozenCaesarParameterCount": (
                0
                if feature_decoder is None
                else sum(
                    parameter.numel()
                    for parameter in feature_decoder.caesar_model.parameters()
                )
            ),
        },
        "data": {
            "artifactDirectories": [str(path) for path in artifact_directories],
            "axisSemantic": block.axis_semantic,
            "train": _block_summary(train_blocks),
            "validation": _block_summary(validation_blocks),
            "test": _block_summary(test_blocks),
            "featureTap": feature_tap,
            "headInitialization": args.head_initialization,
            "caesarModel": (
                None if caesar_model_path is None else str(caesar_model_path)
            ),
            "caesarModelSha256": (
                None if caesar_model_path is None else _sha256(caesar_model_path)
            ),
            "contextDevice": str(context_device),
        },
    }
    metrics_path = output_directory / "plane_decoder_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
