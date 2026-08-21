#!/usr/bin/env python3
"""Train and evaluate the first direct CAESAR latent-to-slice baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np
import torch

from slice_decoder.datasets import (
    MultiBlockPlaneDataset,
    ReferenceBlock,
    discover_reference_artifacts,
    load_reference_collection,
    parse_index_specification,
    split_reference_blocks,
)
from slice_decoder.metrics import error_decomposition
from slice_decoder.point_decoder import (
    PointDecoderConfig,
    PointQueryDecoder,
    save_point_decoder_checkpoint,
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
        description="Train a point-query decoder on CAESAR reference blocks"
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        action="append",
        default=[],
        help="Reference artifact directory; may be repeated",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        action="append",
        default=[],
        help="Recursively discover reference artifacts; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--block-index",
        type=_nonnegative_integer,
        default=None,
        help="Legacy one-artifact mode: restrict training to one block",
    )
    parser.add_argument("--train-sections", default=None)
    parser.add_argument("--validation-sections", default=None)
    parser.add_argument("--test-sections", default=None)
    parser.add_argument(
        "--expected-axis-semantic",
        choices=("spatial_z", "time", "unconfirmed"),
        default=None,
    )
    parser.add_argument("--steps", type=_positive_integer, default=2000)
    parser.add_argument("--batch-size", type=_positive_integer, default=4)
    parser.add_argument("--train-resolution", type=_positive_integer, default=32)
    parser.add_argument("--eval-resolution", type=_positive_integer, default=128)
    parser.add_argument(
        "--eval-planes",
        type=_positive_integer,
        default=8,
        help="Evaluation planes per block",
    )
    parser.add_argument("--hidden-dimension", type=_positive_integer, default=128)
    parser.add_argument("--hidden-layers", type=_positive_integer, default=4)
    parser.add_argument(
        "--positional-frequencies", type=_nonnegative_integer, default=4
    )
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
    parser.add_argument("--log-interval", type=_positive_integer, default=100)
    return parser


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


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


def _evaluate(
    model: PointQueryDecoder,
    dataset: MultiBlockPlaneDataset,
    *,
    device: torch.device,
    output_directory: Path,
    split_name: str,
) -> dict[str, object]:
    model.eval()
    raw_fields: list[np.ndarray] = []
    caesar_fields: list[np.ndarray] = []
    direct_fields: list[np.ndarray] = []
    latencies: list[float] = []
    by_section: dict[int, dict[str, list[np.ndarray]]] = {}

    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            latent = torch.from_numpy(np.array(sample["latent"], copy=True))
            latent = latent.unsqueeze(0).to(device)
            points = torch.from_numpy(sample["points"]).unsqueeze(0).to(device)
            if index == 0:
                model(latent, points)
                _synchronize(device)
            start_time = time.perf_counter()
            prediction_normalized = model(latent, points)
            _synchronize(device)
            latencies.append((time.perf_counter() - start_time) * 1000.0)

            scale = float(sample["scale"])
            offset = float(sample["offset"])
            prediction = prediction_normalized[0, :, 0].detach().cpu().numpy()
            prediction = prediction * scale + offset
            raw = sample["raw_values"][:, 0]
            caesar = sample["caesar_values"][:, 0]
            raw_fields.append(raw)
            caesar_fields.append(caesar)
            direct_fields.append(prediction)
            section = int(sample["source_section_index"])
            section_fields = by_section.setdefault(
                section,
                {"raw": [], "caesar": [], "direct": []},
            )
            section_fields["raw"].append(raw)
            section_fields["caesar"].append(caesar)
            section_fields["direct"].append(prediction)

            if index == 0:
                height, width = dataset.height, dataset.width
                np.savez(
                    output_directory / f"{split_name}_point_decoder_example.npz",
                    raw=raw.reshape(height, width),
                    caesar=caesar.reshape(height, width),
                    direct=prediction.reshape(height, width),
                    points=sample["points"].reshape(height, width, 3),
                    plane_origin=sample["plane_origin"],
                    plane_axis_u=sample["plane_axis_u"],
                    plane_axis_v=sample["plane_axis_v"],
                    plane_bounds=sample["plane_bounds"],
                    source_section_index=sample["source_section_index"],
                    source_frame_start=sample["source_frame_start"],
                )

    raw_values = np.concatenate(raw_fields)
    caesar_values = np.concatenate(caesar_fields)
    direct_values = np.concatenate(direct_fields)
    section_metrics = {
        str(section): error_decomposition(
            np.concatenate(fields["raw"]),
            np.concatenate(fields["caesar"]),
            np.concatenate(fields["direct"]),
        )
        for section, fields in sorted(by_section.items())
    }
    return {
        "aggregate": error_decomposition(raw_values, caesar_values, direct_values),
        "bySection": section_metrics,
        "directInference": {
            "meanMillisecondsPerSlice": float(np.mean(latencies)),
            "medianMillisecondsPerSlice": float(np.median(latencies)),
            "p95MillisecondsPerSlice": float(np.percentile(latencies, 95.0)),
            "outputShape": [dataset.height, dataset.width],
            "planeCount": len(dataset),
            "device": str(device),
            "scope": "point decoder only; excludes CAESAR compression and disk I/O",
        },
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

    (
        artifact_directories,
        train_blocks,
        validation_blocks,
        test_blocks,
        split_mode,
    ) = _resolve_block_sets(args)
    block = train_blocks[0]
    config = PointDecoderConfig(
        latent_channels=int(block.latent.shape[0]),
        hidden_dimension=args.hidden_dimension,
        hidden_layers=args.hidden_layers,
        positional_frequencies=args.positional_frequencies,
    )
    model = PointQueryDecoder(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    train_dataset = MultiBlockPlaneDataset(
        train_blocks,
        sample_count=args.steps * args.batch_size,
        height=args.train_resolution,
        width=args.train_resolution,
        seed=args.seed,
        orientation=args.orientation,
        maximum_offset=args.maximum_offset,
    )
    first_loss: float | None = None
    final_loss = float("nan")
    start_training = time.perf_counter()
    model.train()
    for step in range(args.steps):
        batch = _numpy_batch(train_dataset, step * args.batch_size, args.batch_size)
        latent = torch.from_numpy(batch["latent"]).to(device)
        points = torch.from_numpy(batch["points"]).to(device)
        target = torch.from_numpy(batch["target_normalized"]).to(device)
        prediction = model(latent, points)
        loss = torch.mean((prediction - target) ** 2)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        final_loss = float(loss.detach().cpu())
        if first_loss is None:
            first_loss = final_loss
        if step == 0 or (step + 1) % args.log_interval == 0 or step + 1 == args.steps:
            print(f"step {step + 1:6d}/{args.steps}: normalized MSE={final_loss:.8e}")
    _synchronize(device)
    training_seconds = time.perf_counter() - start_training

    checkpoint_path = save_point_decoder_checkpoint(
        output_directory / "point_decoder.pt",
        model,
        metadata={
            "artifactDirectories": [str(path) for path in artifact_directories],
            "splitMode": split_mode,
            "train": _block_summary(train_blocks),
            "validation": _block_summary(validation_blocks),
            "test": _block_summary(test_blocks),
            "axisSemantic": block.axis_semantic,
            "seed": args.seed,
            "orientation": args.orientation,
        },
    )

    validation_dataset = MultiBlockPlaneDataset(
        validation_blocks,
        sample_count=args.eval_planes * len(validation_blocks),
        height=args.eval_resolution,
        width=args.eval_resolution,
        seed=args.seed + 1_000_000,
        orientation=args.orientation,
        maximum_offset=args.maximum_offset,
    )
    test_dataset = MultiBlockPlaneDataset(
        test_blocks,
        sample_count=args.eval_planes * len(test_blocks),
        height=args.eval_resolution,
        width=args.eval_resolution,
        seed=args.seed + 2_000_000,
        orientation=args.orientation,
        maximum_offset=args.maximum_offset,
    )
    validation_results = _evaluate(
        model,
        validation_dataset,
        device=device,
        output_directory=output_directory,
        split_name="validation",
    )
    test_results = _evaluate(
        model,
        test_dataset,
        device=device,
        output_directory=output_directory,
        split_name="test",
    )
    results = {
        "formatVersion": 2,
        "splitMode": split_mode,
        "validation": validation_results,
        "test": test_results,
        "training": {
            "steps": args.steps,
            "batchSize": args.batch_size,
            "planeShape": [args.train_resolution, args.train_resolution],
            "firstNormalizedMse": first_loss,
            "finalNormalizedMse": final_loss,
            "seconds": training_seconds,
            "orientation": args.orientation,
            "maximumOffset": args.maximum_offset,
            "seed": args.seed,
        },
        "model": {
            **config.to_dict(),
            "parameterCount": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "checkpoint": str(checkpoint_path),
        },
        "data": {
            "artifactDirectories": [str(path) for path in artifact_directories],
            "axisSemantic": block.axis_semantic,
            "train": _block_summary(train_blocks),
            "validation": _block_summary(validation_blocks),
            "test": _block_summary(test_blocks),
        },
    }
    metrics_path = output_directory / "point_decoder_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
