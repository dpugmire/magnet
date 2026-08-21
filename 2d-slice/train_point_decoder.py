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

from slice_decoder.datasets import RandomPlaneDataset, load_reference_block
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
        description="Overfit a point-query decoder to one CAESAR reference block"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-index", type=_nonnegative_integer, default=0)
    parser.add_argument("--steps", type=_positive_integer, default=2000)
    parser.add_argument("--batch-size", type=_positive_integer, default=4)
    parser.add_argument("--train-resolution", type=_positive_integer, default=32)
    parser.add_argument("--eval-resolution", type=_positive_integer, default=128)
    parser.add_argument("--eval-planes", type=_positive_integer, default=8)
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


def _numpy_batch(dataset: RandomPlaneDataset, start: int, count: int) -> dict[str, np.ndarray]:
    samples = [dataset[start + item] for item in range(count)]
    return {
        "points": np.stack([sample["points"] for sample in samples]),
        "target_normalized": np.stack(
            [sample["target_normalized"] for sample in samples]
        ),
    }


def _evaluate(
    model: PointQueryDecoder,
    latent: torch.Tensor,
    dataset: RandomPlaneDataset,
    *,
    device: torch.device,
    scale: float,
    offset: float,
    output_directory: Path,
) -> dict[str, object]:
    model.eval()
    raw_fields: list[np.ndarray] = []
    caesar_fields: list[np.ndarray] = []
    direct_fields: list[np.ndarray] = []
    latencies: list[float] = []

    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            points = torch.from_numpy(sample["points"]).unsqueeze(0).to(device)
            if index == 0:
                model(latent, points)
                _synchronize(device)
            start_time = time.perf_counter()
            prediction_normalized = model(latent, points)
            _synchronize(device)
            latencies.append((time.perf_counter() - start_time) * 1000.0)

            prediction = (
                prediction_normalized[0, :, 0].detach().cpu().numpy() * scale + offset
            )
            raw = sample["raw_values"][:, 0]
            caesar = sample["caesar_values"][:, 0]
            raw_fields.append(raw)
            caesar_fields.append(caesar)
            direct_fields.append(prediction)

            if index == 0:
                height, width = dataset.height, dataset.width
                np.savez(
                    output_directory / "point_decoder_example.npz",
                    raw=raw.reshape(height, width),
                    caesar=caesar.reshape(height, width),
                    direct=prediction.reshape(height, width),
                    points=sample["points"].reshape(height, width, 3),
                    plane_origin=sample["plane_origin"],
                    plane_axis_u=sample["plane_axis_u"],
                    plane_axis_v=sample["plane_axis_v"],
                    plane_bounds=sample["plane_bounds"],
                )

    raw_values = np.concatenate(raw_fields)
    caesar_values = np.concatenate(caesar_fields)
    direct_values = np.concatenate(direct_fields)
    decomposition = error_decomposition(raw_values, caesar_values, direct_values)
    decomposition["directInference"] = {
        "meanMillisecondsPerSlice": float(np.mean(latencies)),
        "medianMillisecondsPerSlice": float(np.median(latencies)),
        "p95MillisecondsPerSlice": float(np.percentile(latencies, 95.0)),
        "outputShape": [dataset.height, dataset.width],
        "planeCount": len(dataset),
        "device": str(device),
        "scope": "point decoder only; excludes CAESAR compression and disk I/O",
    }
    return decomposition


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

    block = load_reference_block(args.artifact_dir, block_index=args.block_index)
    latent = torch.from_numpy(np.array(block.latent, dtype=np.float32, copy=True))
    latent = latent.unsqueeze(0).to(device)
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

    train_dataset = RandomPlaneDataset(
        block,
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
            "artifactDirectory": str(block.artifact_directory),
            "rawSource": str(block.raw_source),
            "blockIndex": args.block_index,
            "blockFrameStart": block.frame_start,
            "blockFrameEnd": block.frame_end,
            "axisSemantic": block.axis_semantic,
            "scale": block.scale,
            "offset": block.offset,
            "seed": args.seed,
            "orientation": args.orientation,
        },
    )

    evaluation_dataset = RandomPlaneDataset(
        block,
        sample_count=args.eval_planes,
        height=args.eval_resolution,
        width=args.eval_resolution,
        seed=args.seed + 1_000_000,
        orientation=args.orientation,
        maximum_offset=args.maximum_offset,
    )
    results = _evaluate(
        model,
        latent,
        evaluation_dataset,
        device=device,
        scale=block.scale,
        offset=block.offset,
        output_directory=output_directory,
    )
    results["formatVersion"] = 1
    results["training"] = {
        "steps": args.steps,
        "batchSize": args.batch_size,
        "planeShape": [args.train_resolution, args.train_resolution],
        "firstNormalizedMse": first_loss,
        "finalNormalizedMse": final_loss,
        "seconds": training_seconds,
        "orientation": args.orientation,
        "maximumOffset": args.maximum_offset,
        "seed": args.seed,
    }
    results["model"] = {
        **config.to_dict(),
        "parameterCount": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint": str(checkpoint_path),
    }
    results["data"] = {
        "artifactDirectory": str(block.artifact_directory),
        "rawSource": str(block.raw_source),
        "blockIndex": args.block_index,
        "frameStart": block.frame_start,
        "frameEnd": block.frame_end,
        "axisSemantic": block.axis_semantic,
        "scale": block.scale,
        "offset": block.offset,
    }
    metrics_path = output_directory / "point_decoder_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
