#!/usr/bin/env python3
"""Evaluate a point decoder against staged CAESAR reconstruction artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from slice_decoder.datasets import MultiBlockPlaneDataset, load_reference_blocks
from slice_decoder.point_decoder import load_point_decoder_checkpoint
from train_point_decoder import evaluate_point_decoder


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose point-decoder error against CAESAR base and final stages"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=_positive_integer, default=128)
    parser.add_argument("--planes-per-block", type=_positive_integer, default=8)
    parser.add_argument(
        "--orientation",
        choices=("axis-aligned", "random", "mixed"),
        default="mixed",
    )
    parser.add_argument("--maximum-offset", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2_002_025)
    parser.add_argument("--device", default="cpu")
    return parser


def _render_comparison(example_path: Path, output_path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    with np.load(example_path) as example:
        fields = [
            np.asarray(example[name]) for name in ("raw", "base", "caesar", "direct")
        ]
        section = int(example["source_section_index"])
        frame_start = int(example["source_frame_start"])

    errors = [
        fields[1] - fields[0],
        fields[2] - fields[1],
        fields[3] - fields[1],
        fields[3] - fields[0],
    ]
    field_names = (
        "Raw ground truth",
        "CAESAR neural base",
        "CAESAR final",
        "Point decoder",
    )
    error_names = (
        "Base - raw",
        "Final - base (residual)",
        "Point - base",
        "Point - raw",
    )
    field_limit = max(float(np.max(np.abs(field))) for field in fields)

    figure, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for axis, name, field in zip(axes[0], field_names, fields):
        image = axis.imshow(
            field,
            cmap="RdBu_r",
            vmin=-field_limit,
            vmax=field_limit,
            origin="lower",
        )
        axis.set_title(f"{name}\nrange [{field.min():.3f}, {field.max():.3f}]")
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, shrink=0.8)

    for axis, name, error in zip(axes[1], error_names, errors):
        limit = float(np.percentile(np.abs(error), 99.0))
        if limit == 0.0:
            limit = 1.0
        rmse = float(np.sqrt(np.mean(error * error)))
        mae = float(np.mean(np.abs(error)))
        image = axis.imshow(
            error,
            cmap="coolwarm",
            vmin=-limit,
            vmax=limit,
            origin="lower",
        )
        axis.set_title(
            f"{name}\nRMSE {rmse:.4f} | MAE {mae:.4f} | color +/-p99 {limit:.3f}"
        )
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, shrink=0.8)

    figure.suptitle(
        "Staged held-out slice diagnostic "
        f"| source section {section} | block start {frame_start}",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not 0.0 <= args.maximum_offset < 1.0:
        raise ValueError("maximum offset must be in [0,1)")

    blocks = load_reference_blocks(args.artifact_dir)
    if any(block.base_volume is None for block in blocks):
        raise ValueError(
            "diagnostic requires staged artifacts containing caesar_base.npy "
            "and caesar_residual.npy"
        )

    device = torch.device(args.device)
    model, checkpoint_metadata = load_point_decoder_checkpoint(
        args.checkpoint,
        device=device,
    )
    if model.config.latent_channels != blocks[0].latent.shape[0]:
        raise ValueError(
            "checkpoint and artifact latent channels differ: "
            f"{model.config.latent_channels} vs {blocks[0].latent.shape[0]}"
        )

    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    dataset = MultiBlockPlaneDataset(
        blocks,
        sample_count=args.planes_per_block * len(blocks),
        height=args.resolution,
        width=args.resolution,
        seed=args.seed,
        orientation=args.orientation,
        maximum_offset=args.maximum_offset,
    )
    evaluation = evaluate_point_decoder(
        model,
        dataset,
        device=device,
        output_directory=output_directory,
        split_name="diagnostic",
    )
    results = {
        "formatVersion": 1,
        "artifactDirectory": str(Path(args.artifact_dir).expanduser().resolve()),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "checkpointMetadata": checkpoint_metadata,
        "evaluation": evaluation,
        "configuration": {
            "resolution": args.resolution,
            "planesPerBlock": args.planes_per_block,
            "orientation": args.orientation,
            "maximumOffset": args.maximum_offset,
            "seed": args.seed,
            "device": str(device),
        },
    }
    metrics_path = output_directory / "staged_diagnostic_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    example_path = output_directory / "diagnostic_point_decoder_example.npz"
    image_path = output_directory / "staged_diagnostic_comparison.png"
    rendered = _render_comparison(example_path, image_path)
    print(metrics_path)
    if rendered:
        print(image_path)
    else:
        print("matplotlib is unavailable; skipped staged comparison PNG")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
