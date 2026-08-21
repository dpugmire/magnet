#!/usr/bin/env python3
"""Plot uncertainty-guided CAESAR frame-fallback results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(error) ** 2)))


def _plot_example(example: dict[str, np.ndarray], output_path: Path) -> None:
    base = example["base"]
    plane = example["plane"]
    uncertainty = example["predicted_95_half_width"]
    pixel_mask = example["pixel_selected_mask"].astype(bool)
    tile_mask = example["tile_selected_mask"].astype(bool)
    pixel_hybrid = example["pixel_hybrid"]
    tile_hybrid = example["tile_hybrid"]
    original_error = plane - base
    pixel_error = pixel_hybrid - base
    tile_error = tile_hybrid - base
    scalar_fields = (base, plane, pixel_hybrid, tile_hybrid)
    value_min = min(float(np.min(values)) for values in scalar_fields)
    value_max = max(float(np.max(values)) for values in scalar_fields)
    error_limit = float(
        np.percentile(
            np.concatenate(
                [
                    np.abs(values).reshape(-1)
                    for values in (original_error, pixel_error, tile_error)
                ]
            ),
            99.0,
        )
    )
    uncertainty_limit = float(np.percentile(uncertainty, 99.0))

    figure, axes = plt.subplots(2, 5, figsize=(16, 6.5), constrained_layout=True)
    scalar_image = axes[0, 0].imshow(
        base,
        origin="lower",
        cmap="viridis",
        vmin=value_min,
        vmax=value_max,
    )
    axes[0, 0].set_title("CAESAR base")
    axes[0, 1].imshow(
        plane,
        origin="lower",
        cmap="viridis",
        vmin=value_min,
        vmax=value_max,
    )
    axes[0, 1].set_title(f"Late prediction\nRMSE {_rmse(original_error):.4f}")
    uncertainty_image = axes[0, 2].imshow(
        uncertainty,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=uncertainty_limit,
    )
    axes[0, 2].set_title("Predicted 95% half-width")

    for axis, mask, name, frames in (
        (
            axes[0, 3],
            pixel_mask,
            "Top uncertain pixels",
            example["pixel_required_depth_frames"],
        ),
        (
            axes[0, 4],
            tile_mask,
            "Uncertain 16x16 tiles",
            example["tile_required_depth_frames"],
        ),
    ):
        axis.imshow(
            plane,
            origin="lower",
            cmap="gray",
            vmin=value_min,
            vmax=value_max,
        )
        axis.imshow(
            np.ma.masked_where(~mask, mask),
            origin="lower",
            cmap="Reds",
            vmin=0.0,
            vmax=1.0,
            alpha=0.75,
        )
        axis.set_title(
            f"{name}\n{100.0 * np.mean(mask):.1f}% pixels; "
            f"{len(frames)}/8 frames"
        )

    error_image = axes[1, 0].imshow(
        original_error,
        origin="lower",
        cmap="coolwarm",
        vmin=-error_limit,
        vmax=error_limit,
    )
    axes[1, 0].set_title("Original signed error")
    axes[1, 1].imshow(
        pixel_hybrid,
        origin="lower",
        cmap="viridis",
        vmin=value_min,
        vmax=value_max,
    )
    axes[1, 1].set_title(f"Pixel hybrid\nRMSE {_rmse(pixel_error):.4f}")
    axes[1, 2].imshow(
        pixel_error,
        origin="lower",
        cmap="coolwarm",
        vmin=-error_limit,
        vmax=error_limit,
    )
    axes[1, 2].set_title("Pixel-hybrid signed error")
    axes[1, 3].imshow(
        tile_hybrid,
        origin="lower",
        cmap="viridis",
        vmin=value_min,
        vmax=value_max,
    )
    axes[1, 3].set_title(f"Tile hybrid\nRMSE {_rmse(tile_error):.4f}")
    axes[1, 4].imshow(
        tile_error,
        origin="lower",
        cmap="coolwarm",
        vmin=-error_limit,
        vmax=error_limit,
    )
    axes[1, 4].set_title("Tile-hybrid signed error")
    for axis in axes.reshape(-1):
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(scalar_image, ax=[axes[0, 0], axes[0, 1], axes[1, 1], axes[1, 3]], shrink=0.78, label="scalar value")
    figure.colorbar(
        uncertainty_image,
        ax=axes[0, 2],
        shrink=0.78,
        label="95% error half-width",
    )
    figure.colorbar(
        error_image,
        ax=[axes[1, 0], axes[1, 2], axes[1, 4]],
        shrink=0.78,
        label="prediction - CAESAR base",
    )
    figure.suptitle("Uncertainty-guided exact CAESAR frame fallback", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_curves(metrics: dict[str, object], output_path: Path) -> None:
    policy_data = metrics["simulation"]["policies"]
    styles = {
        "uncertaintyPixels": ("Uncertainty pixels", "o", "tab:blue"),
        "uncertaintyTiles": ("Uncertainty tiles", "s", "tab:orange"),
        "randomPixels": ("Random pixels", "^", "tab:gray"),
        "errorOraclePixels": ("Actual-error oracle", "D", "tab:green"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for policy, rows in policy_data.items():
        label, marker, color = styles[policy]
        fractions = [100.0 * row["selectedPixelFraction"] for row in rows]
        axes[0].plot(
            fractions,
            [row["rmseVsCaesarBase"] for row in rows],
            label=label,
            marker=marker,
            color=color,
        )
        axes[1].plot(
            fractions,
            [100.0 * row["meanDecodedDepthFraction"] for row in rows],
            label=label,
            marker=marker,
            color=color,
        )
    axes[0].set_xlabel("Refined pixels (%)")
    axes[0].set_ylabel("RMSE vs CAESAR base")
    axes[0].set_title("Accuracy benefit")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("Refined pixels (%)")
    axes[1].set_ylabel("Required depth frames (%)")
    axes[1].set_title("Exact frame-decoding cost proxy")
    axes[1].grid(alpha=0.25)
    axes[1].set_ylim(-2.0, 102.0)
    axes[0].legend()
    figure.suptitle("Hybrid refinement accuracy and decoding footprint", fontsize=14)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metrics_path = args.metrics.expanduser().resolve()
    example_path = args.example.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with np.load(example_path) as archive:
        example = {name: np.asarray(archive[name]) for name in archive.files}
    comparison_path = output_directory / "hybrid_refinement_comparison.png"
    curves_path = output_directory / "hybrid_refinement_curves.png"
    _plot_example(example, comparison_path)
    _plot_curves(metrics, curves_path)
    print(comparison_path)
    print(curves_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
