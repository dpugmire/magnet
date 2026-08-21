#!/usr/bin/env python3
"""Create trust-aware visualization and selective-refinement figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


_TOLERANCES = (0.30, 0.40, 0.50)
_FALLBACK_CMAP = ListedColormap(["#f28e2b"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hybrid-example", type=Path, required=True)
    parser.add_argument("--hybrid-metrics", type=Path, required=True)
    parser.add_argument("--full-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def _limits(fields: Sequence[np.ndarray]) -> tuple[float, float]:
    return (
        min(float(np.min(values)) for values in fields),
        max(float(np.max(values)) for values in fields),
    )


def _error_limit(errors: Sequence[np.ndarray]) -> float:
    return float(
        np.percentile(
            np.concatenate([np.abs(error).reshape(-1) for error in errors]),
            99.0,
        )
    )


def _overlay_mask(
    axis: plt.Axes,
    scalar: np.ndarray,
    mask: np.ndarray,
    *,
    value_min: float,
    value_max: float,
) -> None:
    axis.imshow(
        scalar,
        origin="lower",
        cmap="gray",
        vmin=value_min,
        vmax=value_max,
    )
    axis.imshow(
        np.ma.masked_where(~mask, mask),
        origin="lower",
        cmap=_FALLBACK_CMAP,
        vmin=0.0,
        vmax=1.0,
        alpha=0.72,
    )


def _plot_trust_aware_slice(
    example: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    prediction = example["plane"]
    uncertainty = example["predicted_95_half_width"]
    tolerance = 0.40
    fallback_mask = uncertainty > tolerance
    value_min, value_max = _limits((example["base"], prediction))
    uncertainty_limit = float(np.percentile(uncertainty, 99.0))

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    scalar_image = axes[0].imshow(
        prediction,
        origin="lower",
        cmap="viridis",
        vmin=value_min,
        vmax=value_max,
    )
    axes[0].set_title("Fast approximate scalar slice")
    uncertainty_image = axes[1].imshow(
        uncertainty,
        origin="lower",
        cmap="magma",
        vmin=0.0,
        vmax=uncertainty_limit,
    )
    axes[1].set_title("Predicted 95% error half-width")
    _overlay_mask(
        axes[2],
        prediction,
        fallback_mask,
        value_min=value_min,
        value_max=value_max,
    )
    axes[2].set_title(
        f"Trust-aware view\norange: ±error > {tolerance:.2f}; "
        f"{100.0 * np.mean(fallback_mask):.1f}%"
    )
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(scalar_image, ax=axes[0], shrink=0.82, label="scalar value")
    figure.colorbar(
        uncertainty_image,
        ax=axes[1],
        shrink=0.82,
        label="95% error half-width",
    )
    figure.suptitle(
        "A fast visualization can expose where its values are less trustworthy",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_tolerance_refinement(
    example: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    base = example["base"]
    prediction = example["plane"]
    uncertainty = example["predicted_95_half_width"]
    hybrids: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    for tolerance in _TOLERANCES:
        mask = uncertainty > tolerance
        hybrid = prediction.copy()
        hybrid[mask] = base[mask]
        masks.append(mask)
        hybrids.append(hybrid)
        errors.append(hybrid - base)
    value_min, value_max = _limits((base, prediction, *hybrids))
    error_limit = _error_limit((prediction - base, *errors))

    figure, axes = plt.subplots(
        len(_TOLERANCES),
        3,
        figsize=(10, 10),
        constrained_layout=True,
    )
    scalar_image = None
    error_image = None
    for row, (tolerance, mask, hybrid, error) in enumerate(
        zip(_TOLERANCES, masks, hybrids, errors)
    ):
        _overlay_mask(
            axes[row, 0],
            prediction,
            mask,
            value_min=value_min,
            value_max=value_max,
        )
        scalar_image = axes[row, 1].imshow(
            hybrid,
            origin="lower",
            cmap="viridis",
            vmin=value_min,
            vmax=value_max,
        )
        error_image = axes[row, 2].imshow(
            error,
            origin="lower",
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit,
        )
        axes[row, 0].set_ylabel(
            f"Tolerance ±{tolerance:.2f}\n"
            f"{100.0 * np.mean(mask):.1f}% exact\n"
            f"RMSE {_rmse(error):.4f}"
        )
        for column in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    axes[0, 0].set_title("Requested CAESAR fallback")
    axes[0, 1].set_title("Hybrid scalar slice")
    axes[0, 2].set_title("Remaining signed error")
    if scalar_image is None or error_image is None:
        raise RuntimeError("tolerance-refinement figure has no images")
    figure.colorbar(
        scalar_image,
        ax=axes[:, 1],
        shrink=0.82,
        label="scalar value",
    )
    figure.colorbar(
        error_image,
        ax=axes[:, 2],
        shrink=0.82,
        label="hybrid - CAESAR base",
    )
    figure.suptitle(
        "Scientific tolerance controls where exact CAESAR values are requested",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_accuracy_computation(
    hybrid_metrics: dict[str, object],
    full_metrics: dict[str, object],
    output_path: Path,
) -> None:
    simulation = hybrid_metrics["simulation"]["policies"]
    styles = {
        "uncertaintyPixels": ("Uncertainty pixels", "o", "tab:blue"),
        "uncertaintyTiles": ("Uncertainty tiles", "s", "tab:orange"),
        "randomPixels": ("Random pixels", "^", "tab:gray"),
        "errorOraclePixels": ("Actual-error oracle", "D", "tab:green"),
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for policy, rows in simulation.items():
        label, marker, color = styles[policy]
        axes[0].plot(
            [100.0 * row["selectedPixelFraction"] for row in rows],
            [row["rmseVsCaesarBase"] for row in rows],
            marker=marker,
            color=color,
            label=label,
        )
    axes[0].set_xlabel("Pixels replaced by exact CAESAR values (%)")
    axes[0].set_ylabel("RMSE vs CAESAR base")
    axes[0].set_title("Uncertainty selects useful corrections")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    exact = hybrid_metrics["exactOnDemandFrameFallback"]["policies"]
    full_inference = full_metrics["test"]["inference"]
    operating_points = (
        (
            "Fast approximate",
            hybrid_metrics["directSliceLatency"]["meanMillisecondsPerSlice"],
            simulation["uncertaintyPixels"][0]["rmseVsCaesarBase"],
            "o",
            "tab:blue",
        ),
        (
            "Tile hybrid",
            exact["uncertaintyTiles"]["combinedMeanMillisecondsPerSlice"],
            exact["uncertaintyTiles"]["rmseVsCaesarBase"],
            "s",
            "tab:orange",
        ),
        (
            "Pixel hybrid",
            exact["uncertaintyPixels"]["combinedMeanMillisecondsPerSlice"],
            exact["uncertaintyPixels"]["rmseVsCaesarBase"],
            "^",
            "tab:purple",
        ),
        (
            "Full CAESAR",
            full_inference["estimatedColdMillisecondsPerSlice"],
            full_metrics["test"]["aggregate"]["fullReconstructionVsBase"][
                "rmse"
            ],
            "D",
            "tab:green",
        ),
    )
    annotation_styles = {
        "Fast approximate": ((6, 9), "left", "bottom"),
        "Tile hybrid": ((-6, 11), "right", "bottom"),
        "Pixel hybrid": ((6, 2), "left", "bottom"),
        "Full CAESAR": ((6, 6), "left", "bottom"),
    }
    for label, latency, rmse, marker, color in operating_points:
        axes[1].scatter(latency, rmse, marker=marker, color=color, s=70)
        offset, horizontal_alignment, vertical_alignment = annotation_styles[label]
        axes[1].annotate(
            label,
            (latency, rmse),
            xytext=offset,
            textcoords="offset points",
            ha=horizontal_alignment,
            va=vertical_alignment,
        )
    axes[1].set_xlabel("Measured cold slice latency (ms)")
    axes[1].set_ylabel("RMSE vs CAESAR base")
    axes[1].set_title("Accuracy can be purchased on demand")
    axes[1].grid(alpha=0.25)
    axes[1].set_ylim(-0.008, 0.158)
    axes[1].text(
        0.02,
        0.04,
        "Mac measurements; mixed CPU/MPS paths",
        transform=axes[1].transAxes,
        fontsize=9,
    )
    figure.suptitle(
        "Trust-aware visualization separates fast approximation from exact fallback",
        fontsize=15,
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    example_path = args.hybrid_example.expanduser().resolve()
    hybrid_metrics_path = args.hybrid_metrics.expanduser().resolve()
    full_metrics_path = args.full_metrics.expanduser().resolve()
    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    with np.load(example_path) as archive:
        example = {name: np.asarray(archive[name]) for name in archive.files}
    hybrid_metrics = json.loads(hybrid_metrics_path.read_text(encoding="utf-8"))
    full_metrics = json.loads(full_metrics_path.read_text(encoding="utf-8"))

    trust_path = output_directory / "trust-aware-slice.png"
    tolerance_path = output_directory / "tolerance-driven-refinement.png"
    tradeoff_path = output_directory / "accuracy-computation-options.png"
    _plot_trust_aware_slice(example, trust_path)
    _plot_tolerance_refinement(example, tolerance_path)
    _plot_accuracy_computation(hybrid_metrics, full_metrics, tradeoff_path)
    for path in (trust_path, tolerance_path, tradeoff_path):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
