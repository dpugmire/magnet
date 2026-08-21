#!/usr/bin/env python3
"""Plot scalar, signed-error, and calibrated-uncertainty slice comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single", type=Path, required=True)
    parser.add_argument("--slab", type=Path, required=True)
    parser.add_argument("--early", type=Path, required=True)
    parser.add_argument("--late", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve()) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values) ** 2)))


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    single = _load(args.single)
    slab = _load(args.slab)
    early = _load(args.early)
    late = _load(args.late)
    learned = (single, slab, early, late)
    for candidate in learned[1:]:
        np.testing.assert_allclose(candidate["points"], single["points"])
        np.testing.assert_allclose(candidate["base"], single["base"])

    reference = single["base"]
    fields = (
        ("Raw data", single["raw"]),
        ("CAESAR base", reference),
        ("Single-plane", single["plane"]),
        ("Five-plane slab", slab["plane"]),
        ("Early context", early["plane"]),
        ("Late context", late["plane"]),
    )
    signed_errors = tuple(values - reference for _, values in fields)
    uncertainty_fields: tuple[np.ndarray | None, ...] = (
        None,
        np.zeros_like(reference),
        single["predicted_95_half_width"],
        slab["predicted_95_half_width"],
        early["predicted_95_half_width"],
        late["predicted_95_half_width"],
    )
    value_min = min(float(np.min(values)) for _, values in fields)
    value_max = max(float(np.max(values)) for _, values in fields)
    error_limit = float(
        np.percentile(
            np.concatenate(
                [np.abs(error).reshape(-1) for error in signed_errors]
            ),
            99.0,
        )
    )
    uncertainty_limit = float(
        np.percentile(
            np.concatenate(
                [
                    values.reshape(-1)
                    for values in uncertainty_fields[2:]
                    if values is not None
                ]
            ),
            99.0,
        )
    )

    figure, axes = plt.subplots(
        3,
        len(fields),
        figsize=(19, 9),
        constrained_layout=True,
    )
    value_image = None
    error_image = None
    uncertainty_image = None
    for column, ((name, values), error, uncertainty_values) in enumerate(
        zip(fields, signed_errors, uncertainty_fields)
    ):
        value_image = axes[0, column].imshow(
            values,
            origin="lower",
            cmap="viridis",
            vmin=value_min,
            vmax=value_max,
        )
        axes[0, column].set_title(name)
        error_image = axes[1, column].imshow(
            error,
            origin="lower",
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit,
        )
        axes[1, column].set_title(f"RMSE vs base {_rmse(error):.4f}")
        axes[2, column].imshow(
            values,
            origin="lower",
            cmap="gray",
            vmin=value_min,
            vmax=value_max,
        )
        if column == 0:
            axes[2, column].text(
                0.5,
                0.5,
                "N/A\n(no learned decoder)",
                transform=axes[2, column].transAxes,
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none"},
            )
            axes[2, column].set_title("No predicted interval")
        elif column == 1:
            uncertainty_image = axes[2, column].imshow(
                np.zeros_like(reference),
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=uncertainty_limit,
                alpha=0.72,
            )
            axes[2, column].set_title("Zero by definition")
        else:
            if uncertainty_values is None:
                raise RuntimeError("learned method has no uncertainty field")
            uncertainty_image = axes[2, column].imshow(
                uncertainty_values,
                origin="lower",
                cmap="magma",
                vmin=0.0,
                vmax=uncertainty_limit,
                alpha=0.72,
            )
            coverage = float(
                np.mean(np.abs(error) <= uncertainty_values)
            )
            axes[2, column].set_title(
                f"mean {np.mean(uncertainty_values):.3f}; "
                f"covered {100.0 * coverage:.1f}%"
            )

        for row in range(3):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])

    axes[0, 0].set_ylabel("Scalar slice")
    axes[1, 0].set_ylabel("Signed error")
    axes[2, 0].set_ylabel("Predicted 95%\nerror half-width")
    if value_image is None or error_image is None or uncertainty_image is None:
        raise RuntimeError("comparison images were not created")
    figure.colorbar(value_image, ax=axes[0], shrink=0.82, label="scalar value")
    figure.colorbar(
        error_image,
        ax=axes[1],
        shrink=0.82,
        label="prediction - CAESAR base",
    )
    figure.colorbar(
        uncertainty_image,
        ax=axes[2],
        shrink=0.82,
        label="Predicted 95% error half-width relative to CAESAR base",
    )
    figure.suptitle(
        "Direct plane decoding with calibrated per-pixel uncertainty",
        fontsize=15,
    )
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
