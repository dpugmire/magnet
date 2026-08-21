#!/usr/bin/env python3
"""Plot held-out examples from the contextual plane-decoder ablation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--late", type=Path, required=True)
    parser.add_argument("--caesar-initialized", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path.expanduser().resolve()) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    full = _load(args.full)
    late = _load(args.late)
    initialized = _load(args.caesar_initialized)
    fusion = _load(args.fusion)
    for candidate in (late, initialized, fusion):
        np.testing.assert_allclose(candidate["points"], full["points"])

    fields = (
        ("Raw data", full["raw"]),
        ("CAESAR base", full["base"]),
        ("Full reconstruction", full["full"]),
        ("Late-only head", late["plane"]),
        ("CAESAR-initialized", initialized["plane"]),
        ("Early + late fusion", fusion["plane"]),
    )
    reference = full["base"]
    errors = tuple(values - reference for _, values in fields)
    value_min = min(float(np.min(values)) for _, values in fields)
    value_max = max(float(np.max(values)) for _, values in fields)
    error_limit = float(
        np.percentile(
            np.concatenate([np.abs(error).reshape(-1) for error in errors]),
            99.0,
        )
    )

    figure, axes = plt.subplots(
        2, len(fields), figsize=(18, 6), constrained_layout=True
    )
    value_image = None
    error_image = None
    for column, ((name, values), error) in enumerate(zip(fields, errors)):
        value_image = axes[0, column].imshow(
            values,
            origin="lower",
            cmap="viridis",
            vmin=value_min,
            vmax=value_max,
        )
        rmse = float(np.sqrt(np.mean(error * error)))
        axes[0, column].set_title(name)
        axes[1, column].set_title(f"vs base RMSE {rmse:.4f}")
        error_image = axes[1, column].imshow(
            error,
            origin="lower",
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit,
        )
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    if value_image is None or error_image is None:
        raise RuntimeError("no comparison fields were plotted")
    figure.colorbar(value_image, ax=axes[0], shrink=0.85, label="scalar value")
    figure.colorbar(
        error_image,
        ax=axes[1],
        shrink=0.85,
        label="candidate - CAESAR base",
    )
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
