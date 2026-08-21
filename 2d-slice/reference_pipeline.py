#!/usr/bin/env python3
"""Inspect CAESAR input data and generate Milestone 1 reference artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from slice_decoder.caesar_adapter import (
    inspect_npz,
    prepare_caesar_subset,
    run_caesar_reference,
    save_caesar_reference,
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CAESAR 2D-slice Milestone 1 reference pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Read NPZ metadata without loading the volume payload"
    )
    inspect_parser.add_argument("--data", type=Path, required=True)

    reference_parser = subparsers.add_parser(
        "reference", help="Run CAESAR-V and save reconstruction and latent arrays"
    )
    reference_parser.add_argument("--data", type=Path, required=True)
    reference_parser.add_argument("--model", type=Path, required=True)
    reference_parser.add_argument("--output-dir", type=Path, required=True)
    reference_parser.add_argument("--variable-index", type=int, default=0)
    reference_parser.add_argument("--section-index", type=int, default=0)
    reference_parser.add_argument("--frame-start", type=int, default=0)
    reference_parser.add_argument(
        "--frame-end",
        type=int,
        default=None,
        help="Exclusive index; defaults to the end of the source frame axis",
    )
    reference_parser.add_argument("--n-frame", type=_positive_integer, default=8)
    reference_parser.add_argument("--batch-size", type=_positive_integer, default=8)
    reference_parser.add_argument("--error-bound", type=float, default=0.01)
    reference_parser.add_argument("--device", default="cpu")
    reference_parser.add_argument("--gae-device", default=None)
    reference_parser.add_argument(
        "--axis-semantic",
        choices=("spatial_z", "time", "unconfirmed"),
        default="spatial_z",
        help="Meaning assigned to the source array's third dimension",
    )
    return parser


def _run_inspect(data_path: Path) -> int:
    print(json.dumps(inspect_npz(data_path).to_dict(), indent=2, sort_keys=True))
    return 0


def _run_reference(args: argparse.Namespace) -> int:
    metadata = inspect_npz(args.data)
    shape = metadata.data.shape
    if len(shape) != 5:
        raise ValueError(f"expected source data shape [V,S,D,H,W], got {shape}")

    frame_end = shape[2] if args.frame_end is None else args.frame_end
    if not 0 <= args.variable_index < shape[0]:
        raise ValueError(f"variable index is outside [0,{shape[0] - 1}]")
    if not 0 <= args.section_index < shape[1]:
        raise ValueError(f"section index is outside [0,{shape[1] - 1}]")
    if args.frame_start < 0 or frame_end <= args.frame_start or frame_end > shape[2]:
        raise ValueError(
            f"frame range [{args.frame_start},{frame_end}) is invalid for D={shape[2]}"
        )
    if (frame_end - args.frame_start) % args.n_frame != 0:
        raise ValueError(
            "selected frame count must be divisible by n-frame; this avoids "
            "ambiguous reflected padding in the first reference artifact"
        )
    if args.error_bound <= 0.0:
        raise ValueError("error bound must be positive")

    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    subset_path = prepare_caesar_subset(
        args.data,
        output_directory / "input_subset.npz",
        variable_index=args.variable_index,
        section_index=args.section_index,
        frame_start=args.frame_start,
        frame_end=frame_end,
    )

    # The subset deliberately remaps the selected variable, section, and frame
    # interval to zero-based local coordinates. The original coordinates remain
    # in sourceMetadata below.
    reference = run_caesar_reference(
        subset_path,
        args.model,
        variable_index=0,
        section_index=0,
        frame_start=0,
        frame_end=frame_end - args.frame_start,
        n_frame=args.n_frame,
        batch_size=args.batch_size,
        error_bound=args.error_bound,
        device=args.device,
        gae_device=args.gae_device,
    )
    manifest_path = save_caesar_reference(
        reference,
        output_directory,
        axis_semantic=args.axis_semantic,
        source_metadata={
            "archive": str(metadata.path),
            "archiveBytes": metadata.size_bytes,
            "dataShape": list(shape),
            "variableNames": list(metadata.variable_names),
            "selection": {
                "variableIndex": args.variable_index,
                "sectionIndex": args.section_index,
                "frameStart": args.frame_start,
                "frameEnd": frame_end,
            },
        },
    )
    print(manifest_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "inspect":
        return _run_inspect(args.data)
    if args.command == "reference":
        return _run_reference(args)
    raise AssertionError(f"unhandled command {args.command!r}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
