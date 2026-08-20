#!/usr/bin/env python3
"""Export one ADIOS BP timestep to a legacy VTK STRUCTURED_POINTS file.

This script is intended for batch use on systems like Andes.
It reads numeric variables from an ADIOS2 BP file at a target timestep and
writes a VTK file containing point-data arrays on a uniform grid.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from adios2 import Stream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read one timestep from an ADIOS BP file and export matching grid "
            "variables to a legacy VTK file."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to ADIOS BP file or directory (e.g. run/output.bp).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output VTK file path (typically *.vtk).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=29,
        help="Timestep index to export (default: 29).",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Optional list of variable names to export. Default: all supported numeric variables.",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="Variable names to exclude.",
    )
    parser.add_argument(
        "--origin",
        nargs=3,
        type=float,
        metavar=("OX", "OY", "OZ"),
        default=(0.0, 0.0, 0.0),
        help="Uniform-grid origin (default: 0 0 0).",
    )
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        metavar=("DX", "DY", "DZ"),
        default=(1.0, 1.0, 1.0),
        help="Uniform-grid spacing (default: 1 1 1).",
    )
    parser.add_argument(
        "--transpose-xy",
        action="store_true",
        help="Swap X/Y axes in arrays before writing.",
    )
    parser.add_argument(
        "--force-packed-time-axis",
        action="store_true",
        help=(
            "Force interpretation of the first array axis as time when BP exposes "
            "a single ADIOS step."
        ),
    )
    return parser.parse_args()


def parse_shape(shape_text: str) -> Tuple[int, ...]:
    if not shape_text:
        return ()
    return tuple(int(part.strip()) for part in shape_text.split(",") if part.strip())


def is_numeric_meta_type(type_name: str) -> bool:
    if not type_name:
        return True
    lowered = type_name.lower()
    tokens = ("int", "uint", "float", "double", "long", "short")
    return any(tok in lowered for tok in tokens)


def scan_stream_meta(bp_path: Path) -> Tuple[int, Dict[str, Dict[str, str]]]:
    step_count = 0
    first_vars: Optional[Dict[str, Dict[str, str]]] = None
    with Stream(str(bp_path), "r") as stream:
        for _ in stream.steps():
            step_count += 1
            if first_vars is None:
                first_vars = stream.available_variables()

    if first_vars is None:
        with Stream(str(bp_path), "r") as stream:
            try:
                first_vars = stream.available_variables()
            except Exception:
                first_vars = {}
    return step_count, first_vars or {}


def pick_variable_names(
    meta: Dict[str, Dict[str, str]],
    include: Optional[Sequence[str]],
    exclude: Sequence[str],
) -> List[str]:
    available_names = set(meta.keys())
    exclude_set = set(exclude)

    if include:
        names = [name for name in include if name in available_names]
    else:
        names = sorted(available_names)

    picked: List[str] = []
    for name in names:
        if name in exclude_set:
            continue
        type_name = meta.get(name, {}).get("Type", "")
        if not is_numeric_meta_type(type_name):
            continue
        picked.append(name)
    return picked


def read_stream_step(
    bp_path: Path,
    step_idx: int,
    names: Sequence[str],
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    with Stream(str(bp_path), "r") as stream:
        for _ in stream.steps():
            if int(stream.current_step()) != int(step_idx):
                continue
            for name in names:
                try:
                    out[name] = np.asarray(stream.read(name))
                except Exception:
                    continue
            return out
    return out


def maybe_slice_packed_axis0(
    arrays: Dict[str, np.ndarray],
    step_idx: int,
    force: bool,
) -> Tuple[Dict[str, np.ndarray], int]:
    sliced_count = 0
    out: Dict[str, np.ndarray] = {}
    for name, array in arrays.items():
        if array.ndim >= 2 and array.shape[0] > step_idx and (force or step_idx > 0):
            out[name] = np.asarray(array[step_idx])
            sliced_count += 1
        else:
            out[name] = array
    return out, sliced_count


def to_scalar_or_vector(
    array: np.ndarray,
    transpose_xy: bool,
) -> Tuple[str, np.ndarray]:
    data = np.asarray(array)
    data = np.squeeze(data)

    if data.ndim == 2:
        if transpose_xy:
            data = data.T
        return "scalar", data[np.newaxis, :, :]

    if data.ndim == 3:
        if data.shape[-1] == 3:
            if transpose_xy:
                data = np.swapaxes(data, 0, 1)
            return "vector", data[np.newaxis, :, :, :]
        if transpose_xy:
            data = np.swapaxes(data, -1, -2)
        return "scalar", data

    if data.ndim == 4 and data.shape[-1] == 3:
        if transpose_xy:
            data = np.swapaxes(data, -3, -2)
        return "vector", data

    raise ValueError(f"Unsupported array rank/shape for VTK export: {tuple(data.shape)}")


def sanitize_vtk_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not safe:
        safe = "var"
    if safe[0].isdigit():
        safe = f"v_{safe}"
    return safe


def write_legacy_structured_points(
    out_path: Path,
    spatial_shape_zyx: Tuple[int, int, int],
    scalar_fields: Sequence[Tuple[str, np.ndarray]],
    vector_fields: Sequence[Tuple[str, np.ndarray]],
    origin_xyz: Sequence[float],
    spacing_xyz: Sequence[float],
) -> None:
    nz, ny, nx = spatial_shape_zyx
    point_count = int(nx * ny * nz)

    with out_path.open("wb") as handle:
        def writeln(line: str) -> None:
            handle.write((line + "\n").encode("ascii"))

        writeln("# vtk DataFile Version 3.0")
        writeln("ADIOS BP timestep export")
        writeln("BINARY")
        writeln("DATASET STRUCTURED_POINTS")
        writeln(f"DIMENSIONS {nx} {ny} {nz}")
        writeln(f"ORIGIN {origin_xyz[0]} {origin_xyz[1]} {origin_xyz[2]}")
        writeln(f"SPACING {spacing_xyz[0]} {spacing_xyz[1]} {spacing_xyz[2]}")
        writeln(f"POINT_DATA {point_count}")

        for name, data_zyx in scalar_fields:
            vtk_name = sanitize_vtk_name(name)
            writeln(f"SCALARS {vtk_name} float 1")
            writeln("LOOKUP_TABLE default")
            payload = np.asarray(data_zyx, dtype=">f4").ravel(order="C")
            handle.write(payload.tobytes())
            handle.write(b"\n")

        for name, data_zyxc in vector_fields:
            vtk_name = sanitize_vtk_name(name)
            writeln(f"VECTORS {vtk_name} float")
            payload = np.asarray(data_zyxc, dtype=">f4").reshape((-1, 3), order="C")
            handle.write(payload.tobytes())
            handle.write(b"\n")


def resolve_bp_path(input_arg: str) -> Path:
    input_path = Path(input_arg).expanduser().resolve()
    if input_path.is_dir():
        return input_path
    if input_path.exists():
        return input_path
    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main() -> None:
    args = parse_args()
    if args.step < 0:
        raise ValueError("--step must be >= 0")

    bp_path = resolve_bp_path(args.input)
    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stream_step_count, meta = scan_stream_meta(bp_path)
    if not meta:
        raise RuntimeError(f"No variables found in ADIOS file: {bp_path}")

    names = pick_variable_names(
        meta=meta,
        include=args.variables,
        exclude=args.exclude,
    )
    if not names:
        raise RuntimeError("No candidate numeric variables selected for export.")

    if stream_step_count > args.step:
        arrays = read_stream_step(bp_path, args.step, names)
        mode = "stream_step"
    elif stream_step_count <= 1:
        arrays = read_stream_step(bp_path, 0, names)
        arrays, sliced_count = maybe_slice_packed_axis0(
            arrays=arrays,
            step_idx=args.step,
            force=args.force_packed_time_axis,
        )
        if args.step > 0 and sliced_count == 0:
            raise IndexError(
                "Requested step is out of range for ADIOS stream steps, and no variable "
                "had a packed leading axis large enough to slice."
            )
        mode = "packed_axis0" if sliced_count > 0 else "single_step"
    else:
        raise IndexError(
            f"Requested step {args.step} out of range for file with {stream_step_count} ADIOS steps."
        )

    scalar_fields: List[Tuple[str, np.ndarray]] = []
    vector_fields: List[Tuple[str, np.ndarray]] = []
    spatial_shape: Optional[Tuple[int, int, int]] = None
    skipped: List[str] = []

    for name in names:
        if name not in arrays:
            skipped.append(f"{name} (read failed)")
            continue

        array = arrays[name]
        if not np.issubdtype(array.dtype, np.number):
            skipped.append(f"{name} (non-numeric dtype {array.dtype})")
            continue

        try:
            field_kind, field_data = to_scalar_or_vector(
                array=array,
                transpose_xy=bool(args.transpose_xy),
            )
        except ValueError:
            skipped.append(f"{name} (unsupported shape {tuple(np.asarray(array).shape)})")
            continue

        if field_kind == "scalar":
            field_shape = tuple(int(v) for v in field_data.shape)
        else:
            field_shape = tuple(int(v) for v in field_data.shape[:3])

        if spatial_shape is None:
            spatial_shape = field_shape
        if field_shape != spatial_shape:
            skipped.append(
                f"{name} (shape mismatch {field_shape} != {spatial_shape})"
            )
            continue

        if field_kind == "scalar":
            scalar_fields.append((name, np.asarray(field_data, dtype=np.float32)))
        else:
            vector_fields.append((name, np.asarray(field_data, dtype=np.float32)))

    if spatial_shape is None or (not scalar_fields and not vector_fields):
        raise RuntimeError("No variables with compatible grid shapes were available for VTK export.")

    write_legacy_structured_points(
        out_path=out_path,
        spatial_shape_zyx=spatial_shape,
        scalar_fields=scalar_fields,
        vector_fields=vector_fields,
        origin_xyz=args.origin,
        spacing_xyz=args.spacing,
    )

    print(f"Input: {bp_path}")
    print(f"Output: {out_path}")
    print(f"Requested step: {args.step}")
    print(f"Read mode: {mode}")
    print(f"ADIOS stream steps: {stream_step_count}")
    print(f"Grid shape (z,y,x): {spatial_shape}")
    print(f"Exported scalar fields ({len(scalar_fields)}): {[name for name, _ in scalar_fields]}")
    print(f"Exported vector fields ({len(vector_fields)}): {[name for name, _ in vector_fields]}")
    if skipped:
        print(f"Skipped ({len(skipped)}):")
        for item in skipped:
            print(f"  - {item}")


if __name__ == "__main__":
    main()
