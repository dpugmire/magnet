#!/usr/bin/env python3
import os
import argparse
import numpy as np

import vtk
from vtk.util import numpy_support


def extract_array_from_npz(npz_path: str) -> np.ndarray:
    data = np.load(npz_path)
    keys = list(data.keys())
    if not keys:
        raise RuntimeError(f"No arrays found in {npz_path}")

    key = "data" if "data" in keys else ("arr_0" if "arr_0" in keys else keys[0])
    arr = data[key]

    print(f"Loaded: {npz_path}")
    print(f"Keys: {keys}")
    print(f"Using key: {key}")
    print(f"Array shape: {arr.shape}, dtype: {arr.dtype}")

    return arr


def make_image_data_from_zyx(data_zyx: np.ndarray) -> vtk.vtkImageData:
    """
    data_zyx: numpy array shaped (Z, Y, X)
    Writes point scalars named exactly 'scalars'.
    """
    if data_zyx.ndim != 3:
        raise ValueError(f"Expected (Z,Y,X), got {data_zyx.shape}")

    data_zyx = np.ascontiguousarray(data_zyx)
    zdim, ydim, xdim = data_zyx.shape

    vtk_data = numpy_support.numpy_to_vtk(
        num_array=data_zyx.ravel(order="C"),
        deep=True,
        array_type=numpy_support.get_vtk_array_type(data_zyx.dtype),
    )
    vtk_data.SetName("scalars")

    img = vtk.vtkImageData()
    img.SetDimensions(xdim, ydim, zdim)
    img.SetExtent(0, xdim - 1, 0, ydim - 1, 0, zdim - 1)
    img.SetSpacing(1.0, 1.0, 1.0)
    img.SetOrigin(0.0, 0.0, 0.0)
    img.GetPointData().SetScalars(vtk_data)

    return img


def write_vti(img: vtk.vtkImageData, filename: str):
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(img)
    if writer.Write() != 1:
        raise RuntimeError(f"VTI writer failed for {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert an .npz array (1,C,Z,Y,X) into VTI volumes, or optionally a 2D Z-slice."
    )
    parser.add_argument("npz_file", help="Input .npz file (e.g., ../data/Turb_Rot_testset.npz)")
    parser.add_argument(
        "--slice",
        type=int,
        default=None,
        help="If set, write 2D Z-slice at this index (0..Z-1) instead of full 3D volumes.",
    )
    args = parser.parse_args()

    arr = extract_array_from_npz(args.npz_file)

    if arr.ndim != 5 or arr.shape[0] != 1:
        raise ValueError(f"Expected array shape (1,C,Z,Y,X). Got {arr.shape}")

    base = os.path.splitext(os.path.basename(args.npz_file))[0]
    cdim, zdim = arr.shape[1], arr.shape[2]

    if args.slice is None:
        out_dir = "vtk_volumes"
        os.makedirs(out_dir, exist_ok=True)

        for c in range(cdim):
            vol_zyx = arr[0, c, :, :, :]  # (Z,Y,X)
            img = make_image_data_from_zyx(vol_zyx)
            filename = os.path.join(out_dir, f"{base}.{c}.vti")
            write_vti(img, filename)

        print(f"Wrote {cdim} volumes to: {out_dir}/")
        return

    k = args.slice
    if k < 0 or k >= zdim:
        raise ValueError(f"--slice {k} out of range. Valid: 0..{zdim-1}")

    out_dir = "vtk_slices"
    os.makedirs(out_dir, exist_ok=True)

    for c in range(cdim):
        slice_yx = arr[0, c, k, :, :]          # (Y,X)
        slice_zyx = slice_yx[np.newaxis, :, :] # (1,Y,X)
        img = make_image_data_from_zyx(slice_zyx)
        filename = os.path.join(out_dir, f"{base}_z{k}.{c}.vti")
        write_vti(img, filename)

    print(f"Wrote {cdim} 2D slices at z={k} to: {out_dir}/")


if __name__ == "__main__":
    main()
