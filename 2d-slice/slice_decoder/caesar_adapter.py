"""Data inspection and pyCAESAR reference-volume integration.

The module imports only NumPy at load time. PyTorch and pyCAESAR are imported
lazily by :func:`run_caesar_reference`, allowing metadata and latent-layout tests
to run in lightweight environments.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


_ZIP_LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
_ZIP_LOCAL_SIGNATURE = 0x04034B50


@dataclass(frozen=True)
class ArrayMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: str
    fortran_order: bool
    compressed_bytes: int
    uncompressed_bytes: int
    compression: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "fortranOrder": self.fortran_order,
            "compressedBytes": self.compressed_bytes,
            "uncompressedBytes": self.uncompressed_bytes,
            "zipCompression": self.compression,
        }


@dataclass(frozen=True)
class ArchiveMetadata:
    path: Path
    size_bytes: int
    arrays: tuple[ArrayMetadata, ...]
    variable_names: tuple[str, ...] = ()

    @property
    def data(self) -> ArrayMetadata:
        for array in self.arrays:
            if array.name == "data":
                return array
        raise KeyError("archive does not contain data.npy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sizeBytes": self.size_bytes,
            "arrays": [array.to_dict() for array in self.arrays],
            "variableNames": list(self.variable_names),
        }


@dataclass(frozen=True)
class LatentBlock:
    """One CAESAR-V output block with canonical ``[C,D,H,W]`` latent layout."""

    variable_index: int
    section_index: int
    start_index: int
    end_index: int
    latent: np.ndarray
    scale: np.ndarray
    offset: np.ndarray

    def __post_init__(self) -> None:
        latent = np.asarray(self.latent)
        if latent.ndim != 4:
            raise ValueError(f"latent must have shape [C,D,H,W], got {latent.shape}")
        if self.end_index <= self.start_index:
            raise ValueError("end_index must be greater than start_index")
        object.__setattr__(self, "latent", latent.copy())
        object.__setattr__(self, "scale", np.asarray(self.scale).copy())
        object.__setattr__(self, "offset", np.asarray(self.offset).copy())

    @property
    def output_depth(self) -> int:
        return self.end_index - self.start_index

    @property
    def latent_depth(self) -> int:
        return int(self.latent.shape[1])

    @property
    def depth_downsampling(self) -> float:
        return self.output_depth / self.latent_depth

    def to_dict(self) -> dict[str, Any]:
        return {
            "variableIndex": self.variable_index,
            "sectionIndex": self.section_index,
            "startIndex": self.start_index,
            "endIndex": self.end_index,
            "latentShape": list(self.latent.shape),
            "scale": self.scale.reshape(-1).tolist(),
            "offset": self.offset.reshape(-1).tolist(),
        }


@dataclass(frozen=True)
class LatentVolume:
    variable_index: int
    section_index: int
    start_index: int
    end_index: int
    latent: np.ndarray
    depth_downsampling: float

    def __post_init__(self) -> None:
        latent = np.asarray(self.latent)
        if latent.ndim != 4:
            raise ValueError(f"latent must have shape [C,D,H,W], got {latent.shape}")
        object.__setattr__(self, "latent", latent.copy())


@dataclass(frozen=True)
class CaesarReference:
    original: np.ndarray
    reconstructed: np.ndarray
    latent_blocks: tuple[LatentBlock, ...]
    compressed_bytes: float
    source_data: Path
    model_path: Path
    variable_index: int
    section_index: int
    frame_start: int
    frame_end: int
    n_frame: int
    error_bound: float

    def __post_init__(self) -> None:
        original = np.asarray(self.original)
        reconstructed = np.asarray(self.reconstructed)
        if original.ndim != 3:
            raise ValueError(f"original volume must be [D,H,W], got {original.shape}")
        if reconstructed.shape != original.shape:
            raise ValueError(
                "original and reconstructed shapes differ: "
                f"{original.shape} vs {reconstructed.shape}"
            )
        object.__setattr__(self, "original", original.copy())
        object.__setattr__(self, "reconstructed", reconstructed.copy())


def _read_array_header(stream: Any) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    if version == (3, 0):
        return np.lib.format._read_array_header(stream, version)  # type: ignore[attr-defined]
    raise ValueError(f"unsupported NPY format version {version}")


def inspect_npz(path: str | Path) -> ArchiveMetadata:
    """Inspect NPZ array headers without reading the large array payloads."""

    archive_path = Path(path).expanduser().resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    arrays: list[ArrayMetadata] = []
    with zipfile.ZipFile(archive_path) as archive:
        for info in archive.infolist():
            if not info.filename.endswith(".npy"):
                continue
            with archive.open(info) as stream:
                shape, fortran_order, dtype = _read_array_header(stream)
            arrays.append(
                ArrayMetadata(
                    name=Path(info.filename).stem,
                    shape=tuple(int(value) for value in shape),
                    dtype=np.dtype(dtype).str,
                    fortran_order=bool(fortran_order),
                    compressed_bytes=int(info.compress_size),
                    uncompressed_bytes=int(info.file_size),
                    compression=int(info.compress_type),
                )
            )

    variable_names: tuple[str, ...] = ()
    names_entry = next((array for array in arrays if array.name == "variable_name"), None)
    if names_entry is not None and names_entry.uncompressed_bytes <= 1024 * 1024:
        with np.load(archive_path, allow_pickle=False) as archive:
            names = np.asarray(archive["variable_name"]).reshape(-1)
            variable_names = tuple(str(value) for value in names)

    return ArchiveMetadata(
        path=archive_path,
        size_bytes=archive_path.stat().st_size,
        arrays=tuple(arrays),
        variable_names=variable_names,
    )


def _stored_member_data_offset(archive_path: Path, info: zipfile.ZipInfo) -> int:
    if info.compress_type != zipfile.ZIP_STORED:
        raise ValueError(
            f"{info.filename} is compressed and cannot be memory-mapped directly"
        )

    with archive_path.open("rb") as stream:
        stream.seek(info.header_offset)
        header = stream.read(_ZIP_LOCAL_HEADER.size)
        if len(header) != _ZIP_LOCAL_HEADER.size:
            raise ValueError(f"truncated ZIP local header for {info.filename}")
        fields = _ZIP_LOCAL_HEADER.unpack(header)
        if fields[0] != _ZIP_LOCAL_SIGNATURE:
            raise ValueError(f"invalid ZIP local header for {info.filename}")
        filename_length = fields[-2]
        extra_length = fields[-1]
        return info.header_offset + _ZIP_LOCAL_HEADER.size + filename_length + extra_length


def open_stored_npz_array(
    path: str | Path,
    array_name: str = "data",
) -> np.memmap:
    """Memory-map an uncompressed NPY member stored inside an NPZ archive."""

    archive_path = Path(path).expanduser().resolve()
    member_name = array_name if array_name.endswith(".npy") else f"{array_name}.npy"
    with zipfile.ZipFile(archive_path) as archive:
        try:
            info = archive.getinfo(member_name)
        except KeyError as error:
            raise KeyError(f"{archive_path} does not contain {member_name}") from error
        member_offset = _stored_member_data_offset(archive_path, info)

    with archive_path.open("rb") as stream:
        stream.seek(member_offset)
        shape, fortran_order, dtype = _read_array_header(stream)
        payload_offset = stream.tell()

    return np.memmap(
        archive_path,
        dtype=dtype,
        mode="r",
        offset=payload_offset,
        shape=shape,
        order="F" if fortran_order else "C",
    )


def prepare_caesar_subset(
    source_path: str | Path,
    output_path: str | Path,
    *,
    variable_index: int,
    section_index: int,
    frame_start: int,
    frame_end: int,
) -> Path:
    """Create a small float32 NPZ subset without loading the full source archive."""

    metadata = inspect_npz(source_path)
    if len(metadata.data.shape) != 5:
        raise ValueError(
            f"CAESAR data must have shape [V,S,T,H,W], got {metadata.data.shape}"
        )
    variables, sections, frames, _, _ = metadata.data.shape
    if not 0 <= variable_index < variables:
        raise ValueError(f"variable_index must be in [0,{variables - 1}]")
    if not 0 <= section_index < sections:
        raise ValueError(f"section_index must be in [0,{sections - 1}]")
    if frame_start < 0 or frame_end <= frame_start or frame_end > frames:
        raise ValueError(f"frame range [{frame_start},{frame_end}) is invalid for T={frames}")

    mapped = open_stored_npz_array(source_path, "data")
    selected = np.asarray(
        mapped[
            variable_index : variable_index + 1,
            section_index : section_index + 1,
            frame_start:frame_end,
        ],
        dtype=np.float32,
    )

    if metadata.variable_names and variable_index < len(metadata.variable_names):
        names = np.asarray([metadata.variable_names[variable_index]])
    else:
        names = np.asarray([f"variable_{variable_index}"])

    subset_path = Path(output_path).expanduser().resolve()
    subset_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(subset_path, data=selected, variable_name=names)
    return subset_path


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _batch_value(value: Any, index: int, batch_size: int, name: str) -> np.ndarray:
    array = _as_numpy(value)
    if array.ndim == 0:
        return array.copy()
    if array.shape[0] == batch_size:
        return np.asarray(array[index]).copy()
    if batch_size == 1:
        return array.copy()
    raise ValueError(
        f"{name} has leading dimension {array.shape[0]}, expected {batch_size}"
    )


def extract_latent_blocks(
    compressed: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    expected_depth_downsampling: int | None = 4,
) -> tuple[LatentBlock, ...]:
    """Convert pyCAESAR latent batches into canonical per-block objects."""

    if isinstance(compressed, Mapping):
        batches = compressed.get("latent")
        if batches is None:
            raise KeyError("compressed result does not contain 'latent'")
    else:
        batches = compressed

    blocks: list[LatentBlock] = []
    for batch_number, batch in enumerate(batches):
        if "q_latent" not in batch:
            raise KeyError(f"latent batch {batch_number} does not contain 'q_latent'")
        if "index" not in batch:
            raise KeyError(f"latent batch {batch_number} does not contain 'index'")

        index_fields = batch["index"]
        if not isinstance(index_fields, Sequence) or len(index_fields) != 4:
            raise ValueError("block index must contain variable, section, start, and end")
        index_arrays = [_as_numpy(field).reshape(-1) for field in index_fields]
        batch_size = int(index_arrays[0].size)
        if batch_size == 0 or any(array.size != batch_size for array in index_arrays):
            raise ValueError("block index fields must have the same nonzero length")

        q_latent = _as_numpy(batch["q_latent"])
        if q_latent.ndim == 4:
            # CAESAR-V currently exposes [B*D,C,H,W]. Reconstruct its batch and
            # depth dimensions, then use [B,C,D,H,W] internally.
            if q_latent.shape[0] % batch_size != 0:
                raise ValueError(
                    f"q_latent leading dimension {q_latent.shape[0]} is not "
                    f"divisible by batch size {batch_size}"
                )
            latent_depth = q_latent.shape[0] // batch_size
            channels, height, width = q_latent.shape[1:]
            canonical = q_latent.reshape(
                batch_size, latent_depth, channels, height, width
            ).transpose(0, 2, 1, 3, 4)
        elif q_latent.ndim == 5:
            # Accept a future/native [B,C,D,H,W] representation as well.
            if q_latent.shape[0] != batch_size:
                raise ValueError(
                    f"q_latent batch dimension {q_latent.shape[0]} does not "
                    f"match index batch size {batch_size}"
                )
            canonical = q_latent
            latent_depth = q_latent.shape[2]
        else:
            raise ValueError(
                "q_latent must have shape [B*D,C,H,W] or [B,C,D,H,W], "
                f"got {q_latent.shape}"
            )

        for item in range(batch_size):
            variable_index, section_index, start_index, end_index = (
                int(array[item]) for array in index_arrays
            )
            output_depth = end_index - start_index
            if output_depth <= 0:
                raise ValueError(f"invalid block interval [{start_index},{end_index})")
            if (
                expected_depth_downsampling is not None
                and output_depth != latent_depth * expected_depth_downsampling
            ):
                raise ValueError(
                    f"block depth {output_depth} and latent depth {latent_depth} do "
                    f"not match downsampling {expected_depth_downsampling}"
                )

            blocks.append(
                LatentBlock(
                    variable_index=variable_index,
                    section_index=section_index,
                    start_index=start_index,
                    end_index=end_index,
                    latent=canonical[item],
                    scale=_batch_value(batch["scale"], item, batch_size, "scale"),
                    offset=_batch_value(batch["offset"], item, batch_size, "offset"),
                )
            )

    blocks.sort(
        key=lambda block: (
            block.variable_index,
            block.section_index,
            block.start_index,
        )
    )
    return tuple(blocks)


def stack_latent_depth(
    blocks: Sequence[LatentBlock],
    *,
    variable_index: int,
    section_index: int,
) -> LatentVolume:
    """Concatenate contiguous blocks into one canonical latent-depth tensor."""

    selected = sorted(
        (
            block
            for block in blocks
            if block.variable_index == variable_index
            and block.section_index == section_index
        ),
        key=lambda block: block.start_index,
    )
    if not selected:
        raise ValueError("no latent blocks match the requested variable and section")

    reference_shape = (
        selected[0].latent.shape[0],
        selected[0].latent.shape[2],
        selected[0].latent.shape[3],
    )
    downsampling = selected[0].depth_downsampling
    previous_end = selected[0].start_index
    for block in selected:
        shape = (block.latent.shape[0], block.latent.shape[2], block.latent.shape[3])
        if shape != reference_shape:
            raise ValueError(f"latent block shape mismatch: {shape} vs {reference_shape}")
        if block.start_index != previous_end:
            raise ValueError(
                f"latent blocks are not contiguous: expected {previous_end}, "
                f"got {block.start_index}"
            )
        if not np.isclose(block.depth_downsampling, downsampling):
            raise ValueError("latent blocks use inconsistent depth downsampling")
        previous_end = block.end_index

    return LatentVolume(
        variable_index=variable_index,
        section_index=section_index,
        start_index=selected[0].start_index,
        end_index=selected[-1].end_index,
        latent=np.concatenate([block.latent for block in selected], axis=1),
        depth_downsampling=downsampling,
    )


def _scalar_float(value: Any) -> float:
    array = _as_numpy(value)
    if array.size != 1:
        raise ValueError(f"expected scalar value, got shape {array.shape}")
    return float(array.reshape(-1)[0])


def run_caesar_reference(
    data_path: str | Path,
    model_path: str | Path,
    *,
    variable_index: int = 0,
    section_index: int = 0,
    frame_start: int = 0,
    frame_end: int | None = None,
    n_frame: int = 8,
    batch_size: int = 8,
    error_bound: float = 0.01,
    device: str = "cpu",
    gae_device: str | None = None,
) -> CaesarReference:
    """Run CAESAR-V and return original, reconstructed, and latent block data."""

    try:
        from torch.utils.data import DataLoader
        from CAESAR.compressor import CAESAR
        from dataset import ScientificDataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "pyCAESAR execution requires PyTorch and PYTHONPATH=external/pyCAESAR"
        ) from error

    archive = inspect_npz(data_path)
    if len(archive.data.shape) != 5:
        raise ValueError(f"CAESAR data must be [V,S,T,H,W], got {archive.data.shape}")
    _, _, total_frames, height, width = archive.data.shape
    resolved_end = total_frames if frame_end is None else frame_end
    frame_count = resolved_end - frame_start
    if frame_count <= 0:
        raise ValueError("selected frame range is empty")
    if n_frame <= 0 or frame_count < n_frame:
        raise ValueError("n_frame must be positive and no larger than the frame range")

    data_args = {
        "name": "MAGNET 2D slice reference",
        "data_path": str(Path(data_path).expanduser().resolve()),
        "variable_idx": [variable_index],
        "section_range": [section_index, section_index + 1],
        "frame_range": [frame_start, resolved_end],
        "n_frame": n_frame,
        "n_overlap": 0,
        "test_size": (height, width),
        "train": False,
        "inst_norm": True,
    }
    dataset = ScientificDataset(data_args)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    caesar = CAESAR(
        model_path=str(Path(model_path).expanduser().resolve()),
        use_diffusion=False,
        device=device,
        gae_device=gae_device or device,
        n_frame=n_frame,
    )
    compressed, compressed_size = caesar.compress(dataloader, eb=error_bound)
    reconstructed = dataset.recons_data(caesar.decompress(compressed))
    original = dataset.input_data()

    original_array = _as_numpy(original)
    reconstructed_array = _as_numpy(reconstructed)
    if original_array.shape[:2] != (1, 1):
        raise ValueError(f"expected one variable and section, got {original_array.shape}")

    blocks = extract_latent_blocks(compressed)
    return CaesarReference(
        original=original_array[0, 0],
        reconstructed=reconstructed_array[0, 0],
        latent_blocks=blocks,
        compressed_bytes=_scalar_float(compressed_size),
        source_data=Path(data_path).expanduser().resolve(),
        model_path=Path(model_path).expanduser().resolve(),
        variable_index=variable_index,
        section_index=section_index,
        frame_start=frame_start,
        frame_end=resolved_end,
        n_frame=n_frame,
        error_bound=error_bound,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_caesar_reference(
    reference: CaesarReference,
    output_directory: str | Path,
    *,
    axis_semantic: str,
    source_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write reference arrays, canonical latents, and a JSON manifest."""

    if axis_semantic not in {"spatial_z", "time", "unconfirmed"}:
        raise ValueError("axis_semantic must be spatial_z, time, or unconfirmed")

    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "original.npy", reference.original.astype(np.float32))
    np.save(output / "caesar_reference.npy", reference.reconstructed.astype(np.float32))

    latent_volume = stack_latent_depth(
        reference.latent_blocks,
        variable_index=reference.variable_index,
        section_index=reference.section_index,
    )
    np.save(output / "q_latent.npy", latent_volume.latent.astype(np.float32))

    manifest = {
        "formatVersion": 1,
        "sourceData": str(reference.source_data),
        "sourceDataBytes": reference.source_data.stat().st_size,
        "modelPath": str(reference.model_path),
        "modelSha256": _sha256(reference.model_path),
        "variableIndex": reference.variable_index,
        "sectionIndex": reference.section_index,
        "frameStart": reference.frame_start,
        "frameEnd": reference.frame_end,
        "axisSemantic": axis_semantic,
        "nFrame": reference.n_frame,
        "errorBound": reference.error_bound,
        "compressedBytes": reference.compressed_bytes,
        "originalShape": list(reference.original.shape),
        "reconstructedShape": list(reference.reconstructed.shape),
        "latentShape": list(latent_volume.latent.shape),
        "latentDepthDownsampling": latent_volume.depth_downsampling,
        "blocks": [block.to_dict() for block in reference.latent_blocks],
    }
    if source_metadata is not None:
        manifest["sourceMetadata"] = dict(source_metadata)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path
