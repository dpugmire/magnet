"""Training-pair generation from Milestone 1 reference artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .caesar_adapter import open_stored_npz_array
from .geometry import Plane, sample_volume


PlaneOrientation = Literal["axis-aligned", "random", "mixed"]


@dataclass(frozen=True)
class ReferenceBlock:
    """One CAESAR block and its matching raw/reference dense volumes."""

    artifact_directory: Path
    raw_source: Path
    latent: np.ndarray
    raw_volume: np.ndarray
    caesar_volume: np.ndarray
    scale: float
    offset: float
    frame_start: int
    frame_end: int
    axis_semantic: str
    block_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        latent = np.asarray(self.latent)
        raw_volume = np.asarray(self.raw_volume)
        caesar_volume = np.asarray(self.caesar_volume)
        if latent.ndim != 4:
            raise ValueError(f"latent must be [C,D,H,W], got {latent.shape}")
        if raw_volume.ndim != 3:
            raise ValueError(f"raw volume must be [D,H,W], got {raw_volume.shape}")
        if caesar_volume.shape != raw_volume.shape:
            raise ValueError(
                "raw and CAESAR block shapes differ: "
                f"{raw_volume.shape} vs {caesar_volume.shape}"
            )
        if self.frame_end - self.frame_start != raw_volume.shape[0]:
            raise ValueError("block frame interval does not match its dense depth")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("block scale must be positive and finite")
        if not np.isfinite(self.offset):
            raise ValueError("block offset must be finite")
        object.__setattr__(self, "latent", latent)
        object.__setattr__(self, "raw_volume", raw_volume)
        object.__setattr__(self, "caesar_volume", caesar_volume)

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (np.asarray(values) - self.offset) / self.scale

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values) * self.scale + self.offset


def _scalar_metadata(block: dict[str, Any], name: str) -> float:
    values = np.asarray(block[name], dtype=np.float64).reshape(-1)
    if values.size != 1:
        raise ValueError(
            f"the point baseline requires scalar {name}, got shape {values.shape}"
        )
    return float(values[0])


def load_reference_block(
    artifact_directory: str | Path,
    *,
    block_index: int = 0,
) -> ReferenceBlock:
    """Load one block from a reference artifact without copying dense arrays."""

    directory = Path(artifact_directory).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blocks = manifest.get("blocks", [])
    if not blocks:
        raise ValueError("manifest contains no CAESAR latent blocks")
    if not 0 <= block_index < len(blocks):
        raise IndexError(f"block index {block_index} is outside [0,{len(blocks) - 1}]")

    selected = blocks[block_index]
    variable_index = int(manifest["variableIndex"])
    section_index = int(manifest["sectionIndex"])
    matching_blocks = [
        block
        for block in blocks
        if int(block["variableIndex"]) == variable_index
        and int(block["sectionIndex"]) == section_index
    ]
    if selected not in matching_blocks:
        raise ValueError("selected block is not represented in q_latent.npy")

    latent_depth_offset = 0
    for block in matching_blocks:
        if block is selected:
            break
        latent_depth_offset += int(block["latentShape"][1])
    latent_depth = int(selected["latentShape"][1])

    latent_volume = np.load(directory / "q_latent.npy", mmap_mode="r")
    raw_volume = np.load(directory / "original.npy", mmap_mode="r")
    caesar_volume = np.load(directory / "caesar_reference.npy", mmap_mode="r")
    expected_latent_shape = tuple(int(value) for value in selected["latentShape"])
    latent = latent_volume[
        :,
        latent_depth_offset : latent_depth_offset + latent_depth,
        :,
        :,
    ]
    if latent.shape != expected_latent_shape:
        raise ValueError(
            f"latent block shape {latent.shape} does not match {expected_latent_shape}"
        )

    block_start = int(selected["startIndex"])
    block_end = int(selected["endIndex"])
    dense_start = min(int(block["startIndex"]) for block in matching_blocks)
    local_start = block_start - dense_start
    local_end = block_end - dense_start
    if local_start < 0 or local_end > raw_volume.shape[0]:
        raise ValueError("block interval is outside the saved dense volume")

    raw_source = directory / "original.npy"
    raw_block = raw_volume[local_start:local_end]
    source_metadata = manifest.get("sourceMetadata", {})
    source_archive = Path(str(source_metadata.get("archive", ""))).expanduser()
    selection = source_metadata.get("selection", {})
    if source_archive.is_file() and selection:
        source_array = open_stored_npz_array(source_archive, "data")
        source_frame_start = int(selection["frameStart"]) + local_start
        source_frame_end = int(selection["frameStart"]) + local_end
        raw_block = source_array[
            int(selection["variableIndex"]),
            int(selection["sectionIndex"]),
            source_frame_start:source_frame_end,
        ]
        raw_source = source_archive.resolve()

    return ReferenceBlock(
        artifact_directory=directory,
        raw_source=raw_source,
        latent=latent,
        raw_volume=raw_block,
        caesar_volume=caesar_volume[local_start:local_end],
        scale=_scalar_metadata(selected, "scale"),
        offset=_scalar_metadata(selected, "offset"),
        frame_start=block_start,
        frame_end=block_end,
        axis_semantic=str(manifest.get("axisSemantic", "unconfirmed")),
        block_metadata=dict(selected),
    )


def _random_unit_vector(generator: np.random.Generator) -> np.ndarray:
    while True:
        vector = generator.normal(size=3)
        norm = float(np.linalg.norm(vector))
        if norm > 1.0e-12:
            return vector / norm


def _basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    helper = np.zeros(3, dtype=np.float64)
    helper[int(np.argmin(np.abs(normal)))] = 1.0
    axis_u = np.cross(helper, normal)
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    return axis_u, axis_v


def random_contained_plane(
    generator: np.random.Generator,
    *,
    orientation: PlaneOrientation = "mixed",
    maximum_offset: float = 0.5,
) -> Plane:
    """Generate a deterministic random square plane contained in ``[-1,1]^3``."""

    if orientation not in {"axis-aligned", "random", "mixed"}:
        raise ValueError("orientation must be axis-aligned, random, or mixed")
    if not 0.0 <= maximum_offset < 1.0:
        raise ValueError("maximum_offset must be in [0,1)")
    use_axis_aligned = orientation == "axis-aligned" or (
        orientation == "mixed" and bool(generator.integers(0, 2))
    )

    if use_axis_aligned:
        fixed_axis = int(generator.integers(0, 3))
        normal = np.zeros(3, dtype=np.float64)
        normal[fixed_axis] = 1.0
    else:
        normal = _random_unit_vector(generator)
    axis_u, axis_v = _basis_from_normal(normal)

    signed_offset = float(generator.uniform(-maximum_offset, maximum_offset))
    origin = normal * signed_offset
    extent_limits: list[float] = []
    for component in range(3):
        denominator = abs(axis_u[component]) + abs(axis_v[component])
        if denominator > 1.0e-12:
            extent_limits.append((1.0 - abs(origin[component])) / denominator)
    extent = min(1.0, min(extent_limits)) * (1.0 - 1.0e-9)
    if extent <= 0.0:
        raise RuntimeError("failed to construct a contained plane")
    return Plane(
        origin=origin,
        axis_u=axis_u,
        axis_v=axis_v,
        u_bounds=(-extent, extent),
        v_bounds=(-extent, extent),
    )


class RandomPlaneDataset:
    """Deterministic on-demand planes for one latent/reference block.

    The learned target is the full CAESAR reconstruction in normalized block
    units. Raw samples are retained to measure total end-to-end error.
    """

    def __init__(
        self,
        block: ReferenceBlock,
        *,
        sample_count: int,
        height: int,
        width: int,
        seed: int,
        orientation: PlaneOrientation = "mixed",
        maximum_offset: float = 0.5,
    ) -> None:
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        self.block = block
        self.sample_count = int(sample_count)
        self.height = int(height)
        self.width = int(width)
        self.seed = int(seed)
        self.orientation = orientation
        self.maximum_offset = float(maximum_offset)

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if not 0 <= index < self.sample_count:
            raise IndexError(index)
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        plane = random_contained_plane(
            generator,
            orientation=self.orientation,
            maximum_offset=self.maximum_offset,
        )
        points = plane.points(self.height, self.width)
        caesar_values = sample_volume(self.block.caesar_volume, points)
        raw_values = sample_volume(self.block.raw_volume, points)
        return {
            "points": points.astype(np.float32).reshape(-1, 3),
            "target_normalized": self.block.normalize(caesar_values)
            .astype(np.float32)
            .reshape(-1, 1),
            "caesar_values": caesar_values.reshape(-1, 1),
            "raw_values": raw_values.reshape(-1, 1),
            "plane_origin": plane.origin.astype(np.float32),
            "plane_axis_u": plane.axis_u.astype(np.float32),
            "plane_axis_v": plane.axis_v.astype(np.float32),
            "plane_bounds": np.asarray(
                [*plane.u_bounds, *plane.v_bounds], dtype=np.float32
            ),
        }
