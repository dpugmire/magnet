"""Training-pair generation from Milestone 1 reference artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np

from .caesar_adapter import open_stored_npz_array
from .geometry import Plane, sample_volume


PlaneOrientation = Literal["axis-aligned", "random", "mixed"]


@dataclass(frozen=True)
class ReferenceBlock:
    """One CAESAR block and its matching raw/reference dense volumes."""

    artifact_directory: Path
    artifact_block_index: int
    raw_source: Path
    latent: np.ndarray
    raw_volume: np.ndarray
    caesar_volume: np.ndarray
    scale: float
    offset: float
    frame_start: int
    frame_end: int
    source_variable_index: int
    source_section_index: int
    source_frame_start: int
    source_frame_end: int
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
        if self.source_frame_end - self.source_frame_start != raw_volume.shape[0]:
            raise ValueError("source frame interval does not match its dense depth")
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


def load_reference_blocks(
    artifact_directory: str | Path,
) -> tuple[ReferenceBlock, ...]:
    """Load all represented blocks while sharing the artifact memory maps."""

    directory = Path(artifact_directory).expanduser().resolve()
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_blocks = manifest.get("blocks", [])
    if not manifest_blocks:
        raise ValueError("manifest contains no CAESAR latent blocks")

    variable_index = int(manifest["variableIndex"])
    section_index = int(manifest["sectionIndex"])
    matching_blocks = sorted(
        (
            (index, block)
            for index, block in enumerate(manifest_blocks)
            if int(block["variableIndex"]) == variable_index
            and int(block["sectionIndex"]) == section_index
        ),
        key=lambda entry: int(entry[1]["startIndex"]),
    )
    if not matching_blocks:
        raise ValueError("manifest blocks are not represented in q_latent.npy")

    latent_volume = np.load(directory / "q_latent.npy", mmap_mode="r")
    saved_raw_volume = np.load(directory / "original.npy", mmap_mode="r")
    caesar_volume = np.load(directory / "caesar_reference.npy", mmap_mode="r")
    if saved_raw_volume.shape != caesar_volume.shape:
        raise ValueError("saved raw and CAESAR dense volume shapes differ")

    source_metadata = manifest.get("sourceMetadata", {})
    selection = source_metadata.get("selection", {})
    source_variable_index = int(selection.get("variableIndex", variable_index))
    source_section_index = int(selection.get("sectionIndex", section_index))
    selection_frame_start = int(selection.get("frameStart", manifest["frameStart"]))
    source_archive_text = str(source_metadata.get("archive", ""))
    source_archive = Path(source_archive_text).expanduser()
    if source_archive_text and not source_archive.is_absolute():
        source_archive = directory / source_archive
    raw_source = (directory / "original.npy").resolve()
    source_array: np.ndarray | None = None
    if source_archive.is_file():
        try:
            source_array = open_stored_npz_array(source_archive, "data")
            raw_source = source_archive.resolve()
        except ValueError:
            source_array = None

    dense_start = min(int(block["startIndex"]) for _, block in matching_blocks)
    latent_depth_offset = 0
    loaded: list[ReferenceBlock] = []
    for artifact_block_index, block in matching_blocks:
        expected_latent_shape = tuple(int(value) for value in block["latentShape"])
        latent_depth = expected_latent_shape[1]
        latent = latent_volume[
            :,
            latent_depth_offset : latent_depth_offset + latent_depth,
            :,
            :,
        ]
        latent_depth_offset += latent_depth
        if latent.shape != expected_latent_shape:
            raise ValueError(
                f"latent block shape {latent.shape} does not match "
                f"{expected_latent_shape}"
            )

        block_start = int(block["startIndex"])
        block_end = int(block["endIndex"])
        local_start = block_start - dense_start
        local_end = block_end - dense_start
        if local_start < 0 or local_end > saved_raw_volume.shape[0]:
            raise ValueError("block interval is outside the saved dense volume")

        source_frame_start = selection_frame_start + local_start
        source_frame_end = selection_frame_start + local_end
        raw_block = saved_raw_volume[local_start:local_end]
        if source_array is not None:
            raw_block = source_array[
                source_variable_index,
                source_section_index,
                source_frame_start:source_frame_end,
            ]

        loaded.append(
            ReferenceBlock(
                artifact_directory=directory,
                artifact_block_index=artifact_block_index,
                raw_source=raw_source,
                latent=latent,
                raw_volume=raw_block,
                caesar_volume=caesar_volume[local_start:local_end],
                scale=_scalar_metadata(block, "scale"),
                offset=_scalar_metadata(block, "offset"),
                frame_start=block_start,
                frame_end=block_end,
                source_variable_index=source_variable_index,
                source_section_index=source_section_index,
                source_frame_start=source_frame_start,
                source_frame_end=source_frame_end,
                axis_semantic=str(manifest.get("axisSemantic", "unconfirmed")),
                block_metadata=dict(block),
            )
        )

    if latent_depth_offset != latent_volume.shape[1]:
        raise ValueError(
            f"manifest describes {latent_depth_offset} latent planes, but "
            f"q_latent.npy contains {latent_volume.shape[1]}"
        )
    return tuple(loaded)


def load_reference_block(
    artifact_directory: str | Path,
    *,
    block_index: int = 0,
) -> ReferenceBlock:
    """Load one block from a reference artifact."""

    blocks = load_reference_blocks(artifact_directory)
    if not 0 <= block_index < len(blocks):
        raise IndexError(f"block index {block_index} is outside [0,{len(blocks) - 1}]")
    return blocks[block_index]


def discover_reference_artifacts(
    *,
    artifact_directories: Iterable[str | Path] = (),
    artifact_roots: Iterable[str | Path] = (),
) -> tuple[Path, ...]:
    """Resolve explicit artifacts and recursively scan collection roots."""

    discovered: set[Path] = set()
    for path in artifact_directories:
        directory = Path(path).expanduser().resolve()
        if not (directory / "manifest.json").is_file():
            raise FileNotFoundError(directory / "manifest.json")
        discovered.add(directory)
    for path in artifact_roots:
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        for manifest_path in root.rglob("manifest.json"):
            discovered.add(manifest_path.parent.resolve())
    if not discovered:
        raise ValueError("no reference artifact directories were found")
    return tuple(sorted(discovered))


def load_reference_collection(
    artifact_directories: Sequence[str | Path],
) -> tuple[ReferenceBlock, ...]:
    """Load all blocks from a set of reference artifact directories."""

    loaded: list[ReferenceBlock] = []
    for directory in artifact_directories:
        loaded.extend(load_reference_blocks(directory))
    if not loaded:
        raise ValueError("reference collection contains no latent blocks")
    reference_shape = loaded[0].latent.shape
    identities: set[tuple[Path, int, int, int, int]] = set()
    for block in loaded[1:]:
        if block.latent.shape != reference_shape:
            raise ValueError(
                "point-decoder batches require one latent shape, got "
                f"{reference_shape} and {block.latent.shape}"
            )
    for block in loaded:
        identity = (
            block.raw_source,
            block.source_variable_index,
            block.source_section_index,
            block.source_frame_start,
            block.source_frame_end,
        )
        if identity in identities:
            raise ValueError(f"duplicate source block in reference collection: {identity}")
        identities.add(identity)
    return tuple(loaded)


def parse_index_specification(specification: str) -> frozenset[int]:
    """Parse comma-separated nonnegative indices and inclusive ranges."""

    indices: set[int] = set()
    for part in specification.split(","):
        token = part.strip()
        if not token:
            raise ValueError(f"invalid empty index in {specification!r}")
        if "-" in token:
            lower_text, upper_text = token.split("-", 1)
            lower, upper = int(lower_text), int(upper_text)
            if lower < 0 or upper < lower:
                raise ValueError(f"invalid index range {token!r}")
            indices.update(range(lower, upper + 1))
        else:
            index = int(token)
            if index < 0:
                raise ValueError("indices must be nonnegative")
            indices.add(index)
    if not indices:
        raise ValueError("index specification is empty")
    return frozenset(indices)


@dataclass(frozen=True)
class ReferenceSplit:
    train: tuple[ReferenceBlock, ...]
    validation: tuple[ReferenceBlock, ...]
    test: tuple[ReferenceBlock, ...]


def split_reference_blocks(
    blocks: Sequence[ReferenceBlock],
    *,
    train_sections: Iterable[int],
    validation_sections: Iterable[int],
    test_sections: Iterable[int],
) -> ReferenceSplit:
    """Split blocks by original source section and reject leakage."""

    train_set = frozenset(int(value) for value in train_sections)
    validation_set = frozenset(int(value) for value in validation_sections)
    test_set = frozenset(int(value) for value in test_sections)
    if train_set & validation_set or train_set & test_set or validation_set & test_set:
        raise ValueError("train, validation, and test sections must be disjoint")

    def select(sections: frozenset[int]) -> tuple[ReferenceBlock, ...]:
        return tuple(
            block for block in blocks if block.source_section_index in sections
        )

    split = ReferenceSplit(
        train=select(train_set),
        validation=select(validation_set),
        test=select(test_set),
    )
    available = sorted({block.source_section_index for block in blocks})
    for name, selected in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        if not selected:
            raise ValueError(f"{name} split is empty; available sections are {available}")
    return split


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


class MultiBlockPlaneDataset:
    """Balanced deterministic planes over multiple latent/reference blocks.

    The learned target is the full CAESAR reconstruction in normalized block
    units. Raw samples are retained to measure total end-to-end error.
    """

    def __init__(
        self,
        blocks: Sequence[ReferenceBlock],
        *,
        sample_count: int,
        height: int,
        width: int,
        seed: int,
        orientation: PlaneOrientation = "mixed",
        maximum_offset: float = 0.5,
    ) -> None:
        if not blocks:
            raise ValueError("blocks must not be empty")
        if sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if height <= 0 or width <= 0:
            raise ValueError("height and width must be positive")
        self.blocks = tuple(blocks)
        latent_shape = self.blocks[0].latent.shape
        if any(block.latent.shape != latent_shape for block in self.blocks[1:]):
            raise ValueError("all blocks must have the same latent shape")
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
        block_position = index % len(self.blocks)
        block = self.blocks[block_position]
        generator = np.random.default_rng(np.random.SeedSequence([self.seed, index]))
        plane = random_contained_plane(
            generator,
            orientation=self.orientation,
            maximum_offset=self.maximum_offset,
        )
        points = plane.points(self.height, self.width)
        caesar_values = sample_volume(block.caesar_volume, points)
        raw_values = sample_volume(block.raw_volume, points)
        return {
            "latent": np.asarray(block.latent, dtype=np.float32),
            "points": points.astype(np.float32).reshape(-1, 3),
            "target_normalized": block.normalize(caesar_values)
            .astype(np.float32)
            .reshape(-1, 1),
            "caesar_values": caesar_values.reshape(-1, 1),
            "raw_values": raw_values.reshape(-1, 1),
            "scale": np.asarray(block.scale, dtype=np.float32),
            "offset": np.asarray(block.offset, dtype=np.float32),
            "block_position": np.asarray(block_position, dtype=np.int64),
            "source_section_index": np.asarray(
                block.source_section_index, dtype=np.int64
            ),
            "source_frame_start": np.asarray(
                block.source_frame_start, dtype=np.int64
            ),
            "plane_origin": plane.origin.astype(np.float32),
            "plane_axis_u": plane.axis_u.astype(np.float32),
            "plane_axis_v": plane.axis_v.astype(np.float32),
            "plane_bounds": np.asarray(
                [*plane.u_bounds, *plane.v_bounds], dtype=np.float32
            ),
        }


class RandomPlaneDataset(MultiBlockPlaneDataset):
    """Backward-compatible one-block specialization."""

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
        self.block = block
        super().__init__(
            (block,),
            sample_count=sample_count,
            height=height,
            width=width,
            seed=seed,
            orientation=orientation,
            maximum_offset=maximum_offset,
        )
