"""Core utilities for extracting 2D planes from CAESAR representations."""

from .caesar_adapter import (
    ArchiveMetadata,
    CaesarReference,
    LatentBlock,
    LatentVolume,
    extract_latent_blocks,
    inspect_npz,
    open_stored_npz_array,
    prepare_caesar_subset,
    run_caesar_reference,
    save_caesar_reference,
    stack_latent_depth,
)
from .geometry import (
    Plane,
    axis_aligned_plane,
    index_to_normalized,
    sample_plane,
    sample_volume,
)

__all__ = [
    "ArchiveMetadata",
    "CaesarReference",
    "LatentBlock",
    "LatentVolume",
    "Plane",
    "axis_aligned_plane",
    "extract_latent_blocks",
    "index_to_normalized",
    "inspect_npz",
    "open_stored_npz_array",
    "prepare_caesar_subset",
    "run_caesar_reference",
    "sample_plane",
    "sample_volume",
    "save_caesar_reference",
    "stack_latent_depth",
]
