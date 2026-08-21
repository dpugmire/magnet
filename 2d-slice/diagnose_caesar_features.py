#!/usr/bin/env python3
"""Extract CAESAR-V decoder feature taps and verify exact continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Sequence

import numpy as np
import torch

from slice_decoder.caesar_features import (
    FrozenCaesarVFeatureDecoder,
    caesar_feature_tap_metadata,
)
from slice_decoder.datasets import load_reference_block
from slice_decoder.metrics import field_error_metrics


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify frozen CAESAR-V decoder feature taps"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-index", type=_nonnegative_integer, default=0)
    parser.add_argument("--device", default="cpu")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    model_path = args.model.expanduser().resolve()
    device = torch.device(args.device)

    block = load_reference_block(args.artifact_dir, block_index=args.block_index)
    if block.base_volume is None:
        raise ValueError("feature diagnostics require a staged CAESAR artifact")
    latent = torch.from_numpy(np.array(block.latent, copy=True)).unsqueeze(0)
    decoder = FrozenCaesarVFeatureDecoder(model_path, device=device)
    decoder.decode_latent(latent)
    _synchronize(device)

    tap_results: dict[str, object] = {}
    for tap in ("early", "late"):
        start = time.perf_counter()
        features = decoder.extract(latent, tap)
        _synchronize(device)
        extraction_milliseconds = (time.perf_counter() - start) * 1000.0

        start = time.perf_counter()
        normalized = decoder.continue_decode(features, tap)
        _synchronize(device)
        continuation_milliseconds = (time.perf_counter() - start) * 1000.0

        feature_array = features[0].detach().cpu().numpy().astype(np.float32)
        np.save(output_directory / f"caesar_{tap}_features.npy", feature_array)
        reconstructed = block.denormalize(normalized[0, 0].detach().cpu().numpy())
        metadata = caesar_feature_tap_metadata(tap)
        tap_results[tap] = {
            "shape": list(feature_array.shape),
            "float32Bytes": metadata.float32_bytes,
            "extractionMilliseconds": extraction_milliseconds,
            "continuationMilliseconds": continuation_milliseconds,
            "continuationVsSavedBase": field_error_metrics(
                block.base_volume,
                reconstructed,
            ),
        }

    results = {
        "formatVersion": 1,
        "artifactDirectory": str(Path(args.artifact_dir).expanduser().resolve()),
        "artifactBlockIndex": args.block_index,
        "sourceSectionIndex": block.source_section_index,
        "sourceFrameRange": [block.source_frame_start, block.source_frame_end],
        "model": str(model_path),
        "modelSha256": _sha256(model_path),
        "device": str(device),
        "latentShape": list(block.latent.shape),
        "taps": tap_results,
    }
    result_path = output_directory / "caesar_feature_diagnostic.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
