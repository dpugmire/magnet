"""General utilities for reproducibility, logging, and checkpoint I/O."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def setupLogging(levelName: str = "INFO") -> None:
    """Configure root logger with a compact format."""

    numericLevel = getattr(logging, levelName.upper(), None)
    if not isinstance(numericLevel, int):
        raise ValueError(f"Invalid log level: {levelName}")
    logging.basicConfig(
        level=numericLevel,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setSeeds(seed: int, deterministic: bool = True) -> None:
    """Set seeds across Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:  # pragma: no cover - older torch compatibility
            torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def resolveDevice(deviceName: str) -> torch.device:
    """Resolve device from CLI input."""

    normalized = deviceName.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(normalized)


def ensureDirectory(pathValue: str | Path) -> Path:
    """Create directory if needed and return Path."""

    path = Path(pathValue)
    path.mkdir(parents=True, exist_ok=True)
    return path


def dumpJson(data: Dict[str, Any], outputPath: str | Path) -> None:
    """Write JSON with stable formatting."""

    path = Path(outputPath)
    ensureDirectory(path.parent)
    with path.open("w", encoding="utf-8") as fileObj:
        json.dump(data, fileObj, indent=2, sort_keys=True)


def saveCheckpoint(
    checkpointPath: str | Path,
    epoch: int,
    modelState: Dict[str, Any],
    optimizerState: Optional[Dict[str, Any]],
    schedulerState: Optional[Dict[str, Any]],
    trainConfig: Dict[str, Any],
    normalization: Dict[str, Any],
    modelMeta: Dict[str, Any],
) -> None:
    """Save a full training checkpoint."""

    path = Path(checkpointPath)
    ensureDirectory(path.parent)
    payload = {
        "epoch": int(epoch),
        "modelState": modelState,
        "optimizerState": optimizerState,
        "schedulerState": schedulerState,
        "trainConfig": trainConfig,
        "normalization": normalization,
        "modelMeta": modelMeta,
    }
    torch.save(payload, path)


def loadCheckpoint(
    checkpointPath: str | Path, mapLocation: str | torch.device = "cpu"
) -> Dict[str, Any]:
    """Load checkpoint and validate required keys."""

    path = Path(checkpointPath)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")
    checkpoint = torch.load(path, map_location=mapLocation)
    requiredKeys = {"epoch", "modelState", "trainConfig"}
    missing = sorted(requiredKeys - set(checkpoint.keys()))
    if missing:
        raise KeyError(f"Checkpoint is missing required key(s): {missing}")
    return checkpoint


def countParameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""

    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
