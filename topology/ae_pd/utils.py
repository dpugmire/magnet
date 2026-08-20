"""General utilities for reproducibility, logging, and checkpoints."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


def SetSeed(seed: int) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def EnsureDir(pathText: str | Path) -> Path:
    """Create directory if needed and return resolved path."""

    pathObj = Path(pathText).expanduser().resolve()
    pathObj.mkdir(parents=True, exist_ok=True)
    return pathObj


def SaveJson(dataObj: Dict[str, Any], filePath: str | Path) -> None:
    """Write JSON to disk with stable formatting."""

    pathObj = Path(filePath).expanduser().resolve()
    pathObj.parent.mkdir(parents=True, exist_ok=True)
    with pathObj.open("w", encoding="utf-8") as fileHandle:
        json.dump(dataObj, fileHandle, indent=2, sort_keys=True)


def CreateLogger(outDir: str | Path, loggerName: str = "ae_pd") -> logging.Logger:
    """Create console + file logger."""

    outPath = EnsureDir(outDir)
    logger = logging.getLogger(loggerName)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    while logger.handlers:
        logger.handlers.pop()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    fileHandler = logging.FileHandler(outPath / "train.log")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    return logger


def SaveCheckpoint(stateDict: Dict[str, Any], filePath: str | Path) -> None:
    """Save PyTorch checkpoint."""

    pathObj = Path(filePath).expanduser().resolve()
    pathObj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stateDict, str(pathObj))


def LoadCheckpoint(filePath: str | Path, mapLocation: str | torch.device = "cpu") -> Dict[str, Any]:
    """Load PyTorch checkpoint."""

    pathObj = Path(filePath).expanduser().resolve()
    if not pathObj.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pathObj}")
    return torch.load(str(pathObj), map_location=mapLocation)


def ResolveDevice(deviceText: str) -> torch.device:
    """Resolve a CLI device string to torch.device."""

    deviceLower = deviceText.lower()
    if deviceLower == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if deviceLower == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(deviceLower)
