"""Point sampling and coordinate normalization for vector INR training/eval."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from .adios_dataset import MultiEnsembleDataset


def normalizeSpatialIndices(indices: np.ndarray, size: int) -> np.ndarray:
    """Map integer indices [0, size-1] to continuous coordinates in [-1, 1]."""

    if size <= 1:
        return np.zeros_like(indices, dtype=np.float32)
    return (indices.astype(np.float32) / float(size - 1)) * 2.0 - 1.0


def normalizeTimeIndices(stepIndices: np.ndarray, stepCounts: np.ndarray) -> np.ndarray:
    """Map per-sample step indices to t in [0, 1]."""

    safeDenominator = np.maximum(stepCounts - 1, 1)
    normalizedTime = stepIndices.astype(np.float32) / safeDenominator.astype(np.float32)
    normalizedTime = np.where(stepCounts > 1, normalizedTime, 0.0)
    return normalizedTime.astype(np.float32)


class PointSampler:
    """Uniform random point sampler over ensembles, steps, and grid indices."""

    def __init__(
        self,
        dataset: MultiEnsembleDataset,
        normalizationMean: np.ndarray,
        normalizationStd: np.ndarray,
        batchPoints: int,
        device: torch.device,
        seed: int = 0,
    ) -> None:
        if batchPoints <= 0:
            raise ValueError("batchPoints must be > 0.")
        self.dataset = dataset
        self.normalizationMean = np.asarray(normalizationMean, dtype=np.float32)
        self.normalizationStd = np.asarray(normalizationStd, dtype=np.float32)
        self.batchPoints = int(batchPoints)
        self.device = device
        self.rng = np.random.default_rng(seed)

        if self.normalizationMean.shape != (2,) or self.normalizationStd.shape != (2,):
            raise ValueError(
                "Normalization mean/std must have shape (2,) for (vx, vy) channels."
            )

    def sampleBatch(self, includeVorticity: bool = False) -> Dict[str, torch.Tensor]:
        """Sample one batch and return tensors on the configured device."""

        ensembleCount = self.dataset.getEnsembleCount()
        ensembleIndices = self.rng.integers(
            low=0, high=ensembleCount, size=self.batchPoints, dtype=np.int64
        )
        stepCounts = np.array(
            [self.dataset.getStepCount(int(ensembleIdx)) for ensembleIdx in ensembleIndices],
            dtype=np.int64,
        )
        if np.any(stepCounts <= 0):
            raise RuntimeError("Found ensemble with zero steps, cannot sample training batch.")

        stepIndices = np.array(
            [self.rng.integers(0, int(stepCount)) for stepCount in stepCounts],
            dtype=np.int64,
        )

        ny, nx = self.dataset.getGridShape(0)
        iy = self.rng.integers(low=0, high=ny, size=self.batchPoints, dtype=np.int64)
        ix = self.rng.integers(low=0, high=nx, size=self.batchPoints, dtype=np.int64)

        samplePairs = np.stack([ensembleIndices, stepIndices], axis=1)
        uniquePairs, inverseIndex = np.unique(samplePairs, axis=0, return_inverse=True)

        vxTargets = np.empty(self.batchPoints, dtype=np.float32)
        vyTargets = np.empty(self.batchPoints, dtype=np.float32)
        omegaTargets = np.empty(self.batchPoints, dtype=np.float32) if includeVorticity else None

        for uniquePairIdx, pairRaw in enumerate(uniquePairs):
            ensembleIdx = int(pairRaw[0])
            stepIdx = int(pairRaw[1])
            pairMask = inverseIndex == uniquePairIdx
            sampleIy = iy[pairMask]
            sampleIx = ix[pairMask]

            vxField, vyField = self.dataset.readVelocityStep(ensembleIdx, stepIdx)
            vxTargets[pairMask] = vxField[sampleIy, sampleIx]
            vyTargets[pairMask] = vyField[sampleIy, sampleIx]

            if includeVorticity:
                omegaField = self.dataset.readVorticityStep(
                    ensembleIdx=ensembleIdx,
                    stepIdx=stepIdx,
                    useNormalized=True,
                )
                omegaTargets[pairMask] = omegaField[sampleIy, sampleIx]

        vxTargets = (vxTargets - self.normalizationMean[0]) / self.normalizationStd[0]
        vyTargets = (vyTargets - self.normalizationMean[1]) / self.normalizationStd[1]
        velocityTargets = np.stack([vxTargets, vyTargets], axis=1).astype(np.float32)

        x = normalizeSpatialIndices(ix, nx)
        y = normalizeSpatialIndices(iy, ny)
        t = normalizeTimeIndices(stepIndices, stepCounts)
        coords = np.stack([x, y, t], axis=1).astype(np.float32)

        batch: Dict[str, torch.Tensor] = {
            "coords": torch.from_numpy(coords).to(self.device),
            "ensembleIndices": torch.from_numpy(ensembleIndices).to(self.device),
            "targetVelocity": torch.from_numpy(velocityTargets).to(self.device),
        }
        if includeVorticity and omegaTargets is not None:
            batch["targetOmega"] = torch.from_numpy(omegaTargets).to(self.device)
        return batch


def buildFullGridQuery(
    ny: int,
    nx: int,
    stepIdx: int,
    stepCount: int,
    ensembleIdx: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Build flattened full-grid query tensors for inference."""

    iyGrid, ixGrid = np.indices((ny, nx), dtype=np.int64)
    iy = iyGrid.reshape(-1)
    ix = ixGrid.reshape(-1)
    x = normalizeSpatialIndices(ix, nx)
    y = normalizeSpatialIndices(iy, ny)
    if stepCount <= 1:
        tValue = 0.0
    else:
        tValue = float(stepIdx) / float(stepCount - 1)
    t = np.full_like(x, fill_value=tValue, dtype=np.float32)

    coords = np.stack([x, y, t], axis=1).astype(np.float32)
    ensembleIndices = np.full(coords.shape[0], fill_value=ensembleIdx, dtype=np.int64)
    return {
        "coords": torch.from_numpy(coords).to(device),
        "ensembleIndices": torch.from_numpy(ensembleIndices).to(device),
    }
