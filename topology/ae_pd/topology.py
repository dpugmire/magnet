"""Persistence diagram and persistence image utilities using Gudhi."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import gudhi
import numpy as np
import torch
import torch.nn.functional as F


def DownsampleImage(imageArray: np.ndarray, targetSize: int) -> np.ndarray:
    """Downsample 2D image to (targetSize, targetSize) with bilinear interpolation."""

    imageFloat = imageArray.astype(np.float32, copy=False)
    if targetSize <= 0:
        raise ValueError(f"targetSize must be positive, got {targetSize}")
    if imageFloat.shape == (targetSize, targetSize):
        return imageFloat
    tensorIn = torch.from_numpy(imageFloat[None, None, :, :])
    tensorOut = F.interpolate(
        tensorIn,
        size=(targetSize, targetSize),
        mode="bilinear",
        align_corners=False,
    )
    return tensorOut[0, 0].cpu().numpy().astype(np.float32, copy=False)


def ComputePersistenceDiagrams(
    imageArray: np.ndarray,
    pdDims: Sequence[int],
) -> Dict[int, np.ndarray]:
    """Compute finite birth/death pairs for selected homology dimensions."""

    imageFloat = imageArray.astype(np.float64, copy=False)
    cubicalComplex = gudhi.CubicalComplex(
        dimensions=imageFloat.shape,
        top_dimensional_cells=imageFloat.ravel(order="C"),
    )
    cubicalComplex.persistence()

    diagramByDim: Dict[int, np.ndarray] = {}
    for dimValue in pdDims:
        intervalArray = cubicalComplex.persistence_intervals_in_dimension(int(dimValue))
        intervalArray = np.asarray(intervalArray, dtype=np.float64)
        if intervalArray.size == 0:
            diagramByDim[int(dimValue)] = np.zeros((0, 2), dtype=np.float32)
            continue
        finiteMask = np.isfinite(intervalArray).all(axis=1)
        finitePairs = intervalArray[finiteMask]
        diagramByDim[int(dimValue)] = finitePairs.astype(np.float32, copy=False)
    return diagramByDim


def FilterDiagramByMinPersistence(diagramArray: np.ndarray, minPersistence: float) -> np.ndarray:
    """Filter one birth/death diagram by finite values and persistence threshold."""

    pairArray = np.asarray(diagramArray, dtype=np.float32).reshape(-1, 2)
    if pairArray.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    persistenceArray = pairArray[:, 1] - pairArray[:, 0]
    threshold = max(float(minPersistence), 0.0)
    validMask = np.isfinite(pairArray).all(axis=1) & np.isfinite(persistenceArray) & (persistenceArray > threshold)
    return pairArray[validMask].astype(np.float32, copy=False)


def FilterDiagramDictByMinPersistence(
    diagramByDim: Dict[int, np.ndarray],
    minPersistence: float,
) -> Dict[int, np.ndarray]:
    """Filter all PD dimensions by a minimum persistence threshold."""

    filteredByDim: Dict[int, np.ndarray] = {}
    for dimValue, intervalArray in diagramByDim.items():
        filteredByDim[int(dimValue)] = FilterDiagramByMinPersistence(
            intervalArray,
            minPersistence=minPersistence,
        )
    return filteredByDim


def _CollectBirthPersistence(
    diagramByDim: Dict[int, np.ndarray],
    minPersistence: float,
) -> np.ndarray:
    pointList: List[np.ndarray] = []
    threshold = max(float(minPersistence), 0.0)
    for intervalArray in diagramByDim.values():
        if intervalArray.size == 0:
            continue
        birthArray = intervalArray[:, 0]
        persistenceArray = intervalArray[:, 1] - intervalArray[:, 0]
        validMask = np.isfinite(birthArray) & np.isfinite(persistenceArray) & (persistenceArray > threshold)
        if not np.any(validMask):
            continue
        points = np.stack([birthArray[validMask], persistenceArray[validMask]], axis=1)
        pointList.append(points)
    if not pointList:
        return np.zeros((0, 2), dtype=np.float32)
    return np.concatenate(pointList, axis=0).astype(np.float32, copy=False)


def BuildPersistenceImage(
    diagramByDim: Dict[int, np.ndarray],
    piRes: int,
    piSigma: float,
    weightByPersistence: bool,
    minPersistence: float = 0.0,
    valueRange: Tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Create a persistence image from diagrams by summing Gaussian kernels."""

    if piRes <= 0:
        raise ValueError(f"piRes must be positive, got {piRes}")
    if piSigma <= 0.0:
        raise ValueError(f"piSigma must be positive, got {piSigma}")

    pointArray = _CollectBirthPersistence(
        diagramByDim=diagramByDim,
        minPersistence=minPersistence,
    )
    if pointArray.size == 0:
        return np.zeros((piRes, piRes), dtype=np.float32)

    if valueRange is None:
        birthMin = float(pointArray[:, 0].min())
        birthMax = float(pointArray[:, 0].max())
        persMin = 0.0
        persMax = float(pointArray[:, 1].max())
    else:
        birthMin, birthMax, persMin, persMax = valueRange

    if abs(birthMax - birthMin) < 1e-8:
        birthMax = birthMin + 1e-6
    if abs(persMax - persMin) < 1e-8:
        persMax = persMin + 1e-6

    xAxis = np.linspace(birthMin, birthMax, piRes, dtype=np.float32)
    yAxis = np.linspace(persMin, persMax, piRes, dtype=np.float32)
    xGrid, yGrid = np.meshgrid(xAxis, yAxis, indexing="xy")
    imageArray = np.zeros((piRes, piRes), dtype=np.float32)
    sigma2 = float(piSigma) ** 2

    for pointBirth, pointPersistence in pointArray:
        weightValue = float(pointPersistence) if weightByPersistence else 1.0
        dist2 = (xGrid - pointBirth) ** 2 + (yGrid - pointPersistence) ** 2
        imageArray += weightValue * np.exp(-0.5 * dist2 / sigma2)

    return imageArray.astype(np.float32, copy=False)


def ComputePairedPersistenceImages(
    gtImage: np.ndarray,
    reconImage: np.ndarray,
    pdDims: Sequence[int],
    pdDownsample: int,
    piRes: int,
    piSigma: float,
    weightByPersistence: bool = True,
    minPersistence: float = 0.0,
) -> Dict[str, object]:
    """Compute GT/recon PDs and persistence images with a shared PI range."""

    gtSmall = DownsampleImage(gtImage, pdDownsample)
    reconSmall = DownsampleImage(reconImage, pdDownsample)

    gtDiagramRawByDim = ComputePersistenceDiagrams(gtSmall, pdDims)
    reconDiagramRawByDim = ComputePersistenceDiagrams(reconSmall, pdDims)
    gtDiagramByDim = FilterDiagramDictByMinPersistence(gtDiagramRawByDim, minPersistence=minPersistence)
    reconDiagramByDim = FilterDiagramDictByMinPersistence(reconDiagramRawByDim, minPersistence=minPersistence)

    gtPoints = _CollectBirthPersistence(gtDiagramByDim, minPersistence=minPersistence)
    reconPoints = _CollectBirthPersistence(reconDiagramByDim, minPersistence=minPersistence)
    allPoints = np.concatenate([gtPoints, reconPoints], axis=0) if (gtPoints.size or reconPoints.size) else np.zeros((0, 2), dtype=np.float32)

    if allPoints.size == 0:
        sharedRange = (0.0, 1.0, 0.0, 1.0)
    else:
        birthMin = float(allPoints[:, 0].min())
        birthMax = float(allPoints[:, 0].max())
        persMin = 0.0
        persMax = float(allPoints[:, 1].max())
        if abs(birthMax - birthMin) < 1e-8:
            birthMax = birthMin + 1e-6
        if abs(persMax - persMin) < 1e-8:
            persMax = persMin + 1e-6
        sharedRange = (birthMin, birthMax, persMin, persMax)

    gtPi = BuildPersistenceImage(
        diagramByDim=gtDiagramByDim,
        piRes=piRes,
        piSigma=piSigma,
        weightByPersistence=weightByPersistence,
        minPersistence=minPersistence,
        valueRange=sharedRange,
    )
    reconPi = BuildPersistenceImage(
        diagramByDim=reconDiagramByDim,
        piRes=piRes,
        piSigma=piSigma,
        weightByPersistence=weightByPersistence,
        minPersistence=minPersistence,
        valueRange=sharedRange,
    )

    return {
        "gtPi": gtPi,
        "reconPi": reconPi,
        "gtDiagramByDim": gtDiagramByDim,
        "reconDiagramByDim": reconDiagramByDim,
        "sharedRange": sharedRange,
    }


def ComputeBottleneckDistances(
    gtDiagramByDim: Dict[int, np.ndarray],
    reconDiagramByDim: Dict[int, np.ndarray],
    pdDims: Sequence[int],
) -> Dict[int, float]:
    """Compute bottleneck distance per homology dimension."""

    distanceByDim: Dict[int, float] = {}
    for dimValue in pdDims:
        dimKey = int(dimValue)
        gtDiag = np.asarray(gtDiagramByDim.get(dimKey, np.zeros((0, 2), dtype=np.float32)))
        reconDiag = np.asarray(reconDiagramByDim.get(dimKey, np.zeros((0, 2), dtype=np.float32)))
        try:
            distanceValue = float(gudhi.bottleneck_distance(gtDiag.tolist(), reconDiag.tolist()))
        except Exception:
            distanceValue = float("nan")
        distanceByDim[dimKey] = distanceValue
    return distanceByDim
