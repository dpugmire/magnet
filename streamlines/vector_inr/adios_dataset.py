"""ADIOS2 dataset helpers for ensemble/time-indexed vector fields."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Iterator, List, Optional, Sequence, Tuple, TypeVar

import numpy as np

try:
    import adios2
except ImportError as importError:  # pragma: no cover - environment dependent
    adios2 = None
    _adiosImportError: Optional[Exception] = importError
else:
    _adiosImportError = None


logger = logging.getLogger(__name__)

CacheKeyT = TypeVar("CacheKeyT")
CacheValueT = TypeVar("CacheValueT")


@dataclass(frozen=True)
class NormalizationStats:
    """Global normalization stats for (vx, vy)."""

    mean: np.ndarray
    std: np.ndarray


class LruCache(Generic[CacheKeyT, CacheValueT]):
    """Simple LRU cache with fixed capacity."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("Cache capacity must be >= 0.")
        self.capacity = capacity
        self.items: "OrderedDict[CacheKeyT, CacheValueT]" = OrderedDict()

    def get(self, key: CacheKeyT) -> Optional[CacheValueT]:
        if key not in self.items:
            return None
        value = self.items.pop(key)
        self.items[key] = value
        return value

    def put(self, key: CacheKeyT, value: CacheValueT) -> None:
        if self.capacity == 0:
            return
        if key in self.items:
            self.items.pop(key)
        elif len(self.items) >= self.capacity:
            self.items.popitem(last=False)
        self.items[key] = value

    def clear(self) -> None:
        self.items.clear()


def parseShape(shapeValue: Any) -> Tuple[int, ...]:
    """Parse ADIOS shape metadata into a tuple."""

    if shapeValue is None:
        return ()
    if isinstance(shapeValue, str):
        text = shapeValue.strip()
        if not text:
            return ()
        return tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if isinstance(shapeValue, (list, tuple)):
        return tuple(int(value) for value in shapeValue)
    return ()


def computeFiniteDifferenceVorticity(
    vxField: np.ndarray,
    vyField: np.ndarray,
) -> np.ndarray:
    """Compute omega = d(vy)/dx - d(vx)/dy on normalized x,y in [-1, 1]."""

    if vxField.shape != vyField.shape:
        raise ValueError(
            f"Vorticity requires matching field shapes, got {vxField.shape} and {vyField.shape}."
        )
    if vxField.ndim != 2:
        raise ValueError(f"Expected 2D fields, got ndim={vxField.ndim}.")

    ny, nx = vxField.shape
    dy = 2.0 / max(ny - 1, 1)
    dx = 2.0 / max(nx - 1, 1)
    _, dvyDx = np.gradient(vyField, dy, dx, edge_order=1)
    dvxDy, _ = np.gradient(vxField, dy, dx, edge_order=1)
    omega = dvyDx - dvxDy
    return omega.astype(np.float32, copy=False)


class AdiosEnsembleReader:
    """Reader for one ensemble directory containing one BP dataset."""

    def __init__(
        self,
        ensembleDir: str,
        bpFile: str = "output.bp",
        vxName: str = "vx",
        vyName: str = "vy",
        cacheSteps: int = 8,
        omegaCacheSteps: int = 8,
    ) -> None:
        if adios2 is None:  # pragma: no cover - depends on local env
            raise ImportError(
                "adios2 is required but not installed. Install requirements first."
            ) from _adiosImportError

        self.ensembleDir = Path(ensembleDir)
        self.bpPath = self.ensembleDir / bpFile
        self.vxName = vxName
        self.vyName = vyName
        self.availableVariables: List[str] = []
        self.stepMap: List[Tuple[int, Optional[int]]] = []
        self.layout: Optional[str] = None
        self.gridShape: Optional[Tuple[int, int]] = None
        self.stepCache: LruCache[int, Tuple[np.ndarray, np.ndarray]] = LruCache(cacheSteps)
        self.rawStepCache: LruCache[int, Tuple[np.ndarray, np.ndarray]] = LruCache(cacheSteps)
        self.omegaCache: LruCache[Tuple[int, bool], np.ndarray] = LruCache(omegaCacheSteps)
        self.normalizationMean: Optional[np.ndarray] = None
        self.normalizationStd: Optional[np.ndarray] = None
        self.streamReader: Optional[Any] = None
        self.streamStepIterator: Optional[Iterator[Any]] = None
        self.streamCursor: int = -1
        if not self.bpPath.exists():
            raise FileNotFoundError(f"ADIOS dataset not found: {self.bpPath}")

        self._buildStepIndex()

    def _buildStepIndex(self) -> None:
        seenAnyStep = False
        with adios2.Stream(str(self.bpPath), "r") as stream:
            for streamStepIdx, _ in enumerate(stream.steps()):
                seenAnyStep = True
                variables = stream.available_variables()
                if not self.availableVariables:
                    self.availableVariables = sorted(variables.keys())
                if self.vxName not in variables or self.vyName not in variables:
                    available = ", ".join(sorted(variables.keys()))
                    missingNames = [
                        name
                        for name in (self.vxName, self.vyName)
                        if name not in variables
                    ]
                    raise KeyError(
                        f"Missing variable(s) {missingNames} in {self.bpPath}. "
                        f"Available variables: [{available}]"
                    )

                vxShape = parseShape(variables[self.vxName].get("Shape"))
                vyShape = parseShape(variables[self.vyName].get("Shape"))
                if not vxShape or not vyShape:
                    vxProbe = np.asarray(stream.read(self.vxName))
                    vyProbe = np.asarray(stream.read(self.vyName))
                    vxShape = tuple(int(size) for size in vxProbe.shape)
                    vyShape = tuple(int(size) for size in vyProbe.shape)

                if vxShape != vyShape:
                    raise ValueError(
                        f"vx/vy shape mismatch in {self.bpPath} at stream step {streamStepIdx}: "
                        f"{vxShape} vs {vyShape}."
                    )

                currentLayout, currentGridShape, localStepCount = self._inferLayout(vxShape)
                if self.layout is None:
                    self.layout = currentLayout
                elif self.layout != currentLayout:
                    raise ValueError(
                        f"Inconsistent layout in {self.bpPath}. "
                        f"Saw {self.layout} and {currentLayout}."
                    )

                if self.gridShape is None:
                    self.gridShape = currentGridShape
                elif self.gridShape != currentGridShape:
                    raise ValueError(
                        f"Inconsistent grid shape in {self.bpPath}. "
                        f"Saw {self.gridShape} and {currentGridShape}."
                    )

                if currentLayout == "perStep2d":
                    self.stepMap.append((streamStepIdx, None))
                else:
                    for localStep in range(localStepCount):
                        self.stepMap.append((streamStepIdx, localStep))

        if not seenAnyStep:
            raise RuntimeError(f"No ADIOS steps found in {self.bpPath}.")
        if self.gridShape is None:
            raise RuntimeError(f"Failed to infer grid shape from {self.bpPath}.")
        logger.info(
            "Indexed %d logical steps from %s (layout=%s, grid=%s).",
            len(self.stepMap),
            self.bpPath,
            self.layout,
            self.gridShape,
        )

    @staticmethod
    def _inferLayout(shape: Tuple[int, ...]) -> Tuple[str, Tuple[int, int], int]:
        if len(shape) == 2:
            ny, nx = shape
            return "perStep2d", (int(ny), int(nx)), 1
        if len(shape) == 3:
            stepCount, ny, nx = shape
            return "stacked3d", (int(ny), int(nx)), int(stepCount)
        raise ValueError(
            f"Unsupported variable shape {shape}. Expected 2D [ny,nx] or 3D [step,ny,nx]."
        )

    def close(self) -> None:
        if self.streamReader is None:
            return
        closeMethod = getattr(self.streamReader, "close", None)
        if callable(closeMethod):  # pragma: no branch
            closeMethod()
        self.streamReader = None
        self.streamStepIterator = None
        self.streamCursor = -1

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass

    def getStepCount(self) -> int:
        return len(self.stepMap)

    def getGridShape(self) -> Tuple[int, int]:
        if self.gridShape is None:
            raise RuntimeError("Grid shape unavailable.")
        return self.gridShape

    def _readRawFieldsAtStreamStep(self, streamStepIdx: int) -> Tuple[np.ndarray, np.ndarray]:
        cachedRaw = self.rawStepCache.get(streamStepIdx)
        if cachedRaw is not None:
            return cachedRaw

        if streamStepIdx < 0:
            raise IndexError(f"Stream step index must be >= 0, got {streamStepIdx}.")

        # Keep one open stream and advance sequentially to avoid re-scanning from step 0 on each read.
        if self.streamReader is None or self.streamStepIterator is None:
            self.streamReader = adios2.Stream(str(self.bpPath), "r")
            self.streamStepIterator = iter(self.streamReader.steps())
            self.streamCursor = -1

        if streamStepIdx < self.streamCursor:
            self.close()
            self.streamReader = adios2.Stream(str(self.bpPath), "r")
            self.streamStepIterator = iter(self.streamReader.steps())
            self.streamCursor = -1

        while self.streamCursor < streamStepIdx:
            try:
                next(self.streamStepIterator)
            except StopIteration as stopError:
                raise IndexError(
                    f"Stream step {streamStepIdx} not found in {self.bpPath} when reading velocity."
                ) from stopError
            self.streamCursor += 1

        vxRaw = np.asarray(self.streamReader.read(self.vxName))
        vyRaw = np.asarray(self.streamReader.read(self.vyName))
        value = (vxRaw, vyRaw)
        self.rawStepCache.put(streamStepIdx, value)
        return value

    @staticmethod
    def _to2dField(rawField: np.ndarray, localStep: Optional[int]) -> np.ndarray:
        if rawField.ndim == 2:
            if localStep not in (None, 0):
                raise ValueError(
                    f"Requested local step {localStep} from a 2D field with shape {rawField.shape}."
                )
            return rawField.astype(np.float32, copy=False)

        if rawField.ndim == 3:
            if localStep is None:
                if rawField.shape[0] == 1:
                    return rawField[0].astype(np.float32, copy=False)
                raise ValueError(
                    "Got 3D field but localStep is None and leading dimension is > 1."
                )
            if localStep < 0 or localStep >= rawField.shape[0]:
                raise IndexError(
                    f"localStep {localStep} out of range for field shape {rawField.shape}."
                )
            return rawField[localStep].astype(np.float32, copy=False)

        if rawField.ndim == 4 and rawField.shape[0] == 1:
            return AdiosEnsembleReader._to2dField(rawField[0], localStep)

        raise ValueError(f"Unsupported raw field shape {rawField.shape}.")

    def readVelocityStep(self, stepIdx: int) -> Tuple[np.ndarray, np.ndarray]:
        cachedValue = self.stepCache.get(stepIdx)
        if cachedValue is not None:
            return cachedValue

        if stepIdx < 0 or stepIdx >= len(self.stepMap):
            raise IndexError(
                f"Step index {stepIdx} is out of range [0, {len(self.stepMap) - 1}] "
                f"for dataset {self.bpPath}."
            )

        streamStepIdx, localStep = self.stepMap[stepIdx]
        vxRaw, vyRaw = self._readRawFieldsAtStreamStep(streamStepIdx)
        vxField = self._to2dField(vxRaw, localStep)
        vyField = self._to2dField(vyRaw, localStep)

        if vxField.shape != vyField.shape:
            raise ValueError(
                f"Velocity shape mismatch at logical step {stepIdx}: "
                f"vx={vxField.shape}, vy={vyField.shape}."
            )
        if self.gridShape is not None and vxField.shape != self.gridShape:
            raise ValueError(
                f"Unexpected grid shape at step {stepIdx}: got {vxField.shape}, expected {self.gridShape}."
            )

        vxField = np.asarray(vxField, dtype=np.float32)
        vyField = np.asarray(vyField, dtype=np.float32)
        vxField.setflags(write=False)
        vyField.setflags(write=False)
        value = (vxField, vyField)
        self.stepCache.put(stepIdx, value)
        return value

    def setNormalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        meanArray = np.asarray(mean, dtype=np.float32)
        stdArray = np.asarray(std, dtype=np.float32)
        if meanArray.shape != (2,) or stdArray.shape != (2,):
            raise ValueError(
                f"Normalization arrays must have shape (2,), got {meanArray.shape} and {stdArray.shape}."
            )
        if np.any(stdArray <= 0):
            raise ValueError("Normalization std must be positive for both channels.")
        self.normalizationMean = meanArray
        self.normalizationStd = stdArray
        self.omegaCache.clear()

    def readVorticityStep(self, stepIdx: int, useNormalized: bool = True) -> np.ndarray:
        cacheKey = (stepIdx, bool(useNormalized))
        cachedValue = self.omegaCache.get(cacheKey)
        if cachedValue is not None:
            return cachedValue

        vxField, vyField = self.readVelocityStep(stepIdx)
        localVx = vxField
        localVy = vyField
        if useNormalized:
            if self.normalizationMean is None or self.normalizationStd is None:
                raise RuntimeError(
                    "Normalization stats must be set before requesting normalized vorticity."
                )
            localVx = (localVx - self.normalizationMean[0]) / self.normalizationStd[0]
            localVy = (localVy - self.normalizationMean[1]) / self.normalizationStd[1]

        omegaField = computeFiniteDifferenceVorticity(localVx, localVy)
        omegaField.setflags(write=False)
        self.omegaCache.put(cacheKey, omegaField)
        return omegaField


class MultiEnsembleDataset:
    """Unified API over multiple ensemble directories."""

    def __init__(
        self,
        ensembleDirs: Sequence[str],
        bpFile: str = "output.bp",
        vxName: str = "vx",
        vyName: str = "vy",
        cacheSteps: int = 8,
        omegaCacheSteps: int = 8,
        requireSameGrid: bool = True,
    ) -> None:
        if len(ensembleDirs) == 0:
            raise ValueError("At least one ensemble directory is required.")
        self.ensembleReaders = [
            AdiosEnsembleReader(
                ensembleDir=ensembleDir,
                bpFile=bpFile,
                vxName=vxName,
                vyName=vyName,
                cacheSteps=cacheSteps,
                omegaCacheSteps=omegaCacheSteps,
            )
            for ensembleDir in ensembleDirs
        ]
        self.requireSameGrid = requireSameGrid
        self._validateGridShapes()

    def close(self) -> None:
        for reader in self.ensembleReaders:
            reader.close()

    def __del__(self) -> None:  # pragma: no cover - cleanup only
        try:
            self.close()
        except Exception:
            pass

    def _validateGridShapes(self) -> None:
        if not self.requireSameGrid:
            return
        gridShapes = [reader.getGridShape() for reader in self.ensembleReaders]
        firstShape = gridShapes[0]
        for ensembleIdx, shape in enumerate(gridShapes):
            if shape != firstShape:
                raise ValueError(
                    "All ensembles must share one grid size for this implementation. "
                    f"Ensemble 0 shape={firstShape}, ensemble {ensembleIdx} shape={shape}."
                )

    def getEnsembleCount(self) -> int:
        return len(self.ensembleReaders)

    def getStepCount(self, ensembleIdx: int) -> int:
        return self.ensembleReaders[ensembleIdx].getStepCount()

    def getStepCounts(self) -> List[int]:
        return [reader.getStepCount() for reader in self.ensembleReaders]

    def getGridShape(self, ensembleIdx: int) -> Tuple[int, int]:
        return self.ensembleReaders[ensembleIdx].getGridShape()

    def readVelocityStep(self, ensembleIdx: int, stepIdx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.ensembleReaders[ensembleIdx].readVelocityStep(stepIdx)

    def readVorticityStep(
        self, ensembleIdx: int, stepIdx: int, useNormalized: bool = True
    ) -> np.ndarray:
        return self.ensembleReaders[ensembleIdx].readVorticityStep(
            stepIdx=stepIdx, useNormalized=useNormalized
        )

    def setNormalization(self, mean: np.ndarray, std: np.ndarray) -> None:
        for reader in self.ensembleReaders:
            reader.setNormalization(mean, std)

    # Snake-case aliases for explicit external API compatibility.
    def get_step_count(self, ensemble_idx: int) -> int:
        return self.getStepCount(ensemble_idx)

    def get_grid_shape(self, ensemble_idx: int) -> Tuple[int, int]:
        return self.getGridShape(ensemble_idx)

    def read_velocity_step(
        self, ensemble_idx: int, step_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.readVelocityStep(ensemble_idx, step_idx)

    def computeNormalizationStats(
        self,
        sampleStepsPerEnsemble: int = 8,
        samplePointsPerStep: int = 4096,
        seed: int = 0,
    ) -> NormalizationStats:
        """Estimate global mean/std from a random subsample for speed."""

        if sampleStepsPerEnsemble <= 0:
            raise ValueError("sampleStepsPerEnsemble must be > 0.")
        if samplePointsPerStep <= 0:
            raise ValueError("samplePointsPerStep must be > 0.")

        rng = np.random.default_rng(seed)
        valueSum = np.zeros(2, dtype=np.float64)
        valueSquareSum = np.zeros(2, dtype=np.float64)
        totalSamples = 0

        for ensembleIdx, reader in enumerate(self.ensembleReaders):
            stepCount = reader.getStepCount()
            if stepCount == 0:
                logger.warning("Ensemble %d has zero steps and will be skipped.", ensembleIdx)
                continue
            localStepCount = min(sampleStepsPerEnsemble, stepCount)
            sampledSteps = rng.choice(stepCount, size=localStepCount, replace=False)

            for stepIdxRaw in sampledSteps:
                stepIdx = int(stepIdxRaw)
                vxField, vyField = reader.readVelocityStep(stepIdx)
                ny, nx = vxField.shape
                fieldPointCount = ny * nx
                localPointCount = min(samplePointsPerStep, fieldPointCount)
                sampledFlatIndices = rng.choice(
                    fieldPointCount, size=localPointCount, replace=False
                )
                iy = sampledFlatIndices // nx
                ix = sampledFlatIndices % nx
                sampledValues = np.stack(
                    [vxField[iy, ix], vyField[iy, ix]],
                    axis=1,
                ).astype(np.float64, copy=False)
                valueSum += sampledValues.sum(axis=0)
                valueSquareSum += np.square(sampledValues).sum(axis=0)
                totalSamples += sampledValues.shape[0]

        if totalSamples == 0:
            raise RuntimeError(
                "Failed to compute normalization stats because no samples were collected."
            )

        mean = valueSum / float(totalSamples)
        variance = np.maximum(valueSquareSum / float(totalSamples) - np.square(mean), 1e-12)
        std = np.sqrt(variance)
        stats = NormalizationStats(mean=mean.astype(np.float32), std=std.astype(np.float32))
        logger.info("Normalization mean=%s std=%s", stats.mean.tolist(), stats.std.tolist())
        return stats
