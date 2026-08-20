"""ADIOS2 Stream reader utilities for scalar MHD snapshot data."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from adios2 import Stream


@dataclass
class RunMeta:
    """Metadata describing one simulation run."""

    runName: str
    bpPath: Path
    layout: str
    stepCount: int
    gridShape: Tuple[int, int]


class LruArrayCache:
    """Small in-memory LRU cache for recently read snapshots."""

    def __init__(self, maxSize: int) -> None:
        self.maxSize = max(0, int(maxSize))
        self.cacheDict: "OrderedDict[Tuple[str, int], np.ndarray]" = OrderedDict()

    def get(self, key: Tuple[str, int]) -> Optional[np.ndarray]:
        if key not in self.cacheDict:
            return None
        self.cacheDict.move_to_end(key)
        return self.cacheDict[key]

    def put(self, key: Tuple[str, int], value: np.ndarray) -> None:
        if self.maxSize <= 0:
            return
        self.cacheDict[key] = value
        self.cacheDict.move_to_end(key)
        while len(self.cacheDict) > self.maxSize:
            self.cacheDict.popitem(last=False)


class AdiosScalarArchive:
    """Unified API around ADIOS2 BP scalar fields across multiple runs.

    Supports two scalar layouts:
    - Per-step 2D array: [ny, nx] for each ADIOS step.
    - Packed 3D array: [step, ny, nx] stored in a single array.

    All ADIOS I/O in this class uses the `Stream` API:
    `for _ in stream.steps()`, `stream.current_step()`, `stream.available_variables()`,
    and `stream.read(...)`.
    """

    def __init__(
        self,
        runsDir: str = "runs",
        runDirs: Optional[Sequence[str]] = None,
        bpFile: str = "output.bp",
        scalarName: str = "rho",
        cacheSize: int = 64,
    ) -> None:
        self.runsDir = Path(runsDir).expanduser().resolve()
        self.runDirs = list(runDirs) if runDirs else None
        self.bpFile = bpFile
        self.scalarName = scalarName
        self.cache = LruArrayCache(cacheSize)
        self.runMetaByName: Dict[str, RunMeta] = {}
        self._discoverAndInspect()

    def listRuns(self) -> List[str]:
        """Return sorted run names discovered by this reader."""

        return sorted(self.runMetaByName.keys())

    def getStepCount(self, runName: str) -> int:
        """Return number of timesteps available for a run."""

        return self._requireRun(runName).stepCount

    def getGridShape(self, runName: str) -> Tuple[int, int]:
        """Return (ny, nx) for a run."""

        return self._requireRun(runName).gridShape

    def readScalar(self, runName: str, stepIdx: int) -> np.ndarray:
        """Read scalar snapshot for (runName, stepIdx) as float32 [ny, nx]."""

        runMeta = self._requireRun(runName)
        if stepIdx < 0 or stepIdx >= runMeta.stepCount:
            raise IndexError(
                f"Step {stepIdx} out of range for run '{runName}' "
                f"(0..{runMeta.stepCount - 1})."
            )

        cacheKey = (runName, int(stepIdx))
        cachedArray = self.cache.get(cacheKey)
        if cachedArray is not None:
            return np.array(cachedArray, copy=True)

        imageArray = self._readScalarUncached(runMeta, stepIdx)
        self.cache.put(cacheKey, imageArray)
        return np.array(imageArray, copy=True)

    def _discoverAndInspect(self) -> None:
        runPathByName = self._discoverRunPaths()
        if not runPathByName:
            raise RuntimeError(
                f"No run directories with '{self.bpFile}' found under '{self.runsDir}'."
            )

        for runName in sorted(runPathByName.keys()):
            bpPath = runPathByName[runName]
            self.runMetaByName[runName] = self._inspectRun(runName, bpPath)

    def _discoverRunPaths(self) -> Dict[str, Path]:
        runPathByName: Dict[str, Path] = {}

        if self.runDirs:
            for runEntry in self.runDirs:
                entryPath = Path(runEntry).expanduser()
                if not entryPath.is_absolute():
                    entryPath = (self.runsDir / entryPath).resolve()
                if entryPath.is_dir():
                    bpPath = entryPath / self.bpFile
                    runName = entryPath.name
                elif entryPath.is_file() and entryPath.name == self.bpFile:
                    bpPath = entryPath
                    runName = entryPath.parent.name
                else:
                    raise FileNotFoundError(
                        f"Run entry '{runEntry}' is neither a run directory nor '{self.bpFile}'."
                    )
                if not bpPath.exists():
                    raise FileNotFoundError(f"Missing BP file: {bpPath}")
                runPathByName[runName] = bpPath.resolve()
            return runPathByName

        if not self.runsDir.exists():
            raise FileNotFoundError(f"runsDir does not exist: {self.runsDir}")

        for subDir in sorted(self.runsDir.iterdir()):
            if not subDir.is_dir():
                continue
            bpPath = subDir / self.bpFile
            if bpPath.exists():
                runPathByName[subDir.name] = bpPath.resolve()
        return runPathByName

    def _inspectRun(self, runName: str, bpPath: Path) -> RunMeta:
        availableVars, streamStepCount = self._scanStreamMeta(bpPath)
        if self.scalarName not in availableVars:
            foundNames = ", ".join(sorted(availableVars.keys()))
            raise KeyError(
                f"Scalar '{self.scalarName}' not found in {bpPath}. "
                f"Available variables: {foundNames}"
            )

        varMeta = availableVars[self.scalarName]
        shapeDims = self._parseShape(varMeta.get("Shape", ""))
        availableStepCount = self._safeParseInt(
            varMeta.get("AvailableStepsCount", varMeta.get("StepsCount", "0"))
        )

        # ADIOS layout handling:
        # - [ny, nx] with multiple ADIOS steps: per_step_2d.
        # - [step, ny, nx] in one array with <=1 ADIOS step: packed_3d.
        if len(shapeDims) == 2:
            layout = "per_step_2d"
            gridShape = (shapeDims[0], shapeDims[1])
            stepCount = max(availableStepCount, streamStepCount, 1)
        elif len(shapeDims) == 3 and max(availableStepCount, streamStepCount) <= 1:
            layout = "packed_3d"
            stepCount = max(shapeDims[0], 1)
            gridShape = (shapeDims[1], shapeDims[2])
        else:
            layout, stepCount, gridShape = self._inferLayoutByProbe(
                bpPath=bpPath,
                shapeDims=shapeDims,
                availableStepCount=availableStepCount,
                streamStepCount=streamStepCount,
            )

        return RunMeta(
            runName=runName,
            bpPath=bpPath,
            layout=layout,
            stepCount=int(stepCount),
            gridShape=(int(gridShape[0]), int(gridShape[1])),
        )

    def _scanStreamMeta(self, bpPath: Path) -> Tuple[Dict[str, Dict[str, str]], int]:
        """Read per-stream metadata using Stream API only."""

        firstVars: Optional[Dict[str, Dict[str, str]]] = None
        stepCount = 0
        with Stream(str(bpPath), "r") as stream:
            for _ in stream.steps():
                stepCount += 1
                if firstVars is None:
                    firstVars = stream.available_variables()

        if firstVars is None:
            with Stream(str(bpPath), "r") as stream:
                try:
                    firstVars = stream.available_variables()
                except Exception:
                    firstVars = {}

        return firstVars or {}, stepCount

    def _inferLayoutByProbe(
        self,
        bpPath: Path,
        shapeDims: Sequence[int],
        availableStepCount: int,
        streamStepCount: int,
    ) -> Tuple[str, int, Tuple[int, int]]:
        if len(shapeDims) == 3:
            # Ambiguous metadata: default to packed_3d unless ADIOS reports multiple steps.
            if max(availableStepCount, streamStepCount) > 1:
                return "per_step_2d", max(availableStepCount, streamStepCount), (
                    shapeDims[-2],
                    shapeDims[-1],
                )
            return "packed_3d", shapeDims[0], (shapeDims[1], shapeDims[2])

        probeArray = self._readPerStep2D(bpPath=bpPath, stepIdx=0)
        if probeArray.ndim != 2:
            raise RuntimeError(
                f"Unable to infer scalar layout for '{bpPath}'. Expected 2D probe, got {probeArray.shape}."
            )
        return "per_step_2d", max(availableStepCount, streamStepCount, 1), (
            int(probeArray.shape[0]),
            int(probeArray.shape[1]),
        )

    def _readScalarUncached(self, runMeta: RunMeta, stepIdx: int) -> np.ndarray:
        if runMeta.layout == "per_step_2d":
            return self._readPerStep2D(runMeta.bpPath, stepIdx)
        if runMeta.layout == "packed_3d":
            return self._readPacked3D(runMeta.bpPath, stepIdx, runMeta.gridShape)
        raise RuntimeError(f"Unknown layout '{runMeta.layout}' for run '{runMeta.runName}'.")

    def _readPerStep2D(self, bpPath: Path, stepIdx: int) -> np.ndarray:
        with Stream(str(bpPath), "r") as stream:
            for _ in stream.steps():
                if int(stream.current_step()) != int(stepIdx):
                    continue
                imageArray = np.asarray(stream.read(self.scalarName), dtype=np.float32)
                if imageArray.ndim == 3 and imageArray.shape[0] == 1:
                    imageArray = imageArray[0]
                if imageArray.ndim != 2:
                    raise RuntimeError(
                        f"Expected 2D scalar array, got shape {tuple(imageArray.shape)}."
                    )
                return imageArray

        raise RuntimeError(
            f"Failed to read per-step scalar '{self.scalarName}' at step {stepIdx} from {bpPath}."
        )

    def _readPacked3D(self, bpPath: Path, stepIdx: int, gridShape: Tuple[int, int]) -> np.ndarray:
        with Stream(str(bpPath), "r") as stream:
            for _ in stream.steps():
                # Preferred path: read only one [1, ny, nx] slab.
                try:
                    slabArray = stream.read(
                        self.scalarName,
                        start=[stepIdx, 0, 0],
                        count=[1, gridShape[0], gridShape[1]],
                    )
                    slabArray = np.asarray(slabArray, dtype=np.float32)
                    if slabArray.ndim == 3 and slabArray.shape[0] == 1:
                        return slabArray[0]
                    if slabArray.ndim == 2:
                        return slabArray
                except Exception:
                    pass

                packedArray = np.asarray(stream.read(self.scalarName), dtype=np.float32)
                if packedArray.ndim == 3:
                    if stepIdx >= packedArray.shape[0]:
                        raise IndexError(
                            f"Packed scalar has {packedArray.shape[0]} steps, requested {stepIdx}."
                        )
                    return packedArray[stepIdx]
                if packedArray.ndim == 2 and stepIdx == 0:
                    return packedArray
                raise RuntimeError(
                    f"Expected packed [step, ny, nx] array, got shape {tuple(packedArray.shape)} from {bpPath}."
                )

        raise RuntimeError(f"Unable to read packed scalar '{self.scalarName}' from {bpPath}.")

    def _countStepsByIteration(self, bpPath: Path) -> int:
        stepCount = 0
        with Stream(str(bpPath), "r") as stream:
            for _ in stream.steps():
                stepCount += 1
        return stepCount

    def _requireRun(self, runName: str) -> RunMeta:
        if runName not in self.runMetaByName:
            knownRuns = ", ".join(self.listRuns())
            raise KeyError(f"Unknown run '{runName}'. Known runs: {knownRuns}")
        return self.runMetaByName[runName]

    @staticmethod
    def _parseShape(shapeText: str) -> List[int]:
        cleanedText = shapeText.strip()
        if not cleanedText:
            return []
        return [int(part.strip()) for part in cleanedText.split(",") if part.strip()]

    @staticmethod
    def _safeParseInt(valueText: str) -> int:
        try:
            return int(str(valueText).strip())
        except Exception:
            return 0
