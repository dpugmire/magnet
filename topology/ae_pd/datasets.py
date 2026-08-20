"""PyTorch dataset and sampling helpers for ADIOS snapshots."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .adios_dataset import AdiosScalarArchive

SampleKey = Tuple[str, int]


def BuildAllSamples(archive: AdiosScalarArchive) -> List[SampleKey]:
    """Create a flat list of (runName, stepIdx) pairs."""

    sampleList: List[SampleKey] = []
    for runName in archive.listRuns():
        stepCount = archive.getStepCount(runName)
        for stepIdx in range(stepCount):
            sampleList.append((runName, stepIdx))
    return sampleList


def SplitSamplesByRun(
    sampleList: Sequence[SampleKey],
    valFraction: float,
    seed: int,
) -> Tuple[List[SampleKey], List[SampleKey]]:
    """Split train/val by holding out entire runs."""

    runNameList = sorted({runName for runName, _ in sampleList})
    if len(runNameList) <= 1 or valFraction <= 0.0:
        return list(sampleList), []

    rng = random.Random(seed)
    shuffledRuns = runNameList.copy()
    rng.shuffle(shuffledRuns)

    valRunCount = max(1, int(round(len(shuffledRuns) * valFraction)))
    if valRunCount >= len(shuffledRuns):
        valRunCount = len(shuffledRuns) - 1

    valRunSet = set(shuffledRuns[:valRunCount])
    trainList: List[SampleKey] = []
    valList: List[SampleKey] = []
    for runName, stepIdx in sampleList:
        if runName in valRunSet:
            valList.append((runName, stepIdx))
        else:
            trainList.append((runName, stepIdx))
    return trainList, valList


def SplitSamplesRandom(
    sampleList: Sequence[SampleKey],
    valFraction: float,
    seed: int,
) -> Tuple[List[SampleKey], List[SampleKey]]:
    """Split train/val by random snapshot sampling."""

    if not sampleList:
        return [], []
    if valFraction <= 0.0:
        return list(sampleList), []

    rng = random.Random(seed)
    shuffledList = list(sampleList)
    rng.shuffle(shuffledList)

    valCount = max(1, int(round(len(shuffledList) * valFraction)))
    if valCount >= len(shuffledList):
        valCount = len(shuffledList) - 1
    if valCount <= 0:
        return shuffledList, []
    return shuffledList[valCount:], shuffledList[:valCount]


def NormalizeImage(
    imageArray: np.ndarray,
    normMode: str,
    globalMean: float,
    globalStd: float,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Normalize one 2D image according to configured mode."""

    imageFloat = imageArray.astype(np.float32, copy=False)
    if normMode == "none":
        return imageFloat
    if normMode == "per_image":
        meanValue = float(imageFloat.mean())
        stdValue = float(imageFloat.std())
        if stdValue < epsilon:
            stdValue = 1.0
        return (imageFloat - meanValue) / stdValue
    if normMode == "global":
        stdValue = float(globalStd)
        if stdValue < epsilon:
            stdValue = 1.0
        return (imageFloat - float(globalMean)) / stdValue
    raise ValueError(f"Unsupported normMode: {normMode}")


class SnapshotDataset(Dataset):
    """Dataset yielding full scalar images and sample identifiers."""

    def __init__(
        self,
        archive: AdiosScalarArchive,
        sampleList: Sequence[SampleKey],
        normMode: str = "none",
        globalMean: float = 0.0,
        globalStd: float = 1.0,
    ) -> None:
        self.archive = archive
        self.sampleList = list(sampleList)
        self.normMode = normMode
        self.globalMean = float(globalMean)
        self.globalStd = float(globalStd)

    def __len__(self) -> int:
        return len(self.sampleList)

    def __getitem__(self, index: int) -> Dict[str, object]:
        runName, stepIdx = self.sampleList[index]
        imageArray = self.archive.readScalar(runName, stepIdx)
        imageNorm = NormalizeImage(
            imageArray=imageArray,
            normMode=self.normMode,
            globalMean=self.globalMean,
            globalStd=self.globalStd,
        )
        imageTensor = torch.from_numpy(imageNorm[None, :, :].astype(np.float32, copy=False))
        return {
            "runName": runName,
            "stepIdx": int(stepIdx),
            "image": imageTensor,
        }
