"""Training orchestration for convolutional AE + PD loss."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .adios_dataset import AdiosScalarArchive
from .datasets import BuildAllSamples, SnapshotDataset, SplitSamplesByRun, SplitSamplesRandom
from .losses import ComputeReconLoss, PdLossWrapper
from .models import ConvAutoencoder
from .utils import CreateLogger, EnsureDir, SaveCheckpoint, SaveJson, SetSeed


def EstimateGlobalNorm(
    archive: AdiosScalarArchive,
    sampleList: Sequence[Tuple[str, int]],
    normSamples: int,
    seed: int,
) -> Tuple[float, float]:
    """Estimate global mean/std from random snapshot subset."""

    if not sampleList:
        return 0.0, 1.0

    rng = np.random.default_rng(seed)
    maxCount = len(sampleList)
    useCount = maxCount if normSamples <= 0 else min(normSamples, maxCount)
    chosenIndices = rng.choice(maxCount, size=useCount, replace=False)

    totalCount = 0
    totalSum = 0.0
    totalSumSq = 0.0
    for sampleIdx in chosenIndices:
        runName, stepIdx = sampleList[int(sampleIdx)]
        imageArray = archive.readScalar(runName, int(stepIdx)).astype(np.float64, copy=False)
        totalCount += int(imageArray.size)
        totalSum += float(imageArray.sum())
        totalSumSq += float(np.square(imageArray).sum())

    if totalCount <= 0:
        return 0.0, 1.0

    meanValue = totalSum / totalCount
    varianceValue = max(totalSumSq / totalCount - meanValue * meanValue, 1e-12)
    stdValue = float(np.sqrt(varianceValue))
    return float(meanValue), float(stdValue)


def ResolveTrainingOutDir(configDict: Dict[str, Any], usePd: bool) -> Path:
    """Resolve a non-destructive training output directory."""

    basePath = Path(configDict["outdir"]).expanduser().resolve()
    modeSubdir = bool(configDict.get("mode_subdir", True))
    allowOverwrite = bool(configDict.get("allow_overwrite", False))
    modeName = "with_pd" if usePd else "no_pd"
    candidatePath = (basePath / modeName) if modeSubdir else basePath

    if not allowOverwrite:
        hasExistingArtifacts = any(
            (candidatePath / fileName).exists()
            for fileName in ["best.pt", "last.pt", "config.json", "history.json"]
        )
        if hasExistingArtifacts:
            timeTag = datetime.now().strftime("%Y%m%d_%H%M%S")
            if modeSubdir:
                candidatePath = basePath / f"{modeName}_{timeTag}"
            else:
                candidatePath = basePath.parent / f"{basePath.name}_{modeName}_{timeTag}"

    return EnsureDir(candidatePath)


def _EvaluateEpoch(
    model: ConvAutoencoder,
    valLoader: DataLoader,
    device: torch.device,
    useAmp: bool,
    lambdaPd: float,
    pdEvery: int,
    pdBatchItems: int,
    pdLossWrapper: PdLossWrapper | None,
    usePd: bool,
) -> Dict[str, float]:
    model.eval()
    reconSum = 0.0
    totalSum = 0.0
    pdSum = 0.0
    pdCount = 0
    batchCount = 0

    with torch.no_grad():
        for batchIdx, batchItem in enumerate(valLoader):
            imageTensor = batchItem["image"].to(device=device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=useAmp):
                reconTensor = model(imageTensor)
                reconLoss = ComputeReconLoss(reconTensor, imageTensor)

            shouldComputePd = usePd and (lambdaPd > 0.0) and (pdEvery > 0) and (batchIdx % pdEvery == 0)
            if shouldComputePd:
                if pdLossWrapper is None:
                    raise RuntimeError("pdLossWrapper is required when PD is enabled.")
                pdLoss = pdLossWrapper.ComputeLoss(
                    reconBatch=reconTensor,
                    targetBatch=imageTensor,
                    maxBatchItems=pdBatchItems,
                )
                pdSum += float(pdLoss.item())
                pdCount += 1
            else:
                pdLoss = torch.zeros((), device=device, dtype=torch.float32)

            totalLoss = reconLoss + lambdaPd * pdLoss
            reconSum += float(reconLoss.item())
            totalSum += float(totalLoss.item())
            batchCount += 1

    if batchCount == 0:
        return {"recon": float("nan"), "pd": float("nan"), "total": float("nan"), "pdCount": 0.0}

    return {
        "recon": reconSum / batchCount,
        "pd": (pdSum / pdCount) if pdCount > 0 else 0.0,
        "total": totalSum / batchCount,
        "pdCount": float(pdCount),
    }


def TrainAutoencoder(configDict: Dict[str, Any]) -> Dict[str, Any]:
    """Run end-to-end AE+PD training and checkpointing."""

    usePd = bool(configDict.get("use_pd", True))
    outDir = ResolveTrainingOutDir(configDict=configDict, usePd=usePd)
    logger = CreateLogger(outDir)
    SetSeed(int(configDict["seed"]))
    logger.info("Resolved output directory: %s", str(outDir))

    logger.info("Building ADIOS archive...")
    archive = AdiosScalarArchive(
        runsDir=configDict["runs_dir"],
        runDirs=configDict.get("run_dirs"),
        bpFile=configDict["bp_file"],
        scalarName=configDict["scalar_name"],
        cacheSize=int(configDict["cache_size"]),
    )
    runList = archive.listRuns()
    logger.info("Discovered %d runs: %s", len(runList), ", ".join(runList))

    allSamples = BuildAllSamples(archive)
    if len(allSamples) < 2:
        raise RuntimeError("Need at least 2 snapshots to train/validate.")

    if bool(configDict["split_by_run"]):
        trainSamples, valSamples = SplitSamplesByRun(
            sampleList=allSamples,
            valFraction=float(configDict["val_fraction"]),
            seed=int(configDict["seed"]),
        )
    else:
        trainSamples, valSamples = SplitSamplesRandom(
            sampleList=allSamples,
            valFraction=float(configDict["val_fraction"]),
            seed=int(configDict["seed"]),
        )
    if not trainSamples:
        raise RuntimeError("Train split is empty after sampling.")

    logger.info("Train snapshots: %d | Val snapshots: %d", len(trainSamples), len(valSamples))

    normMode = str(configDict["norm"])
    globalMean = 0.0
    globalStd = 1.0
    if normMode == "global":
        logger.info("Estimating global normalization stats from %d samples...", int(configDict["norm_samples"]))
        globalMean, globalStd = EstimateGlobalNorm(
            archive=archive,
            sampleList=trainSamples,
            normSamples=int(configDict["norm_samples"]),
            seed=int(configDict["seed"]),
        )
        logger.info("Global mean/std: %.6e / %.6e", globalMean, globalStd)

    normStats = {
        "mode": normMode,
        "globalMean": float(globalMean),
        "globalStd": float(globalStd),
    }

    trainDataset = SnapshotDataset(
        archive=archive,
        sampleList=trainSamples,
        normMode=normMode,
        globalMean=globalMean,
        globalStd=globalStd,
    )
    valDataset = SnapshotDataset(
        archive=archive,
        sampleList=valSamples,
        normMode=normMode,
        globalMean=globalMean,
        globalStd=globalStd,
    )

    trainLoader = DataLoader(
        trainDataset,
        batch_size=int(configDict["batch_size"]),
        shuffle=True,
        num_workers=int(configDict["num_workers"]),
        pin_memory=bool(configDict["pin_memory"]),
    )
    valLoader = DataLoader(
        valDataset,
        batch_size=int(configDict["batch_size"]),
        shuffle=False,
        num_workers=int(configDict["num_workers"]),
        pin_memory=bool(configDict["pin_memory"]),
    )

    device = torch.device(configDict["device"])
    useAmp = bool(configDict["amp"]) and device.type == "cuda"
    logger.info("Using device=%s amp=%s", str(device), str(useAmp))

    model = ConvAutoencoder(
        baseChannels=int(configDict["base_channels"]),
        numDown=int(configDict["num_down"]),
        latentChannels=int(configDict["latent_channels"]),
        activation=str(configDict["activation"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(configDict["lr"]),
        weight_decay=float(configDict["weight_decay"]),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=useAmp)

    pdLossWrapper: PdLossWrapper | None = None
    if usePd:
        pdLossWrapper = PdLossWrapper(
            pdDims=configDict["pd_dims"],
            pdDownsample=int(configDict["pd_downsample"]),
            piRes=int(configDict["pi_res"]),
            piSigma=float(configDict["pi_sigma"]),
            pdLossType=str(configDict["pd_loss"]),
            weightByPersistence=bool(configDict["pd_weight_persistence"]),
            minPersistence=float(configDict.get("pd_min_persistence", 0.0)),
        )

    lambdaPd = float(configDict["lambda_pd"])
    pdEvery = int(configDict["pd_every"])
    pdBatchItems = int(configDict["pd_batch_items"])
    epochCount = int(configDict["epochs"])
    if not usePd:
        lambdaPd = 0.0
        pdEvery = 0
    logger.info("PD training term enabled: %s (lambda_pd=%.6f, pd_every=%d)", str(usePd), lambdaPd, pdEvery)

    configToSave = dict(configDict)
    configToSave["outdir"] = str(outDir)
    configToSave["norm_stats"] = normStats
    SaveJson(configToSave, Path(outDir) / "config.json")

    bestValTotal = float("inf")
    historyList: List[Dict[str, float]] = []
    globalStep = 0

    for epochIdx in range(1, epochCount + 1):
        model.train()
        trainReconSum = 0.0
        trainPdSum = 0.0
        trainTotalSum = 0.0
        trainPdCount = 0
        trainBatchCount = 0

        for batchItem in trainLoader:
            imageTensor = batchItem["image"].to(device=device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=useAmp):
                reconTensor = model(imageTensor)
                reconLoss = ComputeReconLoss(reconTensor, imageTensor)

            shouldComputePd = usePd and (lambdaPd > 0.0) and (pdEvery > 0) and (globalStep % pdEvery == 0)
            if shouldComputePd:
                if pdLossWrapper is None:
                    raise RuntimeError("pdLossWrapper is required when PD is enabled.")
                pdLoss = pdLossWrapper.ComputeLoss(
                    reconBatch=reconTensor,
                    targetBatch=imageTensor,
                    maxBatchItems=pdBatchItems,
                )
                trainPdSum += float(pdLoss.item())
                trainPdCount += 1
            else:
                pdLoss = torch.zeros((), device=device, dtype=torch.float32)

            totalLoss = reconLoss + lambdaPd * pdLoss
            scaler.scale(totalLoss).backward()
            scaler.step(optimizer)
            scaler.update()

            trainReconSum += float(reconLoss.item())
            trainTotalSum += float(totalLoss.item())
            trainBatchCount += 1
            globalStep += 1

        trainReconAvg = trainReconSum / max(trainBatchCount, 1)
        trainPdAvg = (trainPdSum / trainPdCount) if trainPdCount > 0 else 0.0
        trainTotalAvg = trainTotalSum / max(trainBatchCount, 1)

        if len(valDataset) > 0:
            valMetrics = _EvaluateEpoch(
                model=model,
                valLoader=valLoader,
                device=device,
                useAmp=useAmp,
                lambdaPd=lambdaPd,
                pdEvery=pdEvery,
                pdBatchItems=pdBatchItems,
                pdLossWrapper=pdLossWrapper,
                usePd=usePd,
            )
            valReconAvg = float(valMetrics["recon"])
            valPdAvg = float(valMetrics["pd"])
            valTotalAvg = float(valMetrics["total"])
        else:
            valReconAvg = trainReconAvg
            valPdAvg = trainPdAvg
            valTotalAvg = trainTotalAvg

        historyEntry = {
            "epoch": float(epochIdx),
            "trainRecon": trainReconAvg,
            "trainPd": trainPdAvg,
            "trainTotal": trainTotalAvg,
            "valRecon": valReconAvg,
            "valPd": valPdAvg,
            "valTotal": valTotalAvg,
        }
        historyList.append(historyEntry)

        checkpointState = {
            "epoch": epochIdx,
            "globalStep": globalStep,
            "modelState": model.state_dict(),
            "optimizerState": optimizer.state_dict(),
            "bestValTotal": bestValTotal,
            "config": configToSave,
            "normStats": normStats,
            "history": historyList,
        }
        SaveCheckpoint(checkpointState, Path(outDir) / "last.pt")

        if np.isfinite(valTotalAvg) and valTotalAvg < bestValTotal:
            bestValTotal = float(valTotalAvg)
            checkpointState["bestValTotal"] = bestValTotal
            SaveCheckpoint(checkpointState, Path(outDir) / "best.pt")

        logger.info(
            "Epoch %d/%d | train(recon=%.6f pd=%.6f total=%.6f) | "
            "val(recon=%.6f pd=%.6f total=%.6f) | best=%.6f",
            epochIdx,
            epochCount,
            trainReconAvg,
            trainPdAvg,
            trainTotalAvg,
            valReconAvg,
            valPdAvg,
            valTotalAvg,
            bestValTotal,
        )

    SaveJson({"history": historyList}, Path(outDir) / "history.json")
    logger.info("Training complete. Outputs written to %s", str(outDir))
    return {
        "outdir": str(outDir),
        "bestCheckpoint": str((Path(outDir) / "best.pt").resolve()),
        "lastCheckpoint": str((Path(outDir) / "last.pt").resolve()),
        "bestValTotal": bestValTotal,
    }
