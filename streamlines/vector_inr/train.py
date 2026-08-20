"""Training loop for vector INR models."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from .adios_dataset import MultiEnsembleDataset
from .losses import computeAutogradVorticity, velocityL1Loss, vorticityL1Loss
from .models import buildVectorInrModel, getModelMetadata
from .sampling import PointSampler
from .utils import (
    countParameters,
    dumpJson,
    ensureDirectory,
    resolveDevice,
    saveCheckpoint,
    setSeeds,
)


logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """All training options."""

    ensembleDirs: List[str]
    bpFile: str = "output.bp"
    vxName: str = "vx"
    vyName: str = "vy"
    model: str = "fourier"
    epochs: int = 200
    stepsPerEpoch: int = 200
    batchPoints: int = 65_536
    embedDim: int = 32
    hidden: int = 256
    layers: int = 6
    freqs: int = 10
    sirenOmega0: float = 30.0
    lr: float = 1e-3
    minLr: float = 1e-5
    weightDecay: float = 1e-4
    cosineSchedule: bool = False
    vorticityLoss: bool = False
    lambdaOmega: float = 0.1
    warmupEpochs: int = 20
    cacheSteps: int = 128
    omegaCacheSteps: int = 128
    normStepsPerEnsemble: int = 8
    normPointsPerStep: int = 4096
    checkpointEvery: int = 10
    outdir: str = "runs/ot_inr"
    device: str = "auto"
    seed: int = 0
    deterministic: bool = True


def _validateTrainConfig(config: TrainConfig) -> None:
    if len(config.ensembleDirs) == 0:
        raise ValueError("At least one --ensemble_dirs path is required.")
    if config.epochs <= 0:
        raise ValueError("--epochs must be > 0.")
    if config.stepsPerEpoch <= 0:
        raise ValueError("--steps_per_epoch must be > 0.")
    if config.batchPoints <= 0:
        raise ValueError("--batch_points must be > 0.")
    if config.embedDim <= 0:
        raise ValueError("--embed_dim must be > 0.")
    if config.hidden <= 0:
        raise ValueError("--hidden must be > 0.")
    if config.layers < 0:
        raise ValueError("--layers must be >= 0.")
    if config.freqs < 0:
        raise ValueError("--freqs must be >= 0.")
    if config.lr <= 0:
        raise ValueError("--lr must be > 0.")
    if config.weightDecay < 0:
        raise ValueError("--weight_decay must be >= 0.")
    if config.checkpointEvery <= 0:
        raise ValueError("--checkpoint_every must be > 0.")
    if config.lambdaOmega < 0:
        raise ValueError("--lambda_omega must be >= 0.")
    if config.warmupEpochs < 0:
        raise ValueError("--warmup_epochs must be >= 0.")


def runTraining(config: TrainConfig) -> Path:
    """Run full training and return last checkpoint path."""

    _validateTrainConfig(config)
    setSeeds(config.seed, deterministic=config.deterministic)
    device = resolveDevice(config.device)
    outputDir = ensureDirectory(config.outdir)

    logger.info("Using device: %s", device)
    logger.info("Building dataset from %d ensemble(s).", len(config.ensembleDirs))
    dataset = MultiEnsembleDataset(
        ensembleDirs=config.ensembleDirs,
        bpFile=config.bpFile,
        vxName=config.vxName,
        vyName=config.vyName,
        cacheSteps=config.cacheSteps,
        omegaCacheSteps=config.omegaCacheSteps,
        requireSameGrid=True,
    )

    gridShape = dataset.getGridShape(0)
    stepCounts = dataset.getStepCounts()
    logger.info("Grid shape: ny=%d nx=%d", gridShape[0], gridShape[1])
    logger.info("Step counts per ensemble: %s", stepCounts)

    normalizationStats = dataset.computeNormalizationStats(
        sampleStepsPerEnsemble=config.normStepsPerEnsemble,
        samplePointsPerStep=config.normPointsPerStep,
        seed=config.seed,
    )
    dataset.setNormalization(mean=normalizationStats.mean, std=normalizationStats.std)

    sampler = PointSampler(
        dataset=dataset,
        normalizationMean=normalizationStats.mean,
        normalizationStd=normalizationStats.std,
        batchPoints=config.batchPoints,
        device=device,
        seed=config.seed + 17,
    )

    ensembleCount = dataset.getEnsembleCount()
    model = buildVectorInrModel(
        modelName=config.model,
        ensembleCount=ensembleCount,
        embedDim=config.embedDim,
        hiddenDim=config.hidden,
        hiddenLayers=config.layers,
        numFrequencies=config.freqs,
        sirenOmega0=config.sirenOmega0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weightDecay
    )
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]
    if config.cosineSchedule:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=config.epochs,
            eta_min=config.minLr,
        )
    else:
        scheduler = None

    trainConfigPath = outputDir / "train_config.json"
    dumpJson(asdict(config), trainConfigPath)

    modelMeta = getModelMetadata(
        modelName=config.model,
        ensembleCount=ensembleCount,
        embedDim=config.embedDim,
        hiddenDim=config.hidden,
        hiddenLayers=config.layers,
        numFrequencies=config.freqs,
        sirenOmega0=config.sirenOmega0,
    )

    logger.info("Model has %d trainable parameters.", countParameters(model))
    logger.info("Starting training for %d epoch(s).", config.epochs)

    latestCheckpointPath = outputDir / "checkpoint_latest.pt"
    lastEpochCheckpointPath = latestCheckpointPath

    for epoch in range(1, config.epochs + 1):
        model.train()
        startTime = time.time()
        useVorticity = config.vorticityLoss and epoch > config.warmupEpochs
        epochVelocityLoss = 0.0
        epochOmegaLoss = 0.0
        epochTotalLoss = 0.0

        for _ in range(config.stepsPerEpoch):
            batch = sampler.sampleBatch(includeVorticity=useVorticity)
            coords = batch["coords"]
            ensembleIndices = batch["ensembleIndices"].long()
            targetVelocity = batch["targetVelocity"]

            if useVorticity:
                coords = coords.clone().detach().requires_grad_(True)

            predVelocity = model(coords, ensembleIndices)
            velocityLossValue = velocityL1Loss(predVelocity, targetVelocity)
            totalLoss = velocityLossValue
            omegaLossValue = None

            if useVorticity:
                targetOmega = batch["targetOmega"]
                predOmega = computeAutogradVorticity(predVelocity, coords, createGraph=True)
                omegaLossValue = vorticityL1Loss(predOmega, targetOmega)
                totalLoss = totalLoss + config.lambdaOmega * omegaLossValue

            optimizer.zero_grad(set_to_none=True)
            totalLoss.backward()
            optimizer.step()

            epochVelocityLoss += float(velocityLossValue.item())
            epochTotalLoss += float(totalLoss.item())
            if omegaLossValue is not None:
                epochOmegaLoss += float(omegaLossValue.item())

        if scheduler is not None:
            scheduler.step()

        epochVelocityLoss /= float(config.stepsPerEpoch)
        epochTotalLoss /= float(config.stepsPerEpoch)
        if useVorticity:
            epochOmegaLoss /= float(config.stepsPerEpoch)

        duration = time.time() - startTime
        currentLr = float(optimizer.param_groups[0]["lr"])
        logger.info(
            "Epoch %d/%d | total=%.6f | vel=%.6f | omega=%s | lr=%.6e | %.2fs",
            epoch,
            config.epochs,
            epochTotalLoss,
            epochVelocityLoss,
            f"{epochOmegaLoss:.6f}" if useVorticity else "disabled",
            currentLr,
            duration,
        )

        if epoch % config.checkpointEvery == 0 or epoch == config.epochs:
            epochCheckpointPath = outputDir / f"checkpoint_epoch_{epoch:04d}.pt"
            checkpointData = {
                "epoch": epoch,
                "trainConfig": asdict(config),
                "modelMeta": modelMeta,
                "normalization": {
                    "mean": normalizationStats.mean.tolist(),
                    "std": normalizationStats.std.tolist(),
                },
                "gridShape": list(gridShape),
                "stepCounts": stepCounts,
            }
            saveCheckpoint(
                checkpointPath=epochCheckpointPath,
                epoch=epoch,
                modelState=model.state_dict(),
                optimizerState=optimizer.state_dict(),
                schedulerState=scheduler.state_dict() if scheduler is not None else None,
                trainConfig=checkpointData["trainConfig"],
                normalization=checkpointData["normalization"],
                modelMeta=modelMeta,
            )
            saveCheckpoint(
                checkpointPath=latestCheckpointPath,
                epoch=epoch,
                modelState=model.state_dict(),
                optimizerState=optimizer.state_dict(),
                schedulerState=scheduler.state_dict() if scheduler is not None else None,
                trainConfig=checkpointData["trainConfig"],
                normalization=checkpointData["normalization"],
                modelMeta=modelMeta,
            )
            metaPath = outputDir / f"checkpoint_epoch_{epoch:04d}.json"
            dumpJson(checkpointData, metaPath)
            lastEpochCheckpointPath = epochCheckpointPath
            logger.info("Saved checkpoint: %s", epochCheckpointPath)

    dataset.close()
    logger.info("Training finished. Latest checkpoint: %s", latestCheckpointPath)
    return lastEpochCheckpointPath
