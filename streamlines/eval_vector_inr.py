#!/usr/bin/env python3
"""CLI inference/evaluation tool for trained vector INR checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from vector_inr.adios_dataset import MultiEnsembleDataset, computeFiniteDifferenceVorticity
from vector_inr.models import buildVectorInrModel
from vector_inr.sampling import buildFullGridQuery
from vector_inr.utils import ensureDirectory, loadCheckpoint, resolveDevice, setupLogging


logger = logging.getLogger(__name__)


def buildArgParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained vector INR checkpoint on selected ensemble step(s)."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ensemble_dirs", dest="ensembleDirs", nargs="+", required=True)
    parser.add_argument("--bp_file", dest="bpFile", default="output.bp")
    parser.add_argument("--vx_name", dest="vxName", default="vx")
    parser.add_argument("--vy_name", dest="vyName", default="vy")
    parser.add_argument("--ensemble_idx", dest="ensembleIdx", type=int, default=0)

    parser.add_argument("--step", type=int, default=None, help="Evaluate exactly one step.")
    parser.add_argument("--step_start", dest="stepStart", type=int, default=None)
    parser.add_argument("--step_end", dest="stepEnd", type=int, default=None)
    parser.add_argument("--step_stride", dest="stepStride", type=int, default=1)

    parser.add_argument("--outdir", default="runs/eval_vector_inr")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--chunk_size", dest="chunkSize", type=int, default=262_144)
    parser.add_argument("--vorticity_metric", dest="vorticityMetric", action="store_true")
    parser.add_argument("--save_png", dest="savePng", action="store_true")
    parser.add_argument(
        "--compare_png",
        dest="comparePng",
        action="store_true",
        help="Save side-by-side GroundTruth vs Prediction PNGs.",
    )
    parser.add_argument(
        "--overlay_png",
        dest="overlayPng",
        action="store_true",
        help="Save a single overlay streamplot with GT and prediction in different colors.",
    )
    parser.add_argument(
        "--plot_mode",
        dest="plotMode",
        choices=["quiver", "streamplot", "both"],
        default="both",
    )
    parser.add_argument("--gt_color", dest="gtColor", default="tab:blue")
    parser.add_argument("--pred_color", dest="predColor", default="tab:orange")
    parser.add_argument(
        "--stream_direction",
        dest="streamDirection",
        choices=["both", "forward", "backward"],
        default="both",
        help="Integration direction for streamlines.",
    )
    parser.add_argument(
        "--show_seed_points",
        dest="showSeedPoints",
        action="store_true",
        help="Draw seed locations as markers on PNG plots.",
    )
    parser.add_argument(
        "--seed_step_length",
        dest="seedStepLength",
        type=float,
        default=4.0,
        help="Length (in index units) of the seed-to-first-step segment; <=0 disables it.",
    )
    parser.add_argument(
        "--stream_density",
        dest="streamDensity",
        type=float,
        default=1.2,
        help="matplotlib streamplot density when explicit seed grid is not set.",
    )
    parser.add_argument(
        "--streamline_seed_nx",
        dest="streamlineSeedNx",
        type=int,
        default=0,
        help="Number of x-direction streamline seeds (requires --streamline_seed_ny).",
    )
    parser.add_argument(
        "--streamline_seed_ny",
        dest="streamlineSeedNy",
        type=int,
        default=0,
        help="Number of y-direction streamline seeds (requires --streamline_seed_nx).",
    )
    parser.add_argument(
        "--random_seeds",
        dest="randomSeeds",
        type=int,
        default=0,
        help="Use N random streamline seed points (mutually exclusive with grid seeds).",
    )
    parser.add_argument("--plot_dpi", dest="plotDpi", type=int, default=150)
    parser.add_argument(
        "--max_quiver_vectors",
        dest="maxQuiverVectors",
        type=int,
        default=32,
        help="Approximate max vectors along one axis in quiver plot.",
    )
    parser.add_argument("--cache_steps", dest="cacheSteps", type=int, default=8)
    parser.add_argument("--omega_cache_steps", dest="omegaCacheSteps", type=int, default=8)
    parser.add_argument("--log_level", dest="logLevel", default="INFO")
    return parser


def selectSteps(
    stepCount: int,
    step: Optional[int],
    stepStart: Optional[int],
    stepEnd: Optional[int],
    stepStride: int,
) -> List[int]:
    if stepStride <= 0:
        raise ValueError("--step_stride must be > 0.")

    if step is not None:
        if step < 0 or step >= stepCount:
            raise IndexError(f"--step={step} is out of range [0, {stepCount - 1}].")
        return [step]

    startValue = 0 if stepStart is None else stepStart
    endValue = (stepCount - 1) if stepEnd is None else stepEnd
    if startValue < 0 or startValue >= stepCount:
        raise IndexError(f"--step_start={startValue} is out of range [0, {stepCount - 1}].")
    if endValue < 0 or endValue >= stepCount:
        raise IndexError(f"--step_end={endValue} is out of range [0, {stepCount - 1}].")
    if endValue < startValue:
        raise ValueError("--step_end must be >= --step_start.")
    return list(range(startValue, endValue + 1, stepStride))


def inferVelocityGrid(
    model: torch.nn.Module,
    coords: torch.Tensor,
    ensembleIndices: torch.Tensor,
    chunkSize: int,
) -> torch.Tensor:
    if chunkSize <= 0:
        raise ValueError("--chunk_size must be > 0.")
    outputs = []
    totalPoints = coords.shape[0]
    with torch.no_grad():
        for startIdx in range(0, totalPoints, chunkSize):
            endIdx = min(startIdx + chunkSize, totalPoints)
            chunkPred = model(coords[startIdx:endIdx], ensembleIndices[startIdx:endIdx])
            outputs.append(chunkPred.detach().cpu())
    return torch.cat(outputs, dim=0)


def computeAngularErrorDeg(predVelocity: np.ndarray, gtVelocity: np.ndarray) -> float:
    predFlat = predVelocity.reshape(-1, 2)
    gtFlat = gtVelocity.reshape(-1, 2)
    dotProducts = np.sum(predFlat * gtFlat, axis=1)
    predNorm = np.linalg.norm(predFlat, axis=1)
    gtNorm = np.linalg.norm(gtFlat, axis=1)
    denominator = predNorm * gtNorm

    angleValues = np.zeros_like(denominator, dtype=np.float64)
    validMask = denominator > 1e-12
    cosineValues = np.clip(dotProducts[validMask] / denominator[validMask], -1.0, 1.0)
    angleValues[validMask] = np.degrees(np.arccos(cosineValues))
    return float(np.mean(angleValues))


def buildStreamStartPoints(
    nx: int,
    ny: int,
    seedNx: int,
    seedNy: int,
    randomSeeds: int = 0,
    rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """Create an explicit start_points grid for streamplot, or None for automatic seeding."""

    if randomSeeds > 0:
        localRng = np.random.default_rng() if rng is None else rng
        xSeeds = localRng.uniform(0.0, float(nx - 1), size=randomSeeds).astype(np.float32)
        ySeeds = localRng.uniform(0.0, float(ny - 1), size=randomSeeds).astype(np.float32)
        return np.stack([xSeeds, ySeeds], axis=1)

    if seedNx == 0 and seedNy == 0:
        return None
    if seedNx <= 0 or seedNy <= 0:
        raise ValueError(
            "--streamline_seed_nx and --streamline_seed_ny must both be > 0 when set."
        )
    xSeeds = np.linspace(0.0, float(nx - 1), num=seedNx, dtype=np.float32)
    ySeeds = np.linspace(0.0, float(ny - 1), num=seedNy, dtype=np.float32)
    xGrid, yGrid = np.meshgrid(xSeeds, ySeeds, indexing="xy")
    startPoints = np.stack([xGrid.reshape(-1), yGrid.reshape(-1)], axis=1)
    return startPoints


def bilinearSample2d(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Bilinear sampling of a 2D field at floating-point (x,y) locations."""

    if field.ndim != 2:
        raise ValueError(f"Expected 2D field, got shape {field.shape}.")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected points [N,2], got shape {points.shape}.")

    ny, nx = field.shape
    x = np.clip(points[:, 0], 0.0, float(nx - 1))
    y = np.clip(points[:, 1], 0.0, float(ny - 1))
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    wx = x - x0.astype(np.float32)
    wy = y - y0.astype(np.float32)

    f00 = field[y0, x0]
    f01 = field[y0, x1]
    f10 = field[y1, x0]
    f11 = field[y1, x1]
    values = (
        (1.0 - wx) * (1.0 - wy) * f00
        + wx * (1.0 - wy) * f01
        + (1.0 - wx) * wy * f10
        + wx * wy * f11
    )
    return values.astype(np.float32, copy=False)


def drawSeedFirstStepSegments(
    axis: object,
    vxField: np.ndarray,
    vyField: np.ndarray,
    startPoints: Optional[np.ndarray],
    streamDirection: str,
    segmentLength: float,
    color: str,
    linewidth: float = 1.2,
) -> None:
    """Draw short line segment(s) from seed to first integration direction."""

    if startPoints is None or segmentLength <= 0.0:
        return
    try:
        from matplotlib.collections import LineCollection  # type: ignore
    except ImportError:
        return

    sampledVx = bilinearSample2d(vxField, startPoints)
    sampledVy = bilinearSample2d(vyField, startPoints)
    speed = np.sqrt(sampledVx**2 + sampledVy**2)
    validMask = speed > 1e-12
    if not np.any(validMask):
        return

    seeds = startPoints[validMask]
    unitVx = sampledVx[validMask] / speed[validMask]
    unitVy = sampledVy[validMask] / speed[validMask]

    segmentList: List[np.ndarray] = []

    if streamDirection in ("forward", "both"):
        endForward = np.stack(
            [seeds[:, 0] + segmentLength * unitVx, seeds[:, 1] + segmentLength * unitVy], axis=1
        )
        segmentList.append(np.stack([seeds, endForward], axis=1))
    if streamDirection in ("backward", "both"):
        endBackward = np.stack(
            [seeds[:, 0] - segmentLength * unitVx, seeds[:, 1] - segmentLength * unitVy], axis=1
        )
        segmentList.append(np.stack([seeds, endBackward], axis=1))

    if not segmentList:
        return
    segments = np.concatenate(segmentList, axis=0)
    collection = LineCollection(segments, colors=color, linewidths=linewidth, alpha=0.95, zorder=3)
    axis.add_collection(collection)


def drawSeedMarkers(axis: object, startPoints: Optional[np.ndarray], markerSize: float) -> None:
    """Draw seed markers without obscuring streamline segments."""

    if startPoints is None:
        return
    axis.scatter(
        startPoints[:, 0],
        startPoints[:, 1],
        s=markerSize,
        facecolors="none",
        edgecolors="black",
        marker="o",
        linewidths=0.9,
        zorder=5,
    )


def drawStreamplot(
    axis: object,
    xAxis: np.ndarray,
    yAxis: np.ndarray,
    vxField: np.ndarray,
    vyField: np.ndarray,
    streamDensity: float,
    startPoints: Optional[np.ndarray],
    linewidth: float = 0.8,
    arrowsize: Optional[float] = None,
    color: Optional[str] = None,
    streamDirection: str = "both",
) -> None:
    """Draw a streamplot with consistent seeded behavior."""

    safeVx = np.nan_to_num(vxField, nan=0.0, posinf=0.0, neginf=0.0)
    safeVy = np.nan_to_num(vyField, nan=0.0, posinf=0.0, neginf=0.0)
    kwargs: Dict[str, object] = {
        "density": streamDensity,
        "linewidth": linewidth,
    }
    if arrowsize is not None:
        kwargs["arrowsize"] = arrowsize
    if color is not None:
        kwargs["color"] = color
    filteredStartPoints: Optional[np.ndarray] = None
    if startPoints is not None:
        sampledVx = bilinearSample2d(safeVx, startPoints)
        sampledVy = bilinearSample2d(safeVy, startPoints)
        speed = np.sqrt(sampledVx**2 + sampledVy**2)
        validMask = np.isfinite(speed) & (speed > 1e-10)
        filteredStartPoints = startPoints[validMask]
        if filteredStartPoints.size == 0:
            logger.warning("All start points are degenerate (near-zero speed); skipping streamplot.")
            return
        if filteredStartPoints.shape[0] < startPoints.shape[0]:
            logger.info(
                "Filtered %d degenerate seed(s); plotting %d valid seed(s).",
                int(startPoints.shape[0] - filteredStartPoints.shape[0]),
                int(filteredStartPoints.shape[0]),
            )
        # Force seeded trajectories to continue through the domain and keep short lines.
        kwargs["start_points"] = filteredStartPoints
        kwargs["broken_streamlines"] = False
        kwargs["minlength"] = 0.0
    kwargs["integration_direction"] = streamDirection
    try:
        axis.streamplot(xAxis, yAxis, safeVx, safeVy, **kwargs)
    except Exception as streamError:
        if filteredStartPoints is None or filteredStartPoints.shape[0] == 0:
            raise
        logger.warning(
            "Batched seeded streamplot failed (%s). Falling back to per-seed drawing.",
            streamError,
        )
        baseKwargs = dict(kwargs)
        baseKwargs.pop("start_points", None)
        for seedPoint in filteredStartPoints:
            perSeedKwargs = dict(baseKwargs)
            perSeedKwargs["start_points"] = seedPoint.reshape(1, 2)
            try:
                axis.streamplot(xAxis, yAxis, safeVx, safeVy, **perSeedKwargs)
            except Exception:
                continue


def savePlot(
    predVx: np.ndarray,
    predVy: np.ndarray,
    pngPath: Path,
    stepIdx: int,
    plotMode: str,
    maxQuiverVectors: int,
    plotDpi: int,
    streamDensity: float,
    seedNx: int,
    seedNy: int,
    randomSeeds: int,
    streamDirection: str,
    showSeedPoints: bool,
    seedStepLength: float,
    startPoints: Optional[np.ndarray] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        logger.warning("matplotlib is not installed; skipping PNG export for step %d.", stepIdx)
        return

    ny, nx = predVx.shape
    yGrid, xGrid = np.mgrid[0:ny, 0:nx]
    if startPoints is None:
        startPoints = buildStreamStartPoints(
            nx=nx, ny=ny, seedNx=seedNx, seedNy=seedNy, randomSeeds=randomSeeds
        )
    subplotCount = 2 if plotMode == "both" else 1
    fig, axes = plt.subplots(1, subplotCount, figsize=(6 * subplotCount, 5), dpi=plotDpi)
    if subplotCount == 1:
        axes = [axes]

    axisCursor = 0
    if plotMode in ("quiver", "both"):
        axis = axes[axisCursor]
        axisCursor += 1
        quiverStrideY = max(1, ny // max(maxQuiverVectors, 1))
        quiverStrideX = max(1, nx // max(maxQuiverVectors, 1))
        axis.quiver(
            xGrid[::quiverStrideY, ::quiverStrideX],
            yGrid[::quiverStrideY, ::quiverStrideX],
            predVx[::quiverStrideY, ::quiverStrideX],
            predVy[::quiverStrideY, ::quiverStrideX],
            angles="xy",
            scale_units="xy",
            scale=None,
        )
        axis.set_title(f"Predicted Quiver (step {stepIdx})")
        axis.set_aspect("equal")

    if plotMode in ("streamplot", "both"):
        axis = axes[axisCursor]
        xAxis = np.arange(nx)
        yAxis = np.arange(ny)
        drawStreamplot(
            axis=axis,
            xAxis=xAxis,
            yAxis=yAxis,
            vxField=predVx,
            vyField=predVy,
            streamDensity=streamDensity,
            startPoints=startPoints,
            linewidth=0.8,
            streamDirection=streamDirection,
        )
        drawSeedFirstStepSegments(
            axis=axis,
            vxField=predVx,
            vyField=predVy,
            startPoints=startPoints,
            streamDirection=streamDirection,
            segmentLength=seedStepLength,
            color="black",
            linewidth=1.0,
        )
        if showSeedPoints and startPoints is not None:
            drawSeedMarkers(axis=axis, startPoints=startPoints, markerSize=18)
        axis.set_title(f"Predicted Streamplot (step {stepIdx})")
        axis.set_aspect("equal")

    for axis in axes:
        axis.set_xlabel("x index")
        axis.set_ylabel("y index")
    fig.tight_layout()
    fig.savefig(pngPath)
    plt.close(fig)


def saveComparisonPlot(
    gtVx: np.ndarray,
    gtVy: np.ndarray,
    predVx: np.ndarray,
    predVy: np.ndarray,
    pngPath: Path,
    stepIdx: int,
    plotMode: str,
    maxQuiverVectors: int,
    plotDpi: int,
    streamDensity: float,
    seedNx: int,
    seedNy: int,
    randomSeeds: int,
    streamDirection: str,
    showSeedPoints: bool,
    seedStepLength: float,
    startPoints: Optional[np.ndarray] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        logger.warning(
            "matplotlib is not installed; skipping comparison PNG export for step %d.", stepIdx
        )
        return

    if gtVx.shape != predVx.shape or gtVy.shape != predVy.shape:
        raise ValueError(
            "Comparison plot requires matching GT/pred shapes: "
            f"gtVx={gtVx.shape}, predVx={predVx.shape}, gtVy={gtVy.shape}, predVy={predVy.shape}."
        )

    ny, nx = predVx.shape
    yGrid, xGrid = np.mgrid[0:ny, 0:nx]
    xAxis = np.arange(nx)
    yAxis = np.arange(ny)
    if startPoints is None:
        startPoints = buildStreamStartPoints(
            nx=nx, ny=ny, seedNx=seedNx, seedNy=seedNy, randomSeeds=randomSeeds
        )
    quiverStrideY = max(1, ny // max(maxQuiverVectors, 1))
    quiverStrideX = max(1, nx // max(maxQuiverVectors, 1))

    if plotMode == "both":
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=plotDpi, sharex=True, sharey=True)
        # Row 0: streamplot
        drawStreamplot(
            axis=axes[0, 0],
            xAxis=xAxis,
            yAxis=yAxis,
            vxField=gtVx,
            vyField=gtVy,
            streamDensity=streamDensity,
            startPoints=startPoints,
            linewidth=0.8,
            streamDirection=streamDirection,
        )
        drawSeedFirstStepSegments(
            axis=axes[0, 0],
            vxField=gtVx,
            vyField=gtVy,
            startPoints=startPoints,
            streamDirection=streamDirection,
            segmentLength=seedStepLength,
            color="black",
            linewidth=1.0,
        )
        if showSeedPoints and startPoints is not None:
            drawSeedMarkers(axis=axes[0, 0], startPoints=startPoints, markerSize=16)
        axes[0, 0].set_title(f"Ground Truth Streamplot (step {stepIdx})")
        drawStreamplot(
            axis=axes[0, 1],
            xAxis=xAxis,
            yAxis=yAxis,
            vxField=predVx,
            vyField=predVy,
            streamDensity=streamDensity,
            startPoints=startPoints,
            linewidth=0.8,
            streamDirection=streamDirection,
        )
        drawSeedFirstStepSegments(
            axis=axes[0, 1],
            vxField=predVx,
            vyField=predVy,
            startPoints=startPoints,
            streamDirection=streamDirection,
            segmentLength=seedStepLength,
            color="black",
            linewidth=1.0,
        )
        if showSeedPoints and startPoints is not None:
            drawSeedMarkers(axis=axes[0, 1], startPoints=startPoints, markerSize=16)
        axes[0, 1].set_title(f"Predicted Streamplot (step {stepIdx})")
        # Row 1: quiver
        axes[1, 0].quiver(
            xGrid[::quiverStrideY, ::quiverStrideX],
            yGrid[::quiverStrideY, ::quiverStrideX],
            gtVx[::quiverStrideY, ::quiverStrideX],
            gtVy[::quiverStrideY, ::quiverStrideX],
            angles="xy",
            scale_units="xy",
            scale=None,
        )
        axes[1, 0].set_title(f"Ground Truth Quiver (step {stepIdx})")
        axes[1, 1].quiver(
            xGrid[::quiverStrideY, ::quiverStrideX],
            yGrid[::quiverStrideY, ::quiverStrideX],
            predVx[::quiverStrideY, ::quiverStrideX],
            predVy[::quiverStrideY, ::quiverStrideX],
            angles="xy",
            scale_units="xy",
            scale=None,
        )
        axes[1, 1].set_title(f"Predicted Quiver (step {stepIdx})")
        flatAxes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=plotDpi, sharex=True, sharey=True)
        if plotMode == "streamplot":
            drawStreamplot(
                axis=axes[0],
                xAxis=xAxis,
                yAxis=yAxis,
                vxField=gtVx,
                vyField=gtVy,
                streamDensity=streamDensity,
                startPoints=startPoints,
                linewidth=0.8,
                streamDirection=streamDirection,
            )
            drawSeedFirstStepSegments(
                axis=axes[0],
                vxField=gtVx,
                vyField=gtVy,
                startPoints=startPoints,
                streamDirection=streamDirection,
                segmentLength=seedStepLength,
                color="black",
                linewidth=1.0,
            )
            if showSeedPoints and startPoints is not None:
                drawSeedMarkers(axis=axes[0], startPoints=startPoints, markerSize=16)
            axes[0].set_title(f"Ground Truth Streamplot (step {stepIdx})")
            drawStreamplot(
                axis=axes[1],
                xAxis=xAxis,
                yAxis=yAxis,
                vxField=predVx,
                vyField=predVy,
                streamDensity=streamDensity,
                startPoints=startPoints,
                linewidth=0.8,
                streamDirection=streamDirection,
            )
            drawSeedFirstStepSegments(
                axis=axes[1],
                vxField=predVx,
                vyField=predVy,
                startPoints=startPoints,
                streamDirection=streamDirection,
                segmentLength=seedStepLength,
                color="black",
                linewidth=1.0,
            )
            if showSeedPoints and startPoints is not None:
                drawSeedMarkers(axis=axes[1], startPoints=startPoints, markerSize=16)
            axes[1].set_title(f"Predicted Streamplot (step {stepIdx})")
        elif plotMode == "quiver":
            axes[0].quiver(
                xGrid[::quiverStrideY, ::quiverStrideX],
                yGrid[::quiverStrideY, ::quiverStrideX],
                gtVx[::quiverStrideY, ::quiverStrideX],
                gtVy[::quiverStrideY, ::quiverStrideX],
                angles="xy",
                scale_units="xy",
                scale=None,
            )
            axes[0].set_title(f"Ground Truth Quiver (step {stepIdx})")
            axes[1].quiver(
                xGrid[::quiverStrideY, ::quiverStrideX],
                yGrid[::quiverStrideY, ::quiverStrideX],
                predVx[::quiverStrideY, ::quiverStrideX],
                predVy[::quiverStrideY, ::quiverStrideX],
                angles="xy",
                scale_units="xy",
                scale=None,
            )
            axes[1].set_title(f"Predicted Quiver (step {stepIdx})")
        else:
            raise ValueError(f"Unexpected plot mode: {plotMode}")
        flatAxes = list(axes)

    for axis in flatAxes:
        axis.set_aspect("equal")
        axis.set_xlabel("x index")
        axis.set_ylabel("y index")
    fig.tight_layout()
    fig.savefig(pngPath)
    plt.close(fig)


def saveOverlayStreamPlot(
    gtVx: np.ndarray,
    gtVy: np.ndarray,
    predVx: np.ndarray,
    predVy: np.ndarray,
    pngPath: Path,
    stepIdx: int,
    plotDpi: int,
    gtColor: str,
    predColor: str,
    streamDensity: float,
    seedNx: int,
    seedNy: int,
    randomSeeds: int,
    streamDirection: str,
    showSeedPoints: bool,
    seedStepLength: float,
    startPoints: Optional[np.ndarray] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.lines import Line2D  # type: ignore
    except ImportError:
        logger.warning(
            "matplotlib is not installed; skipping overlay PNG export for step %d.", stepIdx
        )
        return

    if gtVx.shape != predVx.shape or gtVy.shape != predVy.shape:
        raise ValueError(
            "Overlay plot requires matching GT/pred shapes: "
            f"gtVx={gtVx.shape}, predVx={predVx.shape}, gtVy={gtVy.shape}, predVy={predVy.shape}."
        )

    ny, nx = predVx.shape
    xAxis = np.arange(nx)
    yAxis = np.arange(ny)
    if startPoints is None:
        startPoints = buildStreamStartPoints(
            nx=nx, ny=ny, seedNx=seedNx, seedNy=seedNy, randomSeeds=randomSeeds
        )

    fig, axis = plt.subplots(1, 1, figsize=(7, 6), dpi=plotDpi)
    drawStreamplot(
        axis=axis,
        xAxis=xAxis,
        yAxis=yAxis,
        vxField=gtVx,
        vyField=gtVy,
        streamDensity=streamDensity,
        startPoints=startPoints,
        linewidth=1.0,
        arrowsize=0.8,
        color=gtColor,
        streamDirection=streamDirection,
    )
    drawSeedFirstStepSegments(
        axis=axis,
        vxField=gtVx,
        vyField=gtVy,
        startPoints=startPoints,
        streamDirection=streamDirection,
        segmentLength=seedStepLength,
        color=gtColor,
        linewidth=1.3,
    )
    drawStreamplot(
        axis=axis,
        xAxis=xAxis,
        yAxis=yAxis,
        vxField=predVx,
        vyField=predVy,
        streamDensity=streamDensity,
        startPoints=startPoints,
        linewidth=1.0,
        arrowsize=0.8,
        color=predColor,
        streamDirection=streamDirection,
    )
    drawSeedFirstStepSegments(
        axis=axis,
        vxField=predVx,
        vyField=predVy,
        startPoints=startPoints,
        streamDirection=streamDirection,
        segmentLength=seedStepLength,
        color=predColor,
        linewidth=1.3,
    )
    if showSeedPoints and startPoints is not None:
        drawSeedMarkers(axis=axis, startPoints=startPoints, markerSize=24)
    legendHandles = [
        Line2D([0], [0], color=gtColor, lw=2, label="Ground Truth"),
        Line2D([0], [0], color=predColor, lw=2, label="Prediction"),
    ]
    if showSeedPoints and startPoints is not None:
        legendHandles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="black",
                markerfacecolor="none",
                markeredgecolor="black",
                lw=0,
                label="Seed points",
            )
        )
    axis.legend(handles=legendHandles, loc="upper right")
    axis.set_title(f"Overlay Streamlines (step {stepIdx})")
    axis.set_aspect("equal")
    axis.set_xlabel("x index")
    axis.set_ylabel("y index")
    fig.tight_layout()
    fig.savefig(pngPath)
    plt.close(fig)


def buildModelFromCheckpoint(
    checkpointData: Dict[str, object],
    device: torch.device,
) -> torch.nn.Module:
    modelMetaRaw = checkpointData.get("modelMeta")
    if not isinstance(modelMetaRaw, dict):
        raise KeyError("Checkpoint does not contain valid 'modelMeta'.")
    modelMeta = modelMetaRaw
    model = buildVectorInrModel(
        modelName=str(modelMeta["model"]),
        ensembleCount=int(modelMeta["ensembleCount"]),
        embedDim=int(modelMeta["embedDim"]),
        hiddenDim=int(modelMeta["hidden"]),
        hiddenLayers=int(modelMeta["layers"]),
        numFrequencies=int(modelMeta["freqs"]),
        sirenOmega0=float(modelMeta.get("sirenOmega0", 30.0)),
    )
    modelState = checkpointData.get("modelState")
    if not isinstance(modelState, dict):
        raise KeyError("Checkpoint does not contain valid 'modelState'.")
    model.load_state_dict(modelState, strict=True)
    model.to(device)
    model.eval()
    return model


def main() -> int:
    parser = buildArgParser()
    args = parser.parse_args()
    setupLogging(args.logLevel)

    try:
        device = resolveDevice(args.device)
        checkpointData = loadCheckpoint(args.checkpoint, mapLocation=device)
        if args.streamDensity <= 0:
            raise ValueError("--stream_density must be > 0.")
        if args.seedStepLength < 0:
            raise ValueError("--seed_step_length must be >= 0.")
        if args.randomSeeds < 0:
            raise ValueError("--random_seeds must be >= 0.")
        if (args.streamlineSeedNx == 0) != (args.streamlineSeedNy == 0):
            raise ValueError(
                "Set both --streamline_seed_nx and --streamline_seed_ny, or leave both unset."
            )
        if args.streamlineSeedNx < 0 or args.streamlineSeedNy < 0:
            raise ValueError("--streamline_seed_nx/--streamline_seed_ny must be >= 0.")
        if args.randomSeeds > 0 and (args.streamlineSeedNx > 0 or args.streamlineSeedNy > 0):
            raise ValueError(
                "--random_seeds is mutually exclusive with --streamline_seed_nx/--streamline_seed_ny."
            )
        if args.overlayPng and args.plotMode != "streamplot":
            logger.warning(
                "--overlay_png always renders streamlines; ignoring --plot_mode=%s.",
                args.plotMode,
            )

        dataset = MultiEnsembleDataset(
            ensembleDirs=args.ensembleDirs,
            bpFile=args.bpFile,
            vxName=args.vxName,
            vyName=args.vyName,
            cacheSteps=args.cacheSteps,
            omegaCacheSteps=args.omegaCacheSteps,
            requireSameGrid=True,
        )
        if args.ensembleIdx < 0 or args.ensembleIdx >= dataset.getEnsembleCount():
            raise IndexError(
                f"--ensemble_idx={args.ensembleIdx} is out of range "
                f"[0, {dataset.getEnsembleCount() - 1}]"
            )

        normalizationRaw = checkpointData.get("normalization")
        if not isinstance(normalizationRaw, dict):
            raise KeyError("Checkpoint missing 'normalization'.")
        mean = np.asarray(normalizationRaw["mean"], dtype=np.float32)
        std = np.asarray(normalizationRaw["std"], dtype=np.float32)
        if mean.shape != (2,) or std.shape != (2,):
            raise ValueError("Checkpoint normalization mean/std must each have shape (2,).")
        if np.any(std <= 0):
            raise ValueError("Checkpoint normalization std must be positive.")
        dataset.setNormalization(mean, std)

        model = buildModelFromCheckpoint(checkpointData, device=device)
        modelMeta = checkpointData.get("modelMeta")
        if isinstance(modelMeta, dict):
            expectedEnsembles = int(modelMeta["ensembleCount"])
            if expectedEnsembles != dataset.getEnsembleCount():
                raise ValueError(
                    "Checkpoint ensemble count does not match evaluation dataset: "
                    f"checkpoint={expectedEnsembles}, data={dataset.getEnsembleCount()}."
                )

        ny, nx = dataset.getGridShape(args.ensembleIdx)
        stepCount = dataset.getStepCount(args.ensembleIdx)
        selectedSteps = selectSteps(
            stepCount=stepCount,
            step=args.step,
            stepStart=args.stepStart,
            stepEnd=args.stepEnd,
            stepStride=args.stepStride,
        )

        outputDir = ensureDirectory(args.outdir)
        metricsRows: List[Dict[str, float]] = []
        logger.info(
            "Evaluating %d step(s) on ensemble %d with grid %dx%d.",
            len(selectedSteps),
            args.ensembleIdx,
            ny,
            nx,
        )
        randomSeedRng: Optional[np.random.Generator] = None
        if args.randomSeeds > 0:
            randomSeedRng = np.random.default_rng()
        sharedStartPoints = buildStreamStartPoints(
            nx=nx,
            ny=ny,
            seedNx=args.streamlineSeedNx,
            seedNy=args.streamlineSeedNy,
            randomSeeds=args.randomSeeds,
            rng=randomSeedRng,
        )

        for stepIdx in selectedSteps:
            seedPointsForStep = sharedStartPoints
            if seedPointsForStep is not None:
                seedList = seedPointsForStep.tolist()
                seedCount = len(seedList)
                logger.info("Step %d GT seed locations (%d): %s", stepIdx, seedCount, seedList)
                logger.info(
                    "Step %d Prediction seed locations (%d): %s",
                    stepIdx,
                    seedCount,
                    seedList,
                )

            query = buildFullGridQuery(
                ny=ny,
                nx=nx,
                stepIdx=stepIdx,
                stepCount=stepCount,
                ensembleIdx=args.ensembleIdx,
                device=device,
            )
            predNormalizedFlat = inferVelocityGrid(
                model=model,
                coords=query["coords"],
                ensembleIndices=query["ensembleIndices"].long(),
                chunkSize=args.chunkSize,
            )
            predNormalized = predNormalizedFlat.numpy().reshape(ny, nx, 2)
            predVx = predNormalized[..., 0] * std[0] + mean[0]
            predVy = predNormalized[..., 1] * std[1] + mean[1]

            gtVx, gtVy = dataset.readVelocityStep(args.ensembleIdx, stepIdx)
            gtVelocity = np.stack([gtVx, gtVy], axis=-1).astype(np.float32)
            predVelocity = np.stack([predVx, predVy], axis=-1).astype(np.float32)

            maeVelocity = float(np.mean(np.abs(predVelocity - gtVelocity)))
            angularError = computeAngularErrorDeg(predVelocity, gtVelocity)
            row: Dict[str, float] = {
                "step": float(stepIdx),
                "mae_velocity": maeVelocity,
                "angular_error_deg": angularError,
            }

            if args.vorticityMetric:
                predOmega = computeFiniteDifferenceVorticity(predVx, predVy)
                gtOmega = dataset.readVorticityStep(
                    ensembleIdx=args.ensembleIdx,
                    stepIdx=stepIdx,
                    useNormalized=False,
                )
                row["mae_vorticity"] = float(np.mean(np.abs(predOmega - gtOmega)))

            nameStem = f"e{args.ensembleIdx:03d}_s{stepIdx:05d}"
            np.save(outputDir / f"pred_vx_{nameStem}.npy", predVx.astype(np.float32))
            np.save(outputDir / f"pred_vy_{nameStem}.npy", predVy.astype(np.float32))

            if args.savePng:
                savePlot(
                    predVx=predVx,
                    predVy=predVy,
                    pngPath=outputDir / f"pred_{nameStem}.png",
                    stepIdx=stepIdx,
                    plotMode=args.plotMode,
                    maxQuiverVectors=args.maxQuiverVectors,
                    plotDpi=args.plotDpi,
                    streamDensity=args.streamDensity,
                    seedNx=args.streamlineSeedNx,
                    seedNy=args.streamlineSeedNy,
                    randomSeeds=args.randomSeeds,
                    streamDirection=args.streamDirection,
                    showSeedPoints=args.showSeedPoints,
                    seedStepLength=args.seedStepLength,
                    startPoints=seedPointsForStep,
                )
            if args.comparePng:
                saveComparisonPlot(
                    gtVx=gtVx,
                    gtVy=gtVy,
                    predVx=predVx,
                    predVy=predVy,
                    pngPath=outputDir / f"compare_{nameStem}.png",
                    stepIdx=stepIdx,
                    plotMode=args.plotMode,
                    maxQuiverVectors=args.maxQuiverVectors,
                    plotDpi=args.plotDpi,
                    streamDensity=args.streamDensity,
                    seedNx=args.streamlineSeedNx,
                    seedNy=args.streamlineSeedNy,
                    randomSeeds=args.randomSeeds,
                    streamDirection=args.streamDirection,
                    showSeedPoints=args.showSeedPoints,
                    seedStepLength=args.seedStepLength,
                    startPoints=seedPointsForStep,
                )
            if args.overlayPng:
                saveOverlayStreamPlot(
                    gtVx=gtVx,
                    gtVy=gtVy,
                    predVx=predVx,
                    predVy=predVy,
                    pngPath=outputDir / f"overlay_{nameStem}.png",
                    stepIdx=stepIdx,
                    plotDpi=args.plotDpi,
                    gtColor=args.gtColor,
                    predColor=args.predColor,
                    streamDensity=args.streamDensity,
                    seedNx=args.streamlineSeedNx,
                    seedNy=args.streamlineSeedNy,
                    randomSeeds=args.randomSeeds,
                    streamDirection=args.streamDirection,
                    showSeedPoints=args.showSeedPoints,
                    seedStepLength=args.seedStepLength,
                    startPoints=seedPointsForStep,
                )

            metricsRows.append(row)
            logger.info(
                "Step %d | MAE(v)=%.6f | Angular(deg)=%.4f%s",
                stepIdx,
                row["mae_velocity"],
                row["angular_error_deg"],
                f" | MAE(omega)={row['mae_vorticity']:.6f}"
                if "mae_vorticity" in row
                else "",
            )

        metricsJsonPath = outputDir / "metrics.json"
        with metricsJsonPath.open("w", encoding="utf-8") as fileObj:
            json.dump(metricsRows, fileObj, indent=2)

        metricsCsvPath = outputDir / "metrics.csv"
        csvFields = ["step", "mae_velocity", "angular_error_deg"]
        if args.vorticityMetric:
            csvFields.append("mae_vorticity")
        with metricsCsvPath.open("w", encoding="utf-8", newline="") as fileObj:
            writer = csv.DictWriter(fileObj, fieldnames=csvFields)
            writer.writeheader()
            for row in metricsRows:
                writer.writerow(row)

        dataset.close()
        logger.info("Saved metrics to %s and %s", metricsJsonPath, metricsCsvPath)
        return 0
    except Exception as evalError:
        logger.exception("Evaluation failed: %s", evalError)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
