"""CLI entrypoint for evaluating a trained AE + PD model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from ae_pd.adios_dataset import AdiosScalarArchive
from ae_pd.models import ConvAutoencoder
from ae_pd.topology import ComputeBottleneckDistances, ComputePairedPersistenceImages
from ae_pd.utils import EnsureDir, LoadCheckpoint, ResolveDevice, SaveJson


def BuildArgParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate AE+PD checkpoint on one snapshot.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--run_dirs", type=str, nargs="*", default=None)
    parser.add_argument("--bp_file", type=str, default="output.bp")
    parser.add_argument("--scalar_name", type=str, default="rho")
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--outdir", type=str, default="outputs/ae_pd_eval")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--pd_downsample", type=int, default=128)
    parser.add_argument("--pi_res", type=int, choices=[32, 64], default=64)
    parser.add_argument("--pi_sigma", type=float, default=1.5)
    parser.add_argument("--pd_dims", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--pd_min_persistence", type=float, default=None)
    parser.add_argument("--save_png", action="store_true")
    return parser


def NormalizeForInference(
    imageArray: np.ndarray,
    normMode: str,
    normStats: Dict[str, float],
) -> tuple[np.ndarray, float, float]:
    imageFloat = imageArray.astype(np.float32, copy=False)
    if normMode == "none":
        return imageFloat, 0.0, 1.0
    if normMode == "per_image":
        meanValue = float(imageFloat.mean())
        stdValue = float(imageFloat.std())
        if stdValue < 1e-6:
            stdValue = 1.0
        return (imageFloat - meanValue) / stdValue, meanValue, stdValue
    if normMode == "global":
        meanValue = float(normStats.get("globalMean", 0.0))
        stdValue = float(normStats.get("globalStd", 1.0))
        if stdValue < 1e-6:
            stdValue = 1.0
        return (imageFloat - meanValue) / stdValue, meanValue, stdValue
    raise ValueError(f"Unsupported norm mode: {normMode}")


def DenormalizeImage(imageNorm: np.ndarray, normMode: str, meanValue: float, stdValue: float) -> np.ndarray:
    if normMode == "none":
        return imageNorm.astype(np.float32, copy=False)
    return (imageNorm * stdValue + meanValue).astype(np.float32, copy=False)


def ComputePsnr(gtImage: np.ndarray, reconImage: np.ndarray) -> float:
    """Compute PSNR using GT dynamic range."""

    gtArray = np.asarray(gtImage, dtype=np.float64)
    reconArray = np.asarray(reconImage, dtype=np.float64)
    mseValue = float(np.mean(np.square(reconArray - gtArray)))
    if mseValue <= 1e-20:
        return float("inf")
    dataRange = float(np.max(gtArray) - np.min(gtArray))
    if dataRange <= 1e-20:
        dataRange = 1.0
    return float(20.0 * np.log10(dataRange) - 10.0 * np.log10(mseValue))


def TrySavePng(
    pathObj: Path,
    imageArray: np.ndarray,
    titleText: str,
    xLabel: str | None = None,
    yLabel: str | None = None,
    hideTicks: bool = True,
    extent: tuple[float, float, float, float] | None = None,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, axis = plt.subplots(1, 1, figsize=(5, 4))
    imageHandle = axis.imshow(
        imageArray,
        cmap="viridis",
        origin="lower",
        extent=extent,
        aspect="auto",
    )
    fig.colorbar(imageHandle, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title(titleText)
    if hideTicks:
        axis.set_xticks([])
        axis.set_yticks([])
    if xLabel:
        axis.set_xlabel(xLabel)
    if yLabel:
        axis.set_ylabel(yLabel)
    fig.tight_layout()
    fig.savefig(pathObj, dpi=150)
    plt.close(fig)
    return True


def TrySaveSideBySidePng(
    pathObj: Path,
    leftImage: np.ndarray,
    rightImage: np.ndarray,
    leftTitle: str,
    rightTitle: str,
    figureTitle: str | None = None,
    xLabel: str | None = None,
    yLabel: str | None = None,
    hideTicks: bool = True,
    extent: tuple[float, float, float, float] | None = None,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    leftArray = np.asarray(leftImage, dtype=np.float32)
    rightArray = np.asarray(rightImage, dtype=np.float32)
    vmin = float(min(leftArray.min(), rightArray.min()))
    vmax = float(max(leftArray.max(), rightArray.max()))
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(
        leftArray,
        cmap="viridis",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="auto",
    )
    axes[0].set_title(leftTitle)
    if hideTicks:
        axes[0].set_xticks([])
        axes[0].set_yticks([])
    if xLabel:
        axes[0].set_xlabel(xLabel)
    if yLabel:
        axes[0].set_ylabel(yLabel)

    im1 = axes[1].imshow(
        rightArray,
        cmap="viridis",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=extent,
        aspect="auto",
    )
    axes[1].set_title(rightTitle)
    if hideTicks:
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    if xLabel:
        axes[1].set_xlabel(xLabel)
    if yLabel:
        axes[1].set_ylabel(yLabel)

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    if figureTitle:
        fig.suptitle(figureTitle)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    fig.savefig(pathObj, dpi=150)
    plt.close(fig)
    return True


def TrySavePdComparePng(
    pathObj: Path,
    gtDiagram: np.ndarray,
    reconDiagram: np.ndarray,
    dimValue: int,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    gtPairs = np.asarray(gtDiagram, dtype=np.float32).reshape(-1, 2)
    reconPairs = np.asarray(reconDiagram, dtype=np.float32).reshape(-1, 2)

    plt.figure(figsize=(5, 5))
    if gtPairs.size > 0:
        plt.scatter(
            gtPairs[:, 0],
            gtPairs[:, 1],
            s=18,
            alpha=0.7,
            label=f"GT H{dimValue}",
            c="#1f77b4",
        )
    if reconPairs.size > 0:
        plt.scatter(
            reconPairs[:, 0],
            reconPairs[:, 1],
            s=18,
            alpha=0.7,
            label=f"Recon H{dimValue}",
            c="#ff7f0e",
        )

    if gtPairs.size > 0 or reconPairs.size > 0:
        allPairs = np.vstack([arr for arr in [gtPairs, reconPairs] if arr.size > 0])
        minValue = float(np.min(allPairs))
        maxValue = float(np.max(allPairs))
        if abs(maxValue - minValue) < 1e-8:
            maxValue = minValue + 1e-6
        lineX = np.linspace(minValue, maxValue, 128)
        plt.plot(lineX, lineX, "--", c="k", lw=1.0, label="birth=death")
        plt.xlim(minValue, maxValue)
        plt.ylim(minValue, maxValue)
    else:
        plt.text(0.5, 0.5, "No finite pairs", ha="center", va="center", transform=plt.gca().transAxes)
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)

    plt.title(f"Persistence Diagram Comparison (H{dimValue})")
    plt.xlabel("Birth")
    plt.ylabel("Death")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(pathObj, dpi=150)
    plt.close()
    return True


def main() -> None:
    args = BuildArgParser().parse_args()
    device = ResolveDevice(args.device)
    checkpoint = LoadCheckpoint(args.checkpoint, mapLocation=device)
    configDict = checkpoint.get("config", {})
    normStats = checkpoint.get("normStats", configDict.get("norm_stats", {}))

    model = ConvAutoencoder(
        baseChannels=int(configDict.get("base_channels", 32)),
        numDown=int(configDict.get("num_down", 3)),
        latentChannels=int(configDict.get("latent_channels", 128)),
        activation=str(configDict.get("activation", "silu")),
    ).to(device)
    model.load_state_dict(checkpoint["modelState"])
    model.eval()

    archive = AdiosScalarArchive(
        runsDir=args.runs_dir,
        runDirs=args.run_dirs,
        bpFile=args.bp_file,
        scalarName=args.scalar_name,
        cacheSize=16,
    )

    gtImage = archive.readScalar(args.run, int(args.step)).astype(np.float32, copy=False)
    normMode = str(configDict.get("norm", "none"))
    inputImage, meanValue, stdValue = NormalizeForInference(gtImage, normMode, normStats)

    inputTensor = torch.from_numpy(inputImage[None, None, :, :]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        reconNormTensor = model(inputTensor)
    reconNorm = reconNormTensor[0, 0].detach().cpu().numpy().astype(np.float32, copy=False)
    reconImage = DenormalizeImage(reconNorm, normMode=normMode, meanValue=meanValue, stdValue=stdValue)

    maeValue = float(np.mean(np.abs(reconImage - gtImage)))
    rmseValue = float(np.sqrt(np.mean(np.square(reconImage - gtImage))))
    psnrValue = ComputePsnr(gtImage=gtImage, reconImage=reconImage)
    psnrText = f"{psnrValue:.2f} dB" if np.isfinite(psnrValue) else "inf dB"

    pdDims = [int(dimValue) for dimValue in args.pd_dims]
    pdMinPersistence = (
        float(args.pd_min_persistence)
        if args.pd_min_persistence is not None
        else float(configDict.get("pd_min_persistence", 0.0))
    )
    weightByPersistence = bool(configDict.get("pd_weight_persistence", True))
    pairResult = ComputePairedPersistenceImages(
        gtImage=gtImage,
        reconImage=reconImage,
        pdDims=pdDims,
        pdDownsample=int(args.pd_downsample),
        piRes=int(args.pi_res),
        piSigma=float(args.pi_sigma),
        weightByPersistence=weightByPersistence,
        minPersistence=pdMinPersistence,
    )
    sharedRange = tuple(float(value) for value in pairResult["sharedRange"])
    bottleneckByDim = ComputeBottleneckDistances(
        gtDiagramByDim=pairResult["gtDiagramByDim"],
        reconDiagramByDim=pairResult["reconDiagramByDim"],
        pdDims=pdDims,
    )

    sampleOutDir = EnsureDir(Path(args.outdir) / f"{args.run}_step{int(args.step):05d}")
    np.save(Path(sampleOutDir) / "gt.npy", gtImage)
    np.save(Path(sampleOutDir) / "recon.npy", reconImage)
    np.save(Path(sampleOutDir) / "pi_gt.npy", pairResult["gtPi"])
    np.save(Path(sampleOutDir) / "pi_recon.npy", pairResult["reconPi"])
    for dimValue in pdDims:
        dimKey = int(dimValue)
        gtDiagram = np.asarray(
            pairResult["gtDiagramByDim"].get(dimKey, np.zeros((0, 2), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1, 2)
        reconDiagram = np.asarray(
            pairResult["reconDiagramByDim"].get(dimKey, np.zeros((0, 2), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1, 2)
        np.save(Path(sampleOutDir) / f"pd_gt_dim{dimKey}.npy", gtDiagram)
        np.save(Path(sampleOutDir) / f"pd_recon_dim{dimKey}.npy", reconDiagram)

    metricsDict = {
        "run": args.run,
        "step": int(args.step),
        "mae": maeValue,
        "rmse": rmseValue,
        "psnr": psnrValue,
        "bottleneckByDim": {str(dimKey): float(value) for dimKey, value in bottleneckByDim.items()},
        "normMode": normMode,
        "normMeanUsed": float(meanValue),
        "normStdUsed": float(stdValue),
        "pdMinPersistence": pdMinPersistence,
    }
    SaveJson(metricsDict, Path(sampleOutDir) / "metrics.json")

    pngSaved = False
    if args.save_png:
        pngSaved = True
        pngSaved &= TrySavePng(Path(sampleOutDir) / "gt.png", gtImage, "Ground Truth")
        pngSaved &= TrySavePng(
            Path(sampleOutDir) / "recon.png",
            reconImage,
            f"Reconstruction (PSNR={psnrText})",
        )
        pngSaved &= TrySaveSideBySidePng(
            Path(sampleOutDir) / "gt_recon_side_by_side.png",
            gtImage,
            reconImage,
            "Ground Truth",
            f"Reconstruction (PSNR={psnrText})",
            figureTitle=f"GT vs Reconstruction (PSNR={psnrText})",
        )
        pngSaved &= TrySavePng(
            Path(sampleOutDir) / "pi_gt.png",
            pairResult["gtPi"],
            "PI Ground Truth",
            xLabel="Birth",
            yLabel="Persistence",
            hideTicks=False,
            extent=sharedRange,
        )
        pngSaved &= TrySavePng(
            Path(sampleOutDir) / "pi_recon.png",
            pairResult["reconPi"],
            "PI Reconstruction",
            xLabel="Birth",
            yLabel="Persistence",
            hideTicks=False,
            extent=sharedRange,
        )
        pngSaved &= TrySaveSideBySidePng(
            Path(sampleOutDir) / "pi_gt_recon_side_by_side.png",
            np.asarray(pairResult["gtPi"], dtype=np.float32),
            np.asarray(pairResult["reconPi"], dtype=np.float32),
            "PI Ground Truth",
            "PI Reconstruction",
            xLabel="Birth",
            yLabel="Persistence",
            hideTicks=False,
            extent=sharedRange,
        )
        for dimValue in pdDims:
            dimKey = int(dimValue)
            gtDiagram = np.asarray(
                pairResult["gtDiagramByDim"].get(dimKey, np.zeros((0, 2), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1, 2)
            reconDiagram = np.asarray(
                pairResult["reconDiagramByDim"].get(dimKey, np.zeros((0, 2), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1, 2)
            pngSaved &= TrySavePdComparePng(
                Path(sampleOutDir) / f"pd_compare_dim{dimKey}.png",
                gtDiagram,
                reconDiagram,
                dimKey,
            )

    resultDict = {
        "outdir": str(Path(sampleOutDir).resolve()),
        "mae": maeValue,
        "rmse": rmseValue,
        "psnr": psnrValue,
        "bottleneckByDim": metricsDict["bottleneckByDim"],
        "pngSaved": bool(pngSaved),
    }
    print(json.dumps(resultDict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
