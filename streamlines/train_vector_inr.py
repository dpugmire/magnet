#!/usr/bin/env python3
"""CLI entry point for training vector INR models."""

from __future__ import annotations

import argparse
import logging

from vector_inr.train import TrainConfig, runTraining
from vector_inr.utils import setupLogging


def buildArgParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an implicit neural representation for ADIOS2 vector fields "
            "f(x, y, t, e) -> (vx, vy)."
        )
    )
    parser.add_argument(
        "--ensemble_dirs",
        dest="ensembleDirs",
        nargs="+",
        required=True,
        help="List of ensemble directories. Each should contain a BP dataset path.",
    )
    parser.add_argument("--bp_file", dest="bpFile", default="output.bp")
    parser.add_argument("--vx_name", dest="vxName", default="vx")
    parser.add_argument("--vy_name", dest="vyName", default="vy")

    parser.add_argument("--model", choices=["fourier", "siren"], default="fourier")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--steps_per_epoch", dest="stepsPerEpoch", type=int, default=200)
    parser.add_argument("--batch_points", dest="batchPoints", type=int, default=65_536)
    parser.add_argument("--embed_dim", dest="embedDim", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--freqs", type=int, default=10)
    parser.add_argument("--siren_omega0", dest="sirenOmega0", type=float, default=30.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", dest="minLr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", dest="weightDecay", type=float, default=1e-4)
    parser.add_argument("--cosine_schedule", dest="cosineSchedule", action="store_true")

    parser.add_argument("--vorticity_loss", dest="vorticityLoss", action="store_true")
    parser.add_argument("--lambda_omega", dest="lambdaOmega", type=float, default=0.1)
    parser.add_argument("--warmup_epochs", dest="warmupEpochs", type=int, default=20)

    parser.add_argument("--cache_steps", dest="cacheSteps", type=int, default=128)
    parser.add_argument("--omega_cache_steps", dest="omegaCacheSteps", type=int, default=128)
    parser.add_argument(
        "--norm_steps_per_ensemble", dest="normStepsPerEnsemble", type=int, default=8
    )
    parser.add_argument(
        "--norm_points_per_step", dest="normPointsPerStep", type=int, default=4096
    )

    parser.add_argument("--checkpoint_every", dest="checkpointEvery", type=int, default=10)
    parser.add_argument("--outdir", default="runs/ot_inr")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nondeterministic", action="store_true")
    parser.add_argument("--log_level", dest="logLevel", default="INFO")
    return parser


def main() -> int:
    parser = buildArgParser()
    args = parser.parse_args()
    setupLogging(args.logLevel)

    config = TrainConfig(
        ensembleDirs=args.ensembleDirs,
        bpFile=args.bpFile,
        vxName=args.vxName,
        vyName=args.vyName,
        model=args.model,
        epochs=args.epochs,
        stepsPerEpoch=args.stepsPerEpoch,
        batchPoints=args.batchPoints,
        embedDim=args.embedDim,
        hidden=args.hidden,
        layers=args.layers,
        freqs=args.freqs,
        sirenOmega0=args.sirenOmega0,
        lr=args.lr,
        minLr=args.minLr,
        weightDecay=args.weightDecay,
        cosineSchedule=args.cosineSchedule,
        vorticityLoss=args.vorticityLoss,
        lambdaOmega=args.lambdaOmega,
        warmupEpochs=args.warmupEpochs,
        cacheSteps=args.cacheSteps,
        omegaCacheSteps=args.omegaCacheSteps,
        normStepsPerEnsemble=args.normStepsPerEnsemble,
        normPointsPerStep=args.normPointsPerStep,
        checkpointEvery=args.checkpointEvery,
        outdir=args.outdir,
        device=args.device,
        seed=args.seed,
        deterministic=not args.nondeterministic,
    )

    try:
        checkpointPath = runTraining(config)
        logging.getLogger(__name__).info("Final checkpoint: %s", checkpointPath)
        return 0
    except Exception as runError:
        logging.getLogger(__name__).exception("Training failed: %s", runError)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
