"""CLI entrypoint for training convolutional AE + PD loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ae_pd.train import TrainAutoencoder
from ae_pd.utils import ResolveDevice


def BuildArgParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train AE with persistence-image topology loss.")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--run_dirs", type=str, nargs="*", default=None)
    parser.add_argument("--bp_file", type=str, default="output.bp")
    parser.add_argument("--scalar_name", type=str, default="rho")
    parser.add_argument("--outdir", type=str, default="outputs/ae_pd")
    parser.add_argument(
        "--mode_subdir",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store runs under outdir/with_pd or outdir/no_pd to avoid accidental overwrite.",
    )
    parser.add_argument(
        "--allow_overwrite",
        action="store_true",
        help="Allow writing into existing output directory without creating a new timestamped folder.",
    )

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--split_by_run", action="store_true")
    parser.add_argument("--val_fraction", type=float, default=0.1)

    parser.add_argument("--norm", choices=["none", "per_image", "global"], default="none")
    parser.add_argument("--norm_samples", type=int, default=200)

    parser.add_argument("--cache_size", type=int, default=64)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--num_down", type=int, choices=[3, 4], default=3)
    parser.add_argument("--latent_channels", type=int, default=128)
    parser.add_argument("--activation", choices=["relu", "silu"], default="silu")

    parser.add_argument(
        "--use_pd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable topology term during training (--no-use_pd for AE-only baseline).",
    )
    parser.add_argument("--lambda_pd", type=float, default=0.1)
    parser.add_argument("--pd_every", type=int, default=10)
    parser.add_argument("--pd_batch_items", type=int, default=1)
    parser.add_argument("--pd_downsample", type=int, default=128)
    parser.add_argument("--pd_dims", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--pd_min_persistence", type=float, default=0.0)
    parser.add_argument("--pi_res", type=int, choices=[32, 64], default=64)
    parser.add_argument("--pi_sigma", type=float, default=1.5)
    parser.add_argument("--pd_loss", choices=["l2", "huber"], default="huber")
    parser.add_argument(
        "--pd_weight_persistence",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def NamespaceToConfig(args: argparse.Namespace) -> Dict[str, Any]:
    configDict: Dict[str, Any] = vars(args).copy()
    configDict["run_dirs"] = list(args.run_dirs) if args.run_dirs else None
    configDict["pd_dims"] = [int(dimValue) for dimValue in args.pd_dims]
    configDict["device"] = str(ResolveDevice(args.device))
    if not bool(configDict.get("use_pd", True)):
        configDict["lambda_pd"] = 0.0
        configDict["pd_every"] = 0
    return configDict


def main() -> None:
    parser = BuildArgParser()
    args = parser.parse_args()
    configDict = NamespaceToConfig(args)

    resultDict = TrainAutoencoder(configDict)
    print(json.dumps(resultDict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
