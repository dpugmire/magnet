#!/usr/bin/env python3
"""Compress a Turb-Rot subset with CAESAR and optionally save its first latent batch."""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from CAESAR.compressor import CAESAR
from dataset import ScientificDataset


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=16)
    parser.add_argument("--n-frame", type=int, default=8)
    parser.add_argument("--error-bound", type=float, default=0.1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latents-out", default="all_q_latents_batch0.npz")
    return parser.parse_args()


def main():
    args = parseArgs()
    dataArgs = {
        "data_path": args.data_path,
        "variable_idx": [0],
        "section_range": [0, 1],
        "frame_range": [args.frame_start, args.frame_end],
        "n_frame": args.n_frame,
    }

    compressor = CAESAR(
        model_path=args.model_path,
        use_diffusion=False,
        device=args.device,
        gae_device=args.device,
    )
    dataset = ScientificDataset(dataArgs)
    print("dataset shape:", dataset.data_input.shape)

    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    compressedData, compressedSize = compressor.compress(
        dataloader, eb=args.error_bound
    )
    latents = [batch["q_latent"] for batch in compressedData["latent"]]
    firstLatent = latents[0]
    print(
        "latents batch0 shape, type, min, max:",
        firstLatent.shape,
        firstLatent.dtype,
        torch.min(firstLatent),
        torch.max(firstLatent),
    )
    np.savez(
        args.latents_out,
        all_q_latents_batch0=firstLatent.detach().cpu().numpy(),
    )

    reconstructed = compressor.decompress(compressedData)
    original = dataset.input_data()
    reconstructed = dataset.recons_data(reconstructed)
    nrmse = torch.sqrt(torch.mean((original - reconstructed) ** 2)) / (
        torch.max(original) - torch.min(original)
    )
    compressionRatio = np.prod(original.shape) * 8 / compressedSize
    print("NRMSE:", nrmse.item(), "CR:", compressionRatio.item())
    print(original.shape, reconstructed.shape)


if __name__ == "__main__":
    main()
