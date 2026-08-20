#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CAESAR.compressor import CAESAR
from dataset import ScientificDataset


class FourierFeatures(nn.Module):
    def __init__(self, inDim=3, numFreqs=8):
        super().__init__()
        freqs = 2.0 ** torch.arange(numFreqs)
        self.register_buffer("freqs", freqs)

    def forward(self, x):
        # x: [N,3] in [-1,1]
        x = x[..., None] * self.freqs[None, None, :] * torch.pi  # [N,3,F]
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)       # [N,3,2F]
        return x.reshape(x.shape[0], -1)                          # [N, 3*2F*F]


class LatentInr(nn.Module):
    def __init__(self, outDim=64, hidden=256, layers=5, numFreqs=8):
        super().__init__()
        self.ff = FourierFeatures(3, numFreqs)
        inDim = 3 * 2 * numFreqs

        blocks = []
        d = inDim
        for _ in range(layers - 1):
            blocks.append(nn.Linear(d, hidden))
            blocks.append(nn.ReLU(inplace=True))
            d = hidden
        blocks.append(nn.Linear(d, outDim))
        self.net = nn.Sequential(*blocks)

    def forward(self, coords):
        enc = self.ff(coords)
        return self.net(enc)


def makeLatentCoords(d, h, w, device):
    zz = torch.linspace(-1.0, 1.0, d, device=device)
    yy = torch.linspace(-1.0, 1.0, h, device=device)
    xx = torch.linspace(-1.0, 1.0, w, device=device)
    zGrid, yGrid, xGrid = torch.meshgrid(zz, yy, xx, indexing="ij")  # [D,H,W]
    coords = torch.stack([xGrid, yGrid, zGrid], dim=-1).reshape(-1, 3)  # [N,3]
    return coords


def reshapeQtensorToCdhw(qTensorBtChw, tDim):
    # qTensorBtChw: [BT, C, H, W], with BT = B*(tDim//4)
    bt, c, h, w = qTensorBtChw.shape
    tLat = tDim // 4
    if tLat <= 0:
        raise ValueError(f"tDim must be >= 4, got {tDim}")
    if bt % tLat != 0:
        raise ValueError(f"BT={bt} is not divisible by T/4={tLat}. Wrong tDim?")

    b = bt // tLat
    # [B, TLat, C, H, W] -> [B, C, TLat, H, W]
    q5 = qTensorBtChw.view(b, tLat, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
    return q5, b, tLat


def cdhwToBtChw(latentCdhw):
    # latentCdhw: [C, D, H, W] where D=T/4
    c, d, h, w = latentCdhw.shape
    # -> [D, C, H, W] then flatten D into batch: [D, C, H, W]
    return latentCdhw.permute(1, 0, 2, 3).contiguous().view(d, c, h, w)


def trainInrOnOneLatent(latentCdhw, steps=4000, lr=1e-3, device="cuda"):
    device = torch.device(device)
    latentCdhw = latentCdhw.to(device)

    c, d, h, w = latentCdhw.shape
    coords = makeLatentCoords(d, h, w, device)  # [N,3]
    targets = latentCdhw.permute(1, 2, 3, 0).reshape(-1, c)  # [N,C]

    model = LatentInr(outDim=c, hidden=256, layers=5, numFreqs=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        pred = model(coords)
        loss = torch.mean((pred - targets) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 200 == 0:
            print(f"step {step:5d}  mse {loss.item():.6e}")

    model.eval()
    with torch.no_grad():
        pred = model(coords).reshape(d, h, w, c).permute(3, 0, 1, 2).contiguous()  # [C,D,H,W]
    return model, pred.cpu()


def decodeWithCaesarV(modelPath, latentBtChw, tDim, device="cpu"):
    """
    latentBtChw: [BT, 64, 16, 16] where BT = B*(tDim//4)
    returns: [B, 1, tDim, 256, 256]
    """
    caesar = CAESAR(model_path=modelPath, use_diffusion=False, device=device, gae_device=device)

    tLat = int(tDim) // 4
    BT = int(latentBtChw.shape[0])
    if BT % tLat != 0:
        raise ValueError(f"BT={BT} not divisible by T/4={tLat}. Wrong tDim?")

    batchSize = BT // tLat

    # IMPORTANT: entropy_model.decode() uses entropy_model.t_dim for reshaping
    caesar.compressor_v.entropy_model.t_dim = int(tDim)

    with torch.no_grad():
        y = caesar.compressor_v.decode(latentBtChw.to(device), batchSize)

    return y.cpu()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents-npz", required=True, help="NPZ file containing q_latent (BT,64,16,16)")
    parser.add_argument("--latents-key", default="all_q_latents_batch0", help="Key inside the NPZ")
    parser.add_argument("--model-path", required=True, help="Path to caesar_v.pt")
    parser.add_argument("--t-dim", type=int, default=16, help="Original T used when compressing (e.g., 16 or 8)")
    parser.add_argument("--sample-idx", type=int, default=0, help="Which sample B index to train on")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--out-npz", default="decoded_from_inr.npz")
    args = parser.parse_args()

    npz = np.load(args.latents_npz)
    qLatent = torch.from_numpy(npz[args.latents_key]).float()  # [BT,64,16,16]
    print("loaded q_latent:", tuple(qLatent.shape), qLatent.dtype)

    q5, b, tLat = reshapeQtensorToCdhw(qLatent, args.t_dim)  # [B,64,TLat,16,16]
    print("reshaped to:", tuple(q5.shape), "=> B=", b, "TLat=", tLat)

    if args.sample_idx < 0 or args.sample_idx >= b:
        raise ValueError(f"--sample-idx {args.sample_idx} out of range [0,{b-1}]")

    latentCdhw = q5[args.sample_idx]  # [64,TLat,16,16]
    print("training INR on one latent field:", tuple(latentCdhw.shape))

    _, latentPredCdhw = trainInrOnOneLatent(latentCdhw, steps=args.steps, device=args.device)
    print("INR sampled latent:", tuple(latentPredCdhw.shape))

    latentPredBtChw = cdhwToBtChw(latentPredCdhw)  # [TLat,64,16,16] since B=1 here
    print("latent for decode:", tuple(latentPredBtChw.shape))

    decoded = decodeWithCaesarV(args.model_path, latentPredBtChw, args.t_dim, device=args.device)
    print("decoded shape:", tuple(decoded.shape))  # [B,1,T,256,256]

    decodedT = decoded[0, 0].numpy()  # [T,256,256] for sample 0

    # Reshape decoded back to [T, H, W] for convenience (B=1, C=1)
    tDim = args.t_dim
    decodedBthw = decoded.view(1, tDim, *decoded.shape[-2:])  # [1,T,H,W]
    decodedT = decodedBthw[0].numpy()                         # [T,H,W]

    np.savez(
        args.out_npz,
        decoded=decodedT,
        latentPred=latentPredCdhw.numpy(),
        tDim=np.array([tDim]),
    )
    print("wrote:", args.out_npz)


if __name__ == "__main__":
    main()
