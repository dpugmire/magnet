#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from CAESAR.compressor import CAESAR


class FourierFeatures(nn.Module):
    def __init__(self, inDim, numFreqs):
        super().__init__()
        freqs = 2.0 ** torch.arange(numFreqs)
        self.register_buffer("freqs", freqs)
        self.inDim = int(inDim)
        self.numFreqs = int(numFreqs)

    def forward(self, x):
        # x: [N,inDim] in [-1,1]
        x = x[..., None] * self.freqs[None, None, :] * torch.pi  # [N,inDim,F]
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)       # [N,inDim,2F]
        return x.reshape(x.shape[0], -1)                          # [N,inDim*2F]


class HeatmapRenderer(nn.Module):
    """
    f(x,y,tLocal, latentFeature[64]) -> scalar
    """
    def __init__(self, latentDim=64, numCoordFreqs=8, hidden=256, layers=6):
        super().__init__()
        self.coordFf = FourierFeatures(inDim=3, numFreqs=numCoordFreqs)  # (x,y,tLocal)
        coordDim = 3 * 2 * numCoordFreqs
        inDim = coordDim + latentDim

        blocks = []
        d = inDim
        for _ in range(layers - 1):
            blocks.append(nn.Linear(d, hidden))
            blocks.append(nn.ReLU(inplace=True))
            d = hidden
        blocks.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*blocks)

    def forward(self, coordsXyt, latentFeat):
        # coordsXyt: [N,3], latentFeat: [N,64]
        enc = self.coordFf(coordsXyt)
        h = torch.cat([enc, latentFeat], dim=-1)
        return self.net(h).squeeze(-1)


def decodeTeacherFromLatent(modelPath, qLatentBtChw, tDim, device):
    """
    qLatentBtChw: [TLat,64,16,16] for B=1 (your saved case)
    returns teacher: [T,256,256] (float32)
    """
    caesar = CAESAR(model_path=modelPath, use_diffusion=False, device=device, gae_device=device)
    caesar.compressor_v.entropy_model.t_dim = int(tDim)

    tLat = int(tDim) // 4
    bt = int(qLatentBtChw.shape[0])
    if bt != tLat:
        raise ValueError(f"Expected q_latent BT={tLat} for B=1, got {bt}. Check tDim or file.")

    with torch.no_grad():
        y = caesar.compressor_v.decode(qLatentBtChw.to(device), batch_size=1)  # [1,1,T,256,256] (typical)
    y = y.detach().cpu()
    teacher = y[0, 0].float().numpy()  # [T,256,256]
    return teacher


def sampleLatentFeatures(qLatentBtChw, tIdx, xNorm, yNorm):
    """
    qLatentBtChw: [TLat,64,16,16]
    tIdx: integer frame index in [0,T-1]
    xNorm,yNorm: [N] in [-1,1] (pixel coords)
    returns: latentFeat [N,64]
    """
    tLat = int(tIdx) // 4
    latent2d = qLatentBtChw[tLat].unsqueeze(0)  # [1,64,16,16]

    # grid_sample wants grid in shape [Nbatch, Hout, Wout, 2]
    # We'll sample N points as a "1xN" row: grid [1,1,N,2]
    grid = torch.stack([xNorm, yNorm], dim=-1).view(1, 1, -1, 2)  # [1,1,N,2]
    feat = torch.nn.functional.grid_sample(
        latent2d, grid, mode="bilinear", align_corners=True
    )  # [1,64,1,N]
    feat = feat.squeeze(0).squeeze(1).transpose(0, 1).contiguous()  # [N,64]
    return feat


def trainHeatmapRenderer(qLatentBtChw, teacherT256256, tTrain, steps, batchPixels, lr, device):
    device = torch.device(device)
    qLatentBtChw = qLatentBtChw.to(device)
    teacher = torch.from_numpy(teacherT256256).to(device)  # [T,256,256]

    model = HeatmapRenderer(latentDim=64, numCoordFreqs=8, hidden=256, layers=6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        xs = torch.randint(0, 256, (batchPixels,), device=device)
        ys = torch.randint(0, 256, (batchPixels,), device=device)

        xNorm = (xs.float() / 255.0) * 2.0 - 1.0
        yNorm = (ys.float() / 255.0) * 2.0 - 1.0

        # NEW: sample a random frame index for each pixel
        tIdx = torch.randint(0, teacher.shape[0], (batchPixels,), device=device)  # [N]
        tLocal = (tIdx % 4).float() / 1.5 - 1.0                                    # [N]
        tLat = (tIdx // 4).long()                                                  # [N] in {0,1,2,3}

        # Build latent features by grouping pixels by which latent slice they need
        latentFeat = torch.empty((batchPixels, 64), device=device)

        for k in range(qLatentBtChw.shape[0]):  # qLatentBtChw is [TLat,64,16,16], so k=0..3
            mask = (tLat == k)
            if mask.any():
                # sampleLatentFeatures expects a scalar tIdx; any t with t//4==k works, so use k*4
                latentFeat[mask] = sampleLatentFeatures(qLatentBtChw, int(k * 4), xNorm[mask], yNorm[mask])

        coords = torch.stack([xNorm, yNorm, tLocal], dim=-1)                        # [N,3]
        target = teacher[tIdx, ys, xs].float()                                      # [N]

        pred = model(coords, latentFeat)
        loss = torch.mean((pred - target) ** 2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % 200 == 0:
            print(f"step {step:5d}  mse {loss.item():.6e}")


    return model.cpu()


@torch.no_grad()
def renderFrame(model, qLatentBtChw, tRender, device, chunk=65536):
    device = torch.device(device)
    model = model.to(device).eval()
    qLatentBtChw = qLatentBtChw.to(device)

    tIdx = int(tRender)
    tLocal = (tIdx % 4) / 1.5 - 1.0

    # full grid
    xs = torch.arange(256, device=device)
    ys = torch.arange(256, device=device)
    yGrid, xGrid = torch.meshgrid(ys, xs, indexing="ij")  # [256,256]
    xsFlat = xGrid.reshape(-1)
    ysFlat = yGrid.reshape(-1)

    xNorm = (xsFlat.float() / 255.0) * 2.0 - 1.0
    yNorm = (ysFlat.float() / 255.0) * 2.0 - 1.0
    tLocalTensor = torch.full_like(xNorm, float(tLocal))

    out = torch.empty((256 * 256,), device=device)

    for i in range(0, out.numel(), chunk):
        j = min(i + chunk, out.numel())
        xN = xNorm[i:j]
        yN = yNorm[i:j]
        tN = tLocalTensor[i:j]

        latentFeat = sampleLatentFeatures(qLatentBtChw, tIdx, xN, yN)
        coords = torch.stack([xN, yN, tN], dim=-1)
        out[i:j] = model(coords, latentFeat)

    img = out.view(256, 256).detach().cpu().numpy()
    return img


def saveSideBySideComparison(teacherFrame, predFrame, outPng, clipErrorPercentile=99.0):
    import numpy as np
    import matplotlib.pyplot as plt

    teacher = np.asarray(teacherFrame, dtype=np.float32)
    pred = np.asarray(predFrame, dtype=np.float32)

    vmin = float(np.min(teacher))
    vmax = float(np.max(teacher))
    if vmax <= vmin:
        vmax = vmin + 1.0

    err = np.abs(pred - teacher)
    errMax = float(np.percentile(err, clipErrorPercentile))
    if errMax <= 0.0:
        errMax = float(np.max(err)) if float(np.max(err)) > 0.0 else 1.0

    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean(err * err)))

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    im0 = ax[0].imshow(teacher, origin="lower", vmin=vmin, vmax=vmax)
    ax[0].set_title("Teacher (CAESAR decode)")
    ax[0].axis("off")
    fig.colorbar(im0, ax=ax[0], fraction=0.046, pad=0.04)

    im1 = ax[1].imshow(pred, origin="lower", vmin=vmin, vmax=vmax)
    ax[1].set_title("Predicted (learned renderer)")
    ax[1].axis("off")
    fig.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(err, origin="lower", vmin=0.0, vmax=errMax)
    ax[2].set_title(f"Abs Error |pred-teacher|\n(vmax ≈ p{clipErrorPercentile}={errMax:.4g})")
    ax[2].axis("off")
    fig.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"MAE={mae:.4g}  RMSE={rmse:.4g}  teacher range=[{vmin:.4g},{vmax:.4g}]")
    plt.savefig(outPng, dpi=150, bbox_inches="tight")
    plt.close()



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents-npz", required=True)
    parser.add_argument("--latents-key", default="all_q_latents_batch0")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--t-dim", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--t-train", type=int, default=0, help="frame index to train heatmap on (0..T-1)")
    parser.add_argument("--t-render", type=int, default=0, help="frame index to render (0..T-1)")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-pixels", type=int, default=32768)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out-png", default="heatmap.png")
    args = parser.parse_args()

    npz = np.load(args.latents_npz)
    qLatent = torch.from_numpy(npz[args.latents_key]).float()  # [TLat,64,16,16] in your saved case
    print("q_latent:", tuple(qLatent.shape))

    teacher = decodeTeacherFromLatent(args.model_path, qLatent, args.t_dim, device=args.device)
    print("teacher:", teacher.shape)

    model = trainHeatmapRenderer(
        qLatentBtChw=qLatent,
        teacherT256256=teacher,
        tTrain=args.t_train,
        steps=args.steps,
        batchPixels=args.batch_pixels,
        lr=args.lr,
        device=args.device,
    )

    img = renderFrame(model, qLatent, tRender=args.t_render, device=args.device)

    # Save the predicted heatmap as before
    plt.figure()
    plt.imshow(img, origin="lower")
    plt.colorbar()
    plt.title(f"Heatmap from learned renderer (t={args.t_render})")
    plt.savefig(args.out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print("wrote:", args.out_png)

    # NEW: save teacher vs pred vs error side-by-side
    teacherFrame = teacher[args.t_render]  # [256,256] numpy array
    outCompare = args.out_png.replace(".png", "_compare.png")
    saveSideBySideComparison(teacherFrame, img, outCompare)
    print("wrote:", outCompare)

if __name__ == "__main__":
    main()
