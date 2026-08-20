#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional but nice for "mesh-less contour" extraction from samples
try:
    from skimage import measure
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

# Topology / persistence
try:
    import gudhi as gd
    HAVE_GUDHI = True
except Exception:
    HAVE_GUDHI = False

try:
    from persim import bottleneck, wasserstein
    HAVE_PERSIM = True
except Exception:
    HAVE_PERSIM = False


# ----------------------------
# 1) Synthetic scalar fields
# ----------------------------
def make_synthetic_field(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """
    Produce a smooth-ish scalar field with some blobs + waves + a ridge.
    Returns float32 array in range roughly [0, 1].
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx / (w - 1)) * 2.0 - 1.0  # [-1, 1]
    y = (yy / (h - 1)) * 2.0 - 1.0  # [-1, 1]

    # Random gaussian blobs
    field = np.zeros((h, w), dtype=np.float32)
    num_blobs = rng.integers(2, 5)
    for _ in range(num_blobs):
        cx = rng.uniform(-0.7, 0.7)
        cy = rng.uniform(-0.7, 0.7)
        sx = rng.uniform(0.10, 0.35)
        sy = rng.uniform(0.10, 0.35)
        amp = rng.uniform(0.6, 1.2)
        blob = amp * np.exp(-(((x - cx) ** 2) / (2 * sx**2) + ((y - cy) ** 2) / (2 * sy**2)))
        field += blob.astype(np.float32)

    # Smooth wave component
    fx = rng.uniform(1.0, 3.5)
    fy = rng.uniform(1.0, 3.5)
    phase = rng.uniform(0, 2 * math.pi)
    wave = 0.25 * (np.sin(fx * math.pi * x + phase) * np.cos(fy * math.pi * y - phase)).astype(np.float32)
    field += wave

    # A ridge (like a front)
    angle = rng.uniform(0, math.pi)
    nx, ny = math.cos(angle), math.sin(angle)
    d = (nx * x + ny * y).astype(np.float32)
    ridge = 0.35 * (1.0 / (1.0 + np.exp(-10.0 * (d - rng.uniform(-0.2, 0.2))))).astype(np.float32)
    field += ridge

    # Normalize to [0, 1]
    field -= field.min()
    field /= (field.max() + 1e-8)
    return field.astype(np.float32)


class FieldDataset(Dataset):
    def __init__(self, n: int, h: int, w: int, seed: int = 0):
        self.n = n
        self.h = h
        self.w = w
        self.rng = np.random.default_rng(seed)
        self.data = np.stack([make_synthetic_field(h, w, self.rng) for _ in range(n)], axis=0)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.data[idx][None, :, :]  # [1,H,W]
        return torch.from_numpy(x)


# ----------------------------
# 2) Autoencoder (small CNN)
# ----------------------------
class ConvAE(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 64->32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32->16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16->8
            nn.ReLU(True),
        )
        self.enc_fc = nn.Linear(128 * 8 * 8, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8->16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16->32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 32->64
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        z = self.enc_fc(h)
        return z

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 128, 8, 8)
        return self.dec(h)

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z


# ----------------------------------------
# 3) Implicit f(z,x,y) with Fourier feats
# ----------------------------------------
class FourierFeatures(nn.Module):
    """
    Positional encoding: maps (x,y) -> [sin(Bx), cos(Bx)] features.
    """
    def __init__(self, in_dim=2, num_frequencies=6):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.freqs = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)

    def forward(self, xy):
        freqs = self.freqs.to(xy.device)[None, :]  # [1,F]
        x = xy[:, :, None] * freqs                 # [N,2,F]
        x = math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1).flatten(1)


class ImplicitF(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 128, num_layers: int = 4, num_freq: int = 6):
        super().__init__()
        self.ff = FourierFeatures(in_dim=2, num_frequencies=num_freq)
        ff_dim = 2 * 2 * num_freq
        in_dim = latent_dim + ff_dim

        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(hidden, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z, xy):
        if z.dim() == 1:
            z = z[None, :]
        if z.shape[0] != 1:
            raise ValueError("Pass a single latent vector z ([latent_dim] or [1,latent_dim]).")

        ff = self.ff(xy)                  # [N,ff_dim]
        zrep = z.expand(xy.shape[0], -1)  # [N,latent_dim]
        inp = torch.cat([zrep, ff], dim=1)
        return self.mlp(inp)


# ----------------------------
# Utilities
# ----------------------------
def make_coord_grid(h: int, w: int, device):
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing="ij",
    )
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # [H*W,2]
    return xy


def domain_to_pixel(pt, xmin, xmax, ymin, ymax, w, h):
    x, y = pt
    u = (x - xmin) / (xmax - xmin) * (w - 1)
    v = (y - ymin) / (ymax - ymin) * (h - 1)
    return (u, v)


# ----------------------------
# Persistence helpers
# ----------------------------
def compute_persistence_diagrams_2d(field2d: np.ndarray, superlevel: bool = False):
    if not HAVE_GUDHI:
        raise RuntimeError("gudhi not installed. `pip install gudhi`")
    f = np.asarray(field2d, dtype=np.float64)
    if superlevel:
        f = -f

    cc = gd.CubicalComplex(top_dimensional_cells=f)
    cc.persistence()

    dgm0 = cc.persistence_intervals_in_dimension(0)
    dgm1 = cc.persistence_intervals_in_dimension(1)

    def drop_inf(dgm):
        if dgm.size == 0:
            return dgm
        finite = np.isfinite(dgm[:, 1])
        return dgm[finite]

    return drop_inf(dgm0), drop_inf(dgm1)


def filter_by_persistence(dgm: np.ndarray, min_lifetime: float = 0.0, top_k: int | None = None):
    """
    Keep only features with (death-birth) >= min_lifetime, and optionally keep only top_k by lifetime.
    """
    if dgm.size == 0:
        return dgm

    life = dgm[:, 1] - dgm[:, 0]
    keep = life >= float(min_lifetime)
    dgm2 = dgm[keep]

    if dgm2.size == 0:
        return dgm2

    if top_k is not None:
        k = int(top_k)
        if k > 0 and dgm2.shape[0] > k:
            life2 = dgm2[:, 1] - dgm2[:, 0]
            idx = np.argsort(-life2)[:k]
            dgm2 = dgm2[idx]
    return dgm2


def plot_two_pd(dgmA, dgmB, titleA="Truth", titleB="INR", suptitle="Persistence diagrams"):
    def plot_one(ax, dgm, title):
        ax.set_title(title)
        ax.set_xlabel("birth")
        ax.set_ylabel("death")
        ax.grid(True, alpha=0.3)

        if dgm.size == 0:
            return

        b = dgm[:, 0]
        d = dgm[:, 1]
        mn = float(min(b.min(), d.min()))
        mx = float(max(b.max(), d.max()))
        pad = 0.05 * (mx - mn + 1e-12)
        mn -= pad
        mx += pad

        ax.scatter(b, d, s=12)
        ax.plot([mn, mx], [mn, mx], linewidth=1)
        ax.set_xlim(mn, mx)
        ax.set_ylim(mn, mx)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    plot_one(axs[0], dgmA, titleA)
    plot_one(axs[1], dgmB, titleB)
    fig.suptitle(suptitle)
    plt.show()


def diagram_distances(dgmA, dgmB):
    if not HAVE_PERSIM:
        return None
    bn = bottleneck(dgmA, dgmB)
    w1 = wasserstein(dgmA, dgmB, matching=False)
    return {"bottleneck": float(bn), "wasserstein1": float(w1)}


# ----------------------------
# Regularizers (options)
# ----------------------------
def total_variation_loss(img2d: torch.Tensor):
    """
    img2d: [H,W] or [1,1,H,W]
    Returns isotropic TV-ish penalty.
    """
    if img2d.dim() == 2:
        x = img2d[None, None, :, :]
    else:
        x = img2d
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return (dx.abs().mean() + dy.abs().mean())


def coord_gradient_penalty(f_model: nn.Module, z0: torch.Tensor, xy: torch.Tensor):
    """
    Penalize |df/dx|^2 + |df/dy|^2 on a batch of xy points.
    Helps suppress high-frequency ringing.
    xy: [N,2] in [-1,1], requires_grad will be enabled here.
    """
    xy = xy.clone().detach().requires_grad_(True)
    pred = f_model(z0, xy)  # [N,1]
    g = torch.autograd.grad(
        outputs=pred,
        inputs=xy,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]  # [N,2]
    return (g.pow(2).sum(dim=1)).mean()


@dataclass
class TopologyOptions:
    # PD computation resolution (compare on same grid)
    pd_res: int = 64  # 64 or 256
    # Persistence filtering options (set min_lifetime > 0 to remove near-diagonal noise)
    min_lifetime_h0: float = 0.02
    min_lifetime_h1: float = 0.02
    top_k_h0: int | None = None
    top_k_h1: int | None = None
    # Use sublevel (False) or superlevel (True)
    superlevel: bool = False

@dataclass
class RegularizerOptions:
    # Choose one (or combine): "none", "grad", "tv"
    mode: str = "grad"
    weight: float = 1e-3


# ----------------------------
# Main demo
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    topo = TopologyOptions(
        pd_res=64,             # compare PDs on 64x64 for both
        min_lifetime_h0=0.00,  # filter tiny features
        min_lifetime_h1=0.00,
        top_k_h0=None,         # e.g. 50 if you want
        top_k_h1=None,
        superlevel=False,      # set True to analyze peaks instead of basins
    )
    reg = RegularizerOptions(
        mode="grad",           # "none" | "grad" | "tv"
        weight=1e-3,
    )

    # Data
    H, W = 64, 64
    train_ds = FieldDataset(n=2000, h=H, w=W, seed=1)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)

    # 1) Train AE
    latent_dim = 64
    ae = ConvAE(latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)

    ae.train()
    for epoch in range(8):
        total = 0.0
        for x in train_loader:
            x = x.to(device)
            xhat, _ = ae(x)
            loss = F.mse_loss(xhat, x)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * x.shape[0]
        print(f"AE epoch {epoch+1:02d}  mse={total/len(train_ds):.6f}")

    # Pick one field
    ae.eval()
    with torch.no_grad():
        x0 = train_ds[0].to(device)[None, ...]  # [1,1,H,W]
        x0_hat, z0 = ae(x0)
        z0 = z0[0].detach()  # [latent_dim]
    x0_np = x0[0, 0].cpu().numpy()

    # 2) Train implicit f(z,x,y) for THIS field (fixed z0)
    f = ImplicitF(latent_dim=latent_dim, hidden=128, num_layers=4, num_freq=6).to(device)
    print(f"ImplicitF latent_dim={latent_dim}, total params={sum(p.numel() for p in f.parameters())}")
    opt_f = torch.optim.Adam(f.parameters(), lr=2e-3)

    xy_all = make_coord_grid(H, W, device=device)  # [H*W,2]
    t_all = torch.from_numpy(x0_np.reshape(-1, 1)).to(device)  # [H*W,1]

    f.train()
    batch_n = 4096
    steps = 2000
    for step in range(steps):
        idx = torch.randint(0, xy_all.shape[0], (batch_n,), device=device)
        xy = xy_all[idx]
        t = t_all[idx]

        pred = f(z0, xy)
        mse = F.mse_loss(pred, t)

        # Regularization options
        reg_loss = torch.tensor(0.0, device=device)
        if reg.mode == "grad":
            reg_loss = coord_gradient_penalty(f, z0, xy)  # uses same xy samples
        elif reg.mode == "tv":
            # TV on a small render grid (cheap)
            with torch.no_grad():
                xy_small = make_coord_grid(64, 64, device=device)
            tv_img = f(z0, xy_small).reshape(64, 64)
            reg_loss = total_variation_loss(tv_img)

        loss = mse + reg.weight * reg_loss

        opt_f.zero_grad(set_to_none=True)
        loss.backward()
        opt_f.step()

        if (step + 1) % 400 == 0:
            print(f"f step {step+1:04d}/{steps} mse={float(mse):.6f} reg={float(reg_loss):.6f} total={float(loss):.6f}")

    # 3) Render at 256 for viz
    f.eval()
    with torch.no_grad():
        render_h, render_w = 256, 256
        xy_r = make_coord_grid(render_h, render_w, device=device)
        t_r = f(z0, xy_r).reshape(render_h, render_w).clamp(0, 1).cpu().numpy()

    # 4) Difference (INR - truth upsampled) for plotting
    with torch.no_grad():
        x0_t = torch.from_numpy(x0_np)[None, None, :, :].to(device)
        x0_up = F.interpolate(x0_t, size=(render_h, render_w), mode="bilinear", align_corners=True)
        diff = (torch.from_numpy(t_r)[None, None, :, :].to(device) - x0_up).squeeze().cpu().numpy()

    # ------------------------------------------------------------
    # Persistence: compute on SAME resolution (default: 64x64 for both)
    # ------------------------------------------------------------
    if HAVE_GUDHI:
        if topo.pd_res == 64:
            truth_pd_field = x0_np.astype(np.float64)  # 64x64
            # downsample INR render to 64x64 via avg pooling
            inr_t = torch.from_numpy(t_r)[None, None, :, :].float().to(device)
            inr_ds = F.avg_pool2d(inr_t, kernel_size=4, stride=4)  # 256->64
            inr_pd_field = inr_ds[0, 0].detach().cpu().numpy().astype(np.float64)
        elif topo.pd_res == 256:
            truth_pd_field = x0_up[0, 0].detach().cpu().numpy().astype(np.float64)  # 256x256
            inr_pd_field = t_r.astype(np.float64)
        else:
            raise ValueError("topo.pd_res must be 64 or 256")

        dgm0_truth, dgm1_truth = compute_persistence_diagrams_2d(truth_pd_field, superlevel=topo.superlevel)
        dgm0_inr, dgm1_inr = compute_persistence_diagrams_2d(inr_pd_field, superlevel=topo.superlevel)

        # Filter by persistence/lifetime (and optionally top-k)
        dgm0_truth_f = filter_by_persistence(dgm0_truth, topo.min_lifetime_h0, topo.top_k_h0)
        dgm0_inr_f   = filter_by_persistence(dgm0_inr,   topo.min_lifetime_h0, topo.top_k_h0)

        dgm1_truth_f = filter_by_persistence(dgm1_truth, topo.min_lifetime_h1, topo.top_k_h1)
        dgm1_inr_f   = filter_by_persistence(dgm1_inr,   topo.min_lifetime_h1, topo.top_k_h1)

        print(f"\nPD computation resolution: {topo.pd_res}x{topo.pd_res}")
        print(f"Persistence type: {'superlevel (peaks)' if topo.superlevel else 'sublevel (basins)'}")
        print(f"Raw PD sizes:     H0 truth={len(dgm0_truth)} H0 inr={len(dgm0_inr)}   H1 truth={len(dgm1_truth)} H1 inr={len(dgm1_inr)}")
        print(f"Filtered PD sizes: H0 truth={len(dgm0_truth_f)} H0 inr={len(dgm0_inr_f)}   H1 truth={len(dgm1_truth_f)} H1 inr={len(dgm1_inr_f)}")
        print(f"Filters: H0 min_life={topo.min_lifetime_h0}, top_k={topo.top_k_h0}   H1 min_life={topo.min_lifetime_h1}, top_k={topo.top_k_h1}")

        plot_two_pd(
            dgm0_truth_f, dgm0_inr_f,
            titleA=f"Truth (H0) filtered", titleB=f"INR (H0) filtered",
            suptitle="Persistence Diagram H0 (components)"
        )
        plot_two_pd(
            dgm1_truth_f, dgm1_inr_f,
            titleA=f"Truth (H1) filtered", titleB=f"INR (H1) filtered",
            suptitle="Persistence Diagram H1 (loops)"
        )

        if HAVE_PERSIM:
            d0 = diagram_distances(dgm0_truth_f, dgm0_inr_f)
            d1 = diagram_distances(dgm1_truth_f, dgm1_inr_f)
            print("Diagram distances (filtered):")
            print("  H0:", d0)
            print("  H1:", d1)
        else:
            print("Install persim for distances: pip install persim")
    else:
        print("Install gudhi for persistence diagrams: pip install gudhi")

    # ------------------------------------------------------------
    # Visuals: truth, INR, diff
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(14, 5))
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title("Truth (64×64)")
    ax1.imshow(x0_np, origin="lower")
    plt.colorbar(ax1.images[0], ax=ax1, fraction=0.046, pad=0.04)

    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title("INR render (256×256)")
    ax2.imshow(t_r, origin="lower")
    plt.colorbar(ax2.images[0], ax=ax2, fraction=0.046, pad=0.04)

    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title("Diff: INR - truth(upsampled)")
    imd = ax3.imshow(diff, origin="lower")
    plt.colorbar(imd, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
