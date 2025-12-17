#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
import time

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
        # shape: [N, H, W]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.data[idx][None, :, :]  # add channel dim: [1,H,W]
        return torch.from_numpy(x)


# ----------------------------
# 2) Autoencoder (small CNN)
# ----------------------------
class ConvAE(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: [B,1,64,64] -> [B,128,8,8] -> [B,latent_dim]
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 64->32
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32->16
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16->8
            nn.ReLU(True),
        )
        self.enc_fc = nn.Linear(128 * 8 * 8, latent_dim)

        # Decoder: latent -> [B,128,8,8] -> [B,1,64,64]
        self.dec_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8->16
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16->32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 32->64
            nn.Sigmoid(),  # output in [0,1]
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.flatten(1)
        z = self.enc_fc(h)
        return z

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 128, 8, 8)
        xhat = self.dec(h)
        return xhat

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
    Helps represent sharper detail than raw (x,y).
    """
    def __init__(self, in_dim=2, num_frequencies=6):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        # Frequencies: 1,2,4,... (log scale)
        self.freqs = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32)

    def forward(self, xy):
        # xy: [N,2] in [-1,1]
        # returns [N, 2*in_dim*num_frequencies]
        freqs = self.freqs.to(xy.device)[None, :]  # [1,F]
        x = xy[:, :, None] * freqs  # [N,2,F]
        x = math.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1).flatten(1)


class ImplicitF(nn.Module):
    def __init__(self, latent_dim: int, hidden: int = 128, num_layers: int = 4, num_freq: int = 6):
        super().__init__()
        self.ff = FourierFeatures(in_dim=2, num_frequencies=num_freq)
        ff_dim = 2 * 2 * num_freq  # sin/cos * (x,y) * frequencies

        in_dim = latent_dim + ff_dim
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden, hidden))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(hidden, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, z, xy):
        """
        z:  [B,latent_dim] OR [latent_dim] (we'll expand as needed)
        xy: [N,2] coords in [-1,1]
        returns: [N,1]
        """
        if z.dim() == 1:
            z = z[None, :]
        if z.shape[0] != 1:
            raise ValueError("For this demo, pass a single latent vector z (shape [latent_dim] or [1,latent_dim]).")

        ff = self.ff(xy)                  # [N,ff_dim]
        zrep = z.expand(xy.shape[0], -1)  # [N,latent_dim]
        inp = torch.cat([zrep, ff], dim=1)
        out = self.mlp(inp)
        return out


# ----------------------------
# Utilities
# ----------------------------
def make_coord_grid(h: int, w: int, device):
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=device),
        torch.linspace(-1.0, 1.0, w, device=device),
        indexing="ij",
    )
    xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)  # [H*W,2] (x,y)
    return xy


def marching_squares_segments(data: np.ndarray, level: float):
    """
    Minimal marching-squares that iterates every pixel cell and returns a list of line
    segments approximating the iso-contour at `level`. Each segment is a pair of points
    ((x0, y0), (x1, y1)) in pixel coordinates (x: column, y: row).
    """
    h, w = data.shape
    segments = []

    def interp(p0, p1, v0, v1):
        if v0 == v1:
            t = 0.5
        else:
            t = (level - v0) / (v1 - v0)
        t = min(max(t, 0.0), 1.0)
        return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))

    case_to_edges = {
        0: [],
        1: [(3, 2)],
        2: [(2, 1)],
        3: [(3, 1)],
        4: [(0, 1)],
        5: [(0, 1), (3, 2)],  # saddle; split into two segments
        6: [(0, 2)],
        7: [(3, 0)],
        8: [(3, 0)],
        9: [(0, 2)],
        10: [(0, 1), (2, 3)],  # saddle; split into two segments
        11: [(1, 2)],
        12: [(1, 3)],
        13: [(2, 3)],
        14: [(0, 1)],
        15: [],
    }

    for r in range(h - 1):
        for c in range(w - 1):
            v00 = float(data[r, c])       # top-left
            v10 = float(data[r, c + 1])   # top-right
            v11 = float(data[r + 1, c + 1])  # bottom-right
            v01 = float(data[r + 1, c])   # bottom-left

            idx = 0
            idx |= (v00 >= level) << 3
            idx |= (v10 >= level) << 2
            idx |= (v11 >= level) << 1
            idx |= (v01 >= level) << 0

            edge_pairs = case_to_edges[idx]
            if not edge_pairs:
                continue

            p0 = (c, r)
            p1 = (c + 1, r)
            p2 = (c + 1, r + 1)
            p3 = (c, r + 1)

            edge_points = {
                0: interp(p0, p1, v00, v10),  # top
                1: interp(p1, p2, v10, v11),  # right
                2: interp(p2, p3, v11, v01),  # bottom
                3: interp(p3, p0, v01, v00),  # left
            }

            for e0, e1 in edge_pairs:
                segments.append((edge_points[e0], edge_points[e1]))

    return segments


# --------------------------------------------------------------------
#  Quadtree mesh-less contouring for implicit fields f(z,x,y)
# --------------------------------------------------------------------
def marching_squares_cell_segments(x0, y0, x1, y1, v00, v10, v11, v01, iso):
    """
    Marching squares on ONE cell in domain coordinates.

    Corner layout (x increases right, y increases up):
        (x0,y1) v01 ---- v11 (x1,y1)
                 |        |
                 |        |
        (x0,y0) v00 ---- v10 (x1,y0)

    Returns list of segments: [((xa,ya),(xb,yb)), ...]
    Uses standard 16-case table with a simple center-value decider for ambiguous cases.
    """
    def interp(pa, pb, va, vb):
        if vb == va:
            t = 0.5
        else:
            t = (iso - va) / (vb - va)
        t = float(np.clip(t, 0.0, 1.0))
        return (pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]))

    p00 = (x0, y0)
    p10 = (x1, y0)
    p11 = (x1, y1)
    p01 = (x0, y1)

    b00 = 1 if v00 >= iso else 0
    b10 = 1 if v10 >= iso else 0
    b11 = 1 if v11 >= iso else 0
    b01 = 1 if v01 >= iso else 0
    case_id = (b00 << 0) | (b10 << 1) | (b11 << 2) | (b01 << 3)

    def edge_cross(va, vb):
        return (va < iso and vb >= iso) or (va >= iso and vb < iso)

    e = {}
    if edge_cross(v00, v10):
        e[0] = interp(p00, p10, v00, v10)  # bottom
    if edge_cross(v10, v11):
        e[1] = interp(p10, p11, v10, v11)  # right
    if edge_cross(v01, v11):
        e[2] = interp(p01, p11, v01, v11)  # top
    if edge_cross(v00, v01):
        e[3] = interp(p00, p01, v00, v01)  # left

    table = {
        0: [],
        1: [(3, 0)],
        2: [(0, 1)],
        3: [(3, 1)],
        4: [(1, 2)],
        5: "ambiguous",
        6: [(0, 2)],
        7: [(3, 2)],
        8: [(2, 3)],
        9: [(0, 2)],
        10: "ambiguous",
        11: [(1, 2)],
        12: [(1, 3)],
        13: [(0, 1)],
        14: [(3, 0)],
        15: [],
    }

    entry = table[case_id]
    if entry == []:
        return []

    if entry == "ambiguous":
        # Simple decider using center value
        vc = 0.25 * (v00 + v10 + v11 + v01)
        if case_id == 5:
            pairs = [(3, 2), (0, 1)] if vc >= iso else [(3, 0), (2, 1)]
        else:  # 10
            pairs = [(3, 0), (2, 1)] if vc >= iso else [(3, 2), (0, 1)]
        segs = []
        for a, b in pairs:
            if a in e and b in e:
                segs.append((e[a], e[b]))
        return segs

    segs = []
    for a, b in entry:
        if a in e and b in e:
            segs.append((e[a], e[b]))
    return segs


class QuadtreeContour2D:
    """
    Adaptive quadtree isocontouring for an implicit scalar field T(x,y).

    Refine rule:
      - If iso is within [minCorner,maxCorner] AND cell_size > min_size AND depth < max_depth -> subdivide
      - Otherwise keep as leaf.

    Contour extraction:
      - Marching squares on each leaf cell that intersects iso.
    """
    def __init__(self, eval_func, device=None, batch_size=131072, cache_quant=1e-12):
        self.eval_func = eval_func
        self.device = device if device is not None else torch.device("cpu")
        self.batch_size = int(batch_size)
        self.cache_quant = float(cache_quant)
        self.cache = {}

    def _key(self, x, y):
        q = self.cache_quant
        return (int(round(x / q)), int(round(y / q)))

    def sample_points(self, pts):
        """
        pts: list[(x,y)]
        returns: list[float]
        """
        out = [0.0] * len(pts)
        missing = []
        missing_idx = []

        for i, (x, y) in enumerate(pts):
            k = self._key(x, y)
            if k in self.cache:
                out[i] = self.cache[k]
            else:
                missing.append((x, y))
                missing_idx.append(i)

        if missing:
            xy = torch.tensor(missing, dtype=torch.float32, device=self.device)
            vals = []
            with torch.no_grad():
                for s in range(0, xy.shape[0], self.batch_size):
                    chunk = xy[s : s + self.batch_size]
                    v = self.eval_func(chunk)
                    if v.dim() == 2 and v.shape[1] == 1:
                        v = v[:, 0]
                    vals.append(v.detach().float().cpu())
            vals = torch.cat(vals, dim=0).numpy()

            for j, (x, y) in enumerate(missing):
                k = self._key(x, y)
                vv = float(vals[j])
                self.cache[k] = vv
                out[missing_idx[j]] = vv

        return out

    def contour(
        self,
        iso,
        xmin=-1.0,
        xmax=1.0,
        ymin=-1.0,
        ymax=1.0,
        max_depth=10,
        min_depth=2,          # <<< NEW: force a little refinement
        min_size=None,
        use_midpoints=True,   # <<< NEW: sample edge midpoints + center for intersection test
    ):
        if min_size is None:
            min_size = max((xmax - xmin), (ymax - ymin)) / (2 ** max_depth)

        def sample_corners(x0, y0, x1, y1):
            # order: v00(x0,y0), v10(x1,y0), v11(x1,y1), v01(x0,y1)
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            v00, v10, v11, v01 = self.sample_points(pts)
            return v00, v10, v11, v01

        def sample_range_points(x0, y0, x1, y1):
            # Points used ONLY to decide if iso might exist in this cell.
            xm = 0.5 * (x0 + x1)
            ym = 0.5 * (y0 + y1)

            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (xm, ym)]  # corners + center
            if use_midpoints:
                pts.extend([(xm, y0), (x1, ym), (xm, y1), (x0, ym)])  # edge midpoints

            vals = self.sample_points(pts)
            return min(vals), max(vals)

        # Root node
        v00, v10, v11, v01 = sample_corners(xmin, ymin, xmax, ymax)
        stack = [(xmin, ymin, xmax, ymax, 0, v00, v10, v11, v01)]

        leaves = []
        refined = 0

        while stack:
            x0, y0, x1, y1, depth, v00, v10, v11, v01 = stack.pop()

            size = max(x1 - x0, y1 - y0)
            can_refine = (depth < max_depth) and (size > min_size)

            # Robust intersection test using more samples than corners
            rmin, rmax = sample_range_points(x0, y0, x1, y1)
            intersects = (rmin <= iso <= rmax)

            # Safety: always refine a little at the top so we don't miss interior features
            must_refine = (depth < min_depth)

            if can_refine and (must_refine or intersects):
                xm = 0.5 * (x0 + x1)
                ym = 0.5 * (y0 + y1)

                children = [
                    (x0, y0, xm, ym),  # SW
                    (xm, y0, x1, ym),  # SE
                    (xm, ym, x1, y1),  # NE
                    (x0, ym, xm, y1),  # NW
                ]
                for (cx0, cy0, cx1, cy1) in children:
                    cv00, cv10, cv11, cv01 = sample_corners(cx0, cy0, cx1, cy1)
                    stack.append((cx0, cy0, cx1, cy1, depth + 1, cv00, cv10, cv11, cv01))
                refined += 1
            else:
                leaves.append((x0, y0, x1, y1, v00, v10, v11, v01))

        # Extract segments from leaves (use CORNERS only for marching squares)
        segments = []
        used_leaves = 0
        for x0, y0, x1, y1, v00, v10, v11, v01 in leaves:
            rmin, rmax = sample_range_points(x0, y0, x1, y1)
            if not (rmin <= iso <= rmax):
                continue
            used_leaves += 1
            segments.extend(marching_squares_cell_segments(x0, y0, x1, y1, v00, v10, v11, v01, iso))

        stats = {
            "numLeavesTotal": len(leaves),
            "numLeavesIntersecting": used_leaves,
            "numRefinedNodes": refined,
            "numSegments": len(segments),
            "numCachedSamples": len(self.cache),
        }
        return segments, stats


    def contour_(self, iso, xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0, max_depth=10, min_size=None):
        if min_size is None:
            min_size = max((xmax - xmin), (ymax - ymin)) / (2 ** max_depth)

        def node_corners(x0, y0, x1, y1):
            # order: v00(x0,y0), v10(x1,y0), v11(x1,y1), v01(x0,y1)
            pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            v00, v10, v11, v01 = self.sample_points(pts)
            return v00, v10, v11, v01

        root_v = node_corners(xmin, ymin, xmax, ymax)
        stack = [(xmin, ymin, xmax, ymax, 0, *root_v)]

        leaves = []
        refined = 0

        while stack:
            x0, y0, x1, y1, depth, v00, v10, v11, v01 = stack.pop()

            vmin = min(v00, v10, v11, v01)
            vmax = max(v00, v10, v11, v01)
            size = max(x1 - x0, y1 - y0)

            intersects = (vmin <= iso <= vmax)
            can_refine = (depth < max_depth) and (size > min_size)

            if intersects and can_refine:
                xm = 0.5 * (x0 + x1)
                ym = 0.5 * (y0 + y1)

                children = [
                    (x0, y0, xm, ym),  # SW
                    (xm, y0, x1, ym),  # SE
                    (xm, ym, x1, y1),  # NE
                    (x0, ym, xm, y1),  # NW
                ]
                for (cx0, cy0, cx1, cy1) in children:
                    cv = node_corners(cx0, cy0, cx1, cy1)
                    stack.append((cx0, cy0, cx1, cy1, depth + 1, *cv))
                refined += 1
            else:
                leaves.append((x0, y0, x1, y1, v00, v10, v11, v01))

        segments = []
        used_leaves = 0
        for x0, y0, x1, y1, v00, v10, v11, v01 in leaves:
            vmin = min(v00, v10, v11, v01)
            vmax = max(v00, v10, v11, v01)
            if not (vmin <= iso <= vmax):
                continue
            used_leaves += 1
            segments.extend(marching_squares_cell_segments(x0, y0, x1, y1, v00, v10, v11, v01, iso))

        stats = {
            "numLeavesTotal": len(leaves),
            "numLeavesIntersecting": used_leaves,
            "numRefinedNodes": refined,
            "numSegments": len(segments),
            "numCachedSamples": len(self.cache),
        }
        return segments, stats


def make_eval_func_from_model(f_model, z0, device):
    z0 = z0.detach().to(device)

    def eval_func(xy):
        out = f_model(z0, xy)
        if out.dim() == 2 and out.shape[1] == 1:
            out = out[:, 0]
        return out.clamp(0.0, 1.0)   # <<< IMPORTANT
    return eval_func


def make_eval_func_from_model_(f_model, z0, device):
    """
    Returns eval_func(xy[N,2]) -> tensor[N]
    where xy is in [-1,1] domain coordinates.
    """
    z0 = z0.detach().to(device)

    def eval_func(xy):
        out = f_model(z0, xy)
        if out.dim() == 2 and out.shape[1] == 1:
            out = out[:, 0]
        return out

    return eval_func


def domain_to_pixel(pt, xmin, xmax, ymin, ymax, w, h):
    """
    Map domain coords (x,y) in [xmin,xmax]x[ymin,ymax] to pixel coords (col,row)
    consistent with imshow(origin="lower").
    """
    x, y = pt
    u = (x - xmin) / (xmax - xmin) * (w - 1)
    v = (y - ymin) / (ymax - ymin) * (h - 1)
    return (u, v)


# ----------------------------
# Main demo
# ----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

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

    # Pick one field to demonstrate
    ae.eval()
    with torch.no_grad():
        x0 = train_ds[0].to(device)[None, ...]  # [1,1,H,W]
        x0_hat, z0 = ae(x0)
        z0 = z0[0].detach()  # [latent_dim]
    print(f"latent_dim={latent_dim}, z0 shape={tuple(z0.shape)}, numel={z0.numel()}")
    x0_np = x0[0, 0].cpu().numpy()
    x0_hat_np = x0_hat[0, 0].cpu().numpy()

    # 2) Train implicit f(z,x,y) for THIS field (fixed z0)
    f = ImplicitF(latent_dim=latent_dim, hidden=128, num_layers=4, num_freq=6).to(device)
    implicit_params = sum(p.numel() for p in f.parameters())
    print(f"ImplicitF latent_dim={latent_dim}, total params={implicit_params}")
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
        loss = F.mse_loss(pred, t)

        opt_f.zero_grad(set_to_none=True)
        loss.backward()
        opt_f.step()

        if (step + 1) % 400 == 0:
            print(f"f step {step+1:04d}/{steps} mse={float(loss):.6f}")

    # 3) Mesh-less rendering: evaluate f(z0,x,y) on ANY resolution (for visualization only)
    f.eval()
    with torch.no_grad():
        render_h, render_w = 256, 256
        xy_r = make_coord_grid(render_h, render_w, device=device)
        t_r = f(z0, xy_r).reshape(render_h, render_w).clamp(0, 1).cpu().numpy()

    # 4) Contours setup
    iso_values = [0.1, 0.4, 0.8, 0.9, 0.95]

    # Contours from ORIGINAL GRID ("mesh contour")
    if HAVE_SKIMAGE:
        t0 = time.perf_counter()
        mesh_contours = {v: measure.find_contours(x0_np, level=v) for v in iso_values}
        mesh_contour_time = time.perf_counter() - t0
    else:
        mesh_contours = None
        mesh_contour_time = None

    # Dense mesh-less contours from implicit render (grid marching squares on t_r) - optional comparison
    t0 = time.perf_counter()
    meshless_segments_dense = {v: marching_squares_segments(t_r, level=v) for v in iso_values}
    meshless_dense_time = time.perf_counter() - t0

    # Quadtree mesh-less contours directly from f(z,x,y)
    eval_func = make_eval_func_from_model(f, z0, device)
    # Quick sanity check: sample a small grid and print the value range *as seen by the implicit model*
    with torch.no_grad():
        xy_test = make_coord_grid(64, 64, device=device)
        t_test = eval_func(xy_test).detach().cpu().numpy()
    print(f"implicit eval range: min={t_test.min():.4f} max={t_test.max():.4f}")

    qc = QuadtreeContour2D(eval_func, device=device, batch_size=131072, cache_quant=1e-10)

    qt_stats = {}
    qt_segments = {}
    t0 = time.perf_counter()
    for v in iso_values:
        segs, st = qc.contour(
            iso=v,
            xmin=-1.0, xmax=1.0,
            ymin=-1.0, ymax=1.0,
            max_depth=14,
            min_depth=2,
            min_size=2.0 / 512.0,  # "pixel-ish" tolerance in domain space; tune this
            use_midpoints=True,  # <<< NEW: sample edge midpoints + center for intersection test
        )
        print(f"iso={v:.3f}  segments={len(segs)}  leaves={st['numLeavesTotal']}  cached={st['numCachedSamples']}")

        qt_segments[v] = segs
        qt_stats[v] = st
    quadtree_time = time.perf_counter() - t0

    # Scale factors to map 64×64 contour coordinates into 256×256 pixel space
    scale_x = (render_w - 1) / (W - 1)
    scale_y = (render_h - 1) / (H - 1)

    # Plot
    fig = plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title("Original field (64×64) + mesh contours")
    ax1.imshow(x0_np, origin="lower")
    if mesh_contours is not None:
        for v in iso_values:
            for c in mesh_contours[v]:
                ax1.plot(c[:, 1], c[:, 0], linewidth=1.5)
    else:
        ax1.contour(x0_np, levels=iso_values, linewidths=1.5)

    ax2 = plt.subplot(2, 2, 2)
    ax2.set_title("Implicit render (256×256) + quadtree mesh-less contours")
    ax2.imshow(t_r, origin="lower")

    # Quadtree segments are in domain coords [-1,1]^2. Map them to 256×256 pixel coords for plotting.
    for v in iso_values:
        for (a, b) in qt_segments[v]:
            aPix = domain_to_pixel(a, -1.0, 1.0, -1.0, 1.0, render_w, render_h)
            bPix = domain_to_pixel(b, -1.0, 1.0, -1.0, 1.0, render_w, render_h)
            ax2.plot([aPix[0], bPix[0]], [aPix[1], bPix[1]], linewidth=1.0, color="k")

    ax3 = plt.subplot(2, 2, 3)
    ax3.set_title("Overlay: implicit field + (scaled) mesh contours")
    ax3.imshow(t_r, origin="lower")
    if mesh_contours is not None:
        for v in iso_values:
            for c in mesh_contours[v]:
                x_scaled = c[:, 1] * scale_x
                y_scaled = c[:, 0] * scale_y
                ax3.plot(x_scaled, y_scaled, linewidth=2.0)
    else:
        ax3.contour(
            np.kron(x0_np, np.ones((render_h // H, render_w // W))),
            levels=iso_values,
            linewidths=2.0,
        )

    ax4 = plt.subplot(2, 2, 4)
    ax4.set_title("Difference image: implicit render - original (upsampled)")
    with torch.no_grad():
        x0_t = torch.from_numpy(x0_np)[None, None, :, :].to(device)
        x0_up = F.interpolate(x0_t, size=(render_h, render_w), mode="bilinear", align_corners=True)
        diff = (torch.from_numpy(t_r)[None, None, :, :].to(device) - x0_up).squeeze().cpu().numpy()
    imd = ax4.imshow(diff, origin="lower")
    plt.colorbar(imd, ax=ax4, fraction=0.046, pad=0.04)

    # Timings / stats
    if mesh_contour_time is not None:
        print(f"Mesh contour extraction (skimage, 64x64) time: {mesh_contour_time*1000:.2f} ms")
    else:
        print("Mesh contour extraction: using matplotlib.contour fallback (not timed).")

    print(f"Mesh-less contour (dense marching_squares on 256x256 render) time: {meshless_dense_time*1000:.2f} ms")
    print(f"Mesh-less contour (quadtree from implicit f) total time: {quadtree_time*1000:.2f} ms")
    for v in iso_values:
        st = qt_stats[v]
        print(
            f"  iso={v:0.3f}: segments={st['numSegments']}, leaves={st['numLeavesTotal']}, "
            f"intersecting={st['numLeavesIntersecting']}, cachedSamples={st['numCachedSamples']}"
        )

    print("\nLatent vector z0 (first 10 values):")
    print(z0[:10].cpu().numpy())

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
