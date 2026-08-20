#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import gudhi as gd

def swiss_cheese(n=180, k=10, sigma=0.06, seed=0):
    """
    Smooth bowl with multiple negative Gaussian dents -> lots of holes in sublevel sets.
    """
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(0, n), -1:1:complex(0, n)]

    # Smooth bowl background
    f = 0.35 * (x*x + y*y)

    # Carve "dents"
    for _ in range(k):
        cx, cy = rng.uniform(-0.75, 0.75, size=2)
        amp = rng.uniform(0.8, 1.2)
        f -= amp * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*sigma*sigma))

    # Normalize [0,1]
    f = (f - f.min()) / (f.max() - f.min() + 1e-12)
    return f

def compute_pd(field2d, superlevel=False):
    f = np.asarray(field2d, dtype=np.float64)
    if superlevel:
        f = -f
    cc = gd.CubicalComplex(top_dimensional_cells=f)
    cc.persistence()
    dgm0 = cc.persistence_intervals_in_dimension(0)
    dgm1 = cc.persistence_intervals_in_dimension(1)
    return dgm0, dgm1

def topk_by_lifetime(dgm, k=30, drop_inf=True):
    if dgm.size == 0:
        return dgm
    d = dgm
    if drop_inf:
        finite = np.isfinite(d[:, 1])
        d = d[finite]
        if d.size == 0:
            return d
    life = d[:, 1] - d[:, 0]
    idx = np.argsort(-life)
    return d[idx[:k]]

def plot_pd(ax, dgm, title):
    ax.set_title(title)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.grid(alpha=0.3)

    if dgm.size == 0:
        ax.text(0.5, 0.5, "empty", ha="center", va="center", transform=ax.transAxes)
        return

    b = dgm[:, 0]
    d = dgm[:, 1]
    mn = float(min(b.min(), d[np.isfinite(d)].min()))
    mx = float(max(b.max(), d[np.isfinite(d)].max()))
    pad = 0.05 * (mx - mn + 1e-12)
    mn -= pad
    mx += pad

    ax.scatter(b, d, s=18)
    ax.plot([mn, mx], [mn, mx], "k-", linewidth=1)
    ax.set_xlim(mn, mx)
    ax.set_ylim(mn, mx)

def main():
    f = swiss_cheese(n=180, k=12, sigma=0.06, seed=3)

    # Sublevel persistence works well for this construction
    dgm0, dgm1 = compute_pd(f, superlevel=False)

    # Make PD visually meaningful: keep top-k by lifetime
    dgm0k = topk_by_lifetime(dgm0, k=20, drop_inf=True)
    dgm1k = topk_by_lifetime(dgm1, k=30, drop_inf=True)

    # Plot heatmap + contours + PDs
    fig = plt.figure(figsize=(12, 8))

    ax = plt.subplot(2, 2, 1)
    ax.set_title("Heatmap (swiss cheese)")
    im = ax.imshow(f, origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = plt.subplot(2, 2, 2)
    ax.set_title("Contours")
    ax.imshow(f, origin="lower")
    ax.contour(f, levels=np.linspace(0.15, 0.85, 6), colors="k", linewidths=1)

    ax = plt.subplot(2, 2, 3)
    plot_pd(ax, dgm0k, "PD H0 (top-20 by lifetime)")

    ax = plt.subplot(2, 2, 4)
    plot_pd(ax, dgm1k, "PD H1 (top-30 by lifetime)")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
