#!/usr/bin/env python3
"""
2D scalar NeRF demo with positional encoding, baseline MLP, and spline comparison.

- Builds a synthetic scalar field F(x,y) on a 128x128 grid.
- Trains:
    (1) MLP without positional encoding
    (2) MLP with Fourier positional encoding (NeRF-style)
- Builds a bicubic spline interpolant from the 128x128 grid.
- Evaluates all three at 256x256 and compares to the analytic ground truth.
- Shows images, error maps, and a 1D slice plot.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

np.random.seed(0)
tf.random.set_seed(0)

# ============================================================
# 1) Synthetic scalar field F(x,y)
# ============================================================

def scalar_field(x, y):
    """
    Synthetic F(x,y) with multiple scales:
      - three Gaussian blobs
      - sinusoidal ripple
    x, y: arrays in [-1,1]
    """
    def blob(cx, cy, r, amp):
        return amp * np.exp(-((x - cx)**2 + (y - cy)**2) / (2.0 * r * r))

    s1 = blob(-0.45, -0.15, 0.28, 1.2)
    s2 = blob( 0.30,  0.10, 0.25, 1.0)
    s3 = blob( 0.00,  0.42, 0.22, 0.9)

    # Higher-frequency ripple
    ripple = 0.3 * np.sin(5.0 * np.pi * x) * np.cos(4.0 * np.pi * y)

    f = s1 + s2 + s3 + ripple
    # normalize roughly to [0,1]
    f = f - f.min()
    f = f / (f.max() + 1e-8)
    return f.astype(np.float32)

def make_grid(res):
    xs = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return xs, ys, X, Y

# ============================================================
# 2) Positional encoding (Fourier features)
# ============================================================

def positional_encode(xy, L=8):
    """
    xy: [N,2] tensor in [-1,1]
    L: number of frequency bands
    Returns: [N, 2 + 4*L] features (x,y plus sin/cos for each coord)
    """
    x, y = xy[..., 0:1], xy[..., 1:2]
    outs = [x, y]
    for k in range(L):
        freq = (2.0 ** k) * np.pi
        for coord in (x, y):
            outs.append(tf.sin(freq * coord))
            outs.append(tf.cos(freq * coord))
    return tf.concat(outs, axis=-1)

# ============================================================
# 3) Models: baseline MLP & PE-MLP (NeRF-style)
# ============================================================

def build_mlp(input_dim, hidden=64, depth=3):
    layers = []
    for _ in range(depth):
        layers.append(tf.keras.layers.Dense(hidden, activation="relu"))
    layers.append(tf.keras.layers.Dense(1))  # scalar output
    return tf.keras.Sequential(layers)

class MLPBaseline(tf.keras.Model):
    def __init__(self, hidden=64, depth=3):
        super().__init__()
        self.mlp = build_mlp(input_dim=2, hidden=hidden, depth=depth)

    def call(self, xy):
        # xy: [N,2] in [-1,1]
        return self.mlp(xy)

class MLPPosEnc(tf.keras.Model):
    def __init__(self, L=8, hidden=64, depth=3):
        super().__init__()
        self.L = L
        # input dim = 2 (x,y) + 4*L per coord = 2 + 4*L*2 = 2 + 8L
        input_dim = 2 + 8 * L
        self.mlp = build_mlp(input_dim=input_dim, hidden=hidden, depth=depth)

    def call(self, xy):
        enc = positional_encode(xy, L=self.L)
        return self.mlp(enc)

# ============================================================
# 4) Training utilities
# ============================================================

def psnr(mse):
    return -10.0 * np.log10(mse + 1e-12)

def train_model(model, coords, values, steps=2000, batch_size=4096, lr=1e-3, name="model"):
    opt = tf.keras.optimizers.Adam(lr)

    @tf.function
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            pred = model(xb)  # [B,1]
            loss = tf.reduce_mean((pred - yb) ** 2)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    N = coords.shape[0]
    for step in range(1, steps + 1):
        idx = np.random.randint(0, N, size=batch_size)
        xb = tf.convert_to_tensor(coords[idx])
        yb = tf.convert_to_tensor(values[idx])
        loss = train_step(xb, yb)
        if step % 200 == 0:
            print(f"[{name}] step {step}/{steps}, MSE={loss.numpy():.6f}")
    return model

def eval_on_grid(model, X, Y):
    """
    Evaluate model on a 2D grid of coordinates.
    X,Y: [H,W], in [-1,1]
    Returns: F_pred [H,W]
    """
    H, W = X.shape
    xy = np.stack([X, Y], axis=-1).reshape(-1, 2)  # [H*W,2]
    preds = []
    bs = 16384
    for i in range(0, xy.shape[0], bs):
        chunk = tf.convert_to_tensor(xy[i:i+bs])
        out = model(chunk).numpy()
        preds.append(out)
    f = np.concatenate(preds, axis=0).reshape(H, W)
    return f

# ============================================================
# 5) Main experiment
# ============================================================

def main():
    # -----------------------
    # 5.1 Create training data (128x128)
    # -----------------------
    res_train = 128
    xs_train, ys_train, X_train, Y_train = make_grid(res_train)
    F_train = scalar_field(X_train, Y_train)

    # Train coordinates & values
    coords_train = np.stack([X_train, Y_train], axis=-1).reshape(-1, 2)  # [N,2]
    values_train = F_train.reshape(-1, 1)                                # [N,1]

    # -----------------------
    # 5.2 Build & train models
    # -----------------------
    # simple baseline MLP. Uses (x,y) directly as input. This will struggle to learn high-freq data.
    # MLP with positional encoding: turns x,y into a fourier feature. L: number of freq bands.
    mlp_plain = MLPBaseline(hidden=64, depth=3)
    mlp_pe    = MLPPosEnc(L=8, hidden=64, depth=3)

    print("\n=== Training baseline MLP (no positional encoding) ===")
    train_model(mlp_plain, coords_train, values_train,
                steps=2500, batch_size=4096, lr=1e-3, name="MLP")

    print("\n=== Training NeRF-style MLP with positional encoding ===")
    train_model(mlp_pe, coords_train, values_train,
                steps=2500, batch_size=4096, lr=1e-3, name="MLP+PE")

    # -----------------------
    # 5.3 Build bicubic spline from 128x128 values (baseline interpolant)
    # -----------------------
    spline = RectBivariateSpline(ys_train, xs_train, F_train)  # note (y,x) order

    # -----------------------
    # 5.4 Evaluate on finer grid (256x256)
    # -----------------------
    res_test = 256
    xs_test, ys_test, X_test, Y_test = make_grid(res_test)
    F_gt = scalar_field(X_test, Y_test)

    F_mlp = eval_on_grid(mlp_plain, X_test, Y_test)
    F_pe  = eval_on_grid(mlp_pe,   X_test, Y_test)
    # Spline: interpolation on test grid
    F_sp  = spline(ys_test, xs_test)  # shape [256,256]

    # -----------------------
    # 5.5 Compute errors & PSNR
    # -----------------------
    mse_mlp = np.mean((F_mlp - F_gt)**2)
    mse_pe  = np.mean((F_pe  - F_gt)**2)
    mse_sp  = np.mean((F_sp  - F_gt)**2)

    print("\n=== Evaluation on 256x256 grid ===")
    print(f"MSE  (MLP no PE):   {mse_mlp:.6e}  | PSNR = {psnr(mse_mlp):.2f} dB")
    print(f"MSE  (MLP + PE):    {mse_pe:.6e}   | PSNR = {psnr(mse_pe):.2f} dB")
    print(f"MSE  (Spline):      {mse_sp:.6e}   | PSNR = {psnr(mse_sp):.2f} dB")

    # Error maps
    err_mlp = np.abs(F_mlp - F_gt)
    err_pe  = np.abs(F_pe  - F_gt)
    err_sp  = np.abs(F_sp  - F_gt)

    # Normalize errors for display
    max_err = max(err_mlp.max(), err_pe.max(), err_sp.max())
    eps = 1e-8

    # -----------------------
    # 5.6 Visualizations: fields + error maps
    # -----------------------
    fig, axs = plt.subplots(4, 2, figsize=(10, 14))

    # Row 0: Ground truth
    axs[0,0].imshow(F_gt, origin="lower", extent=(-1,1,-1,1), cmap="viridis")
    axs[0,0].set_title("Ground truth F(x,y)")
    axs[0,0].axis("off")

    axs[0,1].axis("off")

    # Row 1: MLP no PE
    axs[1,0].imshow(F_mlp, origin="lower", extent=(-1,1,-1,1), cmap="viridis")
    axs[1,0].set_title("MLP (no PE) prediction")
    axs[1,0].axis("off")

    im1 = axs[1,1].imshow(err_mlp / (max_err + eps), origin="lower",
                          extent=(-1,1,-1,1), cmap="inferno", vmin=0, vmax=1)
    axs[1,1].set_title("MLP (no PE) |error| (normalized)")
    axs[1,1].axis("off")

    # Row 2: MLP + PE (NeRF-style)
    axs[2,0].imshow(F_pe, origin="lower", extent=(-1,1,-1,1), cmap="viridis")
    axs[2,0].set_title("MLP + Positional Encoding")
    axs[2,0].axis("off")

    im2 = axs[2,1].imshow(err_pe / (max_err + eps), origin="lower",
                          extent=(-1,1,-1,1), cmap="inferno", vmin=0, vmax=1)
    axs[2,1].set_title("MLP+PE |error| (normalized)")
    axs[2,1].axis("off")

    # Row 3: Spline
    axs[3,0].imshow(F_sp, origin="lower", extent=(-1,1,-1,1), cmap="viridis")
    axs[3,0].set_title("Bicubic spline (RectBivariateSpline)")
    axs[3,0].axis("off")

    im3 = axs[3,1].imshow(err_sp / (max_err + eps), origin="lower",
                          extent=(-1,1,-1,1), cmap="inferno", vmin=0, vmax=1)
    axs[3,1].set_title("Spline |error| (normalized)")
    axs[3,1].axis("off")

    plt.tight_layout()
    cbar = fig.colorbar(im3, ax=axs[:,1], fraction=0.02, pad=0.04)
    cbar.set_label("Normalized |error|")
    plt.show()

    # -----------------------
    # 5.7 1D slice comparison (to see PE effect clearly)
    # -----------------------
    # Take horizontal slice at y=0
    y0_idx = res_test // 2
    x_line = xs_test
    gt_line  = F_gt[y0_idx, :]
    mlp_line = F_mlp[y0_idx, :]
    pe_line  = F_pe[y0_idx, :]
    sp_line  = F_sp[y0_idx, :]

    plt.figure(figsize=(8,5))
    plt.plot(x_line, gt_line, "k-", label="Ground truth", linewidth=2)
    plt.plot(x_line, mlp_line, "r--", label="MLP no PE")
    plt.plot(x_line, pe_line,  "b-.", label="MLP + PE")
    plt.plot(x_line, sp_line,  "g:", label="Spline")
    plt.title("1D slice at y=0")
    plt.xlabel("x")
    plt.ylabel("F(x,0)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
