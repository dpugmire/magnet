#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Paths must match those in train_and_encode.py
DATA_FILE = "data/xcompact-TG.npy"
MODEL_DIR = "saved_models"

AE_IMPL_MODEL_PATH      = os.path.join(MODEL_DIR, "ae_implicit.keras")
COMPRESSED_LATENTS_PATH = os.path.join(MODEL_DIR, "compressed_latents_z3.npy")

# Which slice to evaluate
T_IDX = 10
Z_IDX = 3      # must match the Z used for compression


# ---------- Helpers copied from training script ----------

def load_and_normalize(data_file):
    data = np.load(data_file).astype(np.float32)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D array (T,Z,Y,X), got {data.shape}")
    T, Z, Y, X = data.shape
    mean = float(data.mean())
    std  = float(data.std() + 1e-8)
    data_norm = (data - mean) / std
    coord_info = {
        "T": T,
        "Z": Z,
        "Y": Y,
        "X": X,
        "scalar_mean": mean,
        "scalar_std": std,
    }
    return data_norm, coord_info


def positional_encoding_2d_tf(xy, L=6):
    pi = tf.constant(np.pi, dtype=tf.float32)
    x = xy[..., 0:1]
    y = xy[..., 1:2]
    outs = [x, y]
    for k in range(L):
        freq = (2.0 ** k) * pi
        for c in (x, y):
            outs.append(tf.sin(freq * c))
            outs.append(tf.cos(freq * c))
    return tf.concat(outs, axis=-1)


def evaluate_slice_from_latent(
    model,
    data_norm,
    coord_info,
    compressed_latents,
    t_idx,
    z_idx,
    crop_hw=(128, 128),
):
    T, Z, Y, X = coord_info["T"], coord_info["Z"], coord_info["Y"], coord_info["X"]
    Hc, Wc = crop_hw

    # Ground truth
    gt = data_norm[t_idx, z_idx, :Hc, :Wc]

    # Coordinates in [0,1]^2
    xs = np.linspace(0.0, 1.0, Wc, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, Hc, dtype=np.float32)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    coords2 = np.stack([Xg, Yg], axis=-1).reshape(-1, 2)

    # compressed_latents has shape [T, latent_dim]
    latent_vec = compressed_latents[t_idx]  # [C]
    z_rep = np.broadcast_to(latent_vec, (coords2.shape[0], latent_vec.shape[0]))

    preds = model.predict([coords2, z_rep], batch_size=4096, verbose=0).reshape(Hc, Wc)
    return gt, preds


def psnr(mse):
    return -10.0 * np.log10(mse + 1e-12)


# ---------- Main ----------

def main():
    # Load data + normalization
    data_norm, coord_info = load_and_normalize(DATA_FILE)
    mean = coord_info["scalar_mean"]
    std  = coord_info["scalar_std"]
    T, Z, Y, X = coord_info["T"], coord_info["Z"], coord_info["Y"], coord_info["X"]

    # Load compressed latent time series
    compressed_latents = np.load(COMPRESSED_LATENTS_PATH).astype(np.float32)
    print(f"Loaded compressed latents: {compressed_latents.shape}")  # [T, latent_dim]
    T_lat, latent_dim = compressed_latents.shape
    if T_lat != T:
        print(f"WARNING: T mismatch between data ({T}) and compressed latents ({T_lat})")

    # Cropping size (must match AE training)
    Hc, Wc = 128, 128

    # ----- Compression ratio -----
    orig_elements = T * Hc * Wc
    enc_elements  = T * latent_dim

    orig_bytes = orig_elements * 4        # float32
    enc_bytes  = enc_elements * 4

    orig_mb = orig_bytes / (1024**2)
    enc_mb  = enc_bytes  / (1024**2)

    compression_factor = orig_bytes / enc_bytes if enc_bytes > 0 else np.inf

    # Print detailed stats
    print("\n========== Compression statistics ==========")
    print(f"Original GT slice size    : ({T} timesteps, {Hc}x{Wc})")
    print(f"Original GT elements      : {orig_elements:,}")
    print(f"Original GT size (MB)     : {orig_mb:.4f} MB")

    print(f"\nAE latent vector length   : {latent_dim}")
    print(f"Encoded latent elements   : {enc_elements:,}")
    print(f"Encoded AE size (MB)      : {enc_mb:.6f} MB")

    print(f"\nCompression factor        : {compression_factor:.2f}x")
    print("============================================\n")

    # Load AE-conditioned implicit model
    ae_impl_model = tf.keras.models.load_model(
        AE_IMPL_MODEL_PATH,
        custom_objects={"positional_encoding_2d_tf": positional_encoding_2d_tf},
    )

    # Evaluate requested slice
    gt_n, pred_n = evaluate_slice_from_latent(
        ae_impl_model,
        data_norm,
        coord_info,
        compressed_latents,
        T_IDX,
        Z_IDX,
        crop_hw=(Hc, Wc),
    )

    # Convert back to physical units
    gt    = gt_n * std + mean
    pred  = pred_n * std + mean
    err_n = np.abs(pred_n - gt_n)

    mse = float(np.mean((pred_n - gt_n) ** 2))
    print(f"Slice (t={T_IDX}, z={Z_IDX}) MSE={mse:.3e}, PSNR={psnr(mse):.2f} dB")

    # ----------- Plots -----------
    ext = (0, 1, 0, 1)
    vmax_err = err_n.max() + 1e-8

    # Figure 1: fields + error
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    im0 = axs[0].imshow(gt, origin="lower", extent=ext, cmap="viridis")
    axs[0].set_title(f"Ground Truth ({orig_mb:.4f} MB)")
    axs[0].axis("off")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(pred, origin="lower", extent=ext, cmap="viridis")
    axs[1].set_title(
        f"AE-cond implicit ({enc_mb:.6f} MB, {compression_factor:.2f}x reduction)"
    )
    axs[1].axis("off")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(
        err_n / vmax_err,
        origin="lower",
        extent=ext,
        cmap="inferno",
        vmin=0,
        vmax=1,
    )
    axs[2].set_title("|pred - gt| (normalized)")
    axs[2].axis("off")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"t={T_IDX}, z={Z_IDX}")
    plt.tight_layout()
    plt.show()

    # Figure 2: iso-contours
    xs = np.linspace(0, 1, Wc)
    ys = np.linspace(0, 1, Hc)
    Xg, Yg = np.meshgrid(xs, ys, indexing="xy")
    levels = np.linspace(gt.min(), gt.max(), 5)[1:-1]

    plt.figure(figsize=(5, 5))
    plt.imshow(gt, origin="lower", extent=ext, cmap="gray")
    plt.contour(
        Xg,
        Yg,
        gt,
        levels=levels,
        colors="white",
        linewidths=2.0,
        linestyles="solid",
    )
    plt.contour(
        Xg,
        Yg,
        pred,
        levels=levels,
        colors="magenta",
        linewidths=1.5,
        linestyles="dashed",
    )
    plt.title(
        f"Iso-contours (t={T_IDX}, z={Z_IDX}) — {compression_factor:.2f}x "
        f"(GT={orig_mb:.4f} MB → AE={enc_mb:.6f} MB)"
    )
    plt.axis("equal")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
