#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

import sys, traceback


np.random.seed(0)
tf.random.set_seed(0)

# ============================================================
# 0) Configuration for ADIOS input
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]

# Path to your ADIOS BP file
DATA_FILE = REPO_ROOT / "data" / "xcompact-TG.bp"   # <-- change this

# Variable to use as scalar field for AE/NeRF (must exist in the file)
TARGET_VAR = "rho"               # e.g., "pp", "rho", "ux", "uy"

TARGET_XY = (128, 128)           # We crop slices to 128x128

#for now, import the numpy file.
DATA_FILE = REPO_ROOT / "data" / "xcompact-TG.npy"


# ============================================================
# 1) Autoencoder for 128x128 scalar fields
# ============================================================

def build_autoencoder(latent_channels=8, encoder_filters=(32, 64, 64)):
    """
    Conv autoencoder:
      input  : [128,128,1]
      latent : [(128/2^L),(128/2^L),C] where L=len(encoder_filters)
      output : [128,128,1]

    Args:
        latent_channels: number of filters in the latent tensor.
        encoder_filters: iterable of ints, one per stride-2 Conv block.
    """
    if not encoder_filters:
        raise ValueError("encoder_filters must contain at least one value")

    inp = tf.keras.Input(shape=(128, 128, 1))

    # Encoder
    x = inp
    for i, filters in enumerate(encoder_filters):
        layer = tf.keras.layers.Conv2D(
            filters,
            3,
            strides=2,
            padding='same',
            activation='relu',
            name=f'enc_conv_{i}'
        )
        x = layer(x)
    latent_layer = tf.keras.layers.Conv2D(
        latent_channels,
        3,
        strides=1,
        padding='same',
        activation='relu',
        name='latent'
    )
    latent = latent_layer(x)

    # Decoder
    y = latent
    decoder_layers = []
    for i, filters in enumerate(reversed(encoder_filters)):
        layer = tf.keras.layers.Conv2DTranspose(
            filters,
            3,
            strides=2,
            padding='same',
            activation='relu',
            name=f'dec_deconv_{i}'
        )
        y = layer(y)
        decoder_layers.append(layer)
    out_layer = tf.keras.layers.Conv2D(
        1,
        3,
        padding='same',
        activation='sigmoid',
        name='decoder_output'
    )
    out = out_layer(y)

    auto = tf.keras.Model(inp, out, name='autoencoder')
    encoder = tf.keras.Model(inp, latent, name='encoder')

    # Build decoder using the same Conv2DTranspose layers
    latent_shape = latent.shape[1:]
    z_in = tf.keras.Input(shape=latent_shape)
    z = z_in
    for layer in decoder_layers:
        z = layer(z)
    z_out = out_layer(z)
    decoder = tf.keras.Model(z_in, z_out, name='decoder')

    return auto, encoder, decoder


# ============================================================
# 1b) Build AE dataset from ADIOS (real CFD data)
# ============================================================

def load_ae_dataset_from_adios(
    data_file,
    var_name,
    target_xy=(128, 128),
    normalize=True
):
    """
    Read an ADIOS file with CFD data and build training data for a 2D AE.

    Assumes variable var_name has shape per timestep:
        (Nz, Ny, Nx)  OR  (Ny, Nx)
    and that x,y resolution is roughly 129x128 or 128x129.
    We crop each slice to 128x128.

    Returns:
        Xtr : np.ndarray (N_train, 128,128,1)
        Xva : np.ndarray (N_val,   128,128,1)
    """
    slices = []   # list of 2D arrays (cropped)
    vmin, vmax = None, None

    target_h, target_w = target_xy

    print(f"Scanning ADIOS file '{data_file}' for slices of variable '{var_name}'...")
    with adios2.Stream(data_file, "r") as fh:
        for step_idx, step in enumerate(fh.steps()):
            arr = fh.read(var_name)  # shape: (Nz,Ny,Nx) or (Ny,Nx)

            if arr.ndim == 3:
                Nz, Ny, Nx = arr.shape
                for iz in range(Nz):
                    slice2d = arr[iz, :, :]  # (Ny,Nx)
                    h, w = slice2d.shape
                    # Crop to 128x128 from top-left (simple and deterministic)
                    cropped = slice2d[:min(h, target_h), :min(w, target_w)]
                    # If one dimension is 127 or similar, this will be smaller;
                    # but you said 129x128, so min(...) should give 128x128.
                    if cropped.shape != (target_h, target_w):
                        raise ValueError(f"Cropped slice shape {cropped.shape} != {target_xy}")
                    slices.append(cropped.astype(np.float32))
                    mn = float(cropped.min())
                    mx = float(cropped.max())
                    vmin = mn if vmin is None else min(vmin, mn)
                    vmax = mx if vmax is None else max(vmax, mx)
            elif arr.ndim == 2:
                Ny, Nx = arr.shape
                slice2d = arr
                h, w = slice2d.shape
                cropped = slice2d[:min(h, target_h), :min(w, target_w)]
                if cropped.shape != (target_h, target_w):
                    raise ValueError(f"Cropped slice shape {cropped.shape} != {target_xy}")
                slices.append(cropped.astype(np.float32))
                mn = float(cropped.min())
                mx = float(cropped.max())
                vmin = mn if vmin is None else min(vmin, mn)
                vmax = mx if vmax is None else max(vmax, mx)
            else:
                raise ValueError(f"Unexpected ndim={arr.ndim} for variable '{var_name}'")

            if (step_idx + 1) % 10 == 0:
                print(f"  processed {step_idx+1} timesteps...")

    if len(slices) == 0:
        raise RuntimeError(f"No slices were read for variable '{var_name}'")

    print(f"Read {len(slices)} slices for var '{var_name}'. Global min={vmin}, max={vmax}")

    # Normalize and add channel dim
    X_all = []
    for img in slices:
        if normalize:
            img_norm = (img - vmin) / (vmax - vmin + 1e-8)
            img_norm = np.clip(img_norm, 0.0, 1.0)
        else:
            img_norm = img
        X_all.append(img_norm[..., None])  # (128,128,1)

    X_all = np.stack(X_all, axis=0)
    print(f"AE dataset shape from ADIOS: {X_all.shape}")

    # Simple train/val split (90/10)
    n_total = X_all.shape[0]
    n_val = max(1, n_total // 10)
    n_train = n_total - n_val
    Xtr = X_all[:n_train]
    Xva = X_all[n_train:]

    print(f"Train samples: {Xtr.shape[0]}, Val samples: {Xva.shape[0]}")
    return Xtr, Xva

# ============================================================
# 1c) Build AE dataset from NumPy file
# ============================================================

def load_ae_dataset_from_numpy(
    npy_file,
    var_name,
    target_xy=(128, 128),
    normalize=True,
):
    """
    Read a NumPy array saved on disk and build training data for a 2D AE.

    Assumes array has shape (..., Ny, Nx) where the last two dims are spatial.
    For example, a dataset shaped (Nt, Nz, Ny, Nx) with Nt timesteps and Nz slices.
    We crop each slice to 128x128 and flatten the leading dimensions into samples.

    Returns:
        Xtr : np.ndarray (N_train, 128,128,1)
        Xva : np.ndarray (N_val,   128,128,1)
    """
    print(f"Loading NumPy array from '{npy_file}'...")
    data = np.load(npy_file)

    if isinstance(data, np.lib.npyio.NpzFile):
        raise ValueError("Got an .npz archive. Please provide a .npy path or load the array first.")

    if data.ndim < 2:
        raise ValueError(f"Expected at least 2 dims for spatial slice, got shape {data.shape}")

    target_h, target_w = target_xy
    spatial_h, spatial_w = data.shape[-2], data.shape[-1]

    print(f"Array shape: {data.shape}. Treating last two dims ({spatial_h},{spatial_w}) as (y,x).")

    flattened = data.reshape(-1, spatial_h, spatial_w)
    vmin, vmax = None, None
    slices = []
    for idx, slice2d in enumerate(flattened):
        cropped = slice2d[:min(spatial_h, target_h), :min(spatial_w, target_w)]
        if cropped.shape != (target_h, target_w):
            raise ValueError(f"Cropped slice shape {cropped.shape} != {target_xy}")
        cropped = cropped.astype(np.float32, copy=False)
        slices.append(cropped)
        mn = float(cropped.min())
        mx = float(cropped.max())
        vmin = mn if vmin is None else min(vmin, mn)
        vmax = mx if vmax is None else max(vmax, mx)

        if (idx + 1) % 100 == 0:
            print(f"  processed {idx+1} slices...")

    if len(slices) == 0:
        raise RuntimeError("No slices were found in the provided NumPy file.")

    print(f"Collected {len(slices)} slices. Global min={vmin}, max={vmax}")

    X_all = []
    for img in slices:
        if normalize:
            img_norm = (img - vmin) / (vmax - vmin + 1e-8)
            img_norm = np.clip(img_norm, 0.0, 1.0)
        else:
            img_norm = img
        X_all.append(img_norm[..., None])  # (128,128,1)

    X_all = np.stack(X_all, axis=0)
    print(f"AE dataset shape from NumPy: {X_all.shape}")

    n_total = X_all.shape[0]
    n_val = max(1, n_total // 10)
    n_train = n_total - n_val
    Xtr = X_all[:n_train]
    Xva = X_all[n_train:]

    print(f"Train samples: {Xtr.shape[0]}, Val samples: {Xva.shape[0]}")
    return Xtr, Xva


# ============================================================
# 2) NeRF-style latent field (MLP with positional encoding)
# ============================================================

def positional_encode_2d(uv, L=4):
    """
    uv: [N,2] in [-1,1]
    L : number of frequency bands
    returns: [N, 2 + 8L]
    """
    u, v = uv[..., 0:1], uv[..., 1:2]
    outs = [u, v]
    for k in range(L):
        freq = (2.0 ** k) * np.pi
        for coord in (u, v):
            outs.append(tf.sin(freq * coord))
            outs.append(tf.cos(freq * coord))
    return tf.concat(outs, axis=-1)

def build_latent_nerf(L=4, hidden=64, depth=3, latent_channels=8):
    """
    NeRF-style network that maps (u,v) -> latent vector in R^C.
    We'll train it to reproduce the encoder latent grid.
    """
    class LatentNeRF(tf.keras.Model):
        def __init__(self, L=L, hidden=hidden, depth=depth, C=latent_channels):
            super().__init__()
            self.L = L
            layers = []
            for _ in range(depth):
                layers.append(tf.keras.layers.Dense(hidden, activation='relu'))
            layers.append(tf.keras.layers.Dense(C))
            self.net = tf.keras.Sequential(layers)

        def call(self, uv):
            enc = positional_encode_2d(uv, L=self.L)
            return self.net(enc)

    return LatentNeRF()

def render_latent_grid(latent_nerf, grid_res=16, latent_channels=8):
    """
    Evaluate latent NeRF on a 16x16 grid in (u,v) ∈ [-1,1]^2.
    Output shape: [1,16,16,C]
    """
    us = np.linspace(-1.0, 1.0, grid_res, dtype=np.float32)
    vs = np.linspace(-1.0, 1.0, grid_res, dtype=np.float32)
    U, V = np.meshgrid(us, vs, indexing='xy')
    uv = np.stack([U, V], axis=-1).reshape(-1, 2)

    preds = []
    bs = 1024
    for i in range(0, uv.shape[0], bs):
        chunk = tf.convert_to_tensor(uv[i:i+bs])
        out = latent_nerf(chunk)
        preds.append(out.numpy())
    z_flat = np.concatenate(preds, axis=0)  # [256,C]
    z_grid = z_flat.reshape(1, grid_res, grid_res, latent_channels)
    return z_grid


# ============================================================
# 3) Metrics & helpers
# ============================================================

def psnr(mse):
    return -10.0 * np.log10(mse + 1e-12)

def extract_contours(F, levels, extent=(-1,1,-1,1), title=None, color='cyan'):
    """
    Simple helper to plot contours over a scalar field.
    """
    plt.figure(figsize=(6,6))
    plt.imshow(F, origin='lower', extent=extent, cmap='viridis')
    xs = np.linspace(extent[0], extent[1], F.shape[1])
    ys = np.linspace(extent[2], extent[3], F.shape[0])
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    cs = plt.contour(X, Y, F, levels=levels, colors=color, linewidths=2)
    if title:
        plt.title(title)
    plt.axis('equal'); plt.axis('off')
    plt.show()
    return cs


# ============================================================
# 4) Main pipeline
# ============================================================

def main():
    print('hello')
    # --------------------------------------------------------
    # Step 1: Generate AE training data from real ADIOS CFD slices
    # --------------------------------------------------------
    print("Generating AE dataset from ADIOS...")
    Xtr, Xva = load_ae_dataset_from_numpy(
        DATA_FILE,
        TARGET_VAR,
        target_xy=TARGET_XY,
        normalize=True
    )  # [N,128,128,1]

    # --------------------------------------------------------
    # Step 2: Train autoencoder (encoder/decoder)
    # --------------------------------------------------------
    print("Building autoencoder...")
    latent_channels = 8
    # encoder = compressor. It outputs (128/2^L)x(128/2^L)xC latent grid (16x16xC when L=3).
    # decoder = decompressor. It takes that latent grid and reconstructs a 128x128 field.
    # auto = autoencoder model combining encoder and decoder. 128x128 in, 128x128 out.
    encoder_filters = (64,128) # really bad.
    '''
    A stack like (16,16,32,64,128) means five stride‑2 downsampling blocks, so the latent grid becomes 128 / 2^5 = 4 pixels per side. You’re forcing the encoder to cram all spatial detail into a 4×4 grid, so edges/contours get heavily blurred even before NeRF enters the loop. On top of that, the early layers have fewer filters (16) than your previous setups, so their representational power shrinks just when you need more capacity to preserve fine structures. The decoder/NeRF can’t recover detail that was never encoded, so reconstructions look terrible. Stick to fewer downsampling levels (e.g., 3–4 blocks) or raise the input resolution accordingly if you want to use such a deep pyramid.
    '''
    # this looks really good for GT vs AE.
    '''
    (64,128) keeps much more spatial resolution in the latent grid (only 2 downsamples, so 32×32) while offering solid channel capacity, so it makes sense the AE contours look far closer to GT. If you want an even tighter fit, you could experiment with slightly higher latent_channels or minor tweaks to the decoder loss, but otherwise this setup seems like a solid sweet spot.
    '''
    latent_channels = 8
    latent_channels = 4
    encoder_filter_title = "x".join(str(f) for f in encoder_filters)
    encoder_filter_title += f", latent C={latent_channels}"

    auto, encoder, decoder = build_autoencoder(
        latent_channels=latent_channels,
        encoder_filters=encoder_filters
    )
    auto.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

    print("Training autoencoder on real CFD slices...")
    auto.fit(
        Xtr, Xtr,
        validation_data=(Xva, Xva),
        epochs=10,
        batch_size=32,
        verbose=1
    )

    # Freeze encoder/decoder for NeRF training
    encoder.trainable = False
    decoder.trainable = False

    # --------------------------------------------------------
    # Step 3: Choose ONE ground-truth field to approximate with latent NeRF
    # --------------------------------------------------------
    # We'll just take the first validation slice as our "target field".
    print("Selecting a target field F(x,y) from validation set...")
    F_gt_img = Xva[0]          # [128,128,1], normalized [0,1]
    F_gt = F_gt_img[..., 0]    # [128,128]

    # Batch it for AE/decoder calls
    F_gt_img_batch = F_gt_img[None, ...]  # [1,128,128,1]

    # Compute its AE latent representation
    print("Encoding target field into latent space...")
    z_gt = encoder.predict(F_gt_img_batch, verbose=0)   # [1,H_lat,W_lat,C]
    z_gt_grid = z_gt[0]                                 # [H_lat,W_lat,C]
    latent_h, latent_w = z_gt_grid.shape[:2]
    num_latent = latent_h * latent_w
    z_gt_flat = z_gt_grid.reshape(num_latent, latent_channels)  # [num_latent,C]

    # Coordinates in latent space (u,v) for the latent grid
    us = np.linspace(-1.0, 1.0, latent_w, dtype=np.float32)
    vs = np.linspace(-1.0, 1.0, latent_h, dtype=np.float32)
    U, V = np.meshgrid(us, vs, indexing='xy')
    uv_coords = np.stack([U, V], axis=-1).reshape(-1, 2)  # [num_latent,2]

    # --------------------------------------------------------
    # Step 4: Train NeRF-style MLP in latent space
    # --------------------------------------------------------
    print("Building latent NeRF...")
    latent_nerf = build_latent_nerf(L=4, hidden=64, depth=3, latent_channels=latent_channels)
    opt = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def train_step(uv, z_target):
        with tf.GradientTape() as tape:
            z_pred = latent_nerf(uv)    # [N,C]
            loss = tf.reduce_mean((z_pred - z_target)**2)
        grads = tape.gradient(loss, latent_nerf.trainable_variables)
        opt.apply_gradients(zip(grads, latent_nerf.trainable_variables))
        return loss

    uv_tf = tf.convert_to_tensor(uv_coords)            # [num_latent,2]
    z_tf  = tf.convert_to_tensor(z_gt_flat)            # [num_latent,C]

    print("Training latent NeRF to approximate AE latent grid...")
    for step in range(1, 3001):
        loss = train_step(uv_tf, z_tf)
        if step % 300 == 0:
            print(f"[latent NeRF] step {step}/3000, MSE latent = {loss.numpy():.6f}")

    # --------------------------------------------------------
    # Step 5: Use NeRF-latent + decoder to reconstruct field & extract iso-contours
    # --------------------------------------------------------
    print("Rendering latent grid from NeRF and decoding to 128x128 field...")
    # Evaluate the trained latent NeRF on a dense UV grid to recover latent feature tiles
    z_pred_grid = render_latent_grid(
        latent_nerf,
        grid_res=latent_h,
        latent_channels=latent_channels
    )  # [1,H_lat,W_lat,C]
    # Decode the latent grid produced by the NeRF to get the reconstructed scalar field
    F_nerf_img = decoder.predict(z_pred_grid, verbose=0)[0, ..., 0]  # [128,128]
    # Decode the original field through the AE directly for a baseline comparison
    F_ae_img   = auto.predict(F_gt_img_batch, verbose=0)[0, ..., 0]  # AE reconstruction alone

    # --------------------------------------------------------
    # Step 6: Compare to ground truth (in normalized space)
    # --------------------------------------------------------
    mse_ae   = np.mean((F_ae_img   - F_gt)**2)
    mse_nerf = np.mean((F_nerf_img - F_gt)**2)

    print("\n=== Reconstruction quality (normalized units) ===")
    print(f"Autoencoder only: MSE={mse_ae:.6e}, PSNR={psnr(mse_ae):.2f} dB")
    print(f"AE + latent NeRF: MSE={mse_nerf:.6e}, PSNR={psnr(mse_nerf):.2f} dB")

    # Error maps
    err_ae   = np.abs(F_ae_img   - F_gt)
    err_nerf = np.abs(F_nerf_img - F_gt)
    max_err = max(err_ae.max(), err_nerf.max()) + 1e-8

    # Show fields and error maps
    fig, axs = plt.subplots(3, 3, figsize=(12, 10))

    # Ground truth
    axs[0,0].imshow(F_gt, origin='lower', extent=(-1,1,-1,1), cmap='viridis')
    axs[0,0].set_title("Ground truth F(x,y) (normalized)")
    axs[0,0].axis('off')

    axs[0,1].axis('off')
    axs[0,2].axis('off')

    # AE reconstruction
    axs[1,0].imshow(F_ae_img, origin='lower', extent=(-1,1,-1,1), cmap='viridis')
    axs[1,0].set_title("AE reconstruction")
    axs[1,0].axis('off')

    im1 = axs[1,1].imshow(err_ae / max_err, origin='lower', extent=(-1,1,-1,1),
                          vmin=0, vmax=1, cmap='inferno')
    axs[1,1].set_title("|AE - GT| (normalized)")
    axs[1,1].axis('off')

    axs[1,2].axis('off')

    # AE + latent NeRF reconstruction
    axs[2,0].imshow(F_nerf_img, origin='lower', extent=(-1,1,-1,1), cmap='viridis')
    axs[2,0].set_title("AE + latent NeRF reconstruction")
    axs[2,0].axis('off')

    im2 = axs[2,1].imshow(err_nerf / max_err, origin='lower', extent=(-1,1,-1,1),
                          vmin=0, vmax=1, cmap='inferno')
    axs[2,1].set_title("|(AE+NeRF) - GT| (normalized)")
    axs[2,1].axis('off')

    axs[2,2].axis('off')

    plt.tight_layout()
    cbar = fig.colorbar(im2, ax=axs[:,1], fraction=0.025, pad=0.04)
    cbar.set_label("Normalized |error|")
    plt.show()

    # Iso-contour comparison
    # Since F_gt is normalized [0,1], pick iso-levels between 0.2 and 0.8
    levels = np.linspace(0.2, 0.8, 4)

    print("Extracting iso-contours...")
    extract_contours(F_gt,       levels, title="GT iso-contours",         color='white')
    extract_contours(F_ae_img,   levels, title="AE iso-contours",         color='cyan')
    extract_contours(F_nerf_img, levels, title="AE+NeRF iso-contours",    color='magenta')

    xs = np.linspace(-1,1,F_gt.shape[1])
    ys = np.linspace(-1,1,F_gt.shape[0])
    Xg, Yg = np.meshgrid(xs, ys, indexing='xy')

    # Overlay GT vs AE contours
    plt.figure(figsize=(6,6))
    plt.imshow(F_gt, origin='lower', extent=(-1,1,-1,1), cmap='viridis')
    plt.contour(Xg, Yg, F_gt,     levels=levels, colors='white', linewidths=2,   linestyles='solid')
    plt.contour(Xg, Yg, F_ae_img, levels=levels, colors='cyan',  linewidths=1.5, linestyles='dashed')
    plt.title(f"GT vs AE | AE filters {encoder_filter_title}")
    plt.axis('equal'); plt.axis('off')
    plt.show()

    # Overlay GT vs AE+NeRF contours
    plt.figure(figsize=(6,6))
    plt.imshow(F_gt, origin='lower', extent=(-1,1,-1,1), cmap='viridis')
    plt.contour(Xg, Yg, F_gt,       levels=levels, colors='white',   linewidths=2,   linestyles='solid')
    plt.contour(Xg, Yg, F_nerf_img, levels=levels, colors='magenta', linewidths=1.5, linestyles='dashed')
    plt.title(f"GT vs AE+NeRF | AE filters {encoder_filter_title}")
    plt.axis('equal'); plt.axis('off')
    plt.show()

    print("Done.")

if __name__ == "__main__":
    main()
