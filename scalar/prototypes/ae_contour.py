#!/usr/bin/env python3
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

    ripple = 0.3 * np.sin(5.0 * np.pi * x) * np.cos(4.0 * np.pi * y)

    f = s1 + s2 + s3 + ripple
    f = f - f.min()
    f = f / (f.max() + 1e-8)
    return f.astype(np.float32)

def make_grid(res):
    xs = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, res, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    return xs, ys, X, Y

# ============================================================
# 2) Autoencoder for 128x128 scalar fields
# ============================================================

def build_autoencoder(latent_channels=8):
    """
    Conv autoencoder:
      input  : [128,128,1]
      latent : [16,16,C]
      output : [128,128,1]
    """
    inp = tf.keras.Input(shape=(128, 128, 1))

    # Encoder
    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inp)   # 64x64
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)     # 32x32
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)     # 16x16
    latent = tf.keras.layers.Conv2D(latent_channels, 3, strides=1, padding='same',
                                    activation='relu', name='latent')(x)                  # 16x16xC

    # Decoder
    y = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(latent) # 32x32
    y = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(y)      # 64x64
    y = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(y)      # 128x128
    out = tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')(y)

    auto = tf.keras.Model(inp, out, name='autoencoder')
    encoder = tf.keras.Model(inp, latent, name='encoder')

    z_in = tf.keras.Input(shape=(16, 16, latent_channels))
    z = auto.layers[-4](z_in)
    z = auto.layers[-3](z)
    z = auto.layers[-2](z)
    z_out = auto.layers[-1](z)
    decoder = tf.keras.Model(z_in, z_out, name='decoder')

    return auto, encoder, decoder

def make_ae_dataset(n_train=256, n_val=32):
    """
    Generate synthetic dataset of scalar fields for AE training.
    Each sample is [128,128,1].
    """
    Xtr = []
    Xva = []

    for i in range(n_train + n_val):
        xs, ys, X, Y = make_grid(128)
        # vary seeds (e.g., random jitter on parameters) if you want
        F = scalar_field(X, Y)
        img = F[..., None]  # [128,128,1]
        if i < n_train:
            Xtr.append(img)
        else:
            Xva.append(img)

    return np.stack(Xtr, axis=0), np.stack(Xva, axis=0)

# ============================================================
# 3) NeRF-style latent field (MLP with positional encoding)
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
# 4) Metrics & helpers
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
# 5) Main pipeline
# ============================================================

def main():
    # --------------------------------------------------------
    # Step 1: Generate synthetic datasets for AE training
    # --------------------------------------------------------
    print("Generating AE dataset...")
    Xtr, Xva = make_ae_dataset(n_train=256, n_val=32)  # [N,128,128,1]

    # --------------------------------------------------------
    # Step 2: Train autoencoder (encoder/decoder)
    # --------------------------------------------------------
    print("Building autoencoder...")
    latent_channels = 8
    # encoder = compressor. It outputs 16x16xC latent feature grid.
    # decoder = decompressor. It takes 16x16xC latent feature grid and reconstructs 128x128 field.
    # auto = autoencoder model combining encoder and decoder. 128x128 in, 128x128 out.
    auto, encoder, decoder = build_autoencoder(latent_channels=latent_channels)
    auto.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

    print("Training autoencoder...")
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
    # Choose ONE ground-truth field to approximate with latent NeRF
    # --------------------------------------------------------
    print("Creating a target field F(x,y)...")
    xs, ys, X, Y = make_grid(128)
    F_gt = scalar_field(X, Y)       # [128,128]
    # ... keeps everything the same. ",None" adds a channel dim so that it matches for the AE.
    F_gt_img = F_gt[..., None]      # [128,128,1]
    # indexing with None adds a dim to the front.
    F_gt_img_batch = F_gt_img[None] # [1,128,128,1]

    # Compute its AE latent representation
    print("Encoding target field into latent space...")
    z_gt = encoder.predict(F_gt_img_batch, verbose=0)   # [1,16,16,C]
    z_gt_grid = z_gt[0]                                 # [16,16,C]
    # 256 = 16x16
    z_gt_flat = z_gt_grid.reshape(-1, latent_channels)  # [256,C]

    # Coordinates in latent space (u,v) for the 16x16 grid
    us = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    vs = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    U, V = np.meshgrid(us, vs, indexing='xy')
    uv_coords = np.stack([U, V], axis=-1).reshape(-1, 2)  # [256,2]

    # --------------------------------------------------------
    # Step 3: Train NeRF-style MLP in latent space
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

    uv_tf = tf.convert_to_tensor(uv_coords)            # [256,2]
    z_tf  = tf.convert_to_tensor(z_gt_flat)            # [256,C]

    print("Training latent NeRF to approximate AE latent grid...")
    for step in range(1, 3001):
        loss = train_step(uv_tf, z_tf)
        if step % 300 == 0:
            print(f"[latent NeRF] step {step}/3000, MSE latent = {loss.numpy():.6f}")

    # --------------------------------------------------------
    # Step 4: Use NeRF-latent + decoder to reconstruct field & extract iso-contours
    # --------------------------------------------------------
    print("Rendering latent grid from NeRF and decoding to 128x128 field...")
    # Evaluate the trained latent NeRF on a dense UV grid to recover latent feature tiles
    z_pred_grid = render_latent_grid(latent_nerf, grid_res=16, latent_channels=latent_channels)  # [1,16,16,C]
    # Decode the latent grid produced by the NeRF to get the reconstructed scalar field
    F_nerf_img = decoder.predict(z_pred_grid, verbose=0)[0, ..., 0]  # [128,128]
    # Decode the original field through the AE directly for a baseline comparison
    F_ae_img   = auto.predict(F_gt_img_batch, verbose=0)[0, ..., 0]  # AE reconstruction alone

    # --------------------------------------------------------
    # Step 5: Compare to ground truth
    # --------------------------------------------------------
    mse_ae   = np.mean((F_ae_img   - F_gt)**2)
    mse_nerf = np.mean((F_nerf_img - F_gt)**2)

    print("\n=== Reconstruction quality ===")
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
    axs[0,0].set_title("Ground truth F(x,y)")
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
    # Choose a few iso-levels based on ground truth range
    levels = np.linspace(0.2, 0.8, 4) * F_gt.max()

    print("Extracting iso-contours...")
    extract_contours(F_gt,       levels, title="GT iso-contours",         color='white')
    extract_contours(F_ae_img,   levels, title="AE iso-contours",         color='cyan')
    extract_contours(F_nerf_img, levels, title="AE+NeRF iso-contours",    color='magenta')

    # Overlay GT vs AE+NeRF contours for visual comparison
    plt.figure(figsize=(6,6))
    xs = np.linspace(-1,1,F_gt.shape[1])
    ys = np.linspace(-1,1,F_gt.shape[0])
    Xg, Yg = np.meshgrid(xs, ys, indexing='xy')
    plt.imshow(F_gt, origin='lower', extent=(-1,1,-1,1), cmap='viridis')
    plt.contour(Xg, Yg, F_gt,       levels=levels, colors='white',  linewidths=2, linestyles='solid')
    plt.contour(Xg, Yg, F_nerf_img, levels=levels, colors='magenta',linewidths=1.5, linestyles='dashed')
    plt.title("GT (white) vs AE+NeRF (magenta) iso-contours")
    plt.axis('equal'); plt.axis('off')
    plt.show()

    print("Done.")

if __name__ == "__main__":
    main()
