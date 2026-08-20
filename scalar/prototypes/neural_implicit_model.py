#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(0)
tf.random.set_seed(0)

# ============================================================
# Config
# ============================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = REPO_ROOT / "data" / "xcompact-TG.npy"  # shape (T=200, Z=8, Y=129, X=128)

BATCH_SIZE = 4096
EPOCHS = 30
STEPS_PER_EPOCH = 200          # effective samples per epoch: BATCH_SIZE * STEPS_PER_EPOCH
VAL_SAMPLES = 20000            # random samples for validation
IMPORTANCE_PERCENTILE = 75.0   # top X% gradient magnitude treated as "important"
IMPORTANCE_FRACTION = 0.5      # fraction of each batch from important region
#IMPORTANCE_FRACTION = 0.0      # disable importance sampling.


# ============================================================
# 1) Load and normalize data
# ============================================================

def load_and_normalize(data_file):
    print(f"Loading data from {data_file} ...")
    data = np.load(data_file).astype(np.float32)  # shape (T,Z,Y,X)
    if data.ndim != 4:
        raise ValueError(f"Expected 4D array (T,Z,Y,X), got shape {data.shape}")

    T, Z, Y, X = data.shape
    print(f"Data shape: T={T}, Z={Z}, Y={Y}, X={X}")

    # global mean/std
    scalar_mean = float(data.mean())
    scalar_std  = float(data.std() + 1e-8)
    print(f"Scalar mean={scalar_mean:.6g}, std={scalar_std:.6g}")

    data_norm = (data - scalar_mean) / scalar_std

    coord_info = {
        "T": T,
        "Z": Z,
        "Y": Y,
        "X": X,
        "scalar_mean": scalar_mean,
        "scalar_std": scalar_std,
    }

    return data_norm, coord_info


# ============================================================
# 1b) Importance sampling: high gradient regions
# ============================================================

def compute_importance_indices(data_norm, percentile=75.0):
    """
    Compute gradient magnitude in (y,x) directions and return indices of
    voxels with gradient magnitude above given percentile.

    Returns:
      important_indices: np.ndarray of shape [N_imp,4] of (t,z,y,x)
    """
    print("\nComputing gradient-based importance mask...")
    # gradient along y and x axes (2 and 3)
    gy, gx = np.gradient(data_norm, axis=(2, 3))
    grad_mag = np.sqrt(gx**2 + gy**2)

    thresh = np.percentile(grad_mag, percentile)
    important_mask = grad_mag > thresh

    important_indices = np.argwhere(important_mask)
    print(f"Importance threshold (percentile {percentile}): {thresh:.4e}")
    print(f"Number of important voxels: {important_indices.shape[0]}")

    return important_indices


# ============================================================
# 2) Build TF datasets (random sampling)
# ============================================================

def make_training_dataset(data_norm, coord_info, important_indices=None,
                          importance_fraction=0.5):
    """
    Randomly sample (t,z,y,x) to create batch of (coords, value).

    coords: [B,4] = (x_norm, y_norm, t_norm, z_norm) in [0,1]^4
    value : [B,1] normalized scalar

    importance_fraction: fraction of samples per batch drawn from
                         important_indices (if provided).
    """
    T, Z, Y, X = coord_info["T"], coord_info["Z"], coord_info["Y"], coord_info["X"]
    has_imp = important_indices is not None and important_indices.shape[0] > 0
    if not has_imp:
        importance_fraction = 0.0

    def gen():
        while True:
            n_imp = int(BATCH_SIZE * importance_fraction)
            n_uni = BATCH_SIZE - n_imp

            # --- important samples ---
            if n_imp > 0:
                idx = important_indices[
                    np.random.randint(0, important_indices.shape[0], size=n_imp)
                ]
                t_i, z_i, y_i, x_i = idx.T
            else:
                t_i = z_i = y_i = x_i = np.array([], dtype=np.int64)

            # --- uniform samples ---
            t_j = np.random.randint(0, T, size=n_uni)
            z_j = np.random.randint(0, Z, size=n_uni)
            y_j = np.random.randint(0, Y, size=n_uni)
            x_j = np.random.randint(0, X, size=n_uni)

            # concat
            t_idx = np.concatenate([t_i, t_j])
            z_idx = np.concatenate([z_i, z_j])
            y_idx = np.concatenate([y_i, y_j])
            x_idx = np.concatenate([x_i, x_j])

            # normalize to [0,1]
            x_norm = x_idx.astype(np.float32) / (X - 1)
            y_norm = y_idx.astype(np.float32) / (Y - 1)
            t_norm = t_idx.astype(np.float32) / (T - 1)
            z_norm = z_idx.astype(np.float32) / (Z - 1)

            coords = np.stack([x_norm, y_norm, t_norm, z_norm], axis=-1)  # [B,4]
            vals   = data_norm[t_idx, z_idx, y_idx, x_idx][..., None]     # [B,1]

            yield coords.astype(np.float32), vals.astype(np.float32)

    output_signature = (
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_validation_dataset(data_norm, coord_info, num_samples=VAL_SAMPLES):
    T, Z, Y, X = coord_info["T"], coord_info["Z"], coord_info["Y"], coord_info["X"]

    t_idx = np.random.randint(0, T, size=num_samples)
    z_idx = np.random.randint(0, Z, size=num_samples)
    y_idx = np.random.randint(0, Y, size=num_samples)
    x_idx = np.random.randint(0, X, size=num_samples)

    x_norm = x_idx.astype(np.float32) / (X - 1)
    y_norm = y_idx.astype(np.float32) / (Y - 1)
    t_norm = t_idx.astype(np.float32) / (T - 1)
    z_norm = z_idx.astype(np.float32) / (Z - 1)

    coords = np.stack([x_norm, y_norm, t_norm, z_norm], axis=-1).astype(np.float32)
    vals   = data_norm[t_idx, z_idx, y_idx, x_idx][..., None].astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((coords, vals))
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# 3) Positional encoding + implicit model
# ============================================================

def positional_encoding_tf(coords, L_xy=6, L_tz=2):
    """
    coords: [B,4] Tensor = (x,y,t,z) in [0,1]
    Returns concatenated [B, D] of raw coords + sin/cos features.
    Higher L_xy gives more spatial detail; L_tz can be smaller.
    """
    pi = tf.constant(np.pi, dtype=tf.float32)

    x = coords[..., 0:1]
    y = coords[..., 1:2]
    t = coords[..., 2:3]
    z = coords[..., 3:4]

    outs = [x, y, t, z]  # keep raw coords

    # high freq for x,y
    for k in range(L_xy):
        freq = (2.0 ** k) * pi
        for c in (x, y):
            outs.append(tf.sin(freq * c))
            outs.append(tf.cos(freq * c))

    # lower freq for t,z
    for k in range(L_tz):
        freq = (2.0 ** k) * pi
        for c in (t, z):
            outs.append(tf.sin(freq * c))
            outs.append(tf.cos(freq * c))

    return tf.concat(outs, axis=-1)


#L_tz is number of frequency bands used for T and Z.
def build_implicit_model(hidden=256, depth=6, L_xy=6, L_tz=2):
    """
    MLP with positional encoding:
       (x,y,t,z) -> u_norm

    hidden: neurons per layer
    depth : number of hidden layers
    """
    coords_in = tf.keras.Input(shape=(4,), name="coords")  # x,y,t,z in [0,1]

    enc = tf.keras.layers.Lambda(
        lambda c: positional_encoding_tf(c, L_xy=L_xy, L_tz=L_tz),
        name="posenc"
    )(coords_in)

    x = enc
    for i in range(depth):
        x = tf.keras.layers.Dense(hidden, activation='relu', name=f"mlp_{i}")(x)

    out = tf.keras.layers.Dense(1, activation=None, name="u_norm")(x)

    model = tf.keras.Model(inputs=coords_in, outputs=out, name="implicit_scalar_field")
    return model


# ============================================================
# 4) Metrics & plotting
# ============================================================

def psnr(mse):
    return -10.0 * np.log10(mse + 1e-12)


def evaluate_slice(model, data_norm, coord_info, t_idx, z_idx):
    """
    Reconstruct slice at given integer (t_idx,z_idx) using the implicit model,
    and return (gt_norm_slice, pred_norm_slice).
    """
    T, Z, Y, X = coord_info["T"], coord_info["Z"], coord_info["Y"], coord_info["X"]

    gt_slice = data_norm[t_idx, z_idx, :, :]  # [Y,X]

    xs = np.linspace(0.0, 1.0, X, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, Y, dtype=np.float32)
    Xg, Yg = np.meshgrid(xs, ys, indexing='xy')

    t_norm = np.full_like(Xg, t_idx / (T - 1), dtype=np.float32)
    z_norm = np.full_like(Xg, z_idx / (Z - 1), dtype=np.float32)

    coords = np.stack([Xg, Yg, t_norm, z_norm], axis=-1).reshape(-1, 4)

    preds = model.predict(coords, batch_size=4096, verbose=0)
    pred_slice = preds.reshape(Y, X)

    return gt_slice, pred_slice


def plot_comparison(gt_norm, pred_norm, coord_info, title_suffix=""):
    scalar_mean = coord_info["scalar_mean"]
    scalar_std  = coord_info["scalar_std"]

    gt = gt_norm * scalar_std + scalar_mean
    pred = pred_norm * scalar_std + scalar_mean

    mse = float(np.mean((pred_norm - gt_norm) ** 2))
    print(f"MSE (normalized) = {mse:.6e}, PSNR = {psnr(mse):.2f} dB")

    err = np.abs(pred_norm - gt_norm)
    vmax_err = err.max() + 1e-8

    Y, X = gt_norm.shape
    ext = (0, 1, 0, 1)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    im0 = axs[0].imshow(gt, origin='lower', extent=ext, cmap='viridis')
    axs[0].set_title("Ground truth (physical units)")
    axs[0].axis('off')
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(pred, origin='lower', extent=ext, cmap='viridis')
    axs[1].set_title("Implicit model")
    axs[1].axis('off')
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(err / vmax_err, origin='lower', extent=ext,
                        cmap='inferno', vmin=0, vmax=1)
    axs[2].set_title("|pred - gt| (normalized)")
    axs[2].axis('off')
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    fig.suptitle(f"Implicit slice comparison {title_suffix}")
    plt.tight_layout()
    plt.show()

    # Iso-contour comparison
    levels = np.linspace(gt.min(), gt.max(), 5)[1:-1]  # 3 interior levels

    xs = np.linspace(0, 1, X)
    ys = np.linspace(0, 1, Y)
    Xg, Yg = np.meshgrid(xs, ys, indexing='xy')

    plt.figure(figsize=(5,5))
    plt.imshow(gt, origin='lower', extent=(0,1,0,1), cmap='gray')
    plt.contour(Xg, Yg, gt,   levels=levels, colors='cyan',    linewidths=2.0, linestyles='solid')
    plt.contour(Xg, Yg, pred, levels=levels, colors='magenta', linewidths=1.5, linestyles='dashed')
    plt.title(f"Iso-contours {title_suffix}")
    plt.axis('equal'); plt.axis('off')
    plt.show()


# ============================================================
# 5) Main
# ============================================================

def main():
    # Load & normalize
    data_norm, coord_info = load_and_normalize(DATA_FILE)

    # Importance indices
    important_indices = compute_importance_indices(
        data_norm, percentile=IMPORTANCE_PERCENTILE
    )

    # Datasets
    train_ds = make_training_dataset(
        data_norm, coord_info,
        important_indices=important_indices,
        importance_fraction=IMPORTANCE_FRACTION
    )
    val_ds   = make_validation_dataset(data_norm, coord_info, num_samples=VAL_SAMPLES)

    # Model
    model = build_implicit_model(hidden=256, depth=6, L_xy=6, L_tz=2)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='mse'
    )
    model.summary()

    lr_sched = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1
    )

    # Train
    print("\nTraining implicit model...")
    model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=val_ds,
        callbacks=[lr_sched],
        verbose=1
    )

    # Evaluate a chosen slice
    t_idx = 50
    z_idx = 3
    print(f"\nEvaluating slice t={t_idx}, z={z_idx} ...")
    gt_slice, pred_slice = evaluate_slice(model, data_norm, coord_info, t_idx, z_idx)
    plot_comparison(gt_slice, pred_slice, coord_info, title_suffix=f"(t={t_idx}, z={z_idx})")


if __name__ == "__main__":
    main()
