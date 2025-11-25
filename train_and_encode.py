#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(0)
tf.random.set_seed(0)

# ============================================================
# Config
# ============================================================

DATA_FILE = "data/xcompact-TG.npy"  # shape (T=200, Z=8, Y=129, X=128)

# Training toggles
TRAIN_DIRECT_IMPLICIT = True
TRAIN_AE_IMPLICIT     = True

# Direct implicit hyperparams
BATCH_SIZE_DIRECT     = 4096
EPOCHS_DIRECT         = 30
STEPS_PER_EPOCH_DIR   = 200
VAL_SAMPLES_DIRECT    = 20000
IMPORTANCE_PERCENTILE = 75.0
IMPORTANCE_FRACTION   = 0.5

# AE + implicit hyperparams
AE_LATENT_CHANNELS    = 8
AE_ENCODER_FILTERS    = (32, 64, 128)  # 128x128 -> 16x16 latent grid
AE_EPOCHS             = 20
AE_BATCH_SIZE         = 32
BATCH_SIZE_AE_IMPL    = 4096
EPOCHS_AE_IMPL        = 30
STEPS_PER_EPOCH_AE    = 200
VAL_SAMPLES_AE_IMPL   = 20000

# Slice to evaluate / compress
EVAL_T_IDX = 50
EVAL_Z_IDX = 3

# Where to save stuff
MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

AE_AUTO_PATH    = os.path.join(MODEL_DIR, "ae_auto.keras")
AE_ENCODER_PATH = os.path.join(MODEL_DIR, "ae_encoder.keras")
AE_DECODER_PATH = os.path.join(MODEL_DIR, "ae_decoder.keras")

DIRECT_MODEL_PATH  = os.path.join(MODEL_DIR, "direct_implicit.keras")
AE_IMPL_MODEL_PATH = os.path.join(MODEL_DIR, "ae_implicit.keras")

COMPRESSED_LATENTS_PATH = os.path.join(MODEL_DIR, f"compressed_latents_z{EVAL_Z_IDX}.npy")


# ============================================================
# 1) Load and normalize data
# ============================================================

def load_and_normalize(data_file):
    print(f"Loading data from {data_file} ...")
    data = np.load(data_file).astype(np.float32)  # [T,Z,Y,X]
    if data.ndim != 4:
        raise ValueError(f"Expected 4D array (T,Z,Y,X), got {data.shape}")

    T, Z, Y, X = data.shape
    print(f"Data shape: T={T}, Z={Z}, Y={Y}, X={X}")

    scalar_mean = float(data.mean())
    scalar_std  = float(data.std() + 1e-8)
    print(f"Scalar mean={scalar_mean:.6g}, std={scalar_std:.6g}")

    data_norm = (data - scalar_mean) / scalar_std

    coord_info = {
        "T": T, "Z": Z, "Y": Y, "X": X,
        "scalar_mean": scalar_mean,
        "scalar_std": scalar_std,
    }
    return data_norm, coord_info


# ============================================================
# 2) Importance sampling (for direct model)
# ============================================================

def compute_importance_indices(data_norm, percentile=75.0):
    print("\nComputing gradient-based importance mask...")
    gy, gx = np.gradient(data_norm, axis=(2, 3))
    grad_mag = np.sqrt(gx**2 + gy**2)

    thresh = np.percentile(grad_mag, percentile)
    mask = grad_mag > thresh
    idx = np.argwhere(mask)

    print(f"Importance threshold (p{percentile}) = {thresh:.4e}")
    print(f"Important voxels: {idx.shape[0]}")
    return idx


# ============================================================
# 3) Direct implicit model datasets
# ============================================================

def make_training_dataset_direct(data_norm, coord_info,
                                 important_indices=None,
                                 importance_fraction=0.5,
                                 batch_size=4096):
    T, Z, Y, X = coord_info["T"], coord_info["Z"], coord_info["Y"], coord_info["X"]
    has_imp = important_indices is not None and important_indices.shape[0] > 0
    if not has_imp:
        importance_fraction = 0.0

    def gen():
        while True:
            n_imp = int(batch_size * importance_fraction)
            n_uni = batch_size - n_imp

            if n_imp > 0:
                idx = important_indices[
                    np.random.randint(0, important_indices.shape[0], size=n_imp)
                ]
                t_i, z_i, y_i, x_i = idx.T
            else:
                t_i = z_i = y_i = x_i = np.array([], dtype=np.int64)

            t_j = np.random.randint(0, T, size=n_uni)
            z_j = np.random.randint(0, Z, size=n_uni)
            y_j = np.random.randint(0, Y, size=n_uni)
            x_j = np.random.randint(0, X, size=n_uni)

            t_idx = np.concatenate([t_i, t_j])
            z_idx = np.concatenate([z_i, z_j])
            y_idx = np.concatenate([y_i, y_j])
            x_idx = np.concatenate([x_i, x_j])

            x_norm = x_idx.astype(np.float32) / (X - 1)
            y_norm = y_idx.astype(np.float32) / (Y - 1)
            t_norm = t_idx.astype(np.float32) / (T - 1)
            z_norm = z_idx.astype(np.float32) / (Z - 1)

            coords = np.stack([x_norm, y_norm, t_norm, z_norm], axis=-1)
            vals   = data_norm[t_idx, z_idx, y_idx, x_idx][..., None]

            yield coords.astype(np.float32), vals.astype(np.float32)

    out_sig = (
        tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_validation_dataset_direct(data_norm, coord_info,
                                   num_samples=20000,
                                   batch_size=4096):
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
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# 4) Positional encodings + direct implicit model
# ============================================================

def positional_encoding_tf(coords, L_xy=6, L_tz=2):
    """
    coords: [B,4] = (x,y,t,z) in [0,1]
    """
    pi = tf.constant(np.pi, dtype=tf.float32)
    x = coords[..., 0:1]
    y = coords[..., 1:2]
    t = coords[..., 2:3]
    z = coords[..., 3:4]

    outs = [x, y, t, z]
    for k in range(L_xy):
        freq = (2.0 ** k) * pi
        for c in (x, y):
            outs.append(tf.sin(freq * c))
            outs.append(tf.cos(freq * c))
    for k in range(L_tz):
        freq = (2.0 ** k) * pi
        for c in (t, z):
            outs.append(tf.sin(freq * c))
            outs.append(tf.cos(freq * c))

    return tf.concat(outs, axis=-1)


def build_direct_implicit_model(hidden=256, depth=6, L_xy=6, L_tz=2):
    coords_in = tf.keras.Input(shape=(4,), name="coords")  # x,y,t,z

    enc = tf.keras.layers.Lambda(
        positional_encoding_tf,
        arguments={"L_xy": L_xy, "L_tz": L_tz},
        name="posenc"
    )(coords_in)

    x = enc
    for i in range(depth):
        x = tf.keras.layers.Dense(hidden, activation='relu', name=f"mlp_{i}")(x)
    out = tf.keras.layers.Dense(1, activation=None, name="u_norm")(x)

    return tf.keras.Model(coords_in, out, name="direct_implicit")


# ============================================================
# 5) Autoencoder for 2D slices (128x128)
# ============================================================

def build_autoencoder(latent_channels=8, encoder_filters=(32, 64, 128)):
    inp = tf.keras.Input(shape=(128,128,1))

    x = inp
    for i,f in enumerate(encoder_filters):
        x = tf.keras.layers.Conv2D(
            f, 3, strides=2, padding='same', activation='relu', name=f'enc_conv_{i}'
        )(x)
    latent = tf.keras.layers.Conv2D(
        latent_channels, 3, strides=1, padding='same', activation='relu', name='latent'
    )(x)

    y = latent
    decoder_layers = []
    for i,f in enumerate(reversed(encoder_filters)):
        layer = tf.keras.layers.Conv2DTranspose(
            f, 3, strides=2, padding='same', activation='relu', name=f'dec_deconv_{i}'
        )
        y = layer(y)
        decoder_layers.append(layer)

    out_layer = tf.keras.layers.Conv2D(
        1, 3, padding='same', activation=None, name='dec_out'
    )
    out = out_layer(y)

    auto = tf.keras.Model(inp, out, name='autoencoder')
    encoder = tf.keras.Model(inp, latent, name='encoder')

    # standalone decoder
    latent_shape = latent.shape[1:]
    z_in = tf.keras.Input(shape=latent_shape)
    z = z_in
    for layer in decoder_layers:
        z = layer(z)
    z_out = out_layer(z)
    decoder = tf.keras.Model(z_in, z_out, name='decoder')

    return auto, encoder, decoder


def build_ae_dataset_from_volume(data_norm, coord_info):
    """
    All (t,z) slices cropped to 128x128.
    """
    T, Z, Y, X = coord_info["T"], coord_info["Z"], coord_info["Y"], coord_info["X"]
    target_h, target_w = 128, 128
    Yc, Xc = min(target_h, Y), min(target_w, X)

    print("\nBuilding AE dataset from all (t,z) slices...")
    slices = []
    for t in range(T):
        for z in range(Z):
            img = data_norm[t, z, :, :]
            cropped = img[:Yc, :Xc]
            slices.append(cropped[..., None])
    X_ae = np.stack(slices, axis=0).astype(np.float32)
    print(f"AE dataset shape: {X_ae.shape} (N_slices={X_ae.shape[0]})")
    return X_ae, (Yc, Xc)


def compute_slice_latents(encoder, X_ae, coord_info, latent_channels):
    T, Z = coord_info["T"], coord_info["Z"]
    print("\nEncoding all slices to latent space...")
    Z_grid = encoder.predict(X_ae, batch_size=32, verbose=0)  # [N_slices,H_lat,W_lat,C]
    N_slices, H_lat, W_lat, C = Z_grid.shape
    assert N_slices == T * Z

    Z_vec_all = Z_grid.mean(axis=(1,2))  # [N_slices,C]
    latents = Z_vec_all.reshape(T, Z, C)
    print(f"Latent vectors shape: {latents.shape} (C={C})")
    return latents, C


# ============================================================
# 6) AE-conditioned implicit model datasets
# ============================================================

def make_training_dataset_ae_impl(data_norm, coord_info, latents,
                                  crop_hw, batch_size=4096):
    """
    For AE-cond model: sample (t,z,y,x) within cropped area, use latent(t,z)
    and coords2=(x,y).
    """
    T, Z, Y, X = coord_info["T"], coord_info["Z"], coord_info["Y"], coord_info["X"]
    C = latents.shape[-1]
    Hc, Wc = crop_hw

    def gen():
        while True:
            t_idx = np.random.randint(0, T, size=1)[0]
            z_idx = np.random.randint(0, Z, size=1)[0]

            ys = np.random.randint(0, Hc, size=batch_size)
            xs = np.random.randint(0, Wc, size=batch_size)

            x_norm = xs.astype(np.float32) / (Wc - 1)
            y_norm = ys.astype(np.float32) / (Hc - 1)
            coords2 = np.stack([x_norm, y_norm], axis=-1)  # [B,2]

            vals = data_norm[t_idx, z_idx, ys, xs][..., None]  # [B,1]

            latent_vec = latents[t_idx, z_idx]  # [C]
            z_rep = np.broadcast_to(latent_vec, (batch_size, C))

            yield (coords2.astype(np.float32), z_rep.astype(np.float32)), vals.astype(np.float32)

    out_sig = (
        (tf.TensorSpec(shape=(None,2), dtype=tf.float32),
         tf.TensorSpec(shape=(None,latents.shape[-1]), dtype=tf.float32)),
        tf.TensorSpec(shape=(None,1), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_validation_dataset_ae_impl(data_norm, coord_info, latents,
                                    crop_hw, num_samples=20000,
                                    batch_size=4096):
    T, Z, Y, X = coord_info["T"], coord_info["Z"], coord_info["Y"], coord_info["X"]
    Hc, Wc = crop_hw
    C = latents.shape[-1]

    t_idx = np.random.randint(0, T, size=num_samples)
    z_idx = np.random.randint(0, Z, size=num_samples)
    y_idx = np.random.randint(0, Hc, size=num_samples)
    x_idx = np.random.randint(0, Wc, size=num_samples)

    x_norm = x_idx.astype(np.float32) / (Wc - 1)
    y_norm = y_idx.astype(np.float32) / (Hc - 1)
    coords2 = np.stack([x_norm, y_norm], axis=-1).astype(np.float32)

    vals = data_norm[t_idx, z_idx, y_idx, x_idx][..., None].astype(np.float32)
    latent_vecs = latents[t_idx, z_idx]  # [N,C]

    ds = tf.data.Dataset.from_tensor_slices(((coords2, latent_vecs), vals))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ============================================================
# 7) AE-conditioned implicit model
# ============================================================

def positional_encoding_2d_tf(xy, L=6):
    """
    xy: [B,2] = (x,y) in [0,1]
    """
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


def build_ae_implicit_model(latent_dim, hidden=256, depth=6, L_xy=6):
    coords2_in = tf.keras.Input(shape=(2,), name="xy")
    latent_in  = tf.keras.Input(shape=(latent_dim,), name="z_lat")

    enc_xy = tf.keras.layers.Lambda(
        positional_encoding_2d_tf,
        arguments={"L": L_xy},
        name="posenc_xy"
    )(coords2_in)

    h = tf.keras.layers.Concatenate(name="concat_xy_lat")([enc_xy, latent_in])
    for i in range(depth):
        h = tf.keras.layers.Dense(hidden, activation='relu', name=f"mlp_{i}")(h)
    out = tf.keras.layers.Dense(1, activation=None, name="u_norm")(h)

    return tf.keras.Model([coords2_in, latent_in], out, name="ae_cond_implicit")


# ============================================================
# 8) Utilities
# ============================================================

def psnr(mse):
    return -10.0 * np.log10(mse + 1e-12)


def evaluate_slice_ae_impl(model, data_norm, coord_info, latents, crop_hw, t_idx, z_idx):
    T, Z, Y, X = coord_info["T"], coord_info["Z"], coord_info["Y"], coord_info["X"]
    Hc, Wc = crop_hw
    gt = data_norm[t_idx, z_idx, :Hc, :Wc]

    xs = np.linspace(0.0, 1.0, Wc, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, Hc, dtype=np.float32)
    Xg, Yg = np.meshgrid(xs, ys, indexing='xy')
    coords2 = np.stack([Xg, Yg], axis=-1).reshape(-1,2)

    latent_vec = latents[t_idx, z_idx]
    z_rep = np.broadcast_to(latent_vec, (coords2.shape[0], latent_vec.shape[0]))

    preds = model.predict([coords2, z_rep], batch_size=4096, verbose=0).reshape(Hc, Wc)
    return gt, preds


# ============================================================
# 9) Main
# ============================================================

def main():
    data_norm, coord_info = load_and_normalize(DATA_FILE)

    # -------- Direct implicit model --------
    if TRAIN_DIRECT_IMPLICIT:
        important_idx = compute_importance_indices(data_norm, percentile=IMPORTANCE_PERCENTILE)
        train_ds_dir = make_training_dataset_direct(
            data_norm, coord_info,
            important_indices=important_idx,
            importance_fraction=IMPORTANCE_FRACTION,
            batch_size=BATCH_SIZE_DIRECT
        )
        val_ds_dir = make_validation_dataset_direct(
            data_norm, coord_info,
            num_samples=VAL_SAMPLES_DIRECT,
            batch_size=BATCH_SIZE_DIRECT
        )

        direct_model = build_direct_implicit_model()
        direct_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
        direct_model.summary()

        lr_sched = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        )

        print("\nTraining direct implicit model...")
        direct_model.fit(
            train_ds_dir,
            epochs=EPOCHS_DIRECT,
            steps_per_epoch=STEPS_PER_EPOCH_DIR,
            validation_data=val_ds_dir,
            callbacks=[lr_sched],
            verbose=1
        )

        # ----- Save direct implicit model -----
        print(f"\nSaving direct implicit model to {DIRECT_MODEL_PATH}")
        direct_model.save(DIRECT_MODEL_PATH)
    else:
        direct_model = None

    # -------- AE + implicit pipeline --------
    if TRAIN_AE_IMPLICIT:
        # AE training
        X_ae, crop_hw = build_ae_dataset_from_volume(data_norm, coord_info)
        N_slices = X_ae.shape[0]
        n_val = max(1, N_slices // 10)
        X_ae_tr = X_ae[:-n_val]
        X_ae_va = X_ae[-n_val:]

        auto, encoder, decoder = build_autoencoder(
            latent_channels=AE_LATENT_CHANNELS,
            encoder_filters=AE_ENCODER_FILTERS
        )
        auto.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
        auto.summary()

        print("\nTraining autoencoder on slices...")
        auto.fit(
            X_ae_tr, X_ae_tr,
            validation_data=(X_ae_va, X_ae_va),
            epochs=AE_EPOCHS,
            batch_size=AE_BATCH_SIZE,
            verbose=1
        )

        # ----- Save AE models -----
        print(f"\nSaving AE models to {MODEL_DIR}")
        auto.save(AE_AUTO_PATH)
        encoder.save(AE_ENCODER_PATH)
        decoder.save(AE_DECODER_PATH)

        # Latent vectors per (t,z)
        latents, latent_dim = compute_slice_latents(
            encoder, X_ae, coord_info, AE_LATENT_CHANNELS
        )

        # ----- Save compressed timeseries for fixed Z -----
        T = coord_info["T"]
        Z = coord_info["Z"]
        if not (0 <= EVAL_Z_IDX < Z):
            raise ValueError(f"EVAL_Z_IDX={EVAL_Z_IDX} out of range 0..{Z-1}")

        compressed_latents = latents[:, EVAL_Z_IDX, :]  # [T, latent_dim]
        print(f"\nSaving compressed latent timeseries for z={EVAL_Z_IDX} to {COMPRESSED_LATENTS_PATH}")
        np.save(COMPRESSED_LATENTS_PATH, compressed_latents)

        # AE-conditioned implicit model
        train_ds_ae = make_training_dataset_ae_impl(
            data_norm, coord_info, latents, crop_hw,
            batch_size=BATCH_SIZE_AE_IMPL
        )
        val_ds_ae = make_validation_dataset_ae_impl(
            data_norm, coord_info, latents, crop_hw,
            num_samples=VAL_SAMPLES_AE_IMPL,
            batch_size=BATCH_SIZE_AE_IMPL
        )

        ae_impl_model = build_ae_implicit_model(latent_dim)
        ae_impl_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
        ae_impl_model.summary()

        lr_sched2 = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, verbose=1
        )

        print("\nTraining AE-conditioned implicit model...")
        ae_impl_model.fit(
            train_ds_ae,
            epochs=EPOCHS_AE_IMPL,
            steps_per_epoch=STEPS_PER_EPOCH_AE,
            validation_data=val_ds_ae,
            callbacks=[lr_sched2],
            verbose=1
        )

        # ----- Save AE-conditioned implicit model -----
        print(f"\nSaving AE-conditioned implicit model to {AE_IMPL_MODEL_PATH}")
        ae_impl_model.save(AE_IMPL_MODEL_PATH)

        # Small sanity check on one slice (optional)
        gt, pred = evaluate_slice_ae_impl(
            ae_impl_model, data_norm, coord_info, latents, crop_hw,
            EVAL_T_IDX, EVAL_Z_IDX
        )
        mse = float(np.mean((pred - gt)**2))
        print(f"\nAE-cond implicit sanity check slice (t={EVAL_T_IDX}, z={EVAL_Z_IDX}): "
              f"MSE={mse:.3e}, PSNR={psnr(mse):.1f} dB")

    print("\nDone training and saving.")


if __name__ == "__main__":
    main()
