# ae_2d_256.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# ============================================================
# 1) Synthetic dataset: 256x256 scalar fields (grayscale)
#    (Overlapping blobs + sinusoidal ripples -> nontrivial)
# ============================================================
def make_synthetic_image(seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    H = W = 256
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    x = (x / (W-1)) * 2 - 1
    y = (y / (H-1)) * 2 - 1

    def blob(cx, cy, r, amp):
        return amp * np.exp(-((x-cx)**2 + (y-cy)**2) / (2*r*r))

    # Random-ish parameters per image
    amps = rng.uniform(0.6, 1.2, size=3)
    radii = rng.uniform(0.15, 0.35, size=3)
    centers = rng.uniform(-0.6, 0.6, size=(3,2))

    img = (blob(centers[0,0], centers[0,1], radii[0], amps[0]) +
           blob(centers[1,0], centers[1,1], radii[1], amps[1]) +
           blob(centers[2,0], centers[2,1], radii[2], amps[2]))

    # Add ripples
    fx = rng.integers(3, 8)
    fy = rng.integers(2, 6)
    rip = 0.25 * (np.sin(fx*np.pi*x) * np.cos(fy*np.pi*y))

    # Combine & normalize to [0,1]
    img = img + rip
    img -= img.min()
    img /= (img.max() + 1e-8)
    return img.astype(np.float32)

def make_dataset(n_train=512, n_val=32):
    # create training datasets.
    Xtr = np.stack([make_synthetic_image(seed=i) for i in range(n_train)], axis=0)[..., None]
    # create validation datasets.
    Xva = np.stack([make_synthetic_image(seed=10_000+i) for i in range(n_val)], axis=0)[..., None]
    return Xtr, Xva  # shapes: [N,256,256,1]

# ============================================================
# 2) Autoencoder (Conv → bottleneck → ConvTranspose)
# ============================================================
def build_autoencoder(latent_channels=8):
    inputs = tf.keras.Input(shape=(256, 256, 1))

    # Encoder (downsample by strided conv; keep 'same' padding)
    #3x3 kernel across image
    #strides: down sample image by 2.
    # padding: 'same' adds zeros around border to keep size consistent when needed.
    # input is 1 channel (grayscale). Output of layer 1 is 32 channels.
    #   256x256 --> 128x128x32.
    #   Each channel has 10 parameters: (3x3 kernel + bias). Total of 320 parameters.

    x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)   # 128x128x32
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)        # 64x64x64
    x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)        # 32x32x64
    x = tf.keras.layers.Conv2D(latent_channels, 3, strides=2, padding='same', activation='relu', name='latent')(x)  # 16x16x8
    # for an input of 256x256x1 = 65536, the output is 16x16x8 = 2048. This results in a 32X reduction.
    # The 16x16x8 "object" can approximate the input image.

    # Decoder
    # input is 16x16x8 (input to first layer is x).
    y = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)   # 32x32x64
    y = tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(y)   # 64x64x64
    y = tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(y)   # 128x128x32
    outputs = tf.keras.layers.Conv2DTranspose(1, 3, strides=2, padding='same', activation='sigmoid')(y)  # 256x256x1

    #auto is used for training. You feed in an image, and it outputs a reconstructed image. You can use the output to evaluate the quality.
    auto = tf.keras.Model(inputs, outputs, name='autoencoder')

    # Separate encoder/decoder convenience models
    encoder = tf.keras.Model(inputs, auto.get_layer('latent').output, name='encoder')
    encoder.summary()
    for w in encoder.get_weights():
        print(w.shape)

    # To build decoder separately, define a latent Input with 16x16xC:
    z_in = tf.keras.Input(shape=(16, 16, latent_channels))
    z = auto.layers[-4](z_in)  # Conv2DTranspose 64
    z = auto.layers[-3](z)     # Conv2DTranspose 64
    z = auto.layers[-2](z)     # Conv2DTranspose 32
    z_out = auto.layers[-1](z) # Conv2DTranspose 1 (sigmoid)
    decoder = tf.keras.Model(z_in, z_out, name='decoder')

    return auto, encoder, decoder

# ============================================================
# 3) Train
# ============================================================
def train_autoencoder():
    Xtr, Xva = make_dataset(n_train=512, n_val=32)
    auto, enc, dec = build_autoencoder(latent_channels=8)

    #compile and train.
    auto.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                 loss='mse')

    #inputs and targets are both Xtr. We want the output to be the same as the input.
    #after each epoch, test the model on another set of inputs (32 of them).
    #shuffle prevents the model from memorizing the order of the onputs.
    hist = auto.fit(
        Xtr, Xtr,
        validation_data=(Xva, Xva),
        epochs=10, batch_size=32, shuffle=True,
        verbose=1
    )
    return auto, enc, dec, (Xtr, Xva)

# ============================================================
# 4) Metrics & visualization
# ============================================================
def psnr(mse):
    return -10.0 * np.log10(mse + 1e-12)

def eval_and_plot(auto, enc, dec, Xva, sample_index=0, out_dir=None):
    os.makedirs(out_dir or ".", exist_ok=True)
    x = Xva[sample_index:sample_index+1]  # [1,256,256,1]
    x_hat = auto.predict(x, verbose=0)

    # Error maps
    abs_err = np.abs(x_hat - x)[0, ..., 0]         # [256,256]
    sq_err  = ((x_hat - x)**2)[0, ..., 0]          # [256,256]
    mse = float(np.mean(sq_err))
    try:
        ssim = float(tf.image.ssim(tf.convert_to_tensor(x), tf.convert_to_tensor(x_hat), max_val=1.0).numpy()[0])
    except Exception:
        ssim = np.nan

    # Compression ratio (approx): input floats vs latent floats
    # Input: 256*256*1 floats
    # Latent: 16*16*latent_channels floats (from the encoder output)
    latent = enc.predict(x, verbose=0)
    latent_shape = latent.shape[1:]  # (16,16,C)
    in_floats = 256*256*1
    latent_floats = np.prod(latent_shape)
    comp_ratio = in_floats / latent_floats  # how many input floats per latent float

    print(f"MSE:  {mse:.6f}")
    print(f"PSNR: {psnr(mse):.2f} dB")
    print(f"SSIM: {ssim:.4f}")
    print(f"Approx. compression ratio (floats): {comp_ratio:.2f}:1  (latent shape: {latent_shape})")

    # Plots
    fig, axs = plt.subplots(1, 3, figsize=(13,4))

    axs[0].imshow(x[0, ..., 0], cmap='gray', vmin=0, vmax=1)
    axs[0].set_title("Original")
    axs[0].axis('off')

    axs[1].imshow(x_hat[0, ..., 0], cmap='gray', vmin=0, vmax=1)
    axs[1].set_title("Reconstruction")
    axs[1].axis('off')

    im = axs[2].imshow(abs_err, cmap='inferno')
    axs[2].set_title("Per-pixel |error|")
    axs[2].axis('off')
    cbar = fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04)
    cbar.set_label("Absolute error")

    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, "recon_and_error.png"), dpi=150)
    plt.show()

    # Error histogram
    plt.figure(figsize=(5,4))
    plt.hist(abs_err.ravel(), bins=60, color='gray')
    plt.title("Absolute error histogram")
    plt.xlabel("|x̂ - x|"); plt.ylabel("count")
    plt.tight_layout()
    if out_dir:
        plt.savefig(os.path.join(out_dir, "error_hist.png"), dpi=150)
    plt.show()

# ============================================================
# 5) Main
# ============================================================
if __name__ == "__main__":
    auto, enc, dec, (Xtr, Xva) = train_autoencoder()
    eval_and_plot(auto, enc, dec, Xva, sample_index=0, out_dir=None)

    # Example: how to store compressed latent for one sample
    # x = Xva[0:1]                              # [1,256,256,1]
    # z = enc.predict(x, verbose=0).astype(np.float32)  # [1,16,16,C]
    # np.save("latent.npy", z)                  # "compressed" representation
    # # Later, to reconstruct:
    # z_loaded = np.load("latent.npy")
    # x_recon = dec.predict(z_loaded, verbose=0)  # [1,256,256,1]
