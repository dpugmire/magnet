# nerf2d_masked_tf.py
import numpy as np
import tensorflow as tf
from skimage.transform import resize
import matplotlib.pyplot as plt


# -------------------------------------------------
# Helper: high-resolution rendering
# -------------------------------------------------
def render_model_at_res(model, res, L_ray=2.0, batch_size=8192):
    xs = np.linspace(-1, 1, res)
    ys = np.linspace(-1, 1, res)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    xy = np.stack([X, Y], axis=-1).astype(np.float32).reshape(-1, 2)

    preds = []
    for i in range(0, xy.shape[0], batch_size):
        xy_batch = tf.convert_to_tensor(xy[i:i+batch_size])
        sigma, rgb = model(xy_batch)
        color = (1.0 - tf.exp(-sigma * L_ray)) * rgb
        preds.append(color.numpy())
    img = np.concatenate(preds, axis=0).reshape(res, res, 3)
    return np.clip(img, 0, 1)

# ----------------------------
# Ground truth field
# ----------------------------
def gt_sigma_colorOLD(xy):
    x, y = xy[..., 0], xy[..., 1]
    sigma = 8.0 * np.exp(-12*((x-0.3)**2 + (y+0.2)**2))        # density bump
    r = 0.5 + 0.5*np.sin(4*np.pi*x)
    g = 0.5 + 0.5*np.sin(4*np.pi*y)
    b = 0.5 + 0.5*np.sin(4*np.pi*(x+y))
    color = np.stack([r, g, b], axis=-1)                       # RGB in [0,1]
    return sigma.astype(np.float32), color.astype(np.float32)

# Closed-form “rendering” for constant sigma along the ray of length L
def render_from_sigma_colorOLD(sigma, color, L=2.0):
    alpha = 1.0 - np.exp(-sigma * L)                           # in [0,1]
    return (alpha[..., None] * color).astype(np.float32)

# -------------------------------------------------
# New synthetic scene: "Colorful overlapping blobs"
# -------------------------------------------------
def gt_sigma_color(xy):
    """
    Returns a synthetic scene with multiple Gaussian 'lights'
    of different densities and colors. xy is [...,2] in [-1,1]^2.
    """
    x, y = xy[..., 0], xy[..., 1]

    # Three Gaussian blobs with different colors
    def blob(cx, cy, r, color, intensity):
        d2 = (x - cx)**2 + (y - cy)**2
        sigma = intensity * np.exp(-d2 / (2 * r * r))
        col = np.ones(x.shape + (3,), dtype=np.float32) * color
        return sigma, col

    sigma1, col1 = blob(-0.4, -0.2, 0.25, (1.0, 0.2, 0.2), 8.0)  # red-ish
    sigma2, col2 = blob( 0.3,  0.1, 0.3,  (0.2, 0.8, 0.2), 10.0) # green
    sigma3, col3 = blob( 0.0,  0.4, 0.2,  (0.2, 0.3, 1.0), 7.0)  # blue

    # Combine densities (additive)
    sigma = sigma1 + sigma2 + sigma3

    # Weighted color blend proportional to sigma contribution
    color = (
        col1 * (sigma1[..., None]) +
        col2 * (sigma2[..., None]) +
        col3 * (sigma3[..., None])
    ) / np.maximum(sigma[..., None], 1e-6)

    # Add faint background variation to break symmetry
    color += 0.05 * np.stack([
        np.sin(3*np.pi*x)*np.cos(2*np.pi*y),
        np.sin(4*np.pi*y),
        np.cos(5*np.pi*x*y)
    ], axis=-1)

    return sigma.astype(np.float32), np.clip(color, 0.0, 1.0).astype(np.float32)

def render_from_sigma_color(sigma, color, L=2.0):
    """
    Simple “volume” rendering — brightness controlled by density.
    """
    alpha = 1.0 - np.exp(-sigma * L)
    return (alpha[..., None] * color).astype(np.float32)

# ----------------------------
# Positional encoding
# ----------------------------
def positional_encode(x, L=6):
    outs = [x]
    for k in range(L):
        f = (2.0**k) * np.pi
        outs.append(tf.sin(f * x))
        outs.append(tf.cos(f * x))
    return tf.concat(outs, axis=-1)

# ----------------------------
# Model: predicts sigma and RGB
# ----------------------------
class NeRF2D(tf.keras.Model):
    def __init__(self, L=6, hidden=64):
        super().__init__()
        self.L = L
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(4)  # [sigma, r, g, b]
        ])

    def call(self, xy):
        enc = positional_encode(xy, self.L)
        out = self.mlp(enc)
        sigma = tf.nn.softplus(out[..., 0:1])      # >= 0
        rgb   = tf.sigmoid(out[..., 1:4])          # [0,1]
        return sigma, rgb

# ----------------------------
# Setup data
# ----------------------------
res = 64
xs = np.linspace(-1, 1, res, dtype=np.float32)
ys = np.linspace(-1, 1, res, dtype=np.float32)
X, Y = np.meshgrid(xs, ys, indexing='xy')
grid_xy = np.stack([X, Y], axis=-1).reshape(-1, 2)

sigma_gt, color_gt = gt_sigma_color(grid_xy)
img_gt = render_from_sigma_color(sigma_gt.reshape(res,res), color_gt.reshape(res,res,3))

# ----------------------------
# Train on RENDERED pixels
# ----------------------------
model = NeRF2D(L=6, hidden=64)
opt = tf.keras.optimizers.Adam(1e-3)
L_ray = 2.0

@tf.function
def train_step(x, target_rgb):
    with tf.GradientTape() as tape:
        sigma, rgb = model(x)                      # [B,1], [B,3]
        pred_rgb = (1.0 - tf.exp(-sigma * L_ray)) * rgb
        loss = tf.reduce_mean((pred_rgb - target_rgb)**2)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss

coords = grid_xy.astype(np.float32)
targets = img_gt.reshape(-1, 3).astype(np.float32)

for step in range(2000):
    idx = np.random.randint(0, coords.shape[0], size=2048)
    loss = train_step(tf.convert_to_tensor(coords[idx]),
                      tf.convert_to_tensor(targets[idx]))
    if step % 200 == 0:
        print(f"step {step}: loss={loss.numpy():.6f}")

# ----------------------------
# Evaluate & visualize
# ----------------------------
xy_tf = tf.convert_to_tensor(grid_xy)
sigma_pred, rgb_pred = model(xy_tf)
img_pred = ((1.0 - tf.exp(-sigma_pred * L_ray)) * rgb_pred).numpy()
img_pred = img_pred.reshape(res, res, 3)

# Compute per-pixel error (L2 or absolute)
error_map = np.linalg.norm(img_gt - img_pred, axis=-1)  # Euclidean per-pixel RGB error
# or: error_map = np.mean(np.abs(img_gt - img_pred), axis=-1)  # mean absolute error

# Normalize for visualization
error_map /= error_map.max() + 1e-8

# Print a quantitative summary
mse = np.mean((img_gt - img_pred)**2)
psnr = -10.0 * np.log10(mse + 1e-12)
print(f"MSE = {mse:.6f}, PSNR = {psnr:.2f} dB")

# Show all three images side-by-side
plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(img_gt)
plt.title("Ground Truth")

plt.subplot(1,3,2)
plt.imshow(img_pred)
plt.title("Predicted")

plt.subplot(1,3,3)
plt.imshow(error_map, cmap='inferno')
plt.title("Error Map (normalized)")

plt.tight_layout()
plt.show()


res_list = [64, 128, 256, 512]

fig, axs = plt.subplots(1, len(res_list)+1, figsize=(16,4))
axs[0].imshow(img_gt)
axs[0].set_title("Ground Truth (64×64)")
axs[0].axis("off")

for j, res in enumerate(res_list):
    img_pred_hi = render_model_at_res(model, res)
    axs[j+1].imshow(img_pred_hi)
    axs[j+1].set_title(f"Predicted {res}×{res}")
    axs[j+1].axis("off")

plt.tight_layout()
plt.show()


res_ref = 512
img_gt_ref = resize(img_gt, (res_ref, res_ref), order=1, anti_aliasing=True)

res_list = [32, 64, 128, 256, 512]

# First pass: compute all errors to find global max (for shared scale)
error_maps = []
pred_imgs = []
psnrs = []
for res in res_list:
    img_pred = render_model_at_res(model, res)
    img_pred_resized = resize(img_pred, (res_ref, res_ref), order=1, anti_aliasing=True)
    err = np.linalg.norm(img_pred_resized - img_gt_ref, axis=-1)
    mse = np.mean((img_pred_resized - img_gt_ref)**2)
    psnr = -10.0 * np.log10(mse + 1e-12)
    error_maps.append(err)
    pred_imgs.append(img_pred)
    psnrs.append(psnr)

global_err_max = max([e.max() for e in error_maps])

# -------------------------------------------------
# Plot grid: predictions on top, error maps below
# -------------------------------------------------
fig, axs = plt.subplots(2, len(res_list)+1, figsize=(16,6))

# Ground truth
axs[0,0].imshow(img_gt_ref)
axs[0,0].set_title("Ground Truth (ref)")
axs[0,0].axis("off")
axs[1,0].axis("off")

# Render predictions + errors
for j, res in enumerate(res_list):
    axs[0,j+1].imshow(pred_imgs[j])
    axs[0,j+1].set_title(f"Predicted {res}×{res}\nPSNR {psnrs[j]:.2f} dB")
    axs[0,j+1].axis("off")

    axs[1,j+1].imshow(error_maps[j], cmap='inferno', vmin=0, vmax=global_err_max)
    axs[1,j+1].set_title("Error Map (shared scale)")
    axs[1,j+1].axis("off")

# Shared colorbar for error maps
cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.33])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=global_err_max)
cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='inferno'), cax=cbar_ax)
cb.set_label("Error magnitude (shared scale)")

plt.tight_layout(rect=[0,0,0.9,1])
plt.show()
