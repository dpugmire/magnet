import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import measure

# ============================================================
# 1. Synthetic ground truth scene
# ============================================================
def gt_sigma_color(xy):
    """Synthetic scene: overlapping colored blobs."""
    x, y = xy[..., 0], xy[..., 1]

    def blob(cx, cy, r, color, intensity):
        d2 = (x - cx)**2 + (y - cy)**2
        sigma = intensity * np.exp(-d2 / (2 * r * r))
        col = np.ones(x.shape + (3,), dtype=np.float32) * color
        return sigma, col

    s1, c1 = blob(-0.4, -0.2, 0.25, (1.0, 0.2, 0.2), 8.0)
    s2, c2 = blob( 0.3,  0.1, 0.3,  (0.2, 0.8, 0.2), 10.0)
    s3, c3 = blob( 0.0,  0.4, 0.2,  (0.2, 0.3, 1.0), 7.0)
    sigma = s1 + s2 + s3
    color = (
        c1 * s1[..., None] + c2 * s2[..., None] + c3 * s3[..., None]
    ) / np.maximum(sigma[..., None], 1e-6)
    return sigma.astype(np.float32), np.clip(color, 0, 1).astype(np.float32)

def render_from_sigma_color(sigma, color, L=2.0):
    """Render brightness-controlled color image."""
    alpha = 1.0 - np.exp(-sigma * L)
    return (alpha[..., None] * color).astype(np.float32)

# ============================================================
# 2. Positional Encoding
# ============================================================
def positional_encode(x, L=6):
    outs = [x]
    for k in range(L):
        f = 2.0**k * np.pi
        outs.append(tf.sin(f * x))
        outs.append(tf.cos(f * x))
    return tf.concat(outs, axis=-1)

# ============================================================
# 3. Model
# ============================================================
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
        sigma = tf.nn.softplus(out[..., 0:1])
        rgb   = tf.sigmoid(out[..., 1:4])
        return sigma, rgb

# ============================================================
# 4. Create training data
# ============================================================
res = 64
xs = np.linspace(-1, 1, res, dtype=np.float32)
ys = np.linspace(-1, 1, res, dtype=np.float32)
X, Y = np.meshgrid(xs, ys, indexing='xy')
grid_xy = np.stack([X, Y], axis=-1).reshape(-1, 2)
sigma_gt, color_gt = gt_sigma_color(grid_xy)
img_gt = render_from_sigma_color(sigma_gt.reshape(res, res),
                                 color_gt.reshape(res, res, 3))

# ============================================================
# 5. Train the model
# ============================================================
model = NeRF2D(L=6, hidden=64)
opt = tf.keras.optimizers.Adam(1e-3)
L_ray = 2.0

coords = grid_xy.astype(np.float32)
targets = img_gt.reshape(-1, 3).astype(np.float32)

@tf.function
def train_step(x, target):
    with tf.GradientTape() as tape:
        sigma, rgb = model(x)
        pred_rgb = (1.0 - tf.exp(-sigma * L_ray)) * rgb
        loss = tf.reduce_mean((pred_rgb - target)**2)
    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for step in range(2000):
    idx = np.random.randint(0, coords.shape[0], size=2048)
    loss = train_step(tf.convert_to_tensor(coords[idx]),
                      tf.convert_to_tensor(targets[idx]))
    if step % 200 == 0:
        print(f"Step {step}: loss={loss.numpy():.6f}")

# ============================================================
# 6. Render learned field
# ============================================================
res_pred = 256
xs = np.linspace(-1, 1, res_pred)
ys = np.linspace(-1, 1, res_pred)
X, Y = np.meshgrid(xs, ys, indexing='xy')
xy = np.stack([X, Y], axis=-1).astype(np.float32).reshape(-1, 2)

sigma_pred, rgb_pred = model(tf.convert_to_tensor(xy))
sigma_pred = sigma_pred.numpy().reshape(res_pred, res_pred)

img_pred = ((1.0 - np.exp(-sigma_pred[..., None] * L_ray))
            * rgb_pred.numpy().reshape(res_pred, res_pred, 3))


# ============================================================
# 7. Extract contours
# ============================================================
iso_value = 0.5 * sigma_pred.max()  # contour threshold
iso_value = 2.0

# Method 1: matplotlib.contour (quick)
plt.figure(figsize=(6,6))
CS = plt.contour(X, Y, sigma_pred, levels=[iso_value], colors='cyan')
plt.imshow(sigma_pred, extent=(-1,1,-1,1), origin='lower', cmap='inferno')
plt.title(f"σ(x,y) field with contour at {iso_value:.3f}")
plt.axis("equal")
plt.show()

# Method 2: skimage.measure.find_contours (access vertices)
contours = measure.find_contours(sigma_pred, level=iso_value)
plt.figure(figsize=(6,6))
plt.imshow(img_pred, extent=(-1,1,-1,1), origin='lower')
for c in contours:
    cx = -1 + 2*c[:,1]/(res_pred-1)
    cy = -1 + 2*c[:,0]/(res_pred-1)
    plt.plot(cx, cy, color='cyan', linewidth=2)
plt.title("Learned color image with σ contour overlay")
plt.axis("equal")
plt.show()

# Print contour vertex count
for i, c in enumerate(contours):
    print(f"Contour {i}: {len(c)} vertices")
