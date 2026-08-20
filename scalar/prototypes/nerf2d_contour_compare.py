import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.transform import resize

# ============================================================
# 1. Synthetic ground truth scene
# ============================================================
def gt_sigma_color(xy):
    """Synthetic scene with three overlapping colored blobs."""
    x, y = xy[..., 0], xy[..., 1]

    def blob(cx, cy, r, color, intensity):
        d2 = (x - cx)**2 + (y - cy)**2
        sigma = intensity * np.exp(-d2 / (2 * r * r))
        col = np.ones(x.shape + (3,), dtype=np.float32) * color
        return sigma, col

    s1, c1 = blob(-0.4, -0.2, 0.25, (1.0, 0.2, 0.2), 8.0)
    s2, c2 = blob( 0.3,  0.1, 0.3,  (0.2, 0.8, 0.2),10.0)
    s3, c3 = blob( 0.0,  0.4, 0.2,  (0.2, 0.3, 1.0), 7.0)
    sigma = s1 + s2 + s3
    color = (
        c1*s1[...,None] + c2*s2[...,None] + c3*s3[...,None]
    ) / np.maximum(sigma[...,None], 1e-6)
    return sigma.astype(np.float32), np.clip(color,0,1).astype(np.float32)

def render_from_sigma_color(sigma, color, L=2.0):
    alpha = 1.0 - np.exp(-sigma[...,None]*L)
    return alpha * color

# ============================================================
# 2. Positional encoding
# ============================================================
def positional_encode(x, L=6):
    outs = [x]
    for k in range(L):
        f = 2.0**k * np.pi
        outs.append(tf.sin(f*x))
        outs.append(tf.cos(f*x))
    return tf.concat(outs, axis=-1)

# ============================================================
# 3. NeRF-like model
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
        sigma = tf.nn.softplus(out[...,0:1])
        rgb   = tf.sigmoid(out[...,1:4])
        return sigma, rgb

# ============================================================
# 4. Training data (64×64 grid)
# ============================================================
res_train = 64
xs = np.linspace(-1,1,res_train, dtype=np.float32)
ys = np.linspace(-1,1,res_train, dtype=np.float32)
X,Y = np.meshgrid(xs,ys,indexing='xy')
grid_xy = np.stack([X,Y],axis=-1).reshape(-1,2)

sigma_gt, color_gt = gt_sigma_color(grid_xy)
img_gt = render_from_sigma_color(sigma_gt.reshape(res_train,res_train),
                                 color_gt.reshape(res_train,res_train,3))

# ============================================================
# 5. Train model
# ============================================================
model = NeRF2D(L=6, hidden=64)
opt = tf.keras.optimizers.Adam(1e-3)
L_ray = 2.0

coords  = grid_xy.astype(np.float32)
targets = img_gt.reshape(-1,3).astype(np.float32)

@tf.function
def train_step(x, target):
    with tf.GradientTape() as tape:
        sigma, rgb = model(x)
        pred = (1.0 - tf.exp(-sigma * L_ray)) * rgb
        loss = tf.reduce_mean((pred - target)**2)
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
# 6. Utility: render σ & RGB at arbitrary resolution
# ============================================================
def render_field(model, res, L_ray=2.0):
    xs = np.linspace(-1,1,res)
    ys = np.linspace(-1,1,res)
    X,Y = np.meshgrid(xs,ys,indexing='xy')
    xy = np.stack([X,Y],axis=-1).astype(np.float32).reshape(-1,2)
    sigma, rgb = model(tf.convert_to_tensor(xy))
    sigma = sigma.numpy().reshape(res,res)
    rgb   = rgb.numpy().reshape(res,res,3)
    img   = (1.0 - np.exp(-sigma[...,None]*L_ray)) * rgb
    return X,Y,sigma,img

# ============================================================
# 7. Predict & compare at multiple resolutions
# ============================================================
res_list = [64, 128, 256]
n_levels = 4

# Ground truth (high-res reference)
res_ref = 512
xs_ref = np.linspace(-1,1,res_ref)
ys_ref = np.linspace(-1,1,res_ref)
Xr,Yr = np.meshgrid(xs_ref,ys_ref,indexing='xy')
xy_ref = np.stack([Xr,Yr],axis=-1).reshape(-1,2)
sigma_ref,color_ref = gt_sigma_color(xy_ref)
sigma_gt_ref = sigma_ref.reshape(res_ref,res_ref)

fig, axs = plt.subplots(len(res_list), 4, figsize=(16,4*len(res_list)))

for i,res in enumerate(res_list):
    X,Y,sigma_pred,img_pred = render_field(model,res)
    sigma_gt_res = resize(sigma_gt_ref,(res,res),order=1,anti_aliasing=True)
    error = np.abs(sigma_pred - sigma_gt_res)
    levels = np.linspace(0.1*sigma_gt_ref.max(), 0.9*sigma_gt_ref.max(), n_levels)

    # 1️⃣ Predicted σ + contours
    axs[i,0].imshow(sigma_pred, extent=(-1,1,-1,1), origin='lower', cmap='inferno')
    axs[i,0].contour(X,Y,sigma_pred,levels=levels,colors='cyan',linewidths=1)
    axs[i,0].set_title(f"Pred σ (res={res})")
    axs[i,0].axis("off")

    # 2️⃣ Ground truth σ + contours
    xs_gt = np.linspace(-1,1,res)
    ys_gt = np.linspace(-1,1,res)
    Xg,Yg = np.meshgrid(xs_gt,ys_gt,indexing='xy')
    axs[i,1].imshow(sigma_gt_res, extent=(-1,1,-1,1), origin='lower', cmap='inferno')
    axs[i,1].contour(Xg,Yg,sigma_gt_res,levels=levels,colors='lime',linewidths=1)
    axs[i,1].set_title("Ground Truth σ")
    axs[i,1].axis("off")

    # 3️⃣ Error map
    im = axs[i,2].imshow(error, extent=(-1,1,-1,1), origin='lower', cmap='magma')
    axs[i,2].set_title(f"|σ_pred−σ_gt| (res={res})")
    axs[i,2].axis("off")

    # 4️⃣ Overlay of contours (cyan=pred, lime=gt)
    axs[i,3].imshow(sigma_gt_res, extent=(-1,1,-1,1), origin='lower', cmap='gray')
    axs[i,3].contour(X,Y,sigma_pred,levels=levels,colors='cyan',linewidths=1)
    axs[i,3].contour(Xg,Yg,sigma_gt_res,levels=levels,colors='lime',linewidths=1, linestyles='dashed')
    axs[i,3].set_title("Overlay: cyan=pred, lime=gt")
    axs[i,3].axis("off")

fig.colorbar(im, ax=axs[:,2], shrink=0.6, label="Absolute error")
plt.tight_layout()
plt.show()

# ============================================================
# 8. Overlay contours on predicted color image
# ============================================================
res_pred = 256
X,Y,sigma_pred,img_pred = render_field(model,res_pred)
iso_levels = np.linspace(0.2, 0.8, n_levels) * sigma_pred.max()

plt.figure(figsize=(6,6))
plt.imshow(img_pred, extent=(-1,1,-1,1), origin='lower')
for level in iso_levels:
    plt.contour(X,Y,sigma_pred,levels=[level],colors='cyan',linewidths=2)
plt.title("Predicted color image with σ isocontours")
plt.axis("equal")
plt.show()

print("✅ Done: trained, predicted, contoured, compared, and overlayed.")
