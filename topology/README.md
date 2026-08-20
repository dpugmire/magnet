# AE + PD Loss for 2D MHD Scalars (ADIOS2)

This repository trains a convolutional autoencoder (AE) on 2D MHD scalar fields from ADIOS2 BP time-series files and adds a topology term from persistence diagrams (PD) via persistence images (PI).

Scope: **AE + PD loss only** (no INR).

## Data Layout

Default expected tree:

```text
runs/
  run001/output.bp
  run002/output.bp
  run003/output.bp
```

- Default scalar variable: `rho` (override with `--scalar_name`)
- Timesteps: ADIOS steps
- Supported scalar storage layouts:
1. Per-step 2D: `[ny, nx]` at each ADIOS step
2. Packed 3D: `[step, ny, nx]` in a single variable

Run discovery:
- Default: scan `runs/` for subdirectories containing `output.bp`
- Override with `--runs_dir` and/or explicit `--run_dirs`

## Install

Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Entrypoint: `train_ae_pd.py`

Loss:

```text
L_total = L_recon + lambda_pd * L_pd
L_recon: L1(recon, gt)
L_pd: PI distance (L2 or SmoothL1/Huber)
```

Important topology note:
- PD/PI is computed using Gudhi on CPU from detached tensors.
- This PD term is **non-differentiable** in this implementation (no gradient path through PI extraction).
- `L_pd` is still logged and included in the scalar objective value.

### Example

With topology term (AE+PD):

```bash
python train_ae_pd.py \
  --runs_dir runs \
  --bp_file output.bp \
  --scalar_name rho \
  --outdir outputs/ae_pd \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-3 \
  --norm global --norm_samples 200 \
  --pd_downsample 128 --pi_res 64 \
  --pd_dims 0 1 \
  --pd_min_persistence 0.0 \
  --lambda_pd 0.1 \
  --pd_every 10 \
  --seed 0
```

Without topology term (AE-only baseline):

```bash
python train_ae_pd.py \
  --runs_dir runs \
  --bp_file output.bp \
  --scalar_name rho \
  --outdir outputs/ae_only \
  --epochs 100 \
  --batch_size 8 \
  --lr 1e-3 \
  --norm global --norm_samples 200 \
  --no-use_pd \
  --seed 0
```

Useful flags:
- `--split_by_run` to hold out full runs for validation
- `--val_fraction 0.1`
- `--norm none|per_image|global`
- `--pd_loss l2|huber`
- `--pd_min_persistence X` to ignore PD pairs with persistence `<= X`
- `--pd_every N` to compute PD loss every N training steps
- `--pd_batch_items K` to compute PD loss on only first `K` batch items for speed
- `--no-use_pd` to disable topology loss and train AE-only
- `--mode_subdir` (default on) writes to `outdir/with_pd` or `outdir/no_pd`
- `--allow_overwrite` to reuse an existing run folder without auto-creating a timestamped folder
- `--amp` for mixed precision on CUDA

Outputs in `--outdir`:
- `best.pt`
- `last.pt`
- `config.json`
- `history.json`
- `train.log`

By default, training writes into mode-specific folders so PD and no-PD checkpoints are separated.

## Evaluation

Entrypoint: `eval_ae_pd.py`

Given a checkpoint and one `(run, step)`, it saves:
- `gt.npy`
- `recon.npy`
- `pi_gt.npy`
- `pi_recon.npy`
- `pd_gt_dim<d>.npy` and `pd_recon_dim<d>.npy` for each requested PD dimension
- `metrics.json` (MAE, RMSE, PSNR, bottleneck distance per PD dim)
- optional PNGs (`--save_png`, requires matplotlib):
  - `gt.png`, `recon.png`
  - `pi_gt.png`, `pi_recon.png`
  - `pd_compare_dim<d>.png` (GT vs recon persistence diagram scatter)

### Example

```bash
python eval_ae_pd.py \
  --checkpoint outputs/ae_pd/best.pt \
  --runs_dir runs \
  --run rusanov_first_order \
  --step 10 \
  --pd_downsample 128 --pi_res 64 \
  --pd_dims 0 1 \
  --pd_min_persistence 0.0 \
  --save_png
```

## Code Structure

```text
train_ae_pd.py
eval_ae_pd.py

ae_pd/
  __init__.py
  adios_dataset.py
  datasets.py
  models.py
  topology.py
  losses.py
  train.py
  utils.py
```

## ADIOS Reader API

`ae_pd.adios_dataset.AdiosScalarArchive` exposes:
- `listRuns() -> List[str]`
- `getStepCount(runName) -> int`
- `getGridShape(runName) -> (ny, nx)`
- `readScalar(runName, stepIdx) -> np.ndarray [ny, nx] float32`

It includes an LRU cache for recent `(run, step)` reads.
