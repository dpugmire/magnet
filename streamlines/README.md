# Vector INR for ADIOS2 2D Time-Varying Velocity Fields

## Quick start

Expected layout (dummy example paths):

```text
/data/experiments/
  ens_000/
    output.bp/
  ens_001/
    output.bp/
  ens_002/
    output.bp/
```

Each `output.bp` must contain `vx` and `vy` as either:

- `[step, ny, nx]` in one ADIOS step, or
- `[ny, nx]` in each ADIOS step (streaming semantics).

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train (example):

```bash
python train_vector_inr.py \
  --ensemble_dirs /data/experiments/ens_000 /data/experiments/ens_001 /data/experiments/ens_002 \
  --bp_file output.bp \
  --vx_name vx \
  --vy_name vy \
  --epochs 200 \
  --batch_points 65536 \
  --embed_dim 32 \
  --hidden 256 \
  --layers 6 \
  --freqs 10 \
  --lr 1e-3 \
  --outdir runs/ot_inr \
  --vorticity_loss \
  --lambda_omega 0.1 \
  --warmup_epochs 20
```

Evaluate one step:

```bash
python eval_vector_inr.py \
  --checkpoint runs/ot_inr/checkpoint_latest.pt \
  --ensemble_dirs /data/experiments/ens_000 /data/experiments/ens_001 /data/experiments/ens_002 \
  --bp_file output.bp \
  --vx_name vx \
  --vy_name vy \
  --ensemble_idx 0 \
  --step 10 \
  --outdir runs/ot_inr_eval \
  --vorticity_metric \
  --save_png
```

Evaluate a step range:

```bash
python eval_vector_inr.py \
  --checkpoint runs/ot_inr/checkpoint_latest.pt \
  --ensemble_dirs /data/experiments/ens_000 /data/experiments/ens_001 /data/experiments/ens_002 \
  --ensemble_idx 0 \
  --step_start 0 \
  --step_end 49 \
  --step_stride 5 \
  --outdir runs/ot_inr_eval_range
```

## Repository structure

```text
train_vector_inr.py
eval_vector_inr.py
vector_inr/
  __init__.py
  adios_dataset.py
  sampling.py
  models.py
  losses.py
  train.py
  utils.py
requirements.txt
README.md
```

## Notes

- Training samples random points `(ensemble, step, iy, ix)`, not full images.
- Coordinates are normalized as:
  - `x = (ix / (nx - 1)) * 2 - 1`
  - `y = (iy / (ny - 1)) * 2 - 1`
  - `t = step / (nsteps - 1)` (or `0` when `nsteps=1`)
- `vx` and `vy` are normalized by global mean/std estimated via random subsampling.
- Vorticity loss is optional and enabled after warmup epochs.
- Checkpoints include model state, optimizer state, scheduler state, config, and normalization stats.
- Different ensemble grid sizes are rejected with a clear error message.
- PNG export in evaluation uses `matplotlib`; if unavailable, `.npy` outputs and metrics still work.

## Main CLI options

Training:

```bash
python train_vector_inr.py --help
```

Evaluation:

```bash
python eval_vector_inr.py --help
```

# train

python train_vector_inr.py \
 --ensemble_dirs ./runs/hll_first_order ./runs/muscl_hll ./runs/rusanov_first_order \
 --bp_file output.bp --vx_name vx --vy_name vy \
 --outdir runs/ot_inr_retrain_fast \
 --cache_steps 512 --omega_cache_steps 512 --epochs 10 --steps_per_epoch 10

# evaluate

python eval_vector_inr.py \
 --checkpoint runs/ot_inr_retrain_fast/checkpoint_latest.pt \
 --ensemble_dirs ./runs/hll_first_order ./runs/muscl_hll ./runs/rusanov_first_order \
 --ensemble_idx 0 \
 --step 0 \
 --outdir runs/ot_inr_eval \
 --overlay_png \
 --streamline_seed_nx 30 \
 --streamline_seed_ny 30 \
 --gt_color tab:blue \
 --pred_color tab:orange
