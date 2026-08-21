# CAESAR 2D slice decoder

This directory develops direct 2D extraction from CAESAR's latent
representation. Milestone 1 establishes the reference path: a precise plane
convention, a NumPy trilinear oracle, CAESAR latent extraction, and reproducible
reference artifacts. The initial Milestone 2 baseline trains a point-query MLP
against one CAESAR block without reconstructing a dense volume at inference.

The complete staged design is in [PLAN.md](PLAN.md).

## Coordinate convention

A decoded volume is `[D, H, W]`. Normalized spatial coordinates are `(x, y, z)`,
each in `[-1, 1]`, with `align_corners=True` semantics:

- `x` indexes `W`
- `y` indexes `H`
- `z` indexes `D`

A plane is `origin + u * axis_u + v * axis_v`, with orthonormal axes. Samples
are returned as `[v_resolution, u_resolution]`. Axis-aligned helpers reproduce
direct NumPy indexing.

## Initial data

The initial dataset is `Turb_Rot_testset.npz`. Its `data.npy` member is
`float64` with shape `[1, 16, 256, 256, 256]`. Following the project plan, the
first prototype interprets it as `[variable, section, z, y, x]`. Set
`--axis-semantic time` or `unconfirmed` if that interpretation is inappropriate.
Every artifact manifest records this choice.

The archive's `variable_name` array has three names while `data` has one
variable. Milestone 1 uses variable index 0 and preserves the original names in
the manifest instead of guessing how to reconcile that mismatch.

The archive is about 2 GiB. Inspection reads only NPY headers. Reference
generation memory-maps the uncompressed `data.npy` member and writes a selected
`float32` subset, so it does not load the complete source array into memory.

## Commands

From the repository root, inspect the archive without importing PyTorch:

```sh
python3 2d-slice/reference_pipeline.py inspect \
  --data CAESAR/data/Turb_Rot_testset.npz
```

Run the dependency-free tests:

```sh
python3 -m unittest discover -s 2d-slice/tests -v
```

Generate a CAESAR-V reference reconstruction and latent tensor:

```sh
PYTHONPATH=external/pyCAESAR \
python3 2d-slice/reference_pipeline.py reference \
  --data CAESAR/data/Turb_Rot_testset.npz \
  --model CAESAR/data/caesar_v.pt \
  --output-dir 2d-slice/artifacts/turb-rot-section-00 \
  --section-index 0 \
  --frame-start 0 \
  --frame-end 256 \
  --device cuda:0
```

The reference command requires the dependencies used by `external/pyCAESAR`,
notably PyTorch and CompressAI. They are imported lazily, so geometry, archive
inspection, and unit tests work without them. This milestone adds no dependency.

The output directory contains:

- `input_subset.npz`: selected `[1, 1, D, H, W]` input as `float32`
- `original.npy`: the volume given to CAESAR
- `caesar_base.npy`: neural CAESAR reconstruction before residual postprocessing
- `caesar_reference.npy`: the ordinary full CAESAR-V reconstruction
- `caesar_residual.npy`: correction added by CAESAR postprocessing
- `q_latent.npy`: quantized latents in canonical `[C,D_latent,H_latent,W_latent]`
- `manifest.json`: source selection, model hash, parameters, shapes, semantics,
  and full-volume staged error summaries

Generated files under `2d-slice/artifacts/` are ignored by Git.

## Point-query baseline

The direct decoder samples a local feature vector from `q_latent` using
trilinear interpolation, concatenates normalized `(x,y,z)` coordinates and
Fourier features, and maps that vector to one normalized scalar with an MLP.
CAESAR's block scale and offset convert the prediction back to field units.

Only `q_latent`, scale, and offset are decoder inputs. CAESAR's error-bounded
postprocessing residual is not supplied to this baseline, although the full
CAESAR reconstruction remains its training target. The reported slice-decoder
error therefore includes any correction information that cannot be inferred
from `q_latent`.

The one-block mode remains a useful pipeline check. Multi-block mode discovers
all blocks in multiple artifacts and splits them by the original source section,
not by plane, preventing planes from the same volume from leaking across splits.
The target is the full CAESAR reconstruction. When the manifest's source archive
is available, evaluation memory-maps ground truth directly from that raw archive;
otherwise it falls back to the pipeline's pre-compression copy.

Train on the eight-frame reference artifact created above:

```sh
source ~/venv/bin/activate
cd /Users/dpn/proj/MAGNET

python 2d-slice/train_point_decoder.py \
  --artifact-dir 2d-slice/artifacts/test-section-00 \
  --output-dir 2d-slice/artifacts/test-section-00/point-decoder \
  --steps 2000 \
  --batch-size 4 \
  --train-resolution 32 \
  --eval-resolution 128 \
  --eval-planes 8 \
  --orientation mixed \
  --device cpu
```

For a quick pipeline check, use `--steps 10 --eval-planes 2`. The output is:

- `point_decoder.pt`: model weights, architecture, and artifact/split metadata
- `point_decoder_metrics.json`: validation/test errors and inference timing
- `validation_point_decoder_example.npz`: one validation plane triplet
- `test_point_decoder_example.npz`: one test plane triplet

The three error groups answer different questions:

- `compression`: raw source plane versus full CAESAR reconstruction
- `sliceDecoder`: full CAESAR reconstruction versus direct point-query result
- `endToEnd`: raw source plane versus direct point-query result

The recorded latency covers only direct slice inference. It intentionally does
not include CAESAR compression, artifact loading, or disk I/O.

## Multi-section generalization

First confirm that the source third axis is spatial depth before labeling the
artifacts `spatial_z`. pyCAESAR's generic format calls that dimension `T`; if it
is time for this dataset, use `--axis-semantic time` and interpret arbitrary
planes as space-time cuts instead of spatial slices.

Generate one full 256-frame artifact per independent source section:

```sh
source ~/venv/bin/activate
cd /Users/dpn/proj/MAGNET

for section in {0..15}; do
  tag=$(printf "%02d" "$section")
  PYTHONPATH=external/pyCAESAR \
  python 2d-slice/reference_pipeline.py reference \
    --data CAESAR/data/Turb_Rot_testset.npz \
    --model CAESAR/data/caesar_v.pt \
    --output-dir "2d-slice/artifacts/turb-rot-sections/section-${tag}" \
    --section-index "$section" \
    --frame-start 0 \
    --frame-end 256 \
    --axis-semantic spatial_z \
    --device cpu \
    --gae-device cpu
done
```

Each nonconstant full-depth artifact normally contains 32 eight-frame latent
blocks. Train on sections `0–11`, select with `12–13`, and report final results
only on `14–15`:

```sh
python 2d-slice/train_point_decoder.py \
  --artifact-root 2d-slice/artifacts/turb-rot-sections \
  --output-dir 2d-slice/artifacts/turb-rot-point-generalization \
  --train-sections 0-11 \
  --validation-sections 12-13 \
  --test-sections 14-15 \
  --expected-axis-semantic spatial_z \
  --steps 10000 \
  --batch-size 4 \
  --train-resolution 32 \
  --eval-resolution 128 \
  --eval-planes 8 \
  --orientation mixed \
  --device cpu
```

`--eval-planes` is per block. Legacy metrics use format version 2; evaluation
with staged artifacts uses version 3 and adds base/residual comparisons. Both
formats contain aggregate and per-section validation/test results. Training
planes are currently contained within individual eight-frame blocks;
constructing one continuous slice across block boundaries is a separate
stitching step.

## Staged decompression diagnostic

Artifacts generated by the current reference pipeline separate the neural base
reconstruction from CAESAR's residual correction. Legacy artifacts remain
loadable, but they must be regenerated to provide these staged arrays.

Generate one held-out staged artifact and evaluate an existing point decoder:

```sh
PYTHONPATH=external/pyCAESAR \
python 2d-slice/reference_pipeline.py reference \
  --data CAESAR/data/Turb_Rot_testset.npz \
  --model CAESAR/data/caesar_v.pt \
  --output-dir 2d-slice/artifacts/turb-rot-staged/section-14 \
  --section-index 14 \
  --frame-start 0 \
  --frame-end 256 \
  --axis-semantic spatial_z \
  --device cpu \
  --gae-device cpu

python 2d-slice/diagnose_point_decoder.py \
  --artifact-dir 2d-slice/artifacts/turb-rot-staged/section-14 \
  --checkpoint \
    2d-slice/artifacts/turb-rot-point-generalization/point_decoder.pt \
  --output-dir 2d-slice/artifacts/turb-rot-staged-diagnostic/section-14 \
  --planes-per-block 8 \
  --resolution 128 \
  --device cpu
```

The diagnostic reports six comparisons: raw versus neural base, neural base
versus final CAESAR, raw versus final CAESAR, point decoder versus neural base,
point decoder versus final CAESAR, and raw versus point decoder. It also writes
one NPZ example and an optional comparison PNG when Matplotlib is available.

## Plane-aligned convolutional baseline

The plane decoder samples parallel 16x16 feature maps from each
`[64,2,16,16]` latent block around the requested plane. It concatenates the
slab features and 3D positional features, processes the entire map with
residual 2D convolutions, and uses three learned refinement stages to produce a
128x128 slice. Unlike the point decoder, adjacent plane pixels share
latent-neighborhood context. Five slab samples spanning one latent-grid cell
on either side of the requested plane are the default; `--slab-samples 1`
reproduces the original single-plane architecture.

This first experiment deliberately targets `caesar_base.npy`, the output of
CAESAR's neural decoder before residual correction. This isolates whether a
plane-specific network can recover the information already demonstrated to be
present in `q_latent`. Training requires staged artifacts; the older
`turb-rot-sections` artifacts do not contain the neural-base arrays.

Generate staged artifacts for the section split:

```sh
source ~/venv/bin/activate
cd /Users/dpn/proj/MAGNET

for section in {0..15}; do
  tag=$(printf "%02d" "$section")
  PYTHONPATH=external/pyCAESAR \
  python 2d-slice/reference_pipeline.py reference \
    --data CAESAR/data/Turb_Rot_testset.npz \
    --model CAESAR/data/caesar_v.pt \
    --output-dir "2d-slice/artifacts/turb-rot-staged-sections/section-${tag}" \
    --section-index "$section" \
    --frame-start 0 \
    --frame-end 256 \
    --axis-semantic spatial_z \
    --device cpu \
    --gae-device cpu
done
```

Train the 128x128 plane decoder on sections `0-11`, validate on `12-13`, and
test once on `14-15`:

```sh
python 2d-slice/train_plane_decoder.py \
  --artifact-root 2d-slice/artifacts/turb-rot-staged-sections \
  --output-dir 2d-slice/artifacts/turb-rot-plane-slab-generalization \
  --train-sections 0-11 \
  --validation-sections 12-13 \
  --test-sections 14-15 \
  --expected-axis-semantic spatial_z \
  --resolution 128 \
  --coarse-resolution 16 \
  --steps 10000 \
  --batch-size 2 \
  --eval-planes 8 \
  --orientation mixed \
  --slab-samples 5 \
  --slab-radius-cells 1.0 \
  --device mps
```

Use `--device cpu` when MPS is unavailable. PyTorch does not currently provide
3D `grid_sample` on MPS, so the implementation samples the small fixed latent
feature map on CPU and runs the trainable 2D CNN on MPS. The resolution divided
by the coarse resolution must be a power of two. A checkpoint is tied to that
fixed output resolution, which keeps this initial comparison simple and
reproducible.

For a short pipeline check using the staged section already generated by the
diagnostic:

```sh
python 2d-slice/train_plane_decoder.py \
  --artifact-dir 2d-slice/artifacts/turb-rot-staged/section-14 \
  --block-index 0 \
  --output-dir 2d-slice/artifacts/turb-rot-plane-slab-smoke \
  --resolution 32 \
  --coarse-resolution 8 \
  --hidden-channels 16 \
  --minimum-channels 8 \
  --coarse-blocks 1 \
  --slab-samples 5 \
  --slab-radius-cells 1.0 \
  --steps 10 \
  --batch-size 1 \
  --eval-planes 1 \
  --device cpu
```

The output includes `plane_decoder.pt`, `plane_decoder_metrics.json`, and one
NPZ example for each evaluation split. The primary acceptance metric is
`planeDecoderVsBase`; compare it with the point decoder's held-out RMSE of
approximately `0.667`.

### Current held-out results

The 10,000-step section `0-11`/`12-13`/`14-15` experiments produced:

| Decoder | RMSE vs final CAESAR | RMSE vs neural base | Slice latency |
| --- | ---: | ---: | ---: |
| Point-query MLP | 0.6694 | 0.6671* | 17.86 ms CPU |
| Single-plane CNN | 0.6116 | 0.6113 | 4.27 ms MPS |
| Five-plane slab CNN | 0.5971 | 0.5968 | 7.05 ms MPS |

`*` The staged point-versus-base diagnostic currently covers section 14; the
other reported errors cover held-out sections 14 and 15. CPU and MPS latency
numbers are not a device-matched speed comparison. The slab reduces
plane-versus-base RMSE by 2.4% relative to the single-plane CNN, while
increasing MPS latency by about 65%. Its unseen training-section RMSE is 0.5924,
close to its 0.5968 test RMSE, so cross-section generalization is not the
dominant remaining error.
