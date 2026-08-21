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
| Frozen early CAESAR tap + CNN | 0.3368 | 0.3363 | 8.75 ms CPU/MPS |
| Frozen late CAESAR tap + CNN | **0.1442** | **0.1434** | 20.88 ms CPU/MPS |

`*` The staged point-versus-base diagnostic currently covers section 14; the
other reported errors cover held-out sections 14 and 15. CPU and MPS latency
numbers are not a device-matched speed comparison. The slab reduces
plane-versus-base RMSE by 2.4% relative to the single-plane CNN, while
increasing MPS latency by about 65%. Its unseen training-section RMSE is 0.5924,
close to its 0.5968 test RMSE, so cross-section generalization is not the
dominant remaining error.

The contextual latency includes frozen CAESAR feature extraction on CPU and
the plane head on MPS. The late tap reduces plane-versus-base RMSE by 76.0%
relative to the slab and reaches 0.9902 correlation. Its end-to-end raw-data
RMSE is 0.1520. On this Mac, the one-block CPU diagnostic measured about 13.4
ms to reach the late tap and 40.5 ms to continue through the full neural base
decoder; the contextual slice path avoids that full-volume continuation.

## Frozen CAESAR contextual feature taps

CAESAR-V's pretrained neural decoder transforms each `[64,2,16,16]` latent
block through two 3D stages before its per-frame 2D super-resolution model:

```text
q_latent [64,2,16,16]
  -> early [48,4,32,32]
  -> late  [32,8,64,64]
  -> BCRN super-resolution
  -> base  [1,8,256,256]
```

The contextual experiments freeze these pretrained CAESAR stages, sample the
requested plane from either intermediate feature volume, and train only a new
2D plane head. The early tensor is 0.75 MiB per block and the late tensor is
4 MiB per block as float32. They are transient during ordinary training and
inference rather than cached for every block.

Verify both taps on one real block and save example feature artifacts:

```sh
PYTHONPATH=external/pyCAESAR \
python 2d-slice/diagnose_caesar_features.py \
  --artifact-dir 2d-slice/artifacts/turb-rot-staged/section-14 \
  --model CAESAR/data/caesar_v.pt \
  --output-dir 2d-slice/artifacts/turb-rot-caesar-feature-diagnostic \
  --block-index 0 \
  --device cpu
```

Continuing either saved tap through the remaining frozen layers should match
`caesar_base.npy` to floating-point precision. The current section-14 check
has approximately `4.2e-7` RMSE for both taps.

Train the early contextual head:

```sh
PYTHONPATH=external/pyCAESAR \
python 2d-slice/train_plane_decoder.py \
  --artifact-root 2d-slice/artifacts/turb-rot-staged-sections \
  --artifact-dir 2d-slice/artifacts/turb-rot-staged/section-14 \
  --output-dir 2d-slice/artifacts/turb-rot-context-early-generalization \
  --caesar-model CAESAR/data/caesar_v.pt \
  --feature-tap early \
  --coarse-resolution 32 \
  --slab-samples 1 \
  --train-sections 0-11 \
  --validation-sections 12-13 \
  --test-sections 14-15 \
  --expected-axis-semantic spatial_z \
  --resolution 128 \
  --steps 10000 \
  --batch-size 2 \
  --eval-planes 8 \
  --orientation mixed \
  --device mps \
  --context-device cpu
```

For the late head, change the output directory, tap, and coarse resolution:

```text
--output-dir 2d-slice/artifacts/turb-rot-context-late-generalization
--feature-tap late
--coarse-resolution 64
```

This installed PyTorch does not support `Conv3D` on MPS, so Mac training runs
the frozen 3D stages on CPU and the trainable 2D head on MPS. CUDA can use one
GPU for both. Reported contextual latency includes frozen feature extraction
and plane-head inference, but excludes entropy decoding and disk I/O. The saved
checkpoint contains only the plane head; its metadata records the required
CAESAR checkpoint and feature tap.

## Full-reconstruction and contextual-head ablation

The full-reconstruction baseline decodes the complete normalized CAESAR neural
base volume and then samples the identical deterministic planes used to
evaluate the learned heads:

```sh
PYTHONPATH=external/pyCAESAR \
python 2d-slice/benchmark_full_reconstruction.py \
  --artifact-root 2d-slice/artifacts/turb-rot-staged-sections \
  --artifact-dir 2d-slice/artifacts/turb-rot-staged/section-14 \
  --model CAESAR/data/caesar_v.pt \
  --output-dir 2d-slice/artifacts/turb-rot-full-reconstruction-baseline \
  --train-sections 0-11 \
  --validation-sections 12-13 \
  --test-sections 14-15 \
  --expected-axis-semantic spatial_z \
  --resolution 128 \
  --eval-planes 8 \
  --orientation mixed \
  --device cpu
```

Two alternative plane heads were tested against the late-only contextual
head:

- `--feature-tap late --head-initialization caesar` copies CAESAR's final 2D
  residual and BCRN super-resolution modules. A trainable identity-initialized
  adapter adds plane coordinates before all copied weights are fine-tuned. The
  stable run used `--learning-rate 1e-4 --gradient-clip-norm 1.0`; `1e-3`
  became non-finite between steps 1,500 and 2,000.
- `--feature-tap early-late` extracts both tensors in one frozen CAESAR pass,
  samples both on the same `64x64` plane grid, and concatenates 48 early and 32
  late channels before the custom CNN.

The held-out section 14-15 results are:

| Method | RMSE vs neural base | RMSE vs raw | Correlation vs raw | Slice latency |
| --- | ---: | ---: | ---: | ---: |
| Full base reconstruction + sampling | 0.0000015 | **0.0569** | **0.9984** | 66.20 ms cold; 8.50 ms amortized over 8 |
| Late-only custom head, 10k steps | **0.1434** | **0.1520** | **0.9890** | **20.88 ms** CPU/MPS |
| CAESAR-initialized head, 5k stable steps | 0.1528 | 0.1613 | 0.9873 | 26.96 ms CPU |
| Early + late fusion, 10k steps | 0.1539 | 0.1620 | 0.9871 | 27.23 ms CPU/MPS |

The full decoder reproduces the saved neural base to floating-point accuracy.
For a single cold slice, the late-only head is about 3.2 times faster than full
reconstruction on this Mac. If the reconstructed volume is reused, full
reconstruction becomes cheaper at approximately four or more planes and is
already 2.5 times faster per slice when amortized over eight planes. These are
mixed CPU/MPS measurements and must be repeated on CUDA before drawing a
platform-independent performance conclusion.

Neither alternative learned head improves aggregate accuracy. The copied
CAESAR head is parameter-efficient and starts close to the target, but its
axis-aligned 2D prior does not transfer perfectly to arbitrary planes. The
early features are redundant or distracting under simple concatenation. The
late-only custom head therefore remains the preferred learned method.

Generate the held-out image and difference comparison with:

```sh
python 2d-slice/compare_plane_ablation.py \
  --full 2d-slice/artifacts/turb-rot-full-reconstruction-baseline/test_full_reconstruction_example.npz \
  --late 2d-slice/artifacts/turb-rot-context-late-generalization/test_plane_decoder_example.npz \
  --caesar-initialized 2d-slice/artifacts/turb-rot-caesar-initialized-stable-generalization/test_plane_decoder_example.npz \
  --fusion 2d-slice/artifacts/turb-rot-context-fusion-generalization/test_plane_decoder_example.npz \
  --output 2d-slice/artifacts/turb-rot-context-ablation/decoder_ablation_comparison.png
```

## Calibrated per-pixel uncertainty

Each saved convolutional plane decoder can be kept frozen while a small CNN
learns a positive error scale from its final 2D feature map and scalar
prediction. The head is trained against the absolute residual relative to the
CAESAR neural base using Gaussian negative log likelihood. Split-conformal
calibration then turns that learned scale into empirical 90%, 95%, and 99%
error half-widths.

The data separation is deliberately stricter than the original scalar-decoder
split:

- sections `0-11`: already used to train the frozen scalar decoder
- section `12`: train only the uncertainty head
- section `13`: fit conformal scale factors
- sections `14-15`: final held-out uncertainty evaluation

Train and calibrate the late-context head with:

```sh
PYTHONPATH=external/pyCAESAR \
python 2d-slice/train_plane_uncertainty.py \
  --artifact-root 2d-slice/artifacts/turb-rot-staged-sections \
  --artifact-dir 2d-slice/artifacts/turb-rot-staged/section-14 \
  --decoder-checkpoint \
    2d-slice/artifacts/turb-rot-context-late-generalization/plane_decoder.pt \
  --output-dir 2d-slice/artifacts/turb-rot-context-late-uncertainty \
  --uncertainty-train-sections 12 \
  --calibration-sections 13 \
  --test-sections 14-15 \
  --expected-axis-semantic spatial_z \
  --steps 3000 \
  --batch-size 2 \
  --eval-planes 8 \
  --device mps \
  --context-device cpu
```

Use the same command with the single-plane, five-plane slab, and early-context
checkpoints and distinct output directories. `--context-device` is only needed
for contextual decoders. The outputs are:

- `plane_uncertainty.pt`: learned head and conformal scale factors
- `plane_uncertainty_metrics.json`: calibration, coverage, interval width,
  latency, orientation summaries, and uncertainty-guided refinement curves
- `test_plane_uncertainty_metrics.jsonl`: geometry and metrics for every plane
- `test_plane_uncertainty_example.npz`: scalar fields, observed error, predicted
  scale and half-widths, coverage mask, and plane geometry

The 3,000-step experiments produced these aggregate held-out results:

| Decoder | RMSE vs base | 95% coverage | Mean 95% half-width | Error/width correlation |
| --- | ---: | ---: | ---: | ---: |
| Single-plane CNN | 0.6113 | 95.98% | 1.354 | 0.045 |
| Five-plane slab CNN | 0.5968 | 96.04% | 1.329 | 0.053 |
| Frozen early CAESAR tap + CNN | 0.3363 | 95.90% | 0.754 | 0.072 |
| Frozen late CAESAR tap + CNN | **0.1434** | **95.42%** | **0.307** | **0.167** |

The intervals have the intended empirical marginal coverage on held-out
sections, but the pixelwise ranking signal is modest. For the late decoder,
replacing the 10% of pixels with the largest predicted 95% half-width by exact
CAESAR-base
values reduces overall RMSE from `0.1434` to `0.1301`; replacing a random 10%
gives `0.1361`. Thus the estimate is already useful for selective refinement,
while improved error localization remains a clear next target. The reported
coverage treats pixels as calibration samples even though nearby pixels are
spatially correlated, so it should be interpreted as held-out empirical
coverage rather than a strict independent-sample guarantee. It also does not
imply that every individual plane will cover exactly 95% of its pixels.

Generate the scalar, signed-error, and uncertainty comparison with:

```sh
python 2d-slice/plot_plane_uncertainty_comparison.py \
  --single 2d-slice/artifacts/turb-rot-plane-uncertainty/test_plane_uncertainty_example.npz \
  --slab 2d-slice/artifacts/turb-rot-plane-slab-uncertainty/test_plane_uncertainty_example.npz \
  --early 2d-slice/artifacts/turb-rot-context-early-uncertainty/test_plane_uncertainty_example.npz \
  --late 2d-slice/artifacts/turb-rot-context-late-uncertainty/test_plane_uncertainty_example.npz \
  --output 2d-slice/artifacts/turb-rot-uncertainty-comparison/decoder_comparison_uncertainty.png
```

The third row overlays the predicted 95% error half-width on the scalar slice.
All four learned methods share one uncertainty color scale. The raw column is
not applicable, and the CAESAR-base column is zero by definition because that
base is the operational reference.

## Uncertainty-guided CAESAR fallback

The first hybrid experiment treats the CAESAR neural base as exact and replaces
selected late-context predictions with CAESAR values. It supports four
selection policies over several refinement budgets:

- individual pixels with the largest predicted 95% half-width
- complete plane tiles ranked by their largest half-width
- random pixels
- an analysis-only oracle using the largest observed errors

The simulator uses the stored CAESAR base to establish attainable accuracy and
spatial cost. The exact path reuses the late `[32,8,64,64]` feature tensor that
was already computed for the direct slice. It identifies the depth frames
needed for trilinear interpolation, runs only those frames through CAESAR's
remaining frozen 2D decoder, samples the selected points, and replaces them in
the learned slice.

Run the held-out section 14-15 benchmark with:

```sh
PYTHONPATH=external/pyCAESAR \
python 2d-slice/benchmark_hybrid_refinement.py \
  --artifact-root 2d-slice/artifacts/turb-rot-staged-sections \
  --artifact-dir 2d-slice/artifacts/turb-rot-staged/section-14 \
  --decoder-checkpoint \
    2d-slice/artifacts/turb-rot-context-late-generalization/plane_decoder.pt \
  --uncertainty-checkpoint \
    2d-slice/artifacts/turb-rot-context-late-uncertainty/plane_uncertainty.pt \
  --output-dir 2d-slice/artifacts/turb-rot-hybrid-refinement \
  --test-sections 14-15 \
  --expected-axis-semantic spatial_z \
  --eval-planes 8 \
  --fractions 0,0.01,0.05,0.10,0.20,0.50 \
  --tile-size 16 \
  --exact-fallback-fraction 0.10 \
  --device mps \
  --context-device cpu
```

The exact on-demand values match the stored CAESAR base within `9.9e-6` on all
512 test planes. At the 10% budget:

| Policy | Actual refined pixels | RMSE vs base | Required frames | Added fallback time |
| --- | ---: | ---: | ---: | ---: |
| None | 0% | 0.1434 | 0/8 | 0 ms |
| Uncertain pixels | 10.0% | **0.1290** | 6.36/8 | 39.08 ms |
| Uncertain 16x16 tiles | 10.9% | 0.1317 | **5.20/8** | **34.54 ms** |
| Random pixels, simulated | 10.0% | 0.1361 | 6.35/8 | not timed |
| Actual-error oracle, simulated | 10.0% | 0.0958 | 6.35/8 | not available at inference |

The direct late slice plus uncertainty takes approximately `23.46 ms` in this
mixed CPU/MPS run. The complete measured hybrid times are therefore about
`62.54 ms` for pixel selection and `58.00 ms` for tile selection. Selecting
uncertain pixels improves accuracy, but an oblique plane spreads even a small
pixel mask across most depth frames. Tiles sacrifice some accuracy while
reducing disconnected regions and frame decoding.

One CAESAR architectural detail limits the next optimization. Its final BCRN
contains ESA and contrast-aware channel attention that compute statistics over
an entire 2D frame. Therefore, a complete depth frame is currently the smallest
unit that exactly reproduces full CAESAR output. Spatially cropped decoding
would be approximate unless those global operations are changed or their
full-frame statistics are cached.

Generate the example and accuracy/cost figures with:

```sh
python 2d-slice/plot_hybrid_refinement.py \
  --metrics 2d-slice/artifacts/turb-rot-hybrid-refinement/hybrid_refinement_metrics.json \
  --example 2d-slice/artifacts/turb-rot-hybrid-refinement/hybrid_refinement_example.npz \
  --output-dir 2d-slice/artifacts/turb-rot-hybrid-refinement
```

The output directory contains `hybrid_refinement_metrics.json`, an NPZ example,
`hybrid_refinement_comparison.png`, and `hybrid_refinement_curves.png`.
