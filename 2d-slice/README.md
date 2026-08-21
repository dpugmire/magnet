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
- `caesar_reference.npy`: the ordinary full CAESAR-V reconstruction
- `q_latent.npy`: quantized latents in canonical `[C,D_latent,H_latent,W_latent]`
- `manifest.json`: source selection, model hash, parameters, shapes, and semantics

Generated files under `2d-slice/artifacts/` are ignored by Git.

## Point-query baseline

The first direct decoder samples a local feature vector from `q_latent` using
trilinear interpolation, concatenates normalized `(x,y,z)` coordinates and
Fourier features, and maps that vector to one normalized scalar with an MLP.
CAESAR's block scale and offset convert the prediction back to field units.

Only `q_latent`, scale, and offset are decoder inputs. CAESAR's error-bounded
postprocessing residual is not supplied to this baseline, although the full
CAESAR reconstruction remains its training target. The reported slice-decoder
error therefore includes any correction information that cannot be inferred
from `q_latent`.

This is deliberately a one-block overfit experiment. It proves that the direct
query path and error accounting work before adding multi-volume train/validation
splits. The target is the full CAESAR reconstruction. When the manifest's source
archive is available, evaluation memory-maps ground truth directly from that raw
archive; otherwise it falls back to the pipeline's pre-compression copy.

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

- `point_decoder.pt`: model weights, architecture, normalization, and block metadata
- `point_decoder_metrics.json`: compression, slice-decoder, and end-to-end errors
- `point_decoder_example.npz`: one held-out raw/CAESAR/direct plane triplet

The three error groups answer different questions:

- `compression`: raw source plane versus full CAESAR reconstruction
- `sliceDecoder`: full CAESAR reconstruction versus direct point-query result
- `endToEnd`: raw source plane versus direct point-query result

The recorded latency covers only direct slice inference. It intentionally does
not include CAESAR compression, artifact loading, or disk I/O.
