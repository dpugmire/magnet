# MAGNET

MAGNET is a research project exploring AI models as representations for
scientific visualization data and visualization algorithms. The central idea is
to treat a learned representation as a queryable object, rather than only as a
compressed array that must be fully reconstructed before visualization.

```text
simulation data
    -> learned representation (AE latent, CAESAR latent, or INR weights)
    -> continuous spatial and temporal queries
    -> visualization without first reconstructing the original mesh
```

The repository is an experimental workspace containing several related research
strands, historical snapshots, generated data, model checkpoints, and rendered
outputs.

## Research strands

### Scalar neural implicit representations

`scalar/train_and_encode.py` is the original TensorFlow pipeline for a scalar volume
with shape `(time, z, y, x) = (200, 8, 129, 128)`. It compares two
representations:

- A direct implicit model mapping `(x, y, t, z)` to a scalar value.
- An autoencoder-conditioned implicit model mapping `(x, y, latent)` to a
  scalar value.

The direct model uses gradient-based importance sampling to emphasize regions
with high spatial variation. The autoencoder operates on cropped 128x128
slices. Its latent grid is spatially averaged into an 8-value vector per slice,
which conditions a coordinate MLP. `scalar/eval_from_compressed.py` loads this
latent time series and implicit model to reconstruct fields and compare
isocontours.

Earlier scripts under `scalar/prototypes/` develop the same ideas on synthetic
or progressively more realistic data.

### Meshless rendering and contours

`meshless/meshless_render.py` is the clearest prototype of a visualization
algorithm operating directly on a learned field:

1. Train a convolutional autoencoder on synthetic 64x64 scalar fields.
2. Condition a Fourier-feature implicit function `f(z, x, y)` on one latent
   vector.
3. Query the function at arbitrary resolution for meshless rendering.
4. Use an adaptive quadtree and marching squares to extract isocontours directly
   from implicit-function evaluations.

The quadtree caches evaluations and refines cells near requested isovalues.
Topology variants add persistence-diagram comparisons and optional gradient or
total-variation regularization. These files are currently research demos rather
than a reusable contouring library.

### CAESAR latent representations

`external/pyCAESAR` pins the Python CAESAR fork currently used by MAGNET, while
`external/CAESAR` pins the official UFcompressor C++/LibTorch implementation.
CAESAR (*Conditional AutoEncoder with Super-resolution for Augmented Reduction*)
is a learned compressor for spatiotemporal scientific data. CAESAR-V combines
an autoencoder, entropy model, scale hyperprior, and super-resolution decoder.
CAESAR-D additionally uses conditional diffusion to interpolate missing latent
frames.

The MAGNET-specific scripts under `caesar_experiments/` use CAESAR's quantized
latent field in two ways:

- `train_latent_inr_and_decode.py` fits a 3D Fourier INR to a latent tensor,
  samples the INR back onto the latent grid, and sends the result through the
  CAESAR-V decoder.
- `train_heatmap_renderer.py` learns a direct renderer mapping
  `(x, y, local-time, local-latent-feature)` to a scalar. Local 64-component
  features are sampled from the 16x16 CAESAR latent grid, allowing a heatmap to
  be produced without executing the full CAESAR decoder.

The saved heatmap comparison against a CAESAR-decoded teacher reports MAE
`0.01927` and RMSE `0.0252`. Most visible error is concentrated around sharp,
fine-scale structures.

### Vector INRs and streamlines

`streamlines/` is a structured PyTorch implementation for time-varying 2D
velocity fields stored with ADIOS2. It learns

```text
(x, y, t, ensemble) -> (vx, vy)
```

using either a Fourier-feature MLP or SIREN. A learned embedding distinguishes
simulation ensembles. The current data contains three 256x256 Orszag-Tang MHD
runs using first-order HLL, MUSCL-HLL, and first-order Rusanov solvers.

Training samples random points across ensembles, timesteps, and grid locations.
An optional differentiable vorticity loss compares
`d(vy)/dx - d(vx)/dy` against finite-difference targets. Evaluation supports
velocity error, angular error, vorticity error, PNG output, and overlays of
ground-truth and predicted streamlines.

The existing quick-run result at step zero reports velocity MAE `0.242` and
angular error `20.7 degrees`. It demonstrates the end-to-end workflow but is not
yet a high-fidelity result.

### Topology-aware reconstruction experiments

`topology/` trains convolutional autoencoders on 2D MHD scalar snapshots from
ADIOS2. It computes Gudhi cubical-complex persistence diagrams, persistence
images, and per-dimension bottleneck distances for connected components (H0)
and loops (H1).

The current persistence-image loss is computed with NumPy and Gudhi after
detaching the reconstruction from PyTorch. It therefore has no gradient path to
the autoencoder. The saved `with_pd` and `no_pd` experiments have identical
reconstruction histories and evaluation metrics. In its current form, this code
provides topology monitoring and checkpoint scoring, not topology-aware
training.

## Repository organization

```text
scalar/                   Scalar AE + INR pipeline and earlier prototypes
meshless/                 Adaptive implicit rendering and contour experiments
caesar_experiments/       MAGNET experiments using CAESAR latent fields
external/pyCAESAR/        Python CAESAR fork with latent access (submodule)
external/CAESAR/          Official UFcompressor CAESAR (submodule)
streamlines/              ADIOS2 vector INR and streamline evaluation
topology/                 AE reconstruction and persistent-homology evaluation
tools/                    Small data-conversion utilities
saved_models/             TensorFlow models and compressed latent arrays
magnet*/                  Historical or nested MAGNET repository copies
CAESAR-OLD/               Earlier CAESAR snapshot
BACKUP/                   Historical workspace backup
autoresearch/             Unrelated nested autoresearch-mlx checkout
```

This is not yet a unified Python package. The TensorFlow scalar experiments,
PyTorch/ADIOS2 workflows, Gudhi topology code, and CAESAR dependencies use
separate environments and assumptions. There is currently no integrated test
suite.

Generated artifacts and historical working copies are excluded from the main
repository. The Python and official CAESAR implementations remain independent
repositories incorporated through pinned Git submodules.

## Interpretation and current limitations

- Reported latent-only compression ratios generally exclude model weights and
  normalization metadata. This is appropriate when a shared model is amortized
  over a large dataset, but not when measuring a single sample in isolation.
- The original scalar AE pipeline spatially averages its latent grid. The saved
  8-value vector is therefore a compact conditioning code, not a latent accepted
  directly by the standalone convolutional decoder.
- The CAESAR experiments save quantized float latents rather than entropy-coded
  byte streams. Downstream rendering scripts also need consistent frame grouping
  and per-block scale/offset metadata before results can be compared directly in
  physical units.
- `train_heatmap_renderer.py` accepts `--t-train`, but the current training loop
  samples pixels from all frames and does not use that argument.
- Adaptive contour refinement relies on sampled value ranges within each cell.
  Features that fall between those samples can still be missed.

## Existing research ideas

- Apply block decomposition to both autoencoders and neural implicit models.
- Investigate one INR per spatial block.
- Quantify reconstruction and visualization errors more systematically.
- Train representations for selected isovalue or contour ranges.
- Extract isocontours on slice planes through volumes.
- Make topology preservation differentiable so it can influence training.
- Evaluate visualization-space accuracy, such as contour geometry and streamline
  divergence, separately from field reconstruction error.
