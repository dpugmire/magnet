# CAESAR 2D Slice Extraction Plan

## Objective

Build and evaluate learned decoders that extract a scalar-valued 2D plane from
a CAESAR latent representation without first reconstructing and retaining the
entire dense 3D field.

The initial implementation will compare three representations:

1. A point-query implicit decoder.
2. A convolutional decoder operating on a coarse plane-aligned latent feature
   map.
3. A sparse signed anisotropic Gaussian representation of the plane.

A later extension will decode 3D Gaussians once and restrict them analytically
to arbitrary planes.

## Scope and assumptions

- Use `external/pyCAESAR` initially because it exposes CAESAR's quantized latent
  tensor as `q_latent`.
- Treat the compressed tensor's three varying axes as a spatial volume
  `(z, y, x)` for the first prototype. If an input uses the first axis as time,
  record that semantic explicitly and restrict initial tests to constant-time
  `x-y` planes.
- Operate on one CAESAR block or sample first. Generalize to multiple blocks only
  after coordinate and normalization conventions are verified.
- Use the full pyCAESAR decoder followed by trilinear plane sampling to generate
  training targets and the accuracy reference.
- Preserve CAESAR scale, offset, padding, block-index, and axis-order metadata.
- Keep datasets, checkpoints, rendered images, and benchmark output outside Git.
- Add no new dependencies for the first implementation. Use PyTorch and NumPy;
  use existing optional Gudhi support only for evaluation.

## Proposed directory structure

```text
2d-slice/
  PLAN.md
  README.md
  configs/
  slice_decoder/
    __init__.py
    caesar_adapter.py
    geometry.py
    datasets.py
    point_decoder.py
    plane_decoder.py
    gaussian_decoder.py
    rasterizer.py
    losses.py
    metrics.py
  train.py
  evaluate.py
  render.py
  tests/
    test_geometry.py
    test_gaussian_rasterizer.py
    test_caesar_adapter.py
```

The top-level directory retains the requested `2d-slice` name. Importable Python
code lives in `slice_decoder` because hyphens are not valid in Python package
names.

## Data and coordinate contract

Define a plane by:

```text
origin:  o  in R^3
basis:   eu, ev in R^3
bounds:  [umin, umax] x [vmin, vmax]
output:  H x W
```

Plane coordinates map to the volume through

$$
\mathbf{p}(u,v)=\mathbf{o}+u\mathbf{e}_u+v\mathbf{e}_v.
$$

`geometry.py` will own all transformations among:

- physical coordinates;
- normalized CAESAR volume coordinates in `[-1, 1]^3`;
- latent-grid coordinates;
- plane coordinates and output pixels.

The plane basis must be orthonormal within tolerance. Tests will cover axis
order, handedness, corners, boundary sampling, and `align_corners` behavior.

## Phase 1: CAESAR adapter and reference slices

Implement `caesar_adapter.py` to:

1. Load a pyCAESAR checkpoint and input sample.
2. Run compression and retain `q_latent` plus all reconstruction metadata.
3. Convert the latent from CAESAR's flattened batch/time form into a documented
   `[C, D, H, W]` tensor.
4. Decode the complete reference volume when constructing training or
   evaluation data.
5. Sample arbitrary reference planes with PyTorch `grid_sample`.
6. Save only small metadata manifests in Git; keep tensor artifacts ignored.

Initial correctness tests:

- An axis-aligned plane must agree with direct array indexing.
- Reversing a plane basis must produce the expected image flip.
- Identity latent decoding must reproduce pyCAESAR output within floating-point
  tolerance.
- Scale and offset must be applied exactly once.

## Phase 2: baseline slice decoders

### Point-query decoder

Generalize the existing heatmap renderer to evaluate

```text
(x, y, z, local latent feature) -> scalar
```

at requested plane pixels. Trilinearly sample the local latent feature at every
3D query point. This is the flexible baseline, although it requires one network
evaluation per output point.

### Plane-aligned convolutional decoder

1. Sample the CAESAR latent tensor on a coarse plane grid, producing
   `[C, h, w]` features.
2. Optionally sample a small slab of parallel latent planes and concatenate the
   features to provide through-plane context.
3. Use a 2D convolutional decoder to upsample the feature map to `[1, H, W]`.

This establishes the likely speed and accuracy baseline for a fixed output
resolution.

## Phase 3: signed Gaussian slice decoder

Represent the slice as

$$
\hat f(u,v)=b+\sum_{i=1}^{K} a_i
\exp\left[-\frac12(\boldsymbol\xi-\boldsymbol\mu_i)^T
\Sigma_i^{-1}(\boldsymbol\xi-\boldsymbol\mu_i)\right],
$$

where amplitudes `a_i` are signed. This is a Gaussian basis expansion rather
than a probability-density mixture.

### Parameter prediction

Use a spatially anchored decoder rather than an unconstrained unordered set for
the first implementation:

1. Construct the same coarse plane-aligned latent feature map used by the CNN
   baseline.
2. Apply convolutional prediction heads at each coarse cell.
3. Predict `M` Gaussian components per cell, giving `K = h * w * M` maximum
   components.
4. Predict for every component:
   - a center offset relative to its coarse cell;
   - two positive scales;
   - an orientation, or equivalently a Cholesky-parameterized covariance;
   - a signed amplitude;
   - an activation gate used for pruning.
5. Constrain centers with `tanh`, scales with bounded `softplus`, and gates with
   `sigmoid`.

Spatial anchoring avoids permutation matching, discourages duplicate
components, and preserves local correspondence with the CAESAR latent grid.

### Rasterizer

Start with a pure PyTorch reference rasterizer:

- evaluate components in chunks to limit memory;
- truncate support at `3 sigma` for benchmarking sparse evaluation;
- support arbitrary output resolution;
- preserve autograd for training;
- calculate analytic spatial gradients for gradient-loss experiments.

After correctness is established, add tile binning and component culling. A
custom CUDA, Metal, or Triton implementation requires separate justification
and approval.

### Initial model sweep

Evaluate effective component budgets near:

```text
K = 64, 128, 256, 512, 1024
```

For the anchored model, choose `(h, w, M)` combinations that produce comparable
budgets. Record both maximum `K` and effective `K` after gate pruning.

## Phase 4: losses and training protocol

Start with:

$$
L = L_{field} + \lambda_g L_{gradient} + \lambda_s L_{sparsity}
    + \lambda_r L_{regularization}.
$$

- `L_field`: L1 or Charbonnier field reconstruction, with MSE reported as a
  metric.
- `L_gradient`: error between finite-difference or analytic plane gradients.
- `L_sparsity`: penalty on Gaussian gates or amplitudes.
- `L_regularization`: bounds extreme scales and discourages redundant broad
  components.

Training sequence:

1. Axis-aligned planes only.
2. Random offsets along all three axes.
3. Randomly oriented planes fully contained in the volume.
4. Mixed resolutions and zoom levels.
5. Multiple volumes or blocks, split by volume rather than by plane to avoid
   leakage.

Persistence diagrams and contour metrics will initially be evaluation-only.
The existing Gudhi path is non-differentiable and must not be presented as a
training loss without a differentiable replacement.

## Phase 5: hybrid representation

If a pure Gaussian representation needs excessive `K`, add a hybrid decoder:

```text
slice = coarse grid background + sparse Gaussian residual
```

The coarse grid handles broad regions and discontinuities; Gaussians represent
localized residual detail. Compare it at the same output-byte and latency
budgets as the pure models.

## Phase 6: 3D Gaussian representation

Decode the CAESAR latent tensor into 3D signed anisotropic Gaussians:

$$
f(\mathbf{x})=\sum_i a_i
\exp\left[-\frac12(\mathbf{x}-\boldsymbol\mu_i)^T
Q_i(\mathbf{x}-\boldsymbol\mu_i)\right].
$$

For a plane `x = o + E xi`, restrict each component analytically:

$$
A_i=E^TQ_iE,
\qquad
\boldsymbol\xi_i=-A_i^{-1}E^TQ_i(\mathbf{o}-\boldsymbol\mu_i).
$$

The resulting plane covariance is `A_i^-1`; the amplitude is attenuated by the
component's distance from the plane. Test this analytic restriction against
direct evaluation of the same 3D Gaussian on plane sample points.

This phase is preferred for coherent extraction of many arbitrary slices, but
it follows the simpler plane-specific prototype so that its additional
complexity can be justified by measured results.

## Evaluation

Measure each method against full pyCAESAR decoding followed by reference plane
sampling.

### Accuracy

- MAE, RMSE, PSNR, and maximum absolute error;
- gradient magnitude and direction error;
- contour Chamfer or Hausdorff distance at selected isovalues;
- H0 and H1 persistence-diagram bottleneck distance;
- error as a function of plane orientation and zoom.

### Performance

- CAESAR latent preparation time;
- slice-parameter decoding time;
- rasterization or field-evaluation time;
- total latency with synchronized device timing;
- peak device memory;
- maximum and effective Gaussian count;
- bytes required for the intermediate slice representation.

Report cold-start and warmed-up timings separately. Compare methods at matched
accuracy and matched representation size, not only at their default settings.

## Command-line interfaces

Target commands:

```bash
PYTHONPATH=external/pyCAESAR \
  python 2d-slice/train.py --config 2d-slice/configs/gaussian.yaml

PYTHONPATH=external/pyCAESAR \
  python 2d-slice/evaluate.py --checkpoint CHECKPOINT --plane PLANE_JSON

PYTHONPATH=external/pyCAESAR \
  python 2d-slice/render.py --checkpoint CHECKPOINT --plane PLANE_JSON \
    --height 512 --width 512 --output slice.npy
```

Configuration files should record the CAESAR submodule commit, CAESAR model
checksum, dataset identifier, axis semantics, coordinate normalization, plane
sampling policy, and random seed.

## Milestones and acceptance criteria

### Milestone 1: geometry and reference pipeline

- Axis-aligned reference slices pass indexing tests.
- Arbitrary-plane sampling passes analytic test-volume checks.
- CAESAR metadata and latent shapes are recorded and reproducible.

### Milestone 2: baselines

- Point-query and convolutional decoders train end to end.
- Accuracy and synchronized latency are reported on the same held-out planes.

### Milestone 3: Gaussian prototype

- Signed Gaussian rasterization passes value and gradient tests.
- Fixed-budget models train for at least three `K` values.
- Effective component pruning and representation size are measured.

### Milestone 4: visualization evaluation

- Field, gradient, contour, and persistence metrics are generated together.
- Failure cases involving shocks, thin filaments, and dense turbulent detail are
  documented.

### Milestone 5: representation decision

- Select the convolutional, Gaussian, or hybrid design using matched-accuracy
  latency and storage results.
- Proceed to 3D Gaussians only if repeated arbitrary slicing or resolution
  independence provides a measurable advantage.

## Primary risks

- The CAESAR compressed axis may represent time rather than spatial depth.
- CAESAR normalization or block metadata may be omitted from extracted latent
  artifacts.
- A Gaussian basis may require too many components for turbulent or
  discontinuous fields.
- Naive rasterization can cost `O(KHW)` and hide any decoding advantage.
- A separate 2D Gaussian set per plane may be inconsistent across nearby
  slices.
- Low field RMSE does not guarantee accurate contours or preserved topology.

Each risk has a corresponding baseline, metadata check, benchmark, or later 3D
extension in the milestones above.
