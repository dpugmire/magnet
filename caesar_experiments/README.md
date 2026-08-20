# CAESAR experiments

These MAGNET-owned scripts investigate visualization operations on CAESAR latent
representations. Two CAESAR implementations are pinned as Git submodules:

- `external/pyCAESAR` is the Python fork currently used by these experiments. It
  exposes the quantized latent tensor required by the MAGNET renderers.
- `external/CAESAR` is the official UFcompressor C++/LibTorch implementation and
  is the target for future integration work.

Initialize the dependency after cloning MAGNET:

```bash
git submodule update --init --recursive
```

The Python fork is not currently packaged for installation. Add its submodule
root to `PYTHONPATH` when running an experiment from the MAGNET repository root:

```bash
PYTHONPATH=external/pyCAESAR \
  python caesar_experiments/train_heatmap_renderer.py --help
```

The internal Python package remains named `CAESAR`, so experiment imports such
as `from CAESAR.compressor import CAESAR` do not change.

The scripts expect dataset, checkpoint, and latent paths to be supplied on the
command line. Those large artifacts are intentionally excluded from Git.

- `train_heatmap_renderer.py` learns a direct scalar renderer conditioned on
  local features sampled from a CAESAR latent grid.
- `train_latent_inr_and_decode.py` fits an implicit function to a CAESAR latent
  tensor and evaluates the result through the CAESAR decoder.
- `npz_to_vtk.py` converts compatible NPZ volumes or slices to VTK image data.
