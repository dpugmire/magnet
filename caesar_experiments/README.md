# CAESAR experiments

These MAGNET-owned scripts investigate visualization operations on CAESAR latent
representations. The CAESAR implementation is pinned separately as the
`external/CAESAR` Git submodule.

Initialize the dependency after cloning MAGNET:

```bash
git submodule update --init --recursive
```

CAESAR is not currently packaged for installation. Add the submodule root to
`PYTHONPATH` when running an experiment from the MAGNET repository root:

```bash
PYTHONPATH=external/CAESAR \
  python caesar_experiments/train_heatmap_renderer.py --help
```

The scripts expect dataset, checkpoint, and latent paths to be supplied on the
command line. Those large artifacts are intentionally excluded from Git.

- `train_heatmap_renderer.py` learns a direct scalar renderer conditioned on
  local features sampled from a CAESAR latent grid.
- `train_latent_inr_and_decode.py` fits an implicit function to a CAESAR latent
  tensor and evaluates the result through the CAESAR decoder.
- `npz_to_vtk.py` converts compatible NPZ volumes or slices to VTK image data.
