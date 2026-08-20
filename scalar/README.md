# Scalar-field experiments

This directory contains the original TensorFlow experiments for representing a
time-varying scalar volume with autoencoders and neural implicit functions.

## Primary workflow

Run commands from the repository root:

```bash
python scalar/train_and_encode.py
python scalar/eval_from_compressed.py
```

The training script reads `data/xcompact-TG.npy` and writes models and compressed
latents under `saved_models/`. Both directories are intentionally excluded from
Git.

## Prototypes

`prototypes/` contains earlier experiments that developed the autoencoder,
coordinate-network, contour, and importance-sampling ideas used by the primary
workflow. They are retained as research history rather than presented as a
stable API.
