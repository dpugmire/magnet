# magnet

train_and_encode.py:
  Trains an AE on numpy data and saves out AE model
  Trains a neural implicit model (on AE output) and saves it.

eval_from_compressed.py:
   Reads in neural implicit model, compressed data.
   Evaluates the data (in latent space) with the neural implicit and plots.


Ideas:
Do block decomposed for AE and NeRF. What if I train a NeRF for each block?
Look at quantifying the errors.
Train Nerf on different iso contour value ranges.
How about an isocontour plot on a slice plane through a volume?


