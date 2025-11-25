# magnet

train_and_encode.py:
  Trains an AE on numpy data and saves out AE model
  Trains a neural implicit model (on AE output) and saves it.

eval_from_compressed.py:
   Reads in neural implicit model, compressed data.
   Evaluates the data (in latent space) with the neural implicit and plots.
   
