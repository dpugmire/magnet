# dump_adios_to_npy.py
import adios2
import numpy as np
import sys

bpfile = sys.argv[1]
out = sys.argv[2]
var = sys.argv[3]

slices = []
with adios2.Stream(bpfile, "r") as fh:
    for step in fh.steps():
        arr = fh.read(var)   # (Nz,Ny,Nx)
        # crop to 128x128, normalize, etc.
        slices.append(arr.astype("float32"))

np.save(out, np.stack(slices, axis=0))
