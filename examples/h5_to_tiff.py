import h5py
import tifffile

with h5py.File("golden_tomo_with_dose.h5", "r") as f:
    projections = f['projections'][:]

# as stack TIFF
tifffile.imwrite("golden_stack.tif", projections)
