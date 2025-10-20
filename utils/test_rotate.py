import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../msim")))

from LSim_wrap import rotate_volume, build_quaternion


# Define the test parameters
nz, ny, nx = 100, 100, 100
theta = np.deg2rad(45)
q = build_quaternion(np.deg2rad(20), theta)  # tilt axis (alpha=20°), rotate 45°

# Supported dtypes with label and dtype object
test_dtypes = [
    ("uint8",   np.uint8),
    ("uint16",  np.uint16),
    ("float32", np.float32),
    ("float64", np.float64),
]

for label, dtype in test_dtypes:
    print(f"\n[TEST] dtype={label}")

    vol = np.zeros((nz, ny, nx), dtype=dtype)
    vol[nz//2, ny//2, nx//2] = np.iinfo(dtype).max if np.issubdtype(dtype, np.integer) else 1.0
    out = np.zeros_like(vol)

    # Ensure C-contiguous
    vol = np.ascontiguousarray(vol)
    out = np.ascontiguousarray(out)

    # Call wrapper
    rotate_volume(vol, out, q)

    print(f"[OK] Max value in rotated volume: {out.max()} (dtype={out.dtype})")
