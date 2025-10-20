import os
import ctypes
from ctypes import c_void_p, c_int, c_double
import numpy as np
from numpy import sin, cos

# Load the native library
BASE_DIR = os.path.dirname(__file__)
LIB_PATH = os.path.join(BASE_DIR, "cuda", "libmsim.so")
_l = ctypes.CDLL(LIB_PATH)

# C types
_l.rotate_volume.argtypes   = [c_void_p, c_void_p, c_int, c_int, c_int, c_double, c_double, c_double, c_double]
_l.rotate_volume.restype    = None

# Wrapper functions
def build_quaternion(alpha: float, theta: float):
    """
    Build a quaternion (w, x, y, z) for rotation by theta around
axis in the YZ plane tilted by alpha from +Y.
    """
    half = theta / 2.0
    w = cos(half)
    s = sin(half)
    ay = cos(alpha)
    az = sin(alpha)
    return w, 0.0, -ay * s, -az * s

def rotate_volume(vol_in: np.ndarray, vol_out: np.ndarray, quaternion: tuple):
    """
    Rotate a 3D volume using the type-dispatching CUDA kernel.
    Supports uint8, uint16, float32, float64.
    """
    dtype_map = {
        np.uint8:  0,
        np.uint16: 1,
        np.float32: 2,
        np.float64: 3
    }

    dtype_code = dtype_map.get(vol_in.dtype.type, None)
    if dtype_code is None:
        raise ValueError(f"Unsupported dtype: {vol_in.dtype}. Must be one of {list(dtype_map.keys())}")

    if not vol_in.flags['C_CONTIGUOUS']:
        vol_in = np.ascontiguousarray(vol_in)
    if not vol_out.flags['C_CONTIGUOUS']:
        vol_out = np.ascontiguousarray(vol_out)

    qw, qx, qy, qz = quaternion
    nz, ny, nx = vol_in.shape

    _l.rotate_volume(
        vol_in.ctypes.data_as(c_void_p),
        vol_out.ctypes.data_as(c_void_p),
        nx, ny, nz,
        float(qw), float(qx), float(qy), float(qz),
        int(dtype_code)
    )
