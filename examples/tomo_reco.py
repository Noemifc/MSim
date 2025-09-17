#!/usr/bin/env python3
import tomopy
import h5py
import numpy as np

import cProfile
import pstats

# Input / output filenames
INPUT_FILE  = '../Output/Stained_lamino_PA.h5'
OUTPUT_FILE = '../Output/Stained_recon_PA.h5'

# start profiler
profiler = cProfile.Profile()
profiler.enable()

# 1) Load projections, flats, darks and angles from HDF5
with h5py.File(INPUT_FILE, 'r') as f:
    proj  = f['exchange/data'][:]         # shape (n_angles, n_slice, n_det)
    theta = f['angles'][:]        # shape (n_angles,)

# 2) Pre‐processing: normalize and remove any outlier stripes
#proj = tomopy.normalize(proj, flats, darks)
#proj = tomopy.remove_stripe_fw(proj)

theta = np.deg2rad(theta)

proj = tomopy.minus_log(proj)

# 3) Find rotation center (simple auto‐estimate)
#    You can also hard‐code center = proj.shape[2]//2
center = tomopy.find_center(proj, theta, init=proj.shape[2]//2)

# 4) Reconstruct: choose 'gridrec', 'astra', 'sirt', etc.
recon = tomopy.recon(proj,
                     theta,
                     center=center,
                     algorithm='gridrec')

# 5) (Optional) Apply circular mask to each slice
#recon = tomopy.circ_mask(recon, axis=0, ratio=0.95)

# 6) Save the 3D volume to a new HDF5
with h5py.File(OUTPUT_FILE, 'w') as f:
    f.create_dataset('reconstruction', data=recon, compression='gzip')
    f.create_dataset('theta',          data=theta)
    f.attrs['rotation_center'] = center

print(f"Done! Reconstruction written to {OUTPUT_FILE}")
stats = pstats.Stats(profiler).sort_stats('cumtime')
#stats.print_stats(20)
stats.dump_stats('tomo_profile.prof')
#logger.info("Wrote profile to simulate_profile.prof")