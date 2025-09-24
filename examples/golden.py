import numpy as np
import json
import os 
import z5py
import sys

from msim.simulator import XRayScanner, quick_tomography , analyze_dose_only
#export LD_LIBRARY_PATH=/local/cuda-12.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

#setup scanner 
scanner = XRayScanner("enhanced_config.json")
scanner.load_volume("phantom_bone.zarr", "phantom_bone.json")

#golden angles array 
golden_a = 180*(3 - np.sqrt(5)) / 2  # deg
golden_a_rad = golden_a*np.pi / 180  # rad 

num_proj = 360     #select number of projections , int
theta_start = 30   # start angle deg

golden_angles_tomo = np.mod(theta_start + np.arange(num_proj) * golden_a, 180)

print("Primi 10 angoli (float):", golden_angles_tomo[:10])

# scan golden-angle
projections, dose_stats = scanner.tomography_scan(
    golden_angles_tomo,
    "golden_tomo_with_dose.h5",
    calculate_dose=False
)

# golden-angle sorted 
golden_angles_tomo_sorted = np.sort(golden_angles_tomo) #golden angles sorted

print("Primi 10 angoli ordinati:", golden_angles_tomo_sorted[:10])

# scan golden-angle sorted 

projections_sorted, dose_stats_sorted = scanner.tomography_scan(
    golden_angles_tomo_sorted,
    "golden_angles_tomo_sorted.h5",
    calculate_dose=False
)


