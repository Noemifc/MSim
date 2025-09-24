#!/usr/bin/env python3
"""
golden-angle tomography example with phantom generation
"""
#--------------------------------------------
import numpy as np
import sys
import os 
#import z5py
#import json

# path MSIM per moduli import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from msim.simulator import XRayScanner
#--------------------------------------------
#generate phantom if doesn't exist ( sostituire )
phantom_file = "phantom_bone.zarr"
phantom_json = "phantom_bone.json"

if not os.path.exists(phantom_file):
    print("Phantom non trovato. Generazione automatica in corso...")
    from msim.generate_phantom import generate_phantom
    generate_phantom(
        "bone",
        shape=(64, 96, 96),
        voxel_size=(0.5, 0.5, 0.5)
    )
    print("Phantom generato correttamente!")

#--------------------------------------------
#setup scanner 
scanner = XRayScanner("enhanced_config.json")
scanner.load_volume("phantom_bone.zarr", "phantom_bone.json")

#--------------------------------------------
#golden angles array 

golden_a = 180*(3 - np.sqrt(5)) / 2  # deg
golden_a_rad = golden_a*np.pi / 180  # rad 
num_proj = 36     #select number of projections , int
theta_start = 30   # start angle deg

golden_angles_tomo = np.mod(theta_start + np.arange(num_proj) * golden_a, 180)
print(" 10 golden angles (float):", golden_angles_tomo[:10])

# scan golden-angle
projections, dose_stats = scanner.tomography_scan(
    golden_angles_tomo,
    "golden_tomo_with_dose.h5",
    calculate_dose=False
)

# golden-angle sorted 
golden_angles_tomo_sorted = np.sort(golden_angles_tomo) #golden angles sorted
print("10 angles sorted", golden_angles_tomo_sorted[:10])

# scan golden-angle sorted 
projections_sorted, dose_stats_sorted = scanner.tomography_scan(
    golden_angles_tomo_sorted,
    "golden_angles_tomo_sorted.h5",
    calculate_dose=False
)
print("Golden-angle scans complete")
