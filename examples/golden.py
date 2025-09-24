#!/usr/bin/env python3
"""
golden-angle tomography example 
"""
#--------------------------------------------
import numpy as np
#import sys
import os 
#import z5py
#import json
import glob

# path MSIM per moduli import
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from msim.simulator import XRayScanner
#--------------------------------------------
#trovare files phantom json
json_files = glob.glob("*.json")
print("File JSON trovati:", json_files)

#trovare files phantom zarr
phantoms = glob.glob("*.zarr")
print("Phantom Zarr trovati:", phantoms)

phantom_file = "phantom_bone.zarr" if "phantom_bone.zarr" in phantoms else (phantoms[0] if phantoms else None)
phantom_json = "phantom_bone.json" if "phantom_bone.json" in json_files else (json_files[0] if json_files else None)

if phantom_file is None or phantom_json is None:
    raise FileNotFoundError("Nessun phantom o file JSON trovato nella cartella!")

print("Usando phantom:", phantom_file)
print("Usando json metadata:", phantom_json)

#--------------------------------------------
#setup scanner 
config_file = "enhanced_config.json" if "enhanced_config.json" in json_files else json_files[0]
print("Usando configurazione:", config_file)

scanner = XRayScanner(config_file)
scanner.load_volume(phantom_file, phantom_json)


#--------------------------------------------
#golden angles array 

golden_a = 180*(3 - np.sqrt(5)) / 2  # deg
golden_a_rad = golden_a*np.pi / 180  # rad 
num_proj = 360    #select number of projections , int
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

