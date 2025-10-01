#!/usr/bin/env python3
"""
golden-angle tomography example with phantom zip extraction
"""
#--------------------------------------------
import numpy as np
import glob
import os
import zipfile

os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.1/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

from msim.simulator import XRayScanner
#--------------------------------------------
# estrai file Zarr dal zip se esiste
zip_file = "phantom_brain.zarr.zip"
phantom_dir = "phantom_brain.zarr"

if os.path.exists(zip_file):
    print(f"Estrazione {zip_file}...")
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(".")
    print(f"Estrazione completata: {phantom_dir}")
else:
    print(f"Nessun file zip trovato, cercando cartella {phantom_dir}")

#--------------------------------------------
#--------------------------------------------
# trova files phantom json
json_files = glob.glob("*.json")
print("files Json found :", json_files)

# trova cartelle phantom zarr (non solo file!)
phantoms = [d for d in os.listdir(".") if d.endswith(".zarr") and os.path.isdir(d)]
print("phantom Zarr found:", phantoms)

phantom_file = phantom_dir if os.path.isdir(phantom_dir) else (phantoms[0] if phantoms else None)
phantom_json = "phantom_brain.json" if "phantom_brain.json" in json_files else (json_files[0] if json_files else None)

if phantom_file is None or phantom_json is None:
    raise FileNotFoundError("No phantom or JSON files found in the folder")

print("Usando phantom:", phantom_file)
print("Usando json metadata:", phantom_json)

#--------------------------------------------
# setup scanner 
config_file = "enhanced_config.json" if "enhanced_config.json" in json_files else json_files[0]
print("Usando configurazione:", config_file)

scanner = XRayScanner(config_file)
scanner.load_volume(phantom_file, phantom_json)
#--------------------------------------------
# golden angles array 
golden_a = 180*(3 - np.sqrt(5)) / 2  # deg
golden_a_rad = golden_a*np.pi / 180  # rad 
num_proj = 360    # select number of projections
theta_start = 30  # start angle deg

golden_angles_tomo = np.mod(theta_start + np.arange(num_proj) * golden_a, 180)
print(" 10 golden angles (float):", golden_angles_tomo[:10])

# scan golden-angle
projections, dose_stats = scanner.tomography_scan(
    golden_angles_tomo,
    "golden_tomo_with_dose.h5",  # salva i dati della proiez
    calculate_dose=False
)

# golden-angle sorted 
golden_angles_tomo_sorted = np.sort(golden_angles_tomo)  # golden angles sorted
print("10 angles sorted", golden_angles_tomo_sorted[:10])

# scan golden-angle sorted 
projections_sorted, dose_stats_sorted = scanner.tomography_scan(
    golden_angles_tomo_sorted,
    "golden_angles_tomo_sorted.h5",
    calculate_dose=False
)
print("Golden-angle scans complete")
