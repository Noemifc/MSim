#!/usr/bin/env python3
"""
Golden-angle tomography example with automatic phantom generation.
"""

import numpy as np
import os
import sys

# Aggiungi path del modulo MSIM
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from msim.simulator import XRayScanner

# -------------------------------------------------------------------
# Generazione automatica del phantom se non esiste
# -------------------------------------------------------------------
phantom_file = "phantom_bone.zarr"
phantom_json = "phantom_bone.json"

if not os.path.exists(phantom_file):
    print("Phantom non trovato. Generazione automatica in corso...")
    from msim.generate_phantom import generate_phantom
    # Genera un phantom di tipo "bone"
    generate_phantom(
        "bone",
        shape=(64, 96, 96),        # dimensioni (Z, Y, X)
        voxel_size=(0.5, 0.5, 0.5) # voxel size in micron
    )
    print("Phantom generato correttamente!")

# -------------------------------------------------------------------
# Setup scanner
# -------------------------------------------------------------------
scanner = XRayScanner("enhanced_config.json")
scanner.load_volume(phantom_file, phantom_json)

# -------------------------------------------------------------------
# Golden-angle array
# -------------------------------------------------------------------
golden_a = 180*(3 - np.sqrt(5)) / 2  # deg
num_proj = 360     # numero proiezioni
theta_start = 30   # angolo iniziale deg

golden_angles_tomo = np.mod(theta_start + np.arange(num_proj) * golden_a, 180)
print("Primi 10 angoli (float):", golden_angles_tomo[:10])

# Scan golden-angle
projections, dose_stats = scanner.tomography_scan(
    golden_angles_tomo,
    "golden_tomo_with_dose.h5",
    calculate_dose=False
)

# Golden-angle ordinati
golden_angles_tomo_sorted = np.sort(golden_angles_tomo)
print("Primi 10 angoli ordinati:", golden_angles_tomo_sorted[:10])

# Scan golden-angle ordinati
projections_sorted, dose_stats_sorted = scanner.tomography_scan(
    golden_angles_tomo_sorted,
    "golden_angles_tomo_sorted.h5",
    calculate_dose=False
)

print("Golden-angle scans completati correttamente!")
