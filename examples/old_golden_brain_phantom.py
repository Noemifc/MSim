#!/usr/bin/env python3
"""
Golden-angle tomography senza z5py
Carica brain_map.zarr come array NumPy e usa JSON metadati
Salva proiezione di test e stack di proiezioni
"""

import os
import numpy as np
import zarr
import json
import matplotlib.pyplot as plt
from msim.simulator import XRayScanner

#--------------------------------------------
def main():
    base_path = "/data2/Noemi/MSim/examples"
    zarr_file = os.path.join(base_path, "brain_map.zarr")
    json_file = os.path.join(base_path, "brain_lookup.json")
    config_file = os.path.join(base_path, "enhanced_config.json")

    # Controllo esistenza dei file
    for f in [zarr_file, json_file, config_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"{f} non trovato")

    # 1️⃣ Carica Zarr come array NumPy
    print(f"Caricando Zarr originale: {zarr_file}")
    z = zarr.open(zarr_file, mode='r')
    volume_data = z[:]
    print(f"Volume shape: {volume_data.shape}")

    # 2️⃣ Carica metadati JSON
    with open(json_file, 'r') as f:
        metadata = json.load(f)

    lookup = metadata["lookup"]
    voxel_size = metadata.get("voxel_size", [1.0, 1.0, 1.0])

    # 3️⃣ Proiezione di test (vista frontale)
    proj_test = volume_data.sum(axis=0)
    proj_norm = (proj_test - proj_test.min()) / (proj_test.max() - proj_test.min() + 1e-8)
    test_png = os.path.join(base_path, "proiezione_test.png")
    plt.imsave(test_png, proj_norm, cmap='gray')
    print(f"Proiezione di test salvata come: {test_png}")

    # 4️⃣ Setup scanner
    scanner = XRayScanner(config_file)
    scanner.volume = volume_data
    scanner.lookup = lookup
    scanner.voxel_size = voxel_size

    # 5️⃣ Golden-angle scan
    golden_a = 180*(3 - np.sqrt(5)) / 2  # angolo aureo
    num_proj = 10
    theta_start = 30
    golden_angles_tomo = np.mod(theta_start + np.arange(num_proj) * golden_a, 180)
    print("10 golden angles:", golden_angles_tomo[:10])

    # Scansione golden-angle
    out_file = os.path.join(base_path, "golden_tomo_with_dose.h5")
    projections, dose_stats = scanner.tomography_scan(
        golden_angles_tomo,
        out_file,
        calculate_dose=False
    )
    print(f"Golden-angle scan salvato in: {out_file}")

    # Golden-angle ordinato
    golden_angles_sorted = np.sort(golden_angles_tomo)
    projections_sorted, dose_stats_sorted = scanner.tomography_scan(
        golden_angles_sorted,
        os.path.join(base_path, "golden_angles_tomo_sorted.h5"),
        calculate_dose=False
    )

    # 6️⃣ Salvataggio stack di proiezioni per visualizzazione remota
    stack_file = os.path.join(base_path, "stack_projections_sorted.npy")
    np.save(stack_file, projections_sorted)
    print(f"Stack di proiezioni salvato come: {stack_file}")

    print("Golden-angle scans complete")

#--------------------------------------------
if __name__ == "__main__":
    main()
