#!/usr/bin/env python3
"""
Golden-angle tomography senza z5py
Carica direttamente brain_map.zarr come array NumPy e usa JSON metadati
Visualizza proiezioni e volume con Napari
"""
import os
import numpy as np
import zarr
import json
from msim.simulator import XRayScanner

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

    # 3️⃣ Setup scanner
    scanner = XRayScanner(config_file)
    scanner.volume = volume_data
    scanner.lookup = lookup
    scanner.voxel_size = voxel_size

    # 4️⃣ Golden-angle scan
    golden_a = 180*(3 - np.sqrt(5)) / 2
    num_proj = 180
    theta_start = 0
    golden_angles_tomo = np.mod(theta_start + np.arange(num_proj) * golden_a, 180)
    print("10 golden angles:", golden_angles_tomo[:10])

    projections, dose_stats = scanner.tomography_scan(
        golden_angles_tomo,
        os.path.join(base_path, "golden_tomo_with_dose.h5"),
        calculate_dose=False
    )

    golden_angles_tomo_sorted = np.sort(golden_angles_tomo)
    print("10 sorted angles:", golden_angles_tomo_sorted[:10])

    projections_sorted, dose_stats_sorted = scanner.tomography_scan(
        golden_angles_tomo_sorted,
        os.path.join(base_path, "golden_angles_tomo_sorted.h5"),
        calculate_dose=False
    )

    print("Golden-angle scans complete")

    # ----------------------------
    # Visualizzazione interattiva con Napari
    # ----------------------------
    import napari
    viewer = napari.Viewer()
    viewer.add_image(np.array(projections_sorted), name='Golden Projections Sorted')
    viewer.add_image(volume_data, name='Volume Brain', scale=voxel_size)
    napari.run()

if __name__ == "__main__":
    main()
