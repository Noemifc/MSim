#!/usr/bin/env python3
"""
Enhanced X-ray simulation examples with photon statistics, dose calculation,
parameter study, geometry comparison, golden-angle scans, and material-specific dose.
Usa brain_map.h5 come riferimento e brain_map.zarr come volume reale.
"""

import numpy as np
import json
import os
import sys

sys.path.append(r'C:\Users\noemi\Desktop\Noe')  # cartella che contiene msim

from msim.simulator import XRayScanner, quick_tomography, analyze_dose_only

# --- Percorsi dei file ---
H5_SAMPLE = r"C:\Users\noemi\Desktop\Noe\examples\brain_map.h5"
ZARR_FILE = r"C:\Users\noemi\Desktop\Noe\examples\brain_map.zarr"
JSON_FILE = r"C:\Users\noemi\Desktop\Noe\examples\brain_lookup.json"
CONFIG_FILE = r"C:\Users\noemi\Desktop\Noe\examples\enhanced_config.json"

# --- Configurazione ---
def create_enhanced_config():
    config = {
        "ENERGY_KEV": 23.0,
        "DETECTOR_DIST": 0.3,
        "DETECTOR_PIXEL_SIZE": 0.5e-6,
        "PAD": 50,
        "ENABLE_PHASE": True,
        "ENABLE_ABSORPTION": True,
        "ENABLE_SCATTER": True,
        "INCIDENT_PHOTONS": 1e6,
        "DETECTOR_EFFICIENCY": 0.8,
        "DARK_CURRENT": 10,
        "READOUT_NOISE": 5,
        "ENABLE_PHOTON_NOISE": True
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Created {CONFIG_FILE} with photon statistics")
    return config

# --- Funzioni principali ---
def example_basic_scan(volume_file, json_file=None):
    print("\n=== BASIC SCAN EXAMPLE ===")
    projections, _ = quick_tomography(volume_file, json_file, n_projections=360, output_file="basic_tomo_brain.h5")
    print(f"Basic tomography completed: {projections.shape}")

def example_dose_analysis(volume_file, json_file=None):
    print("\n=== DOSE ANALYSIS EXAMPLE ===")
    dose_map, dose_stats = analyze_dose_only(volume_file, json_file, CONFIG_FILE)
    print(f"Dose map calculated: {dose_map.shape}")
    return dose_map, dose_stats

def example_full_scan_with_dose(volume_file, json_file=None):
    print("\n=== FULL SCAN WITH DOSE ===")
    scanner = XRayScanner(CONFIG_FILE)
    scanner.load_volume(volume_file, json_file)
    angles_tomo = np.linspace(0, 180, 360)
    projections, dose_stats = scanner.tomography_scan(
        angles_tomo, "tomo_with_dose_brain.h5", calculate_dose=False
    )
    print(f"Tomography with dose completed: {projections.shape}")
    return projections, dose_stats

def example_parameter_study(volume_file, json_file=None):
    print("\n=== PARAMETER STUDY ===")
    photon_counts = [1e4, 1e5, 1e6, 1e7]
    for i, photon_count in enumerate(photon_counts):
        print(f"\nTesting with {photon_count:.0e} photons...")
        config = create_enhanced_config()
        config["INCIDENT_PHOTONS"] = photon_count
        config_file = f"config_photons_{i}.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        scanner = XRayScanner(config_file)
        scanner.load_volume(volume_file, json_file)
        projections, dose_stats = scanner.tomography_scan(
            np.linspace(0, 180, 36),
            f"tomo_photons_{i}.h5",
            calculate_dose=True
        )
        print(f"  Projection noise level: {np.std(projections):.4f}")

def example_compare_geometries(volume_file, json_file=None):
    print("\n=== GEOMETRY COMPARISON ===")
    scanner = XRayScanner(CONFIG_FILE)
    scanner.load_volume(volume_file, json_file)
    angles = np.linspace(0, 180, 90)
    print("Running tomography...")
    tomo_proj, tomo_dose = scanner.tomography_scan(
        angles, "compare_tomo.h5", calculate_dose=True
    )
    print("Running laminography...")
    lamino_proj, lamino_dose = scanner.laminography_scan(
        angles, tilt_deg=45, output_file="compare_lamino.h5", calculate_dose=True
    )
    print("Geometry comparison completed")

def golden_scan(volume_file, json_file=None, num_proj=100, theta_start=0):
    print("\n=== GOLDEN-ANGLE SCAN ===")
    end_angle = 360.
    golden_a = end_angle * (3 - np.sqrt(5)) / 2
    g_angles = np.mod(theta_start + np.arange(num_proj) * golden_a, end_angle)
    scanner = XRayScanner(CONFIG_FILE)
    scanner.load_volume(volume_file, json_file)
    projections, dose_stats = scanner.tomography_scan(
        g_angles, "golden_tomo_with_dose.h5", calculate_dose=False
    )
    return projections, dose_stats

def golden_scan_interl(volume_file, json_file=None, num_proj_interl=100, theta_start_interl=None):
    print("\n=== INTERLACED GOLDEN-ANGLE SCAN ===")
    end_angle = 360.
    golden_a = end_angle * (3 - np.sqrt(5)) / 2
    if theta_start_interl is None:
        theta_start_interl = np.array([0.])
    theta_start_interl = np.array(theta_start_interl)
    golden_angles = np.mod(
        theta_start_interl[:, None] + np.arange(num_proj_interl) * golden_a,
        end_angle
    ).flatten()
    golden_angles = np.unique(np.sort(golden_angles))
    scanner = XRayScanner(CONFIG_FILE)
    scanner.load_volume(volume_file, json_file)
    projections, dose_stats = scanner.tomography_scan(
        golden_angles, "golden_tomo_interlaced_with_dose.h5", calculate_dose=False
    )
    return projections, dose_stats

def example_material_specific_dose(volume_file, json_file=None):
    print("\n=== MATERIAL-SPECIFIC DOSE ===")
    dose_map, dose_stats = analyze_dose_only(volume_file, json_file, CONFIG_FILE)
    max_dose_material = max(dose_stats.items(), key=lambda x: x[1]['max_dose_gy'])
    print(f"\nHighest dose material: {max_dose_material[1]['material_name']}")
    print(f"Max dose: {max_dose_material[1]['max_dose_gy']:.2e} Gy")
    max_volume_material = max(dose_stats.items(), key=lambda x: x[1]['total_volume_um3'])
    print(f"Largest volume material: {max_volume_material[1]['material_name']}")
    print(f"Volume: {max_volume_material[1]['total_volume_um3']:.1f} μm³")

# --- Main ---
def main():
    print("ENHANCED X-RAY SIMULATION EXAMPLES")
    print("=" * 60)
    print("Features: Photon statistics, dose calculation, parameter study, geometry comparison, golden-angle scans")
    print("=" * 60)

    try:
        create_enhanced_config()
        print(f"Using sample H5 file: {H5_SAMPLE}")
        print(f"Using ZARR file: {ZARR_FILE}")
        print(f"Using JSON metadata: {JSON_FILE}")

        # Esegui tutti gli esempi
        example_basic_scan(ZARR_FILE, JSON_FILE)
        example_dose_analysis(ZARR_FILE, JSON_FILE)
        example_full_scan_with_dose(ZARR_FILE, JSON_FILE)
        example_parameter_study(ZARR_FILE, JSON_FILE)
        example_compare_geometries(ZARR_FILE, JSON_FILE)
        golden_scan(ZARR_FILE, JSON_FILE)
        golden_scan_interl(ZARR_FILE, JSON_FILE)
        example_material_specific_dose(ZARR_FILE, JSON_FILE)

        print("\nAll examples completed successfully!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
