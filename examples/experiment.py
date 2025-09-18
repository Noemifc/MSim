#!/usr/bin/env python3
"""
Enhanced X-ray simulation example with photon statistics and dose calculation.
"""

import numpy as np
import json
from msim.simulator import XRayScanner, quick_tomography, analyze_dose_only

def create_enhanced_config():
    """Create configuration with photon statistics and dose parameters."""
    config = {
        "ENERGY_KEV": 23.0,
        "DETECTOR_DIST": 0.3,
        "DETECTOR_PIXEL_SIZE": 0.5e-6,  # 0.5 micron detector pixels
        "PAD": 50,
        "ENABLE_PHASE": True,
        "ENABLE_ABSORPTION": True,
        "ENABLE_SCATTER": True,
        
        # Photon statistics parameters
        "INCIDENT_PHOTONS": 1e6,       # High quality scan
        "DETECTOR_EFFICIENCY": 0.8,    # 80% quantum efficiency
        "DARK_CURRENT": 10,            # 10 dark counts per pixel
        "READOUT_NOISE": 5,            # 5 RMS readout noise
        "ENABLE_PHOTON_NOISE": True    # Include shot noise
    }
    
    with open("enhanced_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Created enhanced_config.json with photon statistics")
    return config

def example_basic_scan():
    """Basic scan without dose calculation."""
    print("\n=== BASIC SCAN EXAMPLE ===")
    
    # Quick tomography (default settings)
    projections, _ = quick_tomography(
        "phantom_bone.zarr", 
        "phantom_bone.json", 
        n_projections=90,
        output_file="basic_tomo.h5"
    )
    
    print(f"Basic tomography completed: {projections.shape}")

def example_dose_analysis():
    """Dose analysis without simulation."""
    print("\n=== DOSE ANALYSIS EXAMPLE ===")
    
    dose_map, dose_stats = analyze_dose_only(
        "phantom_bone.zarr",
        "phantom_bone.json", 
        "enhanced_config.json"
    )
    
    print(f"Dose map calculated: {dose_map.shape}")
    return dose_map, dose_stats

def example_full_scan_with_dose():
    """Complete scan with dose calculation."""
    print("\n=== FULL SCAN WITH DOSE ===")
    
    scanner = XRayScanner("enhanced_config.json")
    scanner.load_volume("phantom_bone.zarr", "phantom_bone.json")

    
    # Tomography with dose
    angles_tomo = np.linspace(0, 180, 360)   #np.linspace(start, stop, num 360 ) ->da 0 a 180° in 360 step
    projections, dose_stats = scanner.tomography_scan(
        angles_tomo, 
        "tomo_with_dose.h5",
        calculate_dose=False
    )
    
    
    print(f"Tomography with dose completed: {projections.shape}")
    return projections, dose_stats

def example_parameter_study():
    """Study effect of different photon counts on image quality and dose."""
    print("\n=== PARAMETER STUDY ===")
    
    photon_counts = [1e4, 1e5, 1e6, 1e7]  # Low to high photon flux
    
    for i, photon_count in enumerate(photon_counts):
        print(f"\nTesting with {photon_count:.0e} photons...")
        
        # Create config for this photon count
        config = create_enhanced_config()
        config["INCIDENT_PHOTONS"] = photon_count
        
        config_file = f"config_photons_{i}.json"
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        # Run scan
        scanner = XRayScanner(config_file)
        scanner.load_volume("phantom_bone.zarr", "phantom_bone.json")
        
        projections, dose_stats = scanner.tomography_scan(
            np.linspace(0, 180, 36),  # Fewer angles for speed
            f"tomo_photons_{i}.h5",
            calculate_dose=True
        )
        
        print(f"  Projection noise level: {np.std(projections):.4f}")

def example_compare_geometries():
    """Compare tomography vs laminography dose distributions."""
    print("\n=== GEOMETRY COMPARISON ===")
    
    scanner = XRayScanner("enhanced_config.json")
    scanner.load_volume("phantom_bone.zarr", "phantom_bone.json")
    
    angles = np.linspace(0, 180, 90)  # Same angles for fair comparison 
    
    # Tomography
    print("Running tomography...")
    tomo_proj, tomo_dose = scanner.tomography_scan(
        angles, "compare_tomo.h5", calculate_dose=True
    )
    
    # Laminography at 45°
    print("Running laminography...")
    lamino_proj, lamino_dose = scanner.laminography_scan(
        angles, tilt_deg=45, output_file="compare_lamino.h5", calculate_dose=True
    )
    
    print("Geometry comparison completed")
    return tomo_dose, lamino_dose

def example_material_specific_dose():
    """Analyze dose for specific materials."""
    print("\n=== MATERIAL-SPECIFIC DOSE ===")
    
    dose_map, dose_stats = analyze_dose_only(
        "phantom_bone.zarr",
        "phantom_bone.json",
        "enhanced_config.json"
    )
    
    # Find highest dose material
    max_dose_material = max(dose_stats.items(), key=lambda x: x[1]['max_dose_gy'])
    print(f"\nHighest dose material: {max_dose_material[1]['material_name']}")
    print(f"Max dose: {max_dose_material[1]['max_dose_gy']:.2e} Gy")
    
    # Find largest volume material
    max_volume_material = max(dose_stats.items(), key=lambda x: x[1]['total_volume_um3'])
    print(f"\nLargest volume material: {max_volume_material[1]['material_name']}")
    print(f"Volume: {max_volume_material[1]['total_volume_um3']:.1f} μm³")

def main():
    """Run all examples."""
    print("ENHANCED X-RAY SIMULATION EXAMPLES")
    print("=" * 60)
    print("Features: Photon statistics, dose calculation, material analysis")
    print("=" * 60)
    
    try:
        # Setup
        create_enhanced_config()
        
        # Generate test phantom if needed
        import os
        if not os.path.exists("phantom_dose_test.zarr"):
            print("Generating bone phantom...")
            from msim.generate_phantom import generate_phantom
            generate_phantom("bone", shape=(64, 96, 96), voxel_size=(0.5, 0.5, 0.5))
        
        # Run examples
        #example_basic_scan()
        #example_dose_analysis()
        example_full_scan_with_dose()
        #example_parameter_study()
        #example_compare_geometries()
        #example_material_specific_dose()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("\nOutput files created:")
        print("- basic_tomo.h5 (basic tomography)")
        print("- tomo_with_dose.h5 (tomography + dose)")
        print("- tomo_photons_*.h5 (parameter study)")
        print("- compare_*.h5 (geometry comparison)")
        print("- enhanced_config.json (configuration)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
