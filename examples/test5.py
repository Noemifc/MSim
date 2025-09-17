#!/usr/bin/env python3
"""
Test script to debug X-ray simulation physics issues.
Focus on simulation accuracy, not phantom complexity.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from msim.LSim_wrap import rotate_volume, build_quaternion
from msim.physics import projection

def create_simple_block(shape=(64, 64, 64)):
    """Create a simple rectangular block for testing."""
    volume = np.zeros(shape, dtype=np.int32)
    
    # Simple rectangular block in the center
    z_start, z_end = shape[0]//4, 3*shape[0]//4
    y_start, y_end = shape[1]//3, 2*shape[1]//3  
    x_start, x_end = shape[2]//3, 2*shape[2]//3
    
    volume[z_start:z_end, y_start:y_end, x_start:x_end] = 1
    
    print(f"Created block: {z_end-z_start}×{y_end-y_start}×{x_end-x_start} at center")
    return volume

def create_simple_cylinder(shape=(64, 64, 64)):
    """Create a simple cylinder along Z-axis for testing."""
    volume = np.zeros(shape, dtype=np.int32)
    
    center_y = shape[1] // 2
    center_x = shape[2] // 2
    radius = min(shape[1], shape[2]) // 4
    
    yy, xx = np.meshgrid(
        np.arange(shape[1]) - center_y,
        np.arange(shape[2]) - center_x,
        indexing='ij'
    )
    
    cylinder_mask = (yy**2 + xx**2) <= radius**2
    volume[:, cylinder_mask] = 1
    
    print(f"Created cylinder: radius={radius}, height={shape[0]}")
    return volume

def create_simple_sphere(shape=(64, 64, 64)):
    """Create a simple solid sphere for testing."""
    volume = np.zeros(shape, dtype=np.int32)
    
    center_z = shape[0] // 2
    center_y = shape[1] // 2
    center_x = shape[2] // 2
    radius = min(shape) // 3
    
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]) - center_z,
        np.arange(shape[1]) - center_y,
        np.arange(shape[2]) - center_x,
        indexing='ij'
    )
    
    sphere_mask = (zz**2 + yy**2 + xx**2) <= radius**2
    volume[sphere_mask] = 1
    
    print(f"Created sphere: radius={radius}")
    return volume

def setup_simple_materials():
    """Setup simple material properties for testing."""
    lookup = {
        "0": {"composition": {}, "density": 0.0},  # vacuum
        "1": {"composition": {"H": 2, "O": 1}, "density": 1.0}  # water
    }
    return lookup

def setup_test_config():
    """Setup simple simulation config."""
    config = {
        "ENERGY_KEV": 23.0,
        "DETECTOR_DIST": 0.1,  # Close detector to minimize artifacts
        "PAD": 20,  # Reduced padding
        "ENABLE_PHASE": False,   # Test with absorption only first
        "ENABLE_ABSORPTION": True,
        "ENABLE_SCATTER": False
    }
    return config

def test_no_rotation():
    """Test projection without any rotation."""
    print("\n=== TEST 1: NO ROTATION ===")
    
    # Test different geometries
    geometries = {
        "block": create_simple_block(),
        "cylinder": create_simple_cylinder(), 
        "sphere": create_simple_sphere()
    }
    
    lookup = setup_simple_materials()
    voxel_size = (1.0, 1.0, 1.0)
    config = setup_test_config()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, (name, volume) in enumerate(geometries.items()):
        # Show volume slice
        axes[0, i].imshow(volume[:, :, volume.shape[2]//2])
        axes[0, i].set_title(f"{name} - Volume YZ slice")
        
        # Project without rotation
        proj = projection(volume, lookup, voxel_size, config)
        axes[1, i].imshow(proj)
        axes[1, i].set_title(f"{name} - Projection")
        
        print(f"{name}: projection range {proj.min():.4f} to {proj.max():.4f}")
    
    plt.tight_layout()
    plt.savefig("test_no_rotation.png", dpi=150)
    print("Saved test_no_rotation.png")

def test_rotation_comparison():
    """Test the same object at different rotation angles."""
    print("\n=== TEST 2: ROTATION COMPARISON ===")
    
    # Use cylinder for clear rotation test
    volume = create_simple_cylinder(shape=(48, 48, 48))
    lookup = setup_simple_materials()
    voxel_size = (1.0, 1.0, 1.0)
    config = setup_test_config()
    
    angles = [0, 45, 90, 135, 180]
    projections = []
    
    fig, axes = plt.subplots(2, len(angles), figsize=(20, 8))
    
    for i, angle in enumerate(angles):
        print(f"Testing rotation: {angle}°")
        
        # Rotate volume
        quat = build_quaternion(0.0, np.deg2rad(angle))
        rotated = np.empty_like(volume, dtype=volume.dtype)
        volume_contiguous = np.ascontiguousarray(volume, dtype=np.float32)
        rotated_contiguous = np.ascontiguousarray(rotated, dtype=np.float32)
        
        rotate_volume(volume_contiguous, rotated_contiguous, quat)
        rotated_int = rotated_contiguous.astype(np.int32)
        
        # Show rotated volume
        axes[0, i].imshow(rotated_int[:, :, rotated_int.shape[2]//2])
        axes[0, i].set_title(f"Rotated {angle}° - YZ slice")
        
        # Project
        proj = projection(rotated_int, lookup, voxel_size, config)
        axes[1, i].imshow(proj)
        axes[1, i].set_title(f"Projection {angle}°")
        
        projections.append(proj)
        print(f"  Projection range: {proj.min():.4f} to {proj.max():.4f}")
    
    plt.tight_layout()
    plt.savefig("test_rotation_comparison.png", dpi=150)
    print("Saved test_rotation_comparison.png")
    
    return projections

def test_physics_effects():
    """Test different physics effects separately."""
    print("\n=== TEST 3: PHYSICS EFFECTS ===")
    
    volume = create_simple_sphere(shape=(48, 48, 48))
    lookup = setup_simple_materials()
    voxel_size = (1.0, 1.0, 1.0)
    
    # Test different physics combinations
    physics_configs = [
        {"name": "Absorption only", "ENABLE_PHASE": False, "ENABLE_ABSORPTION": True, "ENABLE_SCATTER": False},
        {"name": "Phase only", "ENABLE_PHASE": True, "ENABLE_ABSORPTION": False, "ENABLE_SCATTER": False},
        {"name": "All effects", "ENABLE_PHASE": True, "ENABLE_ABSORPTION": True, "ENABLE_SCATTER": True},
        {"name": "No effects", "ENABLE_PHASE": False, "ENABLE_ABSORPTION": False, "ENABLE_SCATTER": False}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, phys_config in enumerate(physics_configs):
        config = setup_test_config()
        config.update({k: v for k, v in phys_config.items() if k != "name"})
        
        proj = projection(volume, lookup, voxel_size, config)
        
        axes[i].imshow(proj)
        axes[i].set_title(f"{phys_config['name']}\nRange: {proj.min():.4f} to {proj.max():.4f}")
        
        print(f"{phys_config['name']}: range {proj.min():.4f} to {proj.max():.4f}")
    
    plt.tight_layout()
    plt.savefig("test_physics_effects.png", dpi=150)
    print("Saved test_physics_effects.png")

def test_material_effects():
    """Test different material contrasts."""
    print("\n=== TEST 4: MATERIAL EFFECTS ===")
    
    volume = create_simple_sphere(shape=(48, 48, 48))
    voxel_size = (1.0, 1.0, 1.0)
    config = setup_test_config()
    
    # Test different materials
    materials = [
        {"name": "Water", "composition": {"H": 2, "O": 1}, "density": 1.0},
        {"name": "Bone", "composition": {"Ca": 10, "P": 6, "O": 26, "H": 2}, "density": 2.0},
        {"name": "Aluminum", "composition": {"Al": 1}, "density": 2.7},
        {"name": "Iron", "composition": {"Fe": 1}, "density": 7.8}
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, material in enumerate(materials):
        lookup = {
            "0": {"composition": {}, "density": 0.0},
            "1": {"composition": material["composition"], "density": material["density"]}
        }
        
        proj = projection(volume, lookup, voxel_size, config)
        
        axes[i].imshow(proj)
        axes[i].set_title(f"{material['name']} (ρ={material['density']})\nRange: {proj.min():.4f} to {proj.max():.4f}")
        
        print(f"{material['name']}: range {proj.min():.4f} to {proj.max():.4f}")
    
    plt.tight_layout()
    plt.savefig("test_material_effects.png", dpi=150)
    print("Saved test_material_effects.png")

def test_analytical_comparison():
    """Compare simulation with analytical Beer's law."""
    print("\n=== TEST 5: ANALYTICAL COMPARISON ===")
    
    # Create uniform slab
    thickness_voxels = 20
    volume = np.zeros((thickness_voxels, 48, 48), dtype=np.int32)
    volume[:, 16:32, 16:32] = 1  # Uniform square slab
    
    lookup = setup_simple_materials()
    voxel_size = (1.0, 1.0, 1.0)  # 1 micron voxels
    config = setup_test_config()
    
    # Simulate
    proj = projection(volume, lookup, voxel_size, config)
    
    # Calculate analytical result using Beer's law
    import xraylib
    xraylib.XRayInit()
    
    formula = "H2O"
    density = 1.0  # g/cm³
    energy = config["ENERGY_KEV"]
    thickness_cm = thickness_voxels * voxel_size[0] * 1e-4  # Convert µm to cm
    
    mu_mass = xraylib.CS_Total_CP(formula, energy)  # cm²/g
    mu_linear = mu_mass * density  # cm⁻¹
    analytical_transmission = np.exp(-mu_linear * thickness_cm)
    
    # Compare
    simulated_transmission = proj[24, 24]  # Center of projection
    
    print(f"Analytical transmission: {analytical_transmission:.6f}")
    print(f"Simulated transmission: {simulated_transmission:.6f}")
    print(f"Relative error: {abs(simulated_transmission - analytical_transmission)/analytical_transmission * 100:.2f}%")
    
    # Show projection
    plt.figure(figsize=(8, 6))
    plt.imshow(proj)
    plt.title(f"Uniform slab projection\nAnalytical: {analytical_transmission:.4f}, Simulated: {simulated_transmission:.4f}")
    plt.colorbar()
    plt.savefig("test_analytical_comparison.png", dpi=150)
    print("Saved test_analytical_comparison.png")

def main():
    """Run all simulation physics tests."""
    print("X-RAY SIMULATION PHYSICS TEST SUITE")
    print("=" * 60)
    print("Testing simulation accuracy independent of phantom complexity")
    print("=" * 60)
    
    try:
        # Run all tests
        test_no_rotation()
        projections = test_rotation_comparison()
        test_physics_effects()
        test_material_effects()
        test_analytical_comparison()
        
        print("\n" + "=" * 60)
        print("SUMMARY:")
        print("✓ test_no_rotation.png - Check if basic projections look correct")
        print("✓ test_rotation_comparison.png - Check for rotation artifacts") 
        print("✓ test_physics_effects.png - Check physics effect isolation")
        print("✓ test_material_effects.png - Check material contrast")
        print("✓ test_analytical_comparison.png - Check quantitative accuracy")
        
        print("\nLook for:")
        print("- Ring artifacts in rotated projections (indicates simulation bug)")
        print("- Unrealistic contrast patterns")
        print("- Large discrepancy with analytical results")
        print("- Projection shapes that don't match expected geometry")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()