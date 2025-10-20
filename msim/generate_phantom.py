#!/usr/bin/env python3
"""
Generate test phantoms in Zarr format for X-ray simulation with dose calculation.
"""

import numpy as np
import json
import os
import shutil
import z5py

def save_multiscale_zarr(
    data,
    codes,
    out_dir,
    voxel_size=(0.25, 0.25, 0.1),
    n_scales=3,
    base_chunk=(64, 64, 64),
    logger=None
):
    """
    Save a 3D volume as a multiscale Zarr file with Neuroglancer-compatible metadata.
    """
    # Delete existing output if present
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    # Open Zarr root
    f = z5py.File(out_dir, use_zarr_format=True)
    curr = data.copy()
    datasets = []
    for lvl in range(n_scales):
        path = str(lvl)
        if path in f:
            del f[path]
        chunks = tuple(min(c, s) for c, s in zip(base_chunk, curr.shape))
        f.create_dataset(
            name=path,
            data=curr,
            chunks=chunks,
            compression='raw'
        )
        scale = 2 ** lvl
        datasets.append({
            "path": path,
            "coordinateTransformations": [
                {"type": "scale",       "scale": [scale] * 3},
                {"type": "translation", "translation": [scale / 2 - 0.5] * 3}
            ]
        })
        curr = curr[::2, ::2, ::2]
    # Add metadata
    lookup = {v: {'alias': k} for k, v in codes.items()}
    f.attrs['lookup'] = lookup
    f.attrs['voxel_size'] = voxel_size
    multiscale_meta = {
        "version": "0.4",
        "axes": [
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"}
        ],
        "datasets": datasets,
        "type": "image",
        "metadata": {
            "voxel_size": voxel_size
        }
    }
    f.attrs['multiscales'] = [multiscale_meta]
    # Also write JSON file for HTTP-serving compatibility
    with open(os.path.join(out_dir, "multiscale.json"), "w") as fjson:
        json.dump([multiscale_meta], fjson, indent=2)
    if logger:
        logger.info(f"Saved Neuroglancer-ready multiscale Zarr → {out_dir}")

def create_metadata_json(lookup_dict, voxel_size, output_path):
    """
    Create a separate JSON metadata file for simulation compatibility.
    """
    metadata = {
        "voxel_size": list(voxel_size),
        "lookup": lookup_dict
    }
    
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def create_sphere_phantom(shape=(128, 128, 128), voxel_size=(0.5, 0.5, 0.5)):
    """Create a simple sphere phantom - properly centered."""
    nz, ny, nx = shape
    volume = np.zeros(shape, dtype=np.int32)
    
    # Center coordinates - use exact geometric center
    center_z = (nz - 1) / 2.0
    center_y = (ny - 1) / 2.0
    center_x = (nx - 1) / 2.0
    
    # Create coordinate grids
    zz, yy, xx = np.meshgrid(
        np.arange(nz) - center_z,
        np.arange(ny) - center_y,
        np.arange(nx) - center_x,
        indexing='ij'
    )
    
    # Outer sphere - calcium carbonate
    outer_radius = min(shape) // 4
    outer_mask = (zz**2 + yy**2 + xx**2) <= outer_radius**2
    volume[outer_mask] = 1
    
    # Inner sphere - hydroxyapatite (bone)
    inner_radius = outer_radius // 2
    inner_mask = (zz**2 + yy**2 + xx**2) <= inner_radius**2
    volume[inner_mask] = 2
    
    return volume

def create_cylinder_phantom(shape=(128, 128, 128), voxel_size=(0.5, 0.5, 0.5)):
    """Create a phantom with cylindrical features."""
    nz, ny, nx = shape
    volume = np.zeros(shape, dtype=np.int32)
    
    center_y = (ny - 1) / 2.0
    center_x = (nx - 1) / 2.0
    
    # Create coordinate grids
    yy, xx = np.meshgrid(
        np.arange(ny) - center_y,
        np.arange(nx) - center_x,
        indexing='ij'
    )
    
    # Main cylinder - water
    main_radius = min(ny, nx) // 3
    main_mask = (yy**2 + xx**2) <= main_radius**2
    volume[:, main_mask] = 1
    
    # Inner cylinder - bone
    inner_radius = main_radius // 2
    inner_mask = (yy**2 + xx**2) <= inner_radius**2
    volume[:, inner_mask] = 2
    
    # Small cylinders around the edge
    for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
        offset_y = int(main_radius * 0.7 * np.cos(angle))
        offset_x = int(main_radius * 0.7 * np.sin(angle))
        
        small_yy = yy - offset_y
        small_xx = xx - offset_x
        small_radius = main_radius // 8
        small_mask = (small_yy**2 + small_xx**2) <= small_radius**2
        volume[:, small_mask] = 3
    
    return volume

def create_complex_bone_phantom(shape=(128, 128, 128), voxel_size=(0.5, 0.5, 0.5)):
    """Create a complex bone-like phantom with multiple materials and structures."""
    nz, ny, nx = shape
    volume = np.zeros(shape, dtype=np.int32)
    
    # Center coordinates - use exact geometric center
    center_z = (nz - 1) / 2.0
    center_y = (ny - 1) / 2.0
    center_x = (nx - 1) / 2.0
    
    # Create coordinate grids
    zz, yy, xx = np.meshgrid(
        np.arange(nz) - center_z,
        np.arange(ny) - center_y,
        np.arange(nx) - center_x,
        indexing='ij'
    )
    
    # 1. Background soft tissue (water-like)
    main_radius = min(shape) // 2.2
    tissue_mask = (zz**2 + yy**2 + xx**2) <= main_radius**2
    volume[tissue_mask] = 1  # soft tissue
    
    # 2. Central bone cylinder (cortical bone)
    bone_radius = main_radius * 0.4
    bone_mask = (yy**2 + xx**2) <= bone_radius**2
    volume[bone_mask] = 2  # cortical bone
    
    # 3. Bone marrow cavity (fat-like)
    marrow_radius = bone_radius * 0.6
    marrow_mask = (yy**2 + xx**2) <= marrow_radius**2
    volume[marrow_mask] = 3  # bone marrow
    
    # 4. Dense inclusions (calcifications)
    np.random.seed(42)  # Reproducible
    for i in range(8):
        # Random positions around the bone
        angle = i * 2 * np.pi / 8
        offset_y = bone_radius * 1.2 * np.cos(angle)
        offset_x = bone_radius * 1.2 * np.sin(angle)
        
        calc_yy = yy - offset_y
        calc_xx = xx - offset_x
        calc_radius = 3 + i * 0.5  # Varying sizes
        
        calc_mask = (calc_yy**2 + calc_xx**2) <= calc_radius**2
        volume[calc_mask] = 4  # calcifications
    
    # 5. Trabecular bone network (sparse bone)
    trabecular_mask = (yy**2 + xx**2) <= (bone_radius * 1.8)**2
    trabecular_mask &= (yy**2 + xx**2) > (bone_radius * 1.1)**2
    
    # Create trabecular pattern
    spacing = 8
    for y_pos in range(-ny//2, ny//2, spacing):
        for x_pos in range(-nx//2, nx//2, spacing):
            if 0 <= y_pos + ny//2 < ny and 0 <= x_pos + nx//2 < nx:
                if trabecular_mask[nz//2, y_pos + ny//2, x_pos + nx//2]:
                    trab_yy = yy - y_pos
                    trab_xx = xx - x_pos
                    trab_radius = 2
                    trab_sphere = (trab_yy**2 + trab_xx**2 + zz**2) <= trab_radius**2
                    volume[trab_sphere & trabecular_mask] = 5  # trabecular bone
    
    # 6. Air bubbles (contrast features)
    for i in range(5):
        bubble_z = -main_radius * 0.5 + i * main_radius * 0.25
        bubble_y = main_radius * 0.3 * np.cos(i * 1.2)
        bubble_x = main_radius * 0.3 * np.sin(i * 1.2)
        
        bubble_zz = zz - bubble_z
        bubble_yy = yy - bubble_y
        bubble_xx = xx - bubble_x
        bubble_radius = 2 + i * 0.5
        
        bubble_mask = (bubble_zz**2 + bubble_yy**2 + bubble_xx**2) <= bubble_radius**2
        volume[bubble_mask] = 0  # air/vacuum
    
    # 7. Metal implant (titanium screw)
    screw_length = nz * 0.3
    screw_radius = 2
    screw_z_start = center_z - screw_length/2
    screw_z_end = center_z + screw_length/2
    
    screw_y_offset = bone_radius * 0.3
    screw_x_offset = 0
    
    for z_idx in range(int(screw_z_start), int(screw_z_end)):
        if 0 <= z_idx < nz:
            screw_yy = yy[z_idx] - screw_y_offset
            screw_xx = xx[z_idx] - screw_x_offset
            screw_mask = (screw_yy**2 + screw_xx**2) <= screw_radius**2
            volume[z_idx, screw_mask] = 6  # titanium implant
    
    # 8. Contrast agent injection (iodine)
    injection_center_y = main_radius * 0.6
    injection_center_x = 0
    injection_radius = main_radius * 0.15
    
    injection_yy = yy - injection_center_y
    injection_xx = xx - injection_center_x
    injection_mask = (injection_yy**2 + injection_xx**2) <= injection_radius**2
    injection_mask &= tissue_mask  # Only in tissue
    volume[injection_mask] = 7  # contrast agent
    
    return volume

def create_microstructure_phantom(shape=(256, 256, 256), voxel_size=(0.1, 0.1, 0.1)):
    """Create a phantom with fine microstructures for resolution testing."""
    nz, ny, nx = shape
    volume = np.zeros(shape, dtype=np.int32)
    
    center_z = (nz - 1) / 2.0
    center_y = (ny - 1) / 2.0
    center_x = (nx - 1) / 2.0
    
    zz, yy, xx = np.meshgrid(
        np.arange(nz) - center_z,
        np.arange(ny) - center_y,
        np.arange(nx) - center_x,
        indexing='ij'
    )
    
    # Background material
    background_radius = min(shape) // 2.5
    background_mask = (zz**2 + yy**2 + xx**2) <= background_radius**2
    volume[background_mask] = 1  # polymer matrix
    
    # Fiber network (carbon fibers)
    fiber_length = nx * 0.8
    fiber_radius = 1
    n_fibers = 20
    
    np.random.seed(123)
    for i in range(n_fibers):
        # Random fiber orientation
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(0, np.pi)
        
        # Fiber direction
        dir_x = np.sin(phi) * np.cos(theta)
        dir_y = np.sin(phi) * np.sin(theta)
        dir_z = np.cos(phi)
        
        # Random starting point
        start_y = np.random.uniform(-ny//4, ny//4)
        start_x = np.random.uniform(-nx//4, nx//4)
        start_z = np.random.uniform(-nz//4, nz//4)
        
        # Create fiber
        for t in np.linspace(-fiber_length//2, fiber_length//2, 100):
            fiber_y = start_y + t * dir_y
            fiber_x = start_x + t * dir_x
            fiber_z = start_z + t * dir_z
            
            # Check if within volume bounds
            fy_idx = int(fiber_y + center_y)
            fx_idx = int(fiber_x + center_x)
            fz_idx = int(fiber_z + center_z)
            
            if (0 <= fy_idx < ny and 0 <= fx_idx < nx and 0 <= fz_idx < nz):
                # Create small cylinder around fiber
                local_yy = yy - fiber_y
                local_xx = xx - fiber_x
                local_zz = zz - fiber_z
                
                # Distance to fiber axis
                cross_y = local_yy - (local_yy * dir_y + local_xx * dir_x + local_zz * dir_z) * dir_y
                cross_x = local_xx - (local_yy * dir_y + local_xx * dir_x + local_zz * dir_z) * dir_x
                cross_z = local_zz - (local_yy * dir_y + local_xx * dir_x + local_zz * dir_z) * dir_z
                
                fiber_dist = np.sqrt(cross_y**2 + cross_x**2 + cross_z**2)
                fiber_mask = (fiber_dist <= fiber_radius) & background_mask
                volume[fiber_mask] = 2  # carbon fiber
    
    # Nanoparticles (high contrast)
    n_particles = 50
    for i in range(n_particles):
        part_y = np.random.uniform(-background_radius*0.8, background_radius*0.8)
        part_x = np.random.uniform(-background_radius*0.8, background_radius*0.8)
        part_z = np.random.uniform(-background_radius*0.8, background_radius*0.8)
        part_radius = np.random.uniform(0.5, 2.0)
        
        part_yy = yy - part_y
        part_xx = xx - part_x
        part_zz = zz - part_z
        part_mask = (part_yy**2 + part_xx**2 + part_zz**2) <= part_radius**2
        part_mask &= background_mask
        volume[part_mask] = 3  # gold nanoparticles
    
    # Periodic structures (photonic crystal)
    lattice_spacing = 15
    sphere_radius = 3
    for iy in range(-ny//2, ny//2, lattice_spacing):
        for ix in range(-nx//2, nx//2, lattice_spacing):
            for iz in range(-nz//2, nz//2, lattice_spacing):
                lattice_yy = yy - iy
                lattice_xx = xx - ix
                lattice_zz = zz - iz
                
                lattice_mask = (lattice_yy**2 + lattice_xx**2 + lattice_zz**2) <= sphere_radius**2
                lattice_mask &= background_mask
                volume[lattice_mask] = 4  # silicon spheres
    
    return volume

def create_dose_test_phantom(shape=(64, 64, 64), voxel_size=(1.0, 1.0, 1.0)):
    """Create a phantom specifically for dose testing with known materials."""
    nz, ny, nx = shape
    volume = np.zeros(shape, dtype=np.int32)
    
    center_z = (nz - 1) / 2.0
    center_y = (ny - 1) / 2.0
    center_x = (nx - 1) / 2.0
    
    zz, yy, xx = np.meshgrid(
        np.arange(nz) - center_z,
        np.arange(ny) - center_y,
        np.arange(nx) - center_x,
        indexing='ij'
    )
    
    # Water background
    background_radius = min(shape) // 2.5
    background_mask = (zz**2 + yy**2 + xx**2) <= background_radius**2
    volume[background_mask] = 1  # water
    
    # Bone insert (high dose)
    bone_radius = background_radius * 0.4
    bone_mask = (yy**2 + xx**2) <= bone_radius**2
    volume[bone_mask] = 2  # cortical bone
    
    # Metal insert (very high dose)
    metal_radius = bone_radius * 0.3
    metal_mask = (yy**2 + xx**2) <= metal_radius**2
    volume[metal_mask] = 3  # titanium
    
    # Air cavity (no dose)
    air_radius = metal_radius * 0.5
    air_mask = (yy**2 + xx**2) <= air_radius**2
    volume[air_mask] = 0  # air
    
    return volume

def generate_phantom(phantom_type="sphere", shape=(128, 128, 128), voxel_size=(0.5, 0.5, 0.5)):
    """Generate phantom and save files."""
    
    # Extended material definitions with accurate properties for dose calculation
    materials = {
        "vacuum": {"composition": {}, "density": 0.0},
        "air": {"composition": {"N": 0.78, "O": 0.21, "Ar": 0.01}, "density": 0.00129},
        "water": {"composition": {"H": 2, "O": 1}, "density": 1.0},
        "soft_tissue": {"composition": {"H": 10, "C": 5, "N": 1, "O": 4}, "density": 1.06},
        "fat": {"composition": {"H": 11, "C": 6, "O": 1}, "density": 0.92},
        "muscle": {"composition": {"H": 10, "C": 1, "N": 0.3, "O": 4}, "density": 1.05},
        "cortical_bone": {"composition": {"Ca": 10, "P": 6, "O": 26, "H": 2}, "density": 1.92},
        "trabecular_bone": {"composition": {"Ca": 8, "P": 5, "O": 22, "H": 2}, "density": 1.18},
        "bone_marrow": {"composition": {"H": 11, "C": 6, "N": 0.1, "O": 1}, "density": 0.98},
        "calcium_carbonate": {"composition": {"Ca": 1, "C": 1, "O": 3}, "density": 2.71},
        "hydroxyapatite": {"composition": {"Ca": 10, "P": 6, "O": 26, "H": 2}, "density": 3.16},
        "aluminum": {"composition": {"Al": 1}, "density": 2.70},
        "titanium": {"composition": {"Ti": 1}, "density": 4.51},
        "iron": {"composition": {"Fe": 1}, "density": 7.87},
        "lead": {"composition": {"Pb": 1}, "density": 11.34},
        "iodine_contrast": {"composition": {"I": 1, "H": 2, "O": 1}, "density": 1.5},
        "polymer_matrix": {"composition": {"C": 2, "H": 4}, "density": 1.2},
        "carbon_fiber": {"composition": {"C": 1}, "density": 1.8},
        "gold_nanoparticles": {"composition": {"Au": 1}, "density": 19.3},
        "silicon": {"composition": {"Si": 1}, "density": 2.33}
    }
    
    # Generate volume based on type
    if phantom_type == "sphere":
        volume = create_sphere_phantom(shape, voxel_size)
        codes = {"vacuum": 0, "calcium_carbonate": 1, "hydroxyapatite": 2}
        lookup = {
            "0": materials["vacuum"],
            "1": materials["calcium_carbonate"], 
            "2": materials["hydroxyapatite"]
        }
    elif phantom_type == "cylinder":
        volume = create_cylinder_phantom(shape, voxel_size)
        codes = {"vacuum": 0, "water": 1, "hydroxyapatite": 2, "aluminum": 3}
        lookup = {
            "0": materials["vacuum"],
            "1": materials["water"],
            "2": materials["hydroxyapatite"],
            "3": materials["aluminum"]
        }
    elif phantom_type == "bone":
        volume = create_complex_bone_phantom(shape, voxel_size)
        codes = {
            "vacuum": 0, "soft_tissue": 1, "cortical_bone": 2, "bone_marrow": 3,
            "calcium_carbonate": 4, "trabecular_bone": 5, "titanium": 6, "iodine_contrast": 7
        }
        lookup = {
            "0": materials["vacuum"],
            "1": materials["soft_tissue"],
            "2": materials["cortical_bone"],
            "3": materials["bone_marrow"],
            "4": materials["calcium_carbonate"],
            "5": materials["trabecular_bone"],
            "6": materials["titanium"],
            "7": materials["iodine_contrast"]
        }
    elif phantom_type == "microstructure":
        volume = create_microstructure_phantom(shape, voxel_size)
        codes = {
            "vacuum": 0, "polymer_matrix": 1, "carbon_fiber": 2, 
            "gold_nanoparticles": 3, "silicon": 4
        }
        lookup = {
            "0": materials["vacuum"],
            "1": materials["polymer_matrix"],
            "2": materials["carbon_fiber"],
            "3": materials["gold_nanoparticles"],
            "4": materials["silicon"]
        }
    elif phantom_type == "dose_test":
        volume = create_dose_test_phantom(shape, voxel_size)
        codes = {"air": 0, "water": 1, "cortical_bone": 2, "titanium": 3}
        lookup = {
            "0": materials["air"],
            "1": materials["water"],
            "2": materials["cortical_bone"],
            "3": materials["titanium"]
        }
    else:
        raise ValueError(f"Unknown phantom type: {phantom_type}")
    
    # Save files
    zarr_path = f"phantom_{phantom_type}.zarr"
    json_path = f"phantom_{phantom_type}.json"
    
    save_multiscale_zarr(volume, codes, zarr_path, voxel_size)
    create_metadata_json(lookup, voxel_size, json_path)
    
    print(f"Generated {phantom_type} phantom:")
    print(f"  Shape: {volume.shape}")
    print(f"  Voxel size: {voxel_size} µm")
    print(f"  Materials: {list(lookup.keys())}")
    print(f"  Unique labels: {np.unique(volume)}")
    print(f"  Files: {zarr_path}, {json_path}")
    
    # Print material properties for dose calculation
    print(f"\nMaterial properties for dose calculation:")
    for label, props in lookup.items():
        if props.get("composition"):
            formula = ''.join(f"{el}{amt}" for el, amt in props["composition"].items())
            print(f"  Label {label}: {formula}, ρ={props['density']:.3f} g/cm³")
        else:
            print(f"  Label {label}: vacuum/air, ρ={props['density']:.6f} g/cm³")
    
    return zarr_path, json_path

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    phantom_type = sys.argv[1] if len(sys.argv) > 1 else "sphere"
    
    if phantom_type not in ["sphere", "cylinder", "bone", "microstructure", "dose_test"]:
        print("Usage: python generate_phantom.py [sphere|cylinder|bone|microstructure|dose_test]")
        print("Available phantoms:")
        print("  sphere         - Simple nested spheres")
        print("  cylinder       - Cylindrical features")
        print("  bone           - Complex bone with implants, contrast, calcifications")
        print("  microstructure - Fine fibers, nanoparticles, periodic structures")
        print("  dose_test      - Simple phantom optimized for dose testing")
        sys.exit(1)
    
    # Generate phantom with appropriate parameters
    if phantom_type == "bone":
        zarr_path, json_path = generate_phantom(phantom_type, shape=(100, 128, 128), voxel_size=(0.5, 0.5, 0.5))
    elif phantom_type == "microstructure":
        zarr_path, json_path = generate_phantom(phantom_type, shape=(128, 128, 128), voxel_size=(0.2, 0.2, 0.2))
    elif phantom_type == "dose_test":
        zarr_path, json_path = generate_phantom(phantom_type, shape=(64, 64, 64), voxel_size=(1.0, 1.0, 1.0))
    else:
        zarr_path, json_path = generate_phantom(phantom_type, shape=(100, 128, 128), voxel_size=(0.5, 0.5, 0.5))
    
    print(f"\nTo test simulation:")
    print(f"from msim.interface import quick_tomography, analyze_dose_only")
    print(f"projections, dose_stats = quick_tomography('{zarr_path}', '{json_path}', calculate_dose=True)")
    print(f"dose_map, dose_stats = analyze_dose_only('{zarr_path}', '{json_path}')")