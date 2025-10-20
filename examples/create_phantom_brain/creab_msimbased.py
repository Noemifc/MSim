#!/usr/bin/env python3
"""
Generate brain phantom in Zarr format for X-ray simulation with dose calculation.
Outputs multiscale Zarr and JSON with voxel_size, lookup, and regions.
"""

import numpy as np
import json
import os
import shutil
import z5py
import tifffile

# ------------------------
# ZARR SAVE FUNCTION
# ------------------------
def save_multiscale_zarr(
    data,
    codes,
    out_dir,
    voxel_size=(2.0, 2.0, 2.0),
    n_scales=3,
    base_chunk=(64, 128, 128),
    logger=None
):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
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
                {"type": "scale", "scale": [scale] * 3},
                {"type": "translation", "translation": [scale / 2 - 0.5] * 3}
            ]
        })
        curr = curr[::2, ::2, ::2]
    lookup_attr = {v: {'alias': k} for k, v in codes.items()}
    f.attrs['lookup'] = lookup_attr
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
        "metadata": {"voxel_size": voxel_size}
    }
    f.attrs['multiscales'] = [multiscale_meta]
    with open(os.path.join(out_dir, "multiscale.json"), "w") as fjson:
        json.dump([multiscale_meta], fjson, indent=2)
    if logger:
        logger.info(f"Saved Neuroglancer-ready multiscale Zarr → {out_dir}")

# ------------------------
# JSON METADATA FUNCTION
# ------------------------
def create_metadata_json(lookup_dict, voxel_size, output_path):
    """
    Create JSON metadata file including voxel_size, lookup (density and mu), and regions.
    """
    regions = []
    for idx, key in enumerate(sorted(lookup_dict.keys(), key=lambda x: int(x))):
        regions.append({"ID": int(key), "GroupID": int(key)})

    metadata = {
        "voxel_size": list(voxel_size),
        "lookup": lookup_dict,
        "regions": regions
    }

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

# ------------------------
# PHANTOM GENERATION
# ------------------------
def generate_brain_phantom(tiff_file="BCI.tif", json_file="brain_integrated.json"):
    """
    Generate brain phantom from TIFF and JSON lookup.
    """
    # Load TIFF
    img = tifffile.imread(tiff_file)
    print("Loaded brain TIFF:", img.shape)

    # Load JSON lookup
    with open(json_file, "r") as f:
        metadata = json.load(f)
    lookup = metadata["lookup"]

    # Map labels to densities
    mapping = {int(k): float(v["density"]) for k,v in lookup.items()}
    volume = np.zeros_like(img, dtype=np.float32)
    for key, val in mapping.items():
        volume[img == key] = val

    # Codes for Zarr attribute
    codes = {k: k for k in mapping.keys()}

    # Output paths
    zarr_path = "phantom_b.zarr"
    json_path = "phantom_b.json"

    # Save
    save_multiscale_zarr(volume, codes, zarr_path, voxel_size=(2.0, 2.0, 2.0))
    create_metadata_json(lookup, voxel_size=(2.0, 2.0, 2.0), output_path=json_path)

    print(f"Generated brain phantom:")
    print(f"  Shape: {volume.shape}")
    print(f"  Voxel size: [2.0, 2.0, 2.0]")
    print(f"  Unique labels: {np.unique(volume)}")
    print(f"  Files: {zarr_path}, {json_path}")

    return zarr_path, json_path

#
def create_brain_phantom(shape=(64, 128, 128), voxel_size=(2.0, 2.0, 2.0)):
    """
    Create a simplified brain phantom with labeled regions.
    Labels:
      0: air
      1: gyrus corticali
      2: ippocampo + amigdala
      3: insula
      4: globo pallidus + putamen
      5: altro
    """
    nz, ny, nx = shape
    volume = np.zeros(shape, dtype=np.int32)  # default: air

    # Central coordinates
    center_z = (nz - 1) / 2.0
    center_y = (ny - 1) / 2.0
    center_x = (nx - 1) / 2.0

    # Coordinate grids
    zz, yy, xx = np.meshgrid(
        np.arange(nz) - center_z,
        np.arange(ny) - center_y,
        np.arange(nx) - center_x,
        indexing='ij'
    )

    # Define simple ellipsoids for each region (simplified approximation)
    # 1. gyrus corticali
    mask1 = (zz**2 / (nz*0.4)**2 + yy**2 / (ny*0.45)**2 + xx**2 / (nx*0.45)**2) <= 1
    volume[mask1] = 1

    # 2. ippocampo + amigdala
    mask2 = ((zz+10)**2 / (nz*0.15)**2 + (yy-10)**2 / (ny*0.15)**2 + (xx-10)**2 / (nx*0.15)**2) <= 1
    volume[mask2] = 2

    # 3. insula
    mask3 = ((zz-5)**2 / (nz*0.1)**2 + (yy+15)**2 / (ny*0.1)**2 + (xx+5)**2 / (nx*0.1)**2) <= 1
    volume[mask3] = 3

    # 4. globo pallidus + putamen
    mask4 = ((zz-8)**2 / (nz*0.12)**2 + (yy-12)**2 / (ny*0.12)**2 + (xx-8)**2 / (nx*0.12)**2) <= 1
    volume[mask4] = 4

    # 5. altro (resto del tessuto)
    mask5 = (volume == 0)
    volume[mask5] = 5

    return volume

# In generate_phantom(), aggiungi il case "brain":
elif phantom_type == "brain":
    volume = create_brain_phantom(shape, voxel_size)
    codes = {
        "air": 0,
        "gyrus corticali": 1,
        "ippocampo + amigdala": 2,
        "insula": 3,
        "globo pallidus + putamen": 4,
        "altro": 5
    }
    lookup = {
        "0": {"name": "air", "density": 0.0012, "mu": 0.0},
        "1": {"name": "gyrus corticali", "density": 1.03, "mu": 0.19},
        "2": {"name": "ippocampo + amigdala", "density": 1.04, "mu": 0.2},
        "3": {"name": "insula", "density": 1.045, "mu": 0.21},
        "4": {"name": "globo pallidus + putamen", "density": 1.05, "mu": 0.23},
        "5": {"name": "altro", "density": 1.06, "mu": 0.25}
    }

# Poi salva Zarr + JSON come gli altri phantom:
zarr_path = f"phantom_brain.zarr"
json_path = f"phantom_brain.json"

save_multiscale_zarr(volume, codes, zarr_path, voxel_size)
create_metadata_json(lookup, voxel_size, json_path)

print(f"Generated brain phantom: {volume.shape}, voxel size: {voxel_size}")
print(f"Files: {zarr_path}, {json_path}")




#











# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    import sys

    tiff_file = "BCI.tif"
    json_file  = "brain_integrated.json"

    zarr_path, json_path = generate_brain_phantom(tiff_file, json_file)

    # Multilinea sicura per istruzioni
    print(f"""
To test simulation:

from msim.interface import quick_tomography, analyze_dose_only

projections, dose_stats = quick_tomography('{zarr_path}', '{json_path}', calculate_dose=True)

dose_map, dose_stats = analyze_dose_only('{zarr_path}', '{json_path}')
""")
