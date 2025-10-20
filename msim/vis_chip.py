#!/usr/bin/env python3
"""
Chip-specific volume viewer that handles discrete material data properly
"""
import os
import sys
import json
import numpy as np
import h5py
import z5py
from vedo import Volume, Text2D, Plotter, Mesh
from vedo.applications import Slicer3DPlotter

def _looks_like_zarr(path: str) -> bool:
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, ".zgroup"))
        or os.path.exists(os.path.join(path, ".zarray"))
    )

def _find_first_dataset_z5(group: z5py.Group) -> str | None:
    for name, val in group.items():
        if isinstance(val, z5py.Group):
            sub = _find_first_dataset_z5(val)
            if sub:
                return f"{name}/{sub}"
        else:
            return name
    return None

def create_material_meshes(data, material_codes, max_materials=10, palette_idx=0):
    """Create separate meshes for each material with selectable colors"""
    unique_vals = np.unique(data[data > 0])  # Skip background
    print(f"Creating meshes for {min(len(unique_vals), max_materials)} materials...")
    
    # Different color sets based on palette index
    color_sets = [
        ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink'],
        ['darkred', 'darkgreen', 'darkblue', 'gold', 'turquoise', 'violet', 'coral', 'indigo', 'olive', 'navy'],
        ['crimson', 'lime', 'royalblue', 'orange', 'deepskyblue', 'mediumorchid', 'tomato', 'slateblue', 'chocolate', 'hotpink'],
        ['firebrick', 'forestgreen', 'steelblue', 'darkorange', 'lightseagreen', 'mediumpurple', 'sandybrown', 'darkslateblue', 'sienna', 'palevioletred']
    ]
    colors = color_sets[palette_idx % len(color_sets)]
    
    meshes = []
    
    for i, val in enumerate(unique_vals[:max_materials]):
        try:
            # Create binary mask for this material
            mask = (data == val).astype(np.uint8)
            
            # Only process if there's enough material
            if np.sum(mask) < 100:
                continue
                
            # Create volume and extract isosurface
            vol = Volume(mask)
            iso = vol.isosurface(threshold=0.5)
            
            if iso.npoints > 0:  # Check if isosurface was created
                color = colors[i % len(colors)]
                iso.color(color).alpha(0.7)
                iso.name = f"Material_{val}"
                meshes.append(iso)
                print(f"  Material {val}: {iso.npoints} points, color={color}")
            
        except Exception as e:
            print(f"  Failed to create mesh for material {val}: {e}")
            continue
    
    return meshes

def view_volume(volume_path: str) -> None:
    # Load data
    volume_path = volume_path.rstrip(os.sep)
    root, ext = os.path.splitext(volume_path)
    ext = ext.lower()
    json_path = root + ".json"

    if ext in (".h5", ".hdf5", ".hdf"):
        with h5py.File(volume_path, "r") as f:
            if "exchange/data" in f:
                data = f["exchange/data"][:]
            elif "reconstruction" in f:
                data = f["reconstruction"][:]
            else:
                print(f"Error: '/exchange/data' not found in {volume_path}")
                return
        print(f"Loaded HDF5: {volume_path} | shape={data.shape}, dtype={data.dtype}")
    else:
        is_zarr = (ext == ".zarr") or _looks_like_zarr(volume_path)
        f = z5py.File(volume_path, use_zarr_format=is_zarr)
        dset_path = _find_first_dataset_z5(f)
        if not dset_path:
            print(f"No dataset found in {volume_path}")
            return
        data = f[dset_path][:]
        f.close()
        kind = "Zarr" if is_zarr else "N5"
        print(f"Loaded {kind}: {volume_path} → {dset_path} | shape={data.shape}, dtype={data.dtype}")

    # Analyze data
    data = np.squeeze(data)
    if data.ndim != 3:
        print(f"Error: expected 3-D array, got shape {data.shape}")
        return

    unique_vals = np.unique(data)
    n_materials = len(unique_vals)
    print(f"Data range: {data.min()} - {data.max()}")
    print(f"Unique values: {n_materials}")
    print(f"First 10 values: {unique_vals[:10]}")

    # Smart downsampling for very large volumes
    total_voxels = np.prod(data.shape)
    downsample_factor = 1
    if total_voxels > 50_000_000:
        downsample_factor = max(2, int(np.cbrt(total_voxels / 20_000_000)))
        data = data[::downsample_factor, ::downsample_factor, ::downsample_factor]
        print(f"Downsampled by factor {downsample_factor} to {data.shape}")

    # Load material codes if available
    material_codes = {}
    if os.path.exists(json_path):
        with open(json_path) as jf:
            meta = json.load(jf)
        material_codes = meta.get("material_codes", {})

    # Create volume for slicer mode
    vol = Volume(data)
    vol.cmap("tab20").alpha([0, 0.8, 0.8, 0.8, 0.8]).mode(0)

    # State management
    state = {"mode": "slicer", "meshes": None}
    legend = Text2D("Mode: SLICER — M=Mesh, S=Slicer, V=Volume", pos="top-left", c="white", s=1.2)
    desc_label = Text2D("Chip Model", pos="bottom-left", c="white", s=1.0)

    def launch_slicer():
        """Slicer mode - always works"""
        try:
            print("Launching slicer mode...")
            p = Slicer3DPlotter(vol, cmaps=["tab20", "Set3", "viridis"],
                              use_slider3d=False, bg="black", bg2="black")
            legend.text("Mode: SLICER — M=Mesh, S=Slicer, V=Volume")
            p += legend; p += desc_label
            p.add_callback("KeyPress", handle_keypress)
            p.show(viewup="z")
            p.close()
        except Exception as e:
            print(f"Slicer failed: {e}")

    def launch_mesh_mode():
        """Material mesh mode - best for chip visualization"""
        try:
            print("Creating material meshes...")
            palette_idx = getattr(state, 'palette_idx', 0)
            if state["meshes"] is None:
                state["meshes"] = create_material_meshes(data, material_codes, max_materials=8, palette_idx=palette_idx)
            
            if not state["meshes"]:
                print("No meshes created - falling back to slicer")
                launch_slicer()
                return
            
            p = Plotter(bg="black", bg2="black", axes=7)
            
            for mesh in state["meshes"]:
                p += mesh
            
            legend.text(f"Mode: MESH ({len(state['meshes'])} materials) — M=Mesh, S=Slicer, V=Volume, C=Colors")
            p += legend; p += desc_label
            p.add_callback("KeyPress", handle_keypress)
            p.show(viewup="z")
            p.close()
            
        except Exception as e:
            print(f"Mesh mode failed: {e}")
            launch_slicer()

    def launch_volume_mode():
        """Fixed volume rendering with palette cycling"""
        try:
            print("Launching volume mode...")
            p = Plotter(bg="black", bg2="black", axes=7)
            
            # Create a proper volume for rendering
            render_vol = Volume(data)
            
            # Get current palette index
            palettes = ["tab20", "Set3", "Pastel1", "Dark2", "viridis", "plasma", "bone", "cool"]
            current_palette = getattr(state, 'palette_idx', 0)
            palette_name = palettes[current_palette % len(palettes)]
            
            print(f"Using palette: {palette_name}")
            
            # Fix for discrete data: proper colormap setup
            if n_materials <= 20:
                # For discrete materials, map values directly
                render_vol.cmap(palette_name, vmin=data.min(), vmax=data.max())
                render_vol.alpha(0.7)  # Solid alpha for visibility
            else:
                render_vol.cmap(palette_name)
                render_vol.alpha(0.5)
            
            # Use mode 0 for discrete data
            render_vol.mode(0)
            
            p += render_vol
            legend.text(f"Mode: VOLUME ({palette_name}) — M=Mesh, S=Slicer, V=Volume, C=Colors")
            p += legend; p += desc_label
            p.add_callback("KeyPress", handle_keypress)
            p.show(viewup="z")
            p.close()
            
        except Exception as e:
            print(f"Volume mode failed: {e}")
            launch_mesh_mode()

    def handle_keypress(evt):
        k = evt.keypress.lower()
        if k == "m":
            state["mode"] = "mesh"
            launch_mesh_mode()
        elif k == "s":
            state["mode"] = "slicer"
            launch_slicer()
        elif k == "v":
            state["mode"] = "volume"
            launch_volume_mode()
        elif k == "c":
            # Cycle through color palettes
            if not hasattr(state, 'palette_idx'):
                state['palette_idx'] = 0
            state['palette_idx'] = (state['palette_idx'] + 1) % 8
            print(f"Switched to palette {state['palette_idx']}")
            if state["mode"] == "volume":
                launch_volume_mode()
            elif state["mode"] == "mesh":
                state["meshes"] = None  # Force recreation with new colors
                launch_mesh_mode()

    print("\n=== CHIP VOLUME VIEWER ===")
    print("Controls:")
    print("  S = Slicer mode (slice through data)")
    print("  M = Mesh mode (each material as 3D mesh) - RECOMMENDED")
    print("  V = Volume mode (3D volume rendering)")
    print("  C = Change color palette/scheme")
    print("  Mouse = rotate, zoom")
    print("  Mouse wheel = change slices (slicer mode)")
    print()

    # Start with slicer mode
    launch_slicer()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chip_viewer.py <path/to/volume.zarr>")
        print("\nThis viewer is optimized for discrete material data like semiconductor chips.")
    else:
        view_volume(sys.argv[1])