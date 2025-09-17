#!/usr/bin/env python3
"""
Fixed volume viewer optimized for multi-material chip data
"""
import os
import sys
import json
import numpy as np
import h5py
import z5py
from vedo import Volume, Text2D, Plotter
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

def view_volume(volume_path: str) -> None:
    # ---------------------------------------------------------------- I/O ---
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

    # ----------------------------------------------------------- preprocess --
    data = np.squeeze(data)
    if data.ndim != 3:
        print(f"Error: expected 3-D array, got shape {data.shape}")
        return

    # Analyze data for optimization
    unique_vals = np.unique(data)
    n_materials = len(unique_vals)
    print(f"Found {n_materials} unique materials")

    # Downsample if too large for rendering
    total_voxels = np.prod(data.shape)
    if total_voxels > 100_000_000:  # 100M voxels
        factor = max(2, int(np.cbrt(total_voxels / 50_000_000)))
        data = data[::factor, ::factor, ::factor]
        print(f"Downsampled by factor {factor} to {data.shape}")

    # Optimize data type for rendering
    if np.issubdtype(data.dtype, np.floating):
        mn, mx = data.min(), data.max()
        data = np.zeros_like(data) if mx == mn else (data - mn) / (mx - mn)
        data = (data * 255).astype(np.uint8)
    elif data.dtype not in (np.uint8, np.uint16, np.uint32):
        data = data.astype(np.uint8)
    data = np.ascontiguousarray(data)

    # ---------------------------------------------------------- metadata ----
    voxel_size = [1.0, 1.0, 1.0]
    description = "Chip Model"
    if os.path.exists(json_path):
        with open(json_path) as jf:
            meta = json.load(jf)
        voxel_size = meta.get("voxel_size", voxel_size)
        description = meta.get("description", description)

    # ------------------------------------------------------------- vedo -----
    vol = Volume(data).spacing(voxel_size[::-1])
    
    # Optimize volume settings for chip data
    if n_materials > 20:
        vol.cmap("tab20").alpha(0.6).mode(1)  # Composite mode for many materials
    elif n_materials > 5:
        vol.cmap("Set3").alpha(0.7).mode(1)
    else:
        vol.cmap("bone_r").alpha([0, 0.1, 0.3, 0.6, 0.8, 1.0]).mode(0)

    legend = Text2D("Mode: SLICER — R=Render, S=Slicer, I=Iso", pos="top-left", c="white", s=1.2)
    desc_label = Text2D(description, pos="bottom-left", c="white", s=1.0)
    state = {"mode": "slicer"}

    def launch_slicer():
        try:
            p = Slicer3DPlotter(vol, cmaps=["tab20", "Set3", "bone_r", "jet"],
                              use_slider3d=False, bg="black", bg2="black")
            legend.text("Mode: SLICER — R=Render, S=Slicer, I=Iso")
            p += legend; p += desc_label
            p.add_callback("KeyPress", toggle_mode)
            p.show(viewup="z")
            p.close()
        except Exception as e:
            print(f"Slicer failed: {e}")
            launch_fallback()

    def launch_render():
        """Fixed ray-casting render with proper error handling"""
        try:
            # FIXED: Import RayCastPlotter properly
            from vedo.applications import RayCastPlotter
            
            # Create optimized volume for ray-casting
            render_vol = vol.clone()
            
            # FIXED: Optimize settings for multi-material data
            if n_materials > 10:
                # For many materials, use lower alpha and composite mode
                render_vol.mode(1).alpha(0.3)
                render_vol.cmap("tab20")
            else:
                # For fewer materials, use standard settings
                render_vol.mode(0).alpha([0, 0.2, 0.5, 0.8])
                render_vol.cmap("bone_r")
            
            # FIXED: Additional ray-casting optimizations
            render_vol.shade(False)  # Disable shading for complex volumes
            
            p = RayCastPlotter(render_vol, bg="black", bg2="black", axes=7)
            legend.text("Mode: RENDER — R=Render, S=Slicer, I=Iso")
            p += legend; p += desc_label
            p.add_callback("KeyPress", toggle_mode)
            p.show(viewup="z")
            p.close()
            
        except Exception as e:
            print(f"Ray-casting failed: {e}")
            print("Falling back to isosurface rendering...")
            launch_isosurface()

    def launch_isosurface():
        """Isosurface rendering - works well for chip data"""
        try:
            p = Plotter(bg="black", bg2="black", axes=7)
            
            # FIXED: Smart isosurface selection for chip materials
            if n_materials <= 10:
                # Show each material as separate isosurface
                colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
                for i, val in enumerate(unique_vals[1:min(8, len(unique_vals))]):  # Skip background (0)
                    try:
                        iso = vol.isosurface(threshold=val - 0.1)
                        iso.color(colors[i % len(colors)]).alpha(0.7)
                        p += iso
                    except:
                        continue
            else:
                # For many materials, show representative layers
                thresholds = np.percentile(data[data > 0], [25, 50, 75, 90])
                colors = ['red', 'green', 'blue', 'yellow']
                for threshold, color in zip(thresholds, colors):
                    try:
                        iso = vol.isosurface(threshold=threshold)
                        iso.color(color).alpha(0.5)
                        p += iso
                    except:
                        continue
            
            legend.text("Mode: ISOSURFACE — R=Render, S=Slicer, I=Iso")
            p += legend; p += desc_label
            p.add_callback("KeyPress", toggle_mode)
            p.show(viewup="z")
            p.close()
            
        except Exception as e:
            print(f"Isosurface failed: {e}")
            launch_fallback()

    def launch_fallback():
        """Simple 3D plot as last resort"""
        try:
            p = Plotter(bg="black", bg2="black", axes=7)
            simple_vol = vol.clone().alpha(0.3).mode(0)
            p += simple_vol
            legend.text("Mode: SIMPLE — R=Render, S=Slicer, I=Iso")
            p += legend; p += desc_label
            p.add_callback("KeyPress", toggle_mode)
            p.show(viewup="z")
            p.close()
        except Exception as e:
            print(f"All rendering modes failed: {e}")

    def toggle_mode(evt):
        k = evt.keypress.lower()
        if k == "r":
            state["mode"] = "render"
            launch_render()
        elif k == "s":
            state["mode"] = "slicer"
            launch_slicer()
        elif k == "i":
            state["mode"] = "iso"
            launch_isosurface()

    # Print controls
    print("\nControls:")
    print("  S = Slicer mode (recommended for chip data)")
    print("  R = Ray-cast render mode")  
    print("  I = Isosurface mode (good for materials)")
    print("  Mouse wheel = change slices (in slicer mode)")
    print()

    # Start with slicer mode (most reliable)
    launch_slicer()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: view_volume.py <path/to/volume.{h5|n5|zarr}>")
    else:
        view_volume(sys.argv[1])