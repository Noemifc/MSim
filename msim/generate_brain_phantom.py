#prende vol



import tifffile
import numpy as np
import z5py
import os
import shutil
import json

# --- CONFIGURAZIONE ---
input_tiff = 'brain.tiff'
output_zarr = 'brain_cerebral.zarr'
output_json = 'brain_cerebral.json'
voxel_size = (0.1, 0.1, 0.1)  # voxel piccoli
base_chunk = (64, 64, 64)

# --- CARICA IL TIFF ---
volume = tifffile.imread(input_tiff)
print(f"Volume shape: {volume.shape}, dtype: {volume.dtype}")

# --- ETICHETTA I TESSUTI CEREBRALI ---
# Esempio semplice, puoi personalizzare le soglie
brain_labels = np.zeros_like(volume, dtype=np.int32)
brain_labels[(volume > 30) & (volume <= 80)] = 1  # gray matter
brain_labels[(volume > 80) & (volume <= 150)] = 2  # white matter
brain_labels[(volume > 150)] = 3  # ventricles / CSF

# --- MATERIALI / LOOKUP ---
materials = {
    "air": {"composition": {}, "density": 0.0012},
    "gray_matter": {"composition": {"H":10,"C":5,"N":1,"O":4}, "density":1.04},
    "white_matter": {"composition": {"H":11,"C":6,"N":1,"O":5}, "density":1.03},
    "ventricles": {"composition": {"H":2,"O":1}, "density":1.0}
}
lookup = {
    "0": materials["air"],
    "1": materials["gray_matter"],
    "2": materials["white_matter"],
    "3": materials["ventricles"]
}

# --- SALVA ZARR MULTISCALE ---
def save_multiscale_zarr(volume, lookup, out_dir, voxel_size, base_chunk=(64,64,64)):
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    f = z5py.File(out_dir, use_zarr_format=True)
    
    curr = volume.copy()
    datasets = []
    for lvl in range(3):
        path = str(lvl)
        chunks = tuple(min(c, s) for c, s in zip(base_chunk, curr.shape))
        f.create_dataset(name=path, data=curr, chunks=chunks, compression='raw')
        scale = 2 ** lvl
        datasets.append({
            "path": path,
            "coordinateTransformations": [
                {"type": "scale", "scale": [scale]*3},
                {"type": "translation", "translation": [scale/2-0.5]*3}
            ]
        })
        curr = curr[::2, ::2, ::2]
    
    f.attrs['lookup'] = lookup
    f.attrs['voxel_size'] = voxel_size
    f.attrs['multiscales'] = [{
        "version": "0.4",
        "axes":[{"name":"z","type":"space"},{"name":"y","type":"space"},{"name":"x","type":"space"}],
        "datasets": datasets,
        "type": "image",
        "metadata": {"voxel_size": voxel_size}
    }]
    
    with open(os.path.join(out_dir, "multiscale.json"), 'w') as fjson:
        json.dump(f.attrs['multiscales'], fjson, indent=2)

# --- SALVA ---
save_multiscale_zarr(brain_labels, lookup, output_zarr, voxel_size)
with open(output_json, 'w') as fjson:
    json.dump({"voxel_size": list(voxel_size), "lookup": lookup}, fjson, indent=2)

print(f"Zarr e JSON creati:\n  {output_zarr}\n  {output_json}")
print(f"Tessuti presenti: {np.unique(brain_labels)}")
