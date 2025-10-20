import tifffile
import zarr
import numpy as np
import json
import os

# --- INPUT ---
tiff_file = "BCI.tif"
json_file = "phantom_brain.json"
zarr_file = "phantom_brain.zarr"

# --- LEGGE IL TIFF ---
img = tifffile.imread(tiff_file)  # shape: (slices, height, width)
print("Shape TIFF:", img.shape, "dtype:", img.dtype)

# --- LEGGE IL JSON ---
with open(json_file, "r") as f:
    metadata = json.load(f)

voxel_size = metadata["voxel_size"]
lookup = metadata["lookup"]

# --- CREA UNA MAPPA DEI VALORI ---
# Scegli se usare 'density' o 'mu'
use_field = 'density'  # cambia in 'mu' se vuoi
mapping = {int(k): float(v[use_field]) for k, v in lookup.items()}

# --- CREA L'ARRAY NORMALIZZATO ---
img_mapped = np.zeros_like(img, dtype=np.float32)
for key, val in mapping.items():
    img_mapped[img == key] = val

# --- DEFINISCE I CHUNK ---
# 3D volumetrico
chunk_shape = (64, 128, 128)
# Oppure per slice singole: chunk_shape = (1, img.shape[1], img.shape[2])

# --- CREA ZARR ---
z = zarr.open(zarr_file, mode='w', shape=img_mapped.shape, chunks=chunk_shape, dtype=img_mapped.dtype,
              compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2))

# --- SCRIVE I DATI ---
z[:] = img_mapped

# --- SALVA METADATI ---
z.attrs["voxel_size"] = voxel_size
z.attrs["lookup"] = lookup
z.attrs["mapped_field"] = use_field

print("Conversione completata! Zarr salvato in:", zarr_file)
