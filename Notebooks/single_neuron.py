import numpy as np
import z5py
import os

# -----------------------------
# Parameters
# -----------------------------
voxel_size = 0.1  # µm per voxel (higher resolution)

def um_to_vox(um):
    return int(np.round(um / voxel_size))

# Convert sizes to voxels
soma_radius = um_to_vox(7.5)
nucleus_radius = um_to_vox(3.0)
axon_radius = um_to_vox(0.5)
axon_length = um_to_vox(1000)
dendrite_radius = um_to_vox(1.0)  # thinner in µm, but wider in voxels now
dendrite_length = um_to_vox(100)

# -----------------------------
# Create volume
# -----------------------------
vol_shape = (axon_length + 2 * soma_radius, 512, 512)
volume = np.zeros(vol_shape, dtype=np.uint8)
center = (soma_radius, vol_shape[1] // 2, vol_shape[2] // 2)

# -----------------------------
# Soma
# -----------------------------
for z in range(-soma_radius, soma_radius):
    for y in range(-soma_radius, soma_radius):
        for x in range(-soma_radius, soma_radius):
            if x**2 + y**2 + z**2 < soma_radius**2:
                volume[center[0]+z, center[1]+y, center[2]+x] = 100

# -----------------------------
# Nucleus
# -----------------------------
for z in range(-nucleus_radius, nucleus_radius):
    for y in range(-nucleus_radius, nucleus_radius):
        for x in range(-nucleus_radius, nucleus_radius):
            if x**2 + y**2 + z**2 < nucleus_radius**2:
                volume[center[0]+z, center[1]+y, center[2]+x] = 200

# -----------------------------
# Axon
# -----------------------------
for i in range(axon_length):
    for y in range(-axon_radius, axon_radius):
        for x in range(-axon_radius, axon_radius):
            if x**2 + y**2 < axon_radius**2:
                volume[center[0]+soma_radius+i, center[1]+y, center[2]+x] = 150

# -----------------------------
# Local cylinder drawing
# -----------------------------
def add_cylinder_local(volume, start, direction, length, radius, label):
    dz, dy, dx = direction
    for i in range(length):
        zc = int(start[0] + i * dz)
        yc = int(start[1] + i * dy)
        xc = int(start[2] + i * dx)
        if not (0 <= zc < volume.shape[0] and 0 <= yc < volume.shape[1] and 0 <= xc < volume.shape[2]):
            continue
        for z in range(zc - radius, zc + radius + 1):
            if not (0 <= z < volume.shape[0]):
                continue
            for y in range(yc - radius, yc + radius + 1):
                if not (0 <= y < volume.shape[1]):
                    continue
                for x in range(xc - radius, xc + radius + 1):
                    if not (0 <= x < volume.shape[2]):
                        continue
                    if (z - zc)**2 + (y - yc)**2 + (x - xc)**2 <= radius**2:
                        volume[z, y, x] = label

# -----------------------------
# Dendrites from Soma
# -----------------------------
soma_directions = [(0, -1, 0), (0, 0, -1), (0, 0, 1)]
for dir in soma_directions:
    add_cylinder_local(volume, center, dir, dendrite_length, dendrite_radius, 180)

# -----------------------------
# Dendrites from Axon Tip
# -----------------------------
axon_tip_z = center[0] + soma_radius + axon_length - 1
axon_tip_center = (axon_tip_z, center[1], center[2])
axon_directions = [(0, 2, 2), (0, -2, 2), (0, 2, -2), (0, -2, -2)]
for dir in axon_directions:
    add_cylinder_local(volume, axon_tip_center, dir, dendrite_length, dendrite_radius, 180)

# -----------------------------
# Save to N5
# -----------------------------
n5_path = "neuron_model_with_dendrites_01um.n5"
dataset_name = "neuron"

with z5py.File(n5_path, use_zarr_format=False) as f:
    if dataset_name in f:
        del f[dataset_name]
    ds = f.create_dataset(
        name=dataset_name,
        shape=volume.shape,
        chunks=(64, 128, 128),
        dtype='uint8',
        compression='gzip'
    )
    ds[:] = volume

print(f"✅ Saved N5 file at: {n5_path}/{dataset_name}")
