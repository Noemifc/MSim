import zarr
import json
import numpy as np
import napari
import h5py

# ----------------------------
# Percorsi dei file
# ----------------------------
zarr_path = "/home/beams0/NFICO/MSim/examples/mouse_brain/mappa_area/brain_map.zarr"
json_path = "/home/beams0/NFICO/MSim/examples/mouse_brain/mappa_area/brain_lookup.json"

# ----------------------------
# Carica il volume Zarr
# ----------------------------
z = zarr.open(zarr_path, mode='r')
data = z[:]
print("Shape del volume:", data.shape)

# ----------------------------
# Carica JSON integrato
# ----------------------------
with open(json_path) as f:
    phantom = json.load(f)

# Verifica che esista la chiave "regions"
if 'regions' in phantom:
    phantom_regions = phantom['regions']
else:
    raise ValueError("Il JSON non contiene la chiave 'regions'.")

# ----------------------------
# Crea mappa ID -> GroupID
# ----------------------------
id_to_group = {r['ID']: r.get('GroupID', r['ID']) for r in phantom_regions}

# ----------------------------
# Ricostruisci il volume con GroupID
# ----------------------------
volume = np.vectorize(lambda x: id_to_group.get(x, 0))(data)

# ----------------------------
# Visualizza con napari
# ----------------------------
viewer = napari.Viewer()
viewer.add_image(volume, name='Phantom', colormap='tab20', blending='additive')

# ----------------------------
# Facoltativo: aggiungi legenda dei nomi
# ----------------------------
group_names = {}
for r in phantom_regions:
    group_id = r.get('GroupID', r['ID'])
    group_names[group_id] = phantom['lookup'].get(str(group_id), {}).get('name', f'Group {group_id}')

print("Legenda dei gruppi:")
for gid, name in group_names.items():
    print(f"Group {gid}: {name}")

napari.run()

# ----------------------------
# Percorso del file HDF5 da salvare
# ----------------------------
h5_path = "/home/beams/NFICO/MSim/examples/mouse_brain/mappa_area/brain_map.h5"

# ----------------------------
# Salva il volume in HDF5
# ----------------------------
with h5py.File(h5_path, 'w') as f:
    # Crea un dataset chiamato 'volume' con lo stesso shape e tipo del volume
    f.create_dataset('volume', data=volume, compression="gzip")

print(f"Volume salvato correttamente in: {h5_path}")
