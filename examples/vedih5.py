import os
import glob
import h5py
import napari

def open_all_h5_in_napari():
    """Apri tutti i .h5 nella cartella corrente e carica i dataset in Napari."""
    current_dir = os.getcwd()
    print(f"\n🔍 Cercando .h5 in: {current_dir}")

    h5_files = glob.glob(os.path.join(current_dir, "*.h5"))
    if not h5_files:
        print("  Nessun file HDF5 trovato.")
        return

    viewer = napari.Viewer()

    for file in h5_files:
        print(f"\n📂 Aprendo {os.path.basename(file)} ...")
        try:
            with h5py.File(file, "r") as f:
                for name, obj in f.items():
                    if isinstance(obj, h5py.Dataset):
                        print(f"  - Caricando dataset '{name}' shape={obj.shape}, dtype={obj.dtype}")
                        data = obj[()]  # carica il dataset in memoria
                        viewer.add_image(data, name=f"{os.path.basename(file)}::{name}")
        except Exception as e:
            print(f"❌ Errore aprendo {file}: {e}")

    napari.run()

# Esegui
open_all_h5_in_napari()
