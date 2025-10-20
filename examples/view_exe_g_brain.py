#!/usr/bin/env python3
"""
Visualizza le proiezioni dal file HDF5 in Napari con normalizzazione per ridurre il rumore.
"""

import h5py
import napari
import numpy as np

# Percorso del file HDF5
h5_file = "/mnt/c/Users/noemi/Desktop/Noe/examples/basic_tomo.h5"

def view_projections(file_path):
    """Apri dataset 'exchange/data', normalizza e visualizza in Napari."""
    try:
        with h5py.File(file_path, "r") as f:
            dataset_path = "exchange/data"
            if dataset_path not in f:
                print(f"❌ Dataset '{dataset_path}' non trovato nel file.")
                return

            # Carica le proiezioni
            data = f[dataset_path][()]  # shape = (num_angles, height, width)
            print(f"Caricando '{dataset_path}' shape={data.shape}")

            # Normalizzazione 0-1 per ridurre il rumore visivo
            data = (data - data.min()) / (data.max() - data.min())

            # In alternativa, puoi fare una media su più proiezioni per aumentare il contrasto
            # data = np.mean(data, axis=0, keepdims=True)  # mantiene dimensione 3D

            # Apri Napari
            viewer = napari.Viewer()
            viewer.add_image(data, name="projections")
            napari.run()

    except Exception as e:
        print(f"Errore aprendo {file_path}: {e}")

if __name__ == "__main__":
    view_projections(h5_file)
