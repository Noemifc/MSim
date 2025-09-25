#!/usr/bin/env python3
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Crea una label map da uno stack cerebrale TIFF.")
    parser.add_argument("input_file", type=str, help="Percorso completo del file TIFF di input")
    parser.add_argument("--output_file", type=str, default="label_stack.tif", help="Percorso del file TIFF di output")
    args = parser.parse_args()

    # Legge lo stack originale
    stack = tiff.imread(args.input_file)
    print(f"Stack caricato: {stack.shape}, tipo: {stack.dtype}")

    # Crea stack vuoto per le etichette
    label_stack = np.zeros_like(stack, dtype=np.uint16)

    # --- DEFINIZIONE DELLE MASCHERE E VALORI ---
    # Modifica questi intervalli in base al tuo cervello
    masks_values = [
        ((stack > 150) & (stack <= 200), 10),  # area 1
        ((stack > 200) & (stack <= 230), 20),  # area 2
        ((stack > 230), 30),                    # area 3
    ]

    for mask, value in masks_values:
        label_stack[mask] = value

    # Salva lo stack etichettato
    tiff.imwrite(args.output_file, label_stack)
    print(f"Label map salvata in: {os.path.abspath(args.output_file)}")

    # Visualizza il primo slice
    plt.imshow(label_stack[0], cmap='nipy_spectral')
    plt.colorbar()
    plt.title("Slice 0 - Label Map")
    plt.show()

if __name__ == "__main__":
    main()
