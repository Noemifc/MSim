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
    parser.add_argument("--save_pngs", action='store_true', help="Salva alcune slice come PNG colorati")
    parser.add_argument("--step", type=int, default=10, help="Salta slice per salvare PNG (es. 10 = salva uno ogni 10 slice)")
    args = parser.parse_args()

    # Usa backend non grafico per matplotlib (evita errori su server senza GUI)
    import matplotlib
    matplotlib.use("Agg")

    # Legge lo stack originale
    stack = tiff.imread(args.input_file)
    print(f"Stack caricato: {stack.shape}, tipo: {stack.dtype}")

    # Crea stack vuoto per le etichette
    label_stack = np.zeros_like(stack, dtype=np.uint16)

    # --- DEFINIZIONE DELLE MASCHERE E VALORI ---
    masks_values = [
        ((stack > 150) & (stack <= 200), 10),  # area 1
        ((stack > 200) & (stack <= 230), 20),  # area 2
        ((stack > 230), 30),                    # area 3
    ]

    for mask, value in masks_values:
        label_stack[mask] = value

    # Salva lo stack etichettato come TIFF
    tiff.imwrite(args.output_file, label_stack)
    print(f"Label map salvata in: {os.path.abspath(args.output_file)}")

    # Salva alcune slice come PNG colorati se richiesto
    if args.save_pngs:
        png_dir = os.path.join(os.path.dirname(args.output_file), "slices_png")
        os.makedirs(png_dir, exist_ok=True)
        for i in range(0, label_stack.shape[0], args.step):
            png_path = os.path.join(png_dir, f"slice_{i}.png")
            plt.imsave(png_path, label_stack[i], cmap='nipy_spectral')
        print(f"Slice PNG salvati in: {png_dir}")

if __name__ == "__main__":
    main()
