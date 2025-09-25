#!/usr/bin/env python3
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Leggi lo stack originale
    stack = tiff.imread("BCI-DNI_brain.bfc.tif")

    # Crea stack per label map
    label_stack = np.zeros_like(stack, dtype=np.uint16)

    # Esempio di maschere per 3 aree
    label_stack[(stack > 150) & (stack <= 200)] = 10
    label_stack[(stack > 200) & (stack <= 230)] = 20
    label_stack[(stack > 230)] = 30

    # Salva il risultato
    tiff.imwrite("BCI-DNI_brain.bfc.tif", label_stack)

    # Visualizza il primo slice
    plt.imshow(label_stack[0], cmap='nipy_spectral')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()
