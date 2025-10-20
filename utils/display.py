import matplotlib.pyplot as plt

def visualize_2x2(I_orig, I_sim,
                  x_mm, y_mm, hprof, vprof,
                  *, cx_mm, cy_mm):
    fig, ax = plt.subplots(2,2, figsize=(10,8))

    # Original
    im0 = ax[0,0].imshow(
        I_orig, cmap='gray', origin='lower',
        extent=[x_mm[0],x_mm[-1],y_mm[0],y_mm[-1]]
    )
    ax[0,0].axhline(cy_mm, color='red', ls='--')
    ax[0,0].axvline(cx_mm, color='red', ls='--')
    ax[0,0].set(
        title="Original (Beer–Lambert)",
        xlabel="x (mm)", ylabel="y (mm)", aspect='equal'
    )
    plt.colorbar(im0, ax=ax[0,0], fraction=0.046, pad=0.04)

    # Simulated
    im1 = ax[0,1].imshow(
        I_sim, cmap='gray', origin='lower',
        extent=[x_mm[0],x_mm[-1],y_mm[0],y_mm[-1]]
    )
    ax[0,1].axhline(cy_mm, color='red', ls='--')
    ax[0,1].axvline(cx_mm, color='red', ls='--')
    ax[0,1].set(
        title="Simulated I_sim",
        xlabel="x (mm)", ylabel="y (mm)", aspect='equal'
    )
    plt.colorbar(im1, ax=ax[0,1], fraction=0.046, pad=0.04)

    # Horizontal profile
    ax[1,0].plot(x_mm, hprof, 'k-')
    ax[1,0].set(title="Horizontal profile",
                xlabel="x (mm)", ylabel="Intensity")
    ax[1,0].grid(True)

    # Vertical profile
    ax[1,1].plot(y_mm, vprof, 'k-')
    ax[1,1].set(title="Vertical profile",
                xlabel="y (mm)", ylabel="Intensity")
    ax[1,1].grid(True)

    plt.tight_layout()
    plt.show()
