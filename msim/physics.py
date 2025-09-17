import numpy as np
import cupy as cp
import cupyx.scipy.signal as cpxsignal
from scipy.interpolate import interp1d
import xraylib

def projection(volume_labels, lookup, voxel_size, config):
    """
    Simulate detector intensity from a rotated label volume—GPU version.
    
    Parameters:
        volume_labels : (nz, ny, nx) integer labels (NumPy array)
        lookup        : dict mapping label to {composition, density}
        voxel_size    : (dz, dy, dx) in microns (tuple of floats)
        config        : dict containing ENERGY_KEV, DETECTOR_DIST, PAD, ENABLE_*, etc.

    Returns:
        I_sim : (ny, nx) simulated detector intensity (NumPy array)
    """

    # --- 1) Compute δ, μ_abs, μ_scat on CPU ---
    nz, ny, nx = volume_labels.shape
    delta_cpu   = np.zeros_like(volume_labels, dtype='float32')
    mu_abs_cpu  = np.zeros_like(volume_labels, dtype='float32')
    mu_scat_cpu = np.zeros_like(volume_labels, dtype='float32')

    xraylib.XRayInit()
    energy = config["ENERGY_KEV"]
    for k, props in lookup.items():        
        mask    = (volume_labels == int(k))
        if not np.any(mask):
            continue
        
        composition = props.get('composition', {})
        density = props.get('density', 0.0)
        
        # Skip empty compositions (vacuum/air)
        if not composition or density == 0.0:
            continue
            
        formula = ''.join(f"{el}{amt}" for el, amt in composition.items())

        δ_val   = 1 - xraylib.Refractive_Index_Re(formula, energy, density)
        cs_tot  = xraylib.CS_Total_CP(formula, energy)
        cs_rayl = xraylib.CS_Rayl_CP(formula, energy)

        delta_cpu[  mask] = δ_val
        mu_abs_cpu[ mask] = density * (cs_tot - cs_rayl) * 100
        mu_scat_cpu[mask] = density * cs_rayl * 100

    # --- 2) Transfer to GPU and pad ---
    pad = config["PAD"]
    delta_p   = cp.pad(cp.asarray(delta_cpu),   ((0,0),(pad,pad),(pad,pad)), mode='edge')
    mu_abs_p  = cp.pad(cp.asarray(mu_abs_cpu),  ((0,0),(pad,pad),(pad,pad)), mode='edge')
    mu_scat_p = cp.pad(cp.asarray(mu_scat_cpu), ((0,0),(pad,pad),(pad,pad)), mode='edge')
    ny_p, nx_p = ny + 2*pad, nx + 2*pad

    # voxel sizes (m)
    dz, dy, dx = [v * 1e-6 for v in voxel_size]
    
    # Detector pixel size (user-defined, not voxel size!)
    detector_pixel_size = config.get("DETECTOR_PIXEL_SIZE", dy)  # Default to voxel size if not specified

    # --- 3) Build propagation kernels on GPU ---
    # wavenumber
    k0 = 2 * cp.pi / ((6.62607015e-34 * 2.99792458e8) / (energy*1e3*1.602176634e-19))
    kx = cp.fft.fftfreq(nx_p, detector_pixel_size) * 2 * cp.pi
    ky = cp.fft.fftfreq(ny_p, detector_pixel_size) * 2 * cp.pi
    KX, KY = cp.meshgrid(kx, ky)
    H_slice = cp.exp(-1j * (KX**2 + KY**2) * dz / (2*k0))
    H_det   = cp.exp(-1j * (KX**2 + KY**2) * config["DETECTOR_DIST"] / (2*k0))

    # --- 4) Build scattering PSF on CPU, then transfer ---
    r_e   = 2.8179403227e-15
    theta = np.linspace(0, 5e-3, 501)
    cos_t = np.cos(theta)
    dcs_r = r_e**2 * (1 + cos_t**2) / 2
    alpha = energy*1e3/511e3
    num   = 1 + cos_t**2
    den   = (1 + alpha*(1 - cos_t))**2
    dcs_c = (r_e**2/2) * num/den * (1 + alpha*(1 - cos_t)
            - alpha**2*(1 - cos_t)**2/(num*(1 + alpha*(1 - cos_t))))
    psf_r = (dcs_r + dcs_c) * 2 * np.pi * np.sin(theta)
    psf_r /= np.trapz(psf_r, theta)

    r_px = theta * config["DETECTOR_DIST"] / detector_pixel_size
    yy, xx = np.mgrid[-ny_p//2:ny_p//2, -nx_p//2:nx_p//2]
    rgrid = np.hypot(xx, yy)
    psf2d = interp1d(r_px, psf_r, bounds_error=False, fill_value=0)(rgrid)
    psf2d /= psf2d.sum()
    psf2d_gpu = cp.asarray(psf2d, dtype=cp.float32)

    # --- 5) Initialize wavefront on GPU ---
    Psi = cp.ones((ny_p, nx_p), dtype=cp.complex64)
    
    # Optional: Add very small random phase for numerical stability only
    if config.get("ADD_RANDOM_PHASE", False):
        rand_phase = cp.random.uniform(0, 2*cp.pi, (ny_p, nx_p), dtype=cp.float32)
        Psi *= cp.exp(1j * rand_phase * 1e-6)  # Much smaller random phase

    # --- 6) Propagate slice-by-slice ---
    for z in range(nz):
        # free-space to next slice
        Psi = cp.fft.ifft2(cp.fft.fft2(Psi) * H_slice)

        if config["ENABLE_PHASE"]:
            Psi *= cp.exp(1j * k0 * delta_p[z] * dz)
        if config["ENABLE_ABSORPTION"]:
            Psi *= cp.exp(-0.5 * mu_abs_p[z] * dz)
        if config["ENABLE_SCATTER"]:
            I    = cp.abs(Psi)**2
            p    = 1 - cp.exp(-mu_scat_p[z] * dz)
            I_sc = p * I
            I_un = (1 - p) * I
            # FFT-based convolution on GPU
            I_bl = cpxsignal.fftconvolve(I_sc, psf2d_gpu, mode='same')
            Psi  = cp.sqrt(cp.maximum(I_un + I_bl, 0)) * cp.exp(1j*cp.angle(Psi))

    # --- 7) Final propagation to detector ---
    Psi   = cp.fft.ifft2(cp.fft.fft2(Psi) * H_det)
    I_sim = cp.abs(Psi)**2
    
    # --- 8) Convert to photon counts and add noise ---
    I_sim = apply_photon_statistics(I_sim, config)

    # return cropped detector image back on CPU
    return cp.asnumpy(I_sim[pad:pad+ny, pad:pad+nx])

def apply_photon_statistics(intensity, config):
    """
    Convert normalized intensity to photon counts and add quantum noise.
    
    Args:
        intensity: (ny, nx) normalized intensity (0-1 scale from wave simulation)
        config: dict with photon parameters
    
    Returns:
        photon_counts: (ny, nx) array with photon statistics applied
    """
    # Get photon parameters from config
    incident_photons = config.get("INCIDENT_PHOTONS", 1e6)  # Photons per pixel
    detector_efficiency = config.get("DETECTOR_EFFICIENCY", 0.8)  # Quantum efficiency
    dark_current = config.get("DARK_CURRENT", 10)  # Dark counts per pixel
    readout_noise = config.get("READOUT_NOISE", 5)  # RMS electrons
    
    # Convert intensity to transmitted photon flux
    transmitted_photons = intensity * incident_photons
    
    # Apply detector quantum efficiency
    detected_photons = transmitted_photons * detector_efficiency
    
    # Add Poisson noise (photon shot noise)
    if config.get("ENABLE_PHOTON_NOISE", True):
        # Use Poisson statistics for photon counting
        detected_photons = cp.random.poisson(detected_photons).astype(cp.float32)
    
    # Add dark current (also Poisson distributed)
    if dark_current > 0:
        dark_counts = cp.random.poisson(dark_current, size=detected_photons.shape).astype(cp.float32)
        detected_photons += dark_counts
    
    # Add readout noise (Gaussian)
    if readout_noise > 0:
        readout_counts = cp.random.normal(0, readout_noise, size=detected_photons.shape).astype(cp.float32)
        detected_photons += readout_counts
    
    # Ensure non-negative counts
    detected_photons = cp.maximum(detected_photons, 0)
    
    return detected_photons

def calculate_dose_map_accurate(volume_labels, lookup, incident_photons, energy_kev, voxel_size):
    """
    Calculate absorbed dose map accounting for beam attenuation.
    
    Args:
        volume_labels: (nz, ny, nx) integer labels
        lookup: dict mapping label to {composition, density}
        incident_photons: Number of incident photons per pixel
        energy_kev: X-ray energy in keV
        voxel_size: (dz, dy, dx) in microns
    
    Returns:
        dose_map: (nz, ny, nx) absorbed dose in Gray per voxel
    """
    import xraylib
    xraylib.XRayInit()
    
    nz, ny, nx = volume_labels.shape
    dose_map = np.zeros((nz, ny, nx), dtype='float32')
    
    # Physical constants
    photon_energy_J = energy_kev * 1e3 * 1.602176634e-19
    voxel_thickness_cm = voxel_size[0] * 1e-4  # Convert µm to cm
    voxel_volume_cm3 = np.prod(voxel_size) * 1e-12  # Convert µm³ to cm³
    
    # Create material property maps
    mu_total_map = np.zeros((nz, ny, nx), dtype='float32')  # Total attenuation
    mu_en_map = np.zeros((nz, ny, nx), dtype='float32')     # Energy absorption
    density_map = np.zeros((nz, ny, nx), dtype='float32')   # Density
    
    for label_id, props in lookup.items():
        mask = (volume_labels == int(label_id))
        if not np.any(mask):
            continue
        
        composition = props.get('composition', {})
        density = props.get('density', 0.0)
        
        if not composition or density == 0.0:
            continue
        
        formula = ''.join(f"{el}{amt}" for el, amt in composition.items())
        
        try:
            # Mass attenuation coefficients
            mu_total = xraylib.CS_Total_CP(formula, energy_kev)  # cm²/g
            mu_en = xraylib.CS_Energy_CP(formula, energy_kev)    # cm²/g
            
            # Linear attenuation coefficients
            mu_total_map[mask] = mu_total * density  # cm⁻¹
            mu_en_map[mask] = mu_en * density        # cm⁻¹
            density_map[mask] = density              # g/cm³
            
        except Exception as e:
            print(f"Warning: Could not calculate properties for {label_id} ({formula}): {e}")
            continue
    
    # Calculate dose accounting for beam attenuation
    for y in range(ny):
        for x in range(nx):
            # Initial beam intensity
            beam_intensity = incident_photons * photon_energy_J  # J/cm²
            
            # Propagate beam through volume slice by slice
            for z in range(nz):
                if mu_total_map[z, y, x] > 0:
                    # Energy absorbed in this voxel
                    energy_absorbed_per_cm2 = beam_intensity * mu_en_map[z, y, x] * voxel_thickness_cm
                    
                    # Convert to energy per voxel volume
                    energy_absorbed_per_voxel = energy_absorbed_per_cm2  # J (for 1 cm² beam area)
                    
                    # Mass of material in this voxel
                    mass_per_voxel = density_map[z, y, x] * voxel_volume_cm3  # g
                    
                    if mass_per_voxel > 0:
                        # Dose = Energy / Mass
                        dose_gray = energy_absorbed_per_voxel / (mass_per_voxel * 1e-3)  # J/kg = Gy
                        dose_map[z, y, x] = dose_gray
                    
                    # Attenuate beam for next slice
                    attenuation = np.exp(-mu_total_map[z, y, x] * voxel_thickness_cm)
                    beam_intensity *= attenuation
    
    return dose_map

# Use accurate dose calculation
calculate_dose_map = calculate_dose_map_accurate

def calculate_total_dose_statistics(dose_map, volume_labels, lookup):
    """
    Calculate dose statistics for different materials in the phantom.
    
    Args:
        dose_map: (nz, ny, nx) dose in Gray per voxel
        volume_labels: (nz, ny, nx) material labels
        lookup: material properties dict
    
    Returns:
        dose_stats: dict with dose statistics per material
    """
    dose_stats = {}
    
    for label_id, props in lookup.items():
        mask = (volume_labels == int(label_id))
        if not np.any(mask):
            continue
        
        material_doses = dose_map[mask]
        material_doses = material_doses[material_doses > 0]  # Exclude zero doses
        
        if len(material_doses) > 0:
            dose_stats[label_id] = {
                'material_name': ''.join(f"{el}{amt}" for el, amt in props.get('composition', {}).items()),
                'mean_dose_gy': float(np.mean(material_doses)),
                'max_dose_gy': float(np.max(material_doses)),
                'min_dose_gy': float(np.min(material_doses)),
                'total_volume_um3': float(np.sum(mask) * np.prod([0.5, 0.5, 0.5])),  # Assuming voxel size
                'voxel_count': int(np.sum(mask))
            }
    
    return dose_stats