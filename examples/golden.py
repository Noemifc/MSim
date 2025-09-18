from msim.interface import XRayScanner
import numpy as np

scanner = XRayScanner("config.json")
scanner.load_volume("phantom.zarr", "phantom.json")

#def golden angle 
golden_a =180(3-np.sqrt(5))/2 # deg
golden_a_rad = golden_a * np.pi / 180  # rad

# select projections
num_proj = 360    #select number of projections
theta_start = 30  # start angle deg


#golden angles array 
golden_angles_tomo = np.mod(theta_start + np.arange(num_proj) * golden_a, 180)     # generate golden-angle from theta_start  in mod 180 , function numpy.arange([start, ]stop, [step, ]dtype=None)
#golden_angles_tomo_sorted = np.sort( golden_angles_tomo)  # if you need to rearrange 

# Custom GOLDEN angle sequence
golden_angles_tomo = np.mod(theta_start + np.arange(num_proj) * golden_a, 180)  
projections, dose_stats = scanner.tomography_scan(
    golden_angles_tomo, 
     "golden_tomo_with_dose.h5", # PREDE QUESTO FILE O LO GENNERA? SE LO GENERA CAMBIA NOME IN "golden_tomo_with_dose.h5"
    calculate_dose=fALSE
)
