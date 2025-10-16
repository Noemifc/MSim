import numpy as np
import json
import h5py
import z5py
from msim.geometry import simulate_projection_series
from msim.physics import calculate_dose_map, calculate_total_dose_statistics

class XRayScanner:
    """Simple interface for X-ray tomography and laminography with dose calculation."""
    
    def __init__(self, config_file="config.json"):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.volume = None
        self.lookup = None
        self.voxel_size = None
    
    def load_volume(self, volume_path, metadata_path, scale_key="0"):
        """Load volume from your zarr/n5 files."""
        # Load volume data
        try:
            f = z5py.File(volume_path, use_zarr_format=True)
            self.volume = f[scale_key][...]
        except:
            f = z5py.File(volume_path, use_zarr_format=False) 
            self.volume = f[scale_key]['labels'][...]
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        
        self.voxel_size = meta.get("voxel_size", [1.0, 1.0, 1.0])
        self.lookup = meta.get("lookup", meta)
        
        print(f"Loaded volume: {self.volume.shape}")
        print(f"Voxel size: {self.voxel_size} µm")
        print(f"Materials: {len(self.lookup)} types")
    
    def tomography_scan(self, angles_deg, output_file="tomography.h5", calculate_dose=False):
        """Run tomography scan with optional dose calculation."""
        if self.volume is None:
            raise ValueError("Load volume first")
        
        print(f"Running tomography: {len(angles_deg)} projections")
        
        projections = simulate_projection_series(
            self.volume, self.lookup, self.voxel_size,
            angles_deg, tilt_deg=0, config=self.config
        )
        
        dose_map = None
        dose_stats = None
        if calculate_dose:
            print("Calculating dose distribution...")
            dose_map = calculate_dose_map(
                self.volume, self.lookup, 
                self.config.get("INCIDENT_PHOTONS", 1e6),
                self.config.get("ENERGY_KEV", 23.0),
                self.voxel_size
            )
            dose_stats = calculate_total_dose_statistics(dose_map, self.volume, self.lookup)
            self._print_dose_summary(dose_stats)
        
        self._save_results(projections, angles_deg, output_file, tilt_deg=0, dose_map=dose_map)
        print(f"Saved to: {output_file}")
        return projections, dose_stats
    
    def laminography_scan(self, angles_deg, tilt_deg, output_file="laminography.h5", calculate_dose=False):
        """Run laminography scan with optional dose calculation."""
        if self.volume is None:
            raise ValueError("Load volume first")
        
        print(f"Running laminography: tilt={tilt_deg}°, {len(angles_deg)} projections")
        
        projections = simulate_projection_series(
            self.volume, self.lookup, self.voxel_size,
            angles_deg, tilt_deg=tilt_deg, config=self.config
        )
        
        dose_map = None
        dose_stats = None
        if calculate_dose:
            print("Calculating dose distribution...")
            dose_map = calculate_dose_map(
                self.volume, self.lookup,
                self.config.get("INCIDENT_PHOTONS", 1e6),
                self.config.get("ENERGY_KEV", 23.0),
                self.voxel_size
            )
            dose_stats = calculate_total_dose_statistics(dose_map, self.volume, self.lookup)
            self._print_dose_summary(dose_stats)
        
        self._save_results(projections, angles_deg, output_file, tilt_deg=tilt_deg, dose_map=dose_map)
        print(f"Saved to: {output_file}")
        return projections, dose_stats
    
    def single_projection(self, rotation_deg=0, tilt_deg=0):
        """Single projection."""
        projections = simulate_projection_series(
            self.volume, self.lookup, self.voxel_size,
            [rotation_deg], tilt_deg=tilt_deg, config=self.config
        )
        return projections[0]
    
    def calculate_dose_only(self):
        """Calculate dose distribution without projection simulation."""
        if self.volume is None:
            raise ValueError("Load volume first")
        
        print("Calculating dose distribution...")
        dose_map = calculate_dose_map(
            self.volume, self.lookup,
            self.config.get("INCIDENT_PHOTONS", 1e6),
            self.config.get("ENERGY_KEV", 23.0),
            self.voxel_size
        )
        dose_stats = calculate_total_dose_statistics(dose_map, self.volume, self.lookup)
        self._print_dose_summary(dose_stats)
        
        return dose_map, dose_stats
    
    def _print_dose_summary(self, dose_stats):
        """Print dose statistics summary."""
        print("\nDose Summary:")
        print("-" * 50)
        total_dose = 0
        for label, stats in dose_stats.items():
            material_name = stats['material_name'] or f"Material_{label}"
            print(f"{material_name}:")
            print(f"  Mean dose: {stats['mean_dose_gy']:.2e} Gy")
            print(f"  Max dose:  {stats['max_dose_gy']:.2e} Gy")
            print(f"  Volume:    {stats['total_volume_um3']:.1f} μm³")
            total_dose += stats['mean_dose_gy'] * stats['voxel_count']
        
        avg_dose = total_dose / np.sum(self.volume > 0) if np.sum(self.volume > 0) > 0 else 0
        print(f"\nAverage dose across phantom: {avg_dose:.2e} Gy")
        print("-" * 50)
    
    def _save_results(self, projections, angles, output_file, tilt_deg=0, dose_map=None):
        """Save projections and optional dose data to HDF5."""
        with h5py.File(output_file, 'w') as f:
            # Projection data
            exchange = f.create_group("exchange")
            exchange.create_dataset("data", data=projections.astype('float32'), compression='gzip')
            f.create_dataset("angles", data=np.array(angles, dtype='float32'))
            
            # Dose data (if calculated)
            if dose_map is not None:
                dose_group = f.create_group("dose")
                dose_group.create_dataset("dose_map", data=dose_map.astype('float32'), compression='gzip')
                dose_group.attrs['units'] = 'Gray'
                dose_group.attrs['description'] = 'Absorbed dose per voxel'
            
            # Metadata
            f.attrs['tilt_angle_deg'] = tilt_deg
            f.attrs['energy_kev'] = self.config.get("ENERGY_KEV", 23.0)
            f.attrs['incident_photons'] = self.config.get("INCIDENT_PHOTONS", 1e6)
            f.attrs['detector_distance_m'] = self.config.get("DETECTOR_DIST", 0.3)
            f.attrs['voxel_size_um'] = self.voxel_size

# Quick functions with dose calculation
def quick_tomography(volume_path, metadata_path, n_projections=180, output_file="tomo.h5", calculate_dose=False):
    """Quick tomography scan with optional dose calculation."""
    scanner = XRayScanner()
    scanner.load_volume(volume_path, metadata_path)
    angles = np.linspace(0, 180, n_projections)
    return scanner.tomography_scan(angles, output_file, calculate_dose=calculate_dose)

def quick_laminography(volume_path, metadata_path, tilt_deg=45, n_projections=360, output_file="lamino.h5", calculate_dose=False):
    """Quick laminography scan with optional dose calculation."""
    scanner = XRayScanner()
    scanner.load_volume(volume_path, metadata_path)
    angles = np.linspace(0, 360, n_projections)
    return scanner.laminography_scan(angles, tilt_deg, output_file, calculate_dose=calculate_dose)

def analyze_dose_only(volume_path, metadata_path, config_file="config.json"):
    """Calculate dose distribution without running simulation."""
    scanner = XRayScanner(config_file)
    scanner.load_volume(volume_path, metadata_path)
    return scanner.calculate_dose_only()
