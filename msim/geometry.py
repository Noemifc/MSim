import numpy as np
import cupy as cp
from msim.physics import projection
from msim.LSim_wrap import rotate_volume, build_quaternion

class GPUVolumeManager:
    """Manages GPU memory for volume rotations efficiently."""
    
    def __init__(self, volume_labels, lookup, voxel_size, config):
        self.volume_shape = volume_labels.shape
        self.lookup = lookup
        self.voxel_size = voxel_size
        self.config = config
        
        # Keep original volume on GPU
        self.volume_gpu = cp.asarray(volume_labels, dtype=cp.float32)
        self.rotated_gpu = cp.empty_like(self.volume_gpu)
        
        print(f"GPU Volume Manager initialized: {self.volume_shape}")
        print(f"GPU memory used: {self.volume_gpu.nbytes / 1e9:.2f} GB")
    
    def rotate_and_project(self, rotation_deg, tilt_deg=0):
        """Rotate volume on GPU and simulate projection."""
        # Build quaternion
        tilt_rad = np.deg2rad(tilt_deg)
        theta_rad = np.deg2rad(rotation_deg)
        quat = build_quaternion(tilt_rad, theta_rad)
        
        # Rotate volume (stays on GPU)
        # Note: You'll need to modify LSim_wrap to accept GPU arrays directly
        # For now, we do one CPU transfer per rotation
        volume_cpu = cp.asnumpy(self.volume_gpu)
        rotated_cpu = np.empty_like(volume_cpu)
        volume_contiguous = np.ascontiguousarray(volume_cpu, dtype=np.float32)
        rotated_contiguous = np.ascontiguousarray(rotated_cpu, dtype=np.float32)
        
        rotate_volume(volume_contiguous, rotated_contiguous, quat)
        rotated_int = rotated_contiguous.astype(np.int32)
        
        # Project (this happens on GPU in physics.projection)
        return projection(rotated_int, self.lookup, self.voxel_size, self.config)
    
    def cleanup(self):
        """Free GPU memory."""
        del self.volume_gpu
        del self.rotated_gpu
        cp.get_default_memory_pool().free_all_blocks()

def simulate_tomography_projection(volume_labels, lookup, voxel_size, rotation_deg, config):
    """Simulate tomography projection - memory efficient."""
    # For single projections, use simple approach
    quat = build_quaternion(0.0, np.deg2rad(rotation_deg))
    
    rotated = np.empty_like(volume_labels, dtype=volume_labels.dtype)
    volume_contiguous = np.ascontiguousarray(volume_labels, dtype=np.float32)
    rotated_contiguous = np.ascontiguousarray(rotated, dtype=np.float32)
    
    rotate_volume(volume_contiguous, rotated_contiguous, quat)
    rotated_int = rotated_contiguous.astype(np.int32)
    
    return projection(rotated_int, lookup, voxel_size, config)

def simulate_laminography_projection(volume_labels, lookup, voxel_size, rotation_deg, tilt_deg, config):
    """Simulate laminography projection - memory efficient."""
    # For single projections, use simple approach
    quat = build_quaternion(np.deg2rad(tilt_deg), np.deg2rad(rotation_deg))
    
    rotated = np.empty_like(volume_labels, dtype=volume_labels.dtype)
    volume_contiguous = np.ascontiguousarray(volume_labels, dtype=np.float32)
    rotated_contiguous = np.ascontiguousarray(rotated, dtype=np.float32)
    
    rotate_volume(volume_contiguous, rotated_contiguous, quat)
    rotated_int = rotated_contiguous.astype(np.int32)
    
    return projection(rotated_int, lookup, voxel_size, config)

def simulate_projection_series(volume_labels, lookup, voxel_size, angles_deg, tilt_deg, config):
    """
    Simulate series of projections - optimized for large volumes and many angles.
    """
    print(f"Starting projection series: {len(angles_deg)} angles, tilt={tilt_deg}°")
    print(f"Volume size: {volume_labels.shape}, Memory: {volume_labels.nbytes / 1e9:.2f} GB")
    
    # For many projections, use GPU manager to avoid repeated transfers
    if len(angles_deg) > 5:  # Use GPU manager for multiple projections
        gpu_manager = GPUVolumeManager(volume_labels, lookup, voxel_size, config)
        
        projections = []
        try:
            for i, angle in enumerate(angles_deg):
                print(f"Processing angle {i+1}/{len(angles_deg)}: {angle:.1f}°")
                
                proj = gpu_manager.rotate_and_project(angle, tilt_deg)
                projections.append(proj)
                
                # Free GPU memory periodically
                if i % 10 == 0:
                    cp.get_default_memory_pool().free_all_blocks()
        
        finally:
            gpu_manager.cleanup()
    
    else:  # For few projections, use simple approach
        projections = []
        for angle in angles_deg:
            if tilt_deg == 0:
                proj = simulate_tomography_projection(volume_labels, lookup, voxel_size, angle, config)
            else:
                proj = simulate_laminography_projection(volume_labels, lookup, voxel_size, angle, tilt_deg, config)
            
            projections.append(proj)
            print(f"Angle {angle:.1f}° done")
    
    return np.array(projections)

def check_gpu_memory():
    """Check available GPU memory before starting."""
    mempool = cp.get_default_memory_pool()
    total_bytes = mempool.total_bytes()
    used_bytes = mempool.used_bytes()
    free_bytes = total_bytes - used_bytes
    
    print(f"GPU Memory: {used_bytes/1e9:.2f} GB used, {free_bytes/1e9:.2f} GB free")
    return free_bytes

def estimate_memory_needed(volume_shape, n_projections):
    """Estimate GPU memory needed for simulation."""
    volume_bytes = np.prod(volume_shape) * 4  # float32
    # Need: original volume + rotated volume + material maps + projection buffers
    estimated_bytes = volume_bytes * 6  # Conservative estimate
    
    print(f"Estimated GPU memory needed: {estimated_bytes/1e9:.2f} GB")
    return estimated_bytes