import os
import json
import shutil
import z5py


def fill_zarr_meta(root_group, datasets, output_path, metadata_args, mode='w'):
    """
    Fill metadata for the Zarr multiscale datasets and include additional parameters.

    Parameters:
    - root_group (zarr.Group): The root Zarr group.
    - datasets (list): List of datasets with their metadata.
    - output_path (str): Path to save the metadata file.
    - metadata_args (dict): Metadata arguments for custom configurations.
    - mode (str): Mode for metadata handling. Default is 'w'.
    """
    multiscales = [{
        "version": "0.4",
        "name": "example",
        "axes": [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"}
        ],
        "datasets": datasets,
        "type": "gaussian",
        "metadata": {
            "method": "scipy.ndimage.zoom",
            "args": [True],
            "kwargs": {
                "anti_aliasing": True,
                "preserve_range": True
            }
        }
    }]

    # Update Zarr group attributes
    if mode == 'w':
        root_group.attrs.update({"multiscales": multiscales})

        # Save metadata as JSON
        metadata_file = os.path.join(output_path, 'multiscales.json')
        with open(metadata_file, 'w') as f:
            json.dump({"multiscales": multiscales}, f, indent=4)


def save_multiscale_zarr(
    data,
    codes,
    out_dir,
    voxel_size=(0.25, 0.25, 0.1),
    n_scales=3,
    base_chunk=(64, 64, 64),
    logger=None
):
    """
    Save a 3D volume as a multiscale Zarr file with Neuroglancer-compatible metadata.

    Parameters:
    - data: 3D NumPy array to save (e.g., tiled volume)
    - codes: dict mapping label names to integer codes (for lookup metadata)
    - out_dir: target directory for the Zarr dataset (e.g., 'chip_model.zarr')
    - voxel_size: tuple of 3 floats (z, y, x) voxel size in µm
    - n_scales: number of pyramid levels (default: 3)
    - base_chunk: chunk size to use for Zarr compression (default: (64, 64, 64))
    - logger: optional logger for output messages
    """

    # Delete existing output if present
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    # Open Zarr root
    f = z5py.File(out_dir, use_zarr_format=True)

    curr = data.copy()
    datasets = []

    for lvl in range(n_scales):
        path = str(lvl)

        if path in f:
            del f[path]

        chunks = tuple(min(c, s) for c, s in zip(base_chunk, curr.shape))

        f.create_dataset(
            name=path,
            data=curr,
            chunks=chunks,
            compression='raw'
        )

        scale = 2 ** lvl
        datasets.append({
            "path": path,
            "coordinateTransformations": [
                {"type": "scale",       "scale": [scale] * 3},
                {"type": "translation", "translation": [scale / 2 - 0.5] * 3}
            ]
        })

        curr = curr[::2, ::2, ::2]

    # Add metadata
    lookup = {v: {'alias': k} for k, v in codes.items()}
    f.attrs['lookup'] = lookup
    f.attrs['voxel_size'] = voxel_size

    multiscale_meta = {
        "version": "0.4",
        "axes": [
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"}
        ],
        "datasets": datasets,
        "type": "image",
        "metadata": {
            "voxel_size": voxel_size
        }
    }

    f.attrs['multiscales'] = [multiscale_meta]

    # Also write JSON file for HTTP-serving compatibility
    with open(os.path.join(out_dir, "multiscale.json"), "w") as fjson:
        json.dump([multiscale_meta], fjson, indent=2)

    if logger:
        logger.info(f"Saved Neuroglancer-ready multiscale Zarr → {out_dir}")
