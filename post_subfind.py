import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

dir_out = "../data"

folder = Path(dir_out)
SubFind_files = list(folder.glob('snapshot_*.hdf5'))
GIZMO_files = list(folder.glob('bak-snapshot_*.hdf5'))

assert len(SubFind_files) == len(GIZMO_files)

for snap in tqdm(np.arange(len(SubFind_files)), total=len(SubFind_files), desc="Processing files"):
    with h5py.File(dir_out+f"/bak-snapshot_{snap:03d}.hdf5", 'r') as fsrc, h5py.File(dir_out+f"/snapshot_{snap:03d}.hdf5",'a') as fdst:
        for ptype in ['PartType0', 'PartType1', 'PartType4']:
            if ptype not in fsrc:
                continue  # STARS not always present!

            grp_src = fsrc[ptype]
            grp_dst = fdst.require_group(ptype)

            # Get the ParticleIDs for this PartType
            pid_gizmo = grp_src['ParticleIDs'][...]  # Unordered IDs from GIZMO
            pid_subfind = grp_dst['ParticleIDs'][...]  # Ordered IDs from SubFind

            sorted_idx = np.argsort(pid_gizmo)
            index_mask = sorted_idx[np.searchsorted(pid_gizmo[sorted_idx], pid_subfind)]

            # Sanity check
            assert np.all(pid_gizmo[index_mask] == pid_subfind), f"Indexing mismatch for {ptype}"
            
            # Loop through datasets in this PartType
            for key in grp_src.keys():
                if key in grp_dst:
                    print(f"Skipping existing dataset: {ptype}/{key}")
                    continue

                data = grp_src[key][...]
                reordered_data = data[index_mask]
                grp_dst.create_dataset(key, data=reordered_data)
