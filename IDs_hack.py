import numpy as np
import h5py as h5

from pathlib import Path
from tqdm import tqdm

dir_out =  "../data"

folder = Path(dir_out)
snap_files = list(folder.glob('snapshot_*.hdf5'))

for snap_file in tqdm(snap_files):
    with h5.File(snap_file, "a") as hf:
        for Type in [0, 1, 4]: #GAS, DM, STARS

            PartType = f"PartType{Type}"

            if PartType not in hf: #But STARS not always present
                continue

            IDs = hf[f"{PartType}/ParticleIDs"][()]
            childIDs = hf[f"{PartType}/ParticleChildIDsNumber"][()]

            # this is a bit-shift opertation which is super-duper fast. I left-shift IDs 32 bit and then add childIDs to the last 32 bits.
            # The methods is robust for 1024^3 runs as long as we are using 32bit for IDs.
            # TODO: Might not work for 64-bit ids. Though note that script won't complain in this case.
            
            new_IDs = (IDs.astype(np.uint64) << 32) | childIDs.astype(np.uint64)

            del hf[f"{PartType}/ParticleIDs"]
            del hf[f"{PartType}/ParticleChildIDsNumber"]
            del hf[f"{PartType}/ParticleIDGenerationNumber"]

            hf.create_dataset(f"{PartType}/ParticleIDs", data=new_IDs, chunks=True)

# How to invert the mapping

original_ParticleIDs = (new_IDs >> 32).astype(np.uint32) # right-shift by 32-bits
original_ChildIDs = (new_IDs & 0xFFFFFFFF).astype(np.uint32) # that is the hex for (2^32 - 1); The AND picks out the last 32 bits.
