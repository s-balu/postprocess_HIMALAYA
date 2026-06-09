#!/bin/bash

## You wil need to tweak some of these for the specific cluster that you're using.

#### Hercules-CICA ########

#SBATCH --job-name=subfind
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=standard
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=48
#SBATCH --hint=nomultithread
#SBATCH --threads-per-core=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bsreedhar@us.es

#### Hercules-CICA ########

# Abort on error / unset var / failed pipe so a crashed FOF/SUBFIND run
# never silently feeds incomplete catalogues into modes 7/8.
# If `source activate` trips on set -u, drop the `u`: set -eo pipefail
set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

## Load the appropriate modules

### Hercules-CICA modules ###
module purge
module load OpenMPI/5.0.7-GCC-14.2.0 GSL/2.8-GCC-14.2.0 HDF5/1.14.6-gompi-2025a FFTW/3.3.10-GCC-14.2.0

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
### Hercules-CICA modules ###

RUN_DIR=./
EXEC=Gadget4
PARAMFILE=input.params
OUTFILE=log.out
NTASKS="${SLURM_NTASKS}"

executable="${RUN_DIR}/${EXEC}"

# Single helper function; TODO: srun instead of mpirun?
run() { mpirun -np "${NTASKS}" "${executable}" "${PARAMFILE}" "$@" >>"${OUTFILE}" 2>&1; }

# Fresh log per job.
: >"${OUTFILE}"
echo "=== Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date) ===" >>"${OUTFILE}"

# Count snapshots (single-file layout). For NumFilesPerSnapshot > 1 the snaps
# live in snapdir_XXX/ -- point the find at ../data/snapdir_* and count dirs.
total_snapshots=$(find ../data -maxdepth 1 -type f -name "snapshot_*.hdf5" | wc -l)
if ((total_snapshots == 0)); then
  echo "ERROR: no snapshot_*.hdf5 found in ../data -- check path / output layout" >>"${OUTFILE}"
  exit 1
fi

LAST_SNAP=$((total_snapshots - 1))
echo "Found ${total_snapshots} snapshot HDF5 files (0..${LAST_SNAP})" >>"${OUTFILE}"

# --- Back up the original snapshots BEFORE IDs_hack.py rewrites them in place ---
# --ignore-existing makes this rerun-safe: it never overwrites an existing backup
# with already-hacked files, and rsync's atomic rename means a partial first copy
# is safely resumed. Under set -e, a failed backup aborts before anything mutates
# ../data. NB: backup is keyed on filename -- if you regenerate the snapshots,
# clear or rename ${BACKUP_DIR} first. For multi-file output (snapdir_XXX/),
# copy the directories instead of the snapshot_*.hdf5 glob.
BACKUP_DIR=../snapshots_orig
mkdir -p "${BACKUP_DIR}"
echo ">>> backing up snapshots to ${BACKUP_DIR}" >>"${OUTFILE}"
rsync -a --ignore-existing ../data/snapshot_*.hdf5 "${BACKUP_DIR}/" >>"${OUTFILE}" 2>&1
echo "Backup complete ($(find "${BACKUP_DIR}" -maxdepth 1 -name 'snapshot_*.hdf5' | wc -l) files)" >>"${OUTFILE}"

## Import your pyenv or conda env here
source ~/.him/bin/activate

python ./IDs_hack.py >>"${OUTFILE}" 2>&1
echo "IDs fixed!" >>"${OUTFILE}"
sleep 7

# 3 = FOF/SUBFIND on every snapshot
for N in $(seq 0 "${LAST_SNAP}"); do
  echo ">>> FOF/SUBFIND snapshot ${N}" >>"${OUTFILE}"
  run 3 "${N}"
done

# 7 = descendant links (each snapshot -> its predecessor), so 1..LAST
for N in $(seq 1 "${LAST_SNAP}"); do
  echo ">>> descendants snapshot ${N}" >>"${OUTFILE}"
  run 7 "${N}"
done

# 8 = assemble merger trees, anchored at the final snapshot
echo ">>> make trees" >>"${OUTFILE}"
run 8 "${LAST_SNAP}"
echo "Subfind finished!" >>"${OUTFILE}"
sleep 7

python ./post_subfind.py >>"${OUTFILE}" 2>&1
echo "datasets indexed!" >>"${OUTFILE}"
