#!/bin/bash
#SBATCH -p all
#SBATCH --exclude=worker003,worker088,worker075
######SBATCH -p AMD
#SBATCH --nodes=1
#SBATCH --ntasks=10
#SBATCH --cpus-per-task=1
#SBATCH --job-name=subfind
#SBATCH --time=02:30:00
#SBATCH --mem=50G

cd ${SLURM_SUBMIT_DIR}
ntasks=10

module purge
# module load openmpi hdf5-openmpi fftw
module load openmpi hdf5-openmpi fftw/2.1.5/intel/mpich

RUN_DIR=./
EXEC=Gadget4
PARAMFILE=input.params
OUTFILE=log.out
RESTART= #--restart

mpirun_command="mpirun -np ${ntasks}"
executable=${RUN_DIR}/${EXEC}
params=${PARAMFILE}

total_snapshots=$(find ../data -maxdepth 1 -type f -name "snapshot_*.hdf5" | wc -l)

LAST_SNAP=$((total_snapshots - 1))
echo "Found ${total_snapshots} snapshot HDF5 files" >>${OUTFILE}

source ~/.him/bin/activate

python ./IDs_hack.py >>${OUTFILE}

echo "IDs fixed!" >>${OUTFILE}
sleep 1m

for N in $(seq 0 $LAST_SNAP); do
  ${mpirun_command} ${executable} ${params} 3 $N >>${OUTFILE}
done

for N in $(seq 1 $LAST_SNAP); do
  ${mpirun_command} ${executable} ${params} 7 $N >>${OUTFILE}
done

${mpirun_command} ${executable} ${params} 8 $LAST_SNAP >>${OUTFILE}

echo "Subfind finished!" >>${OUTFILE}
sleep 1m

python ./post_subfind.py >>${OUTFILE}

echo "datasets indexed!" >>${OUTFILE}
