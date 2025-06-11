#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --array=0-9
#SBATCH --job-name="agglom-alamo"
#SBATCH --output="tests/Agglomeration/%A-%a.log"
#SBATCH --mail-user=mehrtens@iastate.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

GAMMA_VALUES=(1e10 9e9 8e9 7e9 6e9 5e9 4e9 3e9 2e9 1e9)
GAMMA=${GAMMA_VALUES[$SLURM_ARRAY_TASK_ID]}

echo "Starting job for gamma=${GAMMA} at $(date)"
echo "Array Job ID: ${SLURM_ARRAY_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"

module purge
module load openmpi
srun --mpi=pmix ./bin/alamo-2d-g++ tests/Agglomeration/input gamma=${GAMMA} plot_file=tests/Agglomeration/output-${GAMMA}

echo "Job completed at $(date)"
