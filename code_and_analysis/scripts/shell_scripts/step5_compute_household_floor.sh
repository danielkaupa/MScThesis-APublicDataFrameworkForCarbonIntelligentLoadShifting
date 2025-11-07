#PBS -l select=1:ncpus=36:mem=1400gb
#PBS -l walltime=1:00:00
#PBS -N step5_compute_household_floor
set -euo pipefail

cd $PBS_O_WORKDIR
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module purge
module load tools/prod
module load GCC/14.2.0
module load miniforge/3
# (OpenMPI not needed)
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_4

export POLARS_MAX_THREADS=36 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) starting on $(hostname) at $(date) ==="
python -u step5_compute_household_floor.py
echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="
