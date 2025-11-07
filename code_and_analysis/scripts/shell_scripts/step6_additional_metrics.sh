#!/bin/bash
#PBS -l select=1:ncpus=8:mem=700gb
#PBS -l walltime=01:30:00
#PBS -N  step6_additional_metrics.py

set -euo pipefail
trap 'echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) failed on $(hostname) at $(date) ==="' ERR

cd "$PBS_O_WORKDIR"
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${PBS_JOBNAME}.o${PBS_JOBID}
exec 2>${PBS_JOBNAME}.e${PBS_JOBID}

module purge
module load tools/prod
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_4

# ----- Threads: use what PBS gave us (fallback to nproc) -----
NP="${PBS_NP:-${NCPUS:-$(nproc)}}"
export POLARS_MAX_THREADS="${NP}"
export RAYON_NUM_THREADS="${NP}"

# Prevent BLAS from oversubscribing Polars' threads
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONUNBUFFERED=1

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) starting on $(hostname) at $(date) ==="
python -u step6_additional_metrics.py
echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="
