#!/bin/bash
#PBS -l select=1:ncpus=20:mpiprocs=16:mem=900gb:cpu_type=rome
#PBS -l walltime=6:00:00
#PBS -N step1_meter_readings_processing_split_to_quarterly

set -euo pipefail
trap 'echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) failed on $(hostname) at $(date) ==="' ERR

cd $PBS_O_WORKDIR
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module load tools/prod
module load GCC/14.2.0
module load OpenMPI
module load Boost
module load ICU
module load XZ
module load zstd
module load zlib
module load bzip2

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_4

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) started on $(hostname) at $(date) ==="

mpiexec -n 16 python step1_meter_readings_processing_split_to_quarterly.py

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="
