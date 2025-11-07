#!/bin/bash
#PBS -l select=1:ncpus=64:mem=700gb
#PBS -l walltime=4:00:00
#PBS -N reducing_data_size_v2

cd $PBS_O_WORKDIR
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module load OpenMPI
module load Boost
module load GCC
module load ICU
module load XZ
module load zstd
module load zlib
module load bzip2
module load tools/prod
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_1

mpiexec -n 64 python reducing_data_size_v2.py

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="

