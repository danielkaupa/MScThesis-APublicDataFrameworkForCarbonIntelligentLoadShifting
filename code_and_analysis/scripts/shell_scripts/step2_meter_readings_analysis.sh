#!/bin/bash
#PBS -l select=1:ncpus=32:mpiprocs=10:mem=1800gb
#PBS -l walltime=8:00:00
#PBS -N step2_meter_readings_analysis

cd $PBS_O_WORKDIR
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module purge
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

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=3 POLARS_MAX_THREADS=3

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) started on $(hostname) at $(date) ==="

mpiexec --map-by ppr:10:node:pe=3 --bind-to core -n 10 \
        -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS \
        -x VECLIB_MAXIMUM_THREADS -x RAYON_NUM_THREADS -x POLARS_MAX_THREADS \
        python step2_meter_readings_analysis.py

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="
