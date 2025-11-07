#!/bin/bash
#PBS -l select=1:ncpus=16:mpiprocs=12:mem=700gb:cpu_type=rome
#PBS -l walltime=4:00:00
#PBS -N step1_meter_readings_processing_split_by_city

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

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=1

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) started on $(hostname) at $(date) ==="

mpiexec -n 12 \
        -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS \
        -x VECLIB_MAXIMUM_THREADS -x RAYON_NUM_THREADS -x POLARS_MAX_THREADS \
        python step1_meter_readings_processing_split_by_city.py


echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="
