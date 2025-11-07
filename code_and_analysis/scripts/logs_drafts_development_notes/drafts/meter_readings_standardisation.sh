#!/bin/bash
#PBS -l select=1:ncpus=64:mem=700gb
#PBS -l walltime=4:00:00
#PBS -N meter_readings_standardisation

cd $PBS_O_WORKDIR
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module purge

module load GCC/14.2.0

module load OpenMPI/5.0.7-GCC-14.2.0
module load Boost/1.88.0-GCC-14.2.0
module load zlib/1.3.1-GCCcore-14.2.0
module load XZ/5.6.3-GCCcore-14.2.0
module load ICU/76.1-GCCcore-14.2.0
module load bzip2/1.0.8-GCCcore-14.2.0
module load zstd/1.5.6-GCCcore-14.2.0
module load tools/prod
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_1

mpiexec -n 64 python meter_readings_standardisation.py

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="
