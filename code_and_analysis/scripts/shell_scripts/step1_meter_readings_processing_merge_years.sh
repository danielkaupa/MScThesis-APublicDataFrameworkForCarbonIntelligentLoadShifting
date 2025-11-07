#!/bin/bash
#PBS -l select=1:ncpus=24:mem=850gb
#PBS -l walltime=4:00:00
#PBS -N step1_meter_readings_processing_merge_years

cd $PBS_O_WORKDIR
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module load tools/prod
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_4

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) started on $(hostname) at $(date) ==="

python step1_meter_readings_processing_merge_years.py

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="
