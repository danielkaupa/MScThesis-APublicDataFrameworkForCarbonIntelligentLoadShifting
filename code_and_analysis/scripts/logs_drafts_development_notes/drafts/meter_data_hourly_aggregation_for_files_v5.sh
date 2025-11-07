#!/bin/bash
#PBS -l select=1:ncpus=64:mem=900gb
#PBS -l walltime=8:00:00
#PBS -N meter_data_hourly_aggregation_for_files_v5

cd $PBS_O_WORKDIR
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module load tools/prod
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_1

python meter_data_hourly_aggregation_for_files_v5.py

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="
