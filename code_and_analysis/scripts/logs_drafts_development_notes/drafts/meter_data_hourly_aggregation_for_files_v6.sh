#!/bin/bash
#PBS -l select=2:ncpus=64:mem=1000gb:cpu_type=rome
#PBS -l walltime=4:00:00
#PBS -N meter_data_hourly_aggregation_for_files_v6

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
