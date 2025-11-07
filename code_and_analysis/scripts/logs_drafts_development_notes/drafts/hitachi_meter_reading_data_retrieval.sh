#!/bin/bash
#PBS -l select=1:ncpus=36:mem=900gb:cpu_type=rome
#PBS -l walltime=24:00:00
#PBS -N hitachi_meter_reading_data_retrieval

cd $PBS_O_WORKDIR
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_1

python hitachi_meter_reading_data_retrieval.py

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="
