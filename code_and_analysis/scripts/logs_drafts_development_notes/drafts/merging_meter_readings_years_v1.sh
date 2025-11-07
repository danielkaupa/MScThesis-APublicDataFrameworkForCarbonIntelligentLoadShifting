#!/bin/bash
#PBS -l select=1:ncpus=32:mem=400gb:cpu_type=rome             
#PBS -l walltime=2:00:00                      
#PBS -N merging_meter_readings_years_v1.py

cd $PBS_O_WORKDIR
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module load tools/prod
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_1

python merging_meter_readings_years_v1.py                  

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="