#!/bin/bash
#PBS -l select=1:ncpus=12:mem=850gb
#PBS -l walltime=2:00:00
#PBS -N step5_joining_customer_meter_emissions_full

cd $PBS_O_WORKDIR
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module load tools/prod
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_4

python step5_joining_customer_meter_emissions_full.py

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="
