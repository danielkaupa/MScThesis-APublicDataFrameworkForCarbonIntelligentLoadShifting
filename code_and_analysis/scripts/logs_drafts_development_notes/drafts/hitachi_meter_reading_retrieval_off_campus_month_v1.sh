#!/bin/bash
#PBS -l select=1:ncpus=12:mem=300gb:cpu_type=rome             
#PBS -l walltime=24:00:00                      
#PBS -N hitachi_meter_reading_retrieval_off_campus_month_v1
#PBS -o error_and_output/%N.o%J
#PBS -e error_and_output/%N.e%J      

cd $PBS_O_WORKDIR

module load tools/prod
module load PostgreSQL/16.4-GCCcore-13.3.0
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_1

export PGPASSWORD="Iamdaniel00!"

which psql
psql --version

python hitachi_meter_reading_retrieval_off_campus_month_v1.py

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="