#!/bin/bash
#PBS -l select=1:ncpus=8:mem=300gb         
#PBS -l walltime=24:00:00                      
#PBS -N hitachi_meter_reading_retrieval_on_campus_v1           
#PBS -M daniel.kaupa24@imperial.ac.uk
#PBS -m abe

cd $PBS_O_WORKDIR

module load tools/prod
module load PostgreSQL/16.4-GCCcore-13.3.0
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_1

export PGPASSWORD="Iamdaniel00!"

which psql
psql --version

python hitachi_meter_reading_retrieval_on_campus_v1.py





echo "=== Test job finished on $(hostname) at $(date) ==="