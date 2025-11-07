#!/bin/bash
#PBS -l select=1:ncpus=32:mem=500gb         
#PBS -l walltime=24:00:00                      
#PBS -N hitachi_meter_reading_retrieval_off_campus_v3            
#PBS -M daniel.kaupa24@imperial.ac.uk
#PBS -m abe

cd $PBS_O_WORKDIR

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_1

export PGPASSWORD="Iamdaniel00!"

python hitachi_meter_reading_retrieval_off_campus_v3.py                         

echo "=== Test job finished on $(hostname) at $(date) ==="