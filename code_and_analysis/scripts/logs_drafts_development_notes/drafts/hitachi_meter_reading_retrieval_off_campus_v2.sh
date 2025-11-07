#!/bin/bash
#PBS -l select=1:ncpus=128:mem=900gb:cpu_type=rome             
#PBS -l walltime=24:00:00                      
#PBS -N hitachi_meter_Reading_retrieval_off_campus_v2            
#PBS -M daniel.kaupa24@imperial.ac.uk
#PBS -m abe

cd $PBS_O_WORKDIR

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_1

python hitachi_meter_reading_retrieval_off_campus_v2.py                         

echo "=== Test job finished on $(hostname) at $(date) ==="