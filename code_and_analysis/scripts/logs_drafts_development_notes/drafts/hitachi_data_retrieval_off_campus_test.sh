#!/bin/bash
#PBS -l select=1:ncpus=8:mem=32gb               
#PBS -l walltime=00:30:00                       
#PBS -N hitachi_data_retrieval_off_campus_test            
#PBS -M daniel.kaupa24@imperial.ac.uk
#PBS -m abe

cd $PBS_O_WORKDIR

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_1

python hitachi_data_retrieval_off_campus_test.py                         

echo "=== Test job finished on $(hostname) at $(date) ==="
