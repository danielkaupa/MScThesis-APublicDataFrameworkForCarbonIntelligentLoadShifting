#!/bin/bash
#PBS -l select=1:ncpus=12:mem=400gb:cpu_type=rome             
#PBS -l walltime=16:00:00                      
#PBS -N hitachi_meter_reading_retrieval_off_campus_year_v1
#PBS -o error_and_output/%N.o%J
#PBS -e error_and_output/%N.e%J      

cd $PBS_O_WORKDIR

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_1

python hitachi_meter_reading_retrieval_off_campus_year_v1_2023.py
python hitachi_meter_reading_retrieval_off_campus_year_v1_2022.py
python hitachi_meter_reading_retrieval_off_campus_year_v1_2021.py                       

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="