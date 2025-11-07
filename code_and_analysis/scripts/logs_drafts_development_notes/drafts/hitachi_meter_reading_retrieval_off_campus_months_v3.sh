#!/bin/bash
#PBS -l select=1:ncpus=4:mem=600gb
#PBS -l walltime=12:00:00
#PBS -N hitachi_meter_reading_retrieval_off_campus_months_v3

cd $PBS_O_WORKDIR
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module load tools/prod
module load PostgreSQL/16.4-GCCcore-13.3.0
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_1

export PGPASSWORD="Iamdaniel00!"

which psql
psql --version

python hitachi_meter_reading_retrieval_off_campus_months_v3_2023.py
python hitachi_meter_reading_retrieval_off_campus_months_v3_2022.py
python hitachi_meter_reading_retrieval_off_campus_months_v3_2021.py

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="
