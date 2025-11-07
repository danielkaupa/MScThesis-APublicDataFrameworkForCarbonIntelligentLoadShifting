#!/bin/bash
#PBS -l select=1:ncpus=24:mem=850gb
#PBS -l walltime=8:00:00
#PBS -N meter_readings_analysis

cd $PBS_O_WORKDIR
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module load tools/prod
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_1

python meter_readings_analysis.py \
  --base-dir ../.. \
  --meter-dir data/hitachi_copy/meter_primary_files \
  --out-dir data/outputs \
  --images-dir images \
  --files-2021 meter_readings_2021_20250714_2015.parquet \
  --files-2022 meter_readings_2022_20250714_2324.parquet \
  --files-2023 meter_readings_2023_20250714_2039.parquet \
  --files-2021-formatted meter_readings_2021_20250714_2015_formatted.parquet \
  --files-2022-formatted meter_readings_2022_20250714_2324_formatted.parquet \
  --files-2023-formatted meter_readings_2023_20250714_2039_formatted.parquet \
  --delhi-2021 meter_readings_delhi_2021_20250714_2015_formatted.parquet \
  --delhi-2022 meter_readings_delhi_2022_20250714_2324_formatted.parquet \
  --mumbai-2022 meter_readings_mumbai_2022_20250714_2324_formatted.parquet \
  --mumbai-2023 meter_readings_mumbai_2023_20250714_2039_formatted.parquet \
  --all-years-formatted meter_readings_all_years_20250714_formatted.parquet

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="