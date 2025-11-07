#!/bin/bash
#PBS -l select=1:ncpus=5:mem=900gb
#PBS -l walltime=9:00:00
#PBS -N step4_marginal_emissions_pygam_grid

cd $PBS_O_WORKDIR
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module purge
module load tools/prod
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_4

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 POLARS_MAX_THREADS=1

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) started on $(hostname) at $(date) ==="

python step4_marginal_emissions_pygam_grid.py
# mpiexec --map-by ppr:10:node:pe=3 --bind-to core -n 10 \
#         -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS \
#         -x VECLIB_MAXIMUM_THREADS -x RAYON_NUM_THREADS -x POLARS_MAX_THREADS \


echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="
