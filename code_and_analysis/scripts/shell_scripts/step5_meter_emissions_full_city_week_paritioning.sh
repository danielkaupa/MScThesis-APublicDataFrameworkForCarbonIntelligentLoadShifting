#PBS -l select=1:ncpus=36:mpiprocs=6:mem=1800gb
#PBS -l walltime=4:00:00
#PBS -N step5_meter_emissions_full_city_week_paritioning
set -euo pipefail

cd $PBS_O_WORKDIR
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module purge
module load tools/prod
module load GCC/14.2.0
module load OpenMPI
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_4

# Threads per rank (make 'pe' match this)
export POLARS_MAX_THREADS=6
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1

# Ranks = mpiprocs
NP="${PBS_NP:-}"
if [[ -z "$NP" && -f "$PBS_NODEFILE" ]]; then NP=$(wc -l < "$PBS_NODEFILE"); fi
NP=${NP:-6}

MAP="--bind-to core --map-by slot:pe=${OMP_NUM_THREADS}"   # pe MUST equal threads/rank (6)
HOST=""
[[ -f "$PBS_NODEFILE" ]] && HOST="--hostfile $PBS_NODEFILE"

mpiexec $MAP $HOST -n "$NP" \
  -x POLARS_MAX_THREADS -x OMP_NUM_THREADS \
  -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x VECLIB_MAXIMUM_THREADS \
  -x PYTHONUNBUFFERED \
  python step5_meter_emissions_full_city_week_paritioning.py
