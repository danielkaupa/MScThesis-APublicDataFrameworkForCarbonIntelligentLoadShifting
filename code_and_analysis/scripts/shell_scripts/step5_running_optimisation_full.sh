#!/bin/bash
#PBS -l select=1:ncpus=128:mpiprocs=82:mem=2600gb
#PBS -l walltime=4:00:00
#PBS -N step5_running_optimisation_full

set -euo pipefail
trap 'echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) failed on $(hostname) at $(date) ==="' ERR

cd "$PBS_O_WORKDIR"
JOBNAME=${PBS_JOBNAME}; JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module purge
module load tools/prod
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_4

# single-thread per rank
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAYON_NUM_THREADS=1
export POLARS_MAX_THREADS=1
export PYTHONUNBUFFERED=1

# runner knobs
export GROUP_TARGET_MB=256     # ~256MB micro-batches
export FLUSH_EVERY=15          # write every 15 batches
export WRITE_MOVES=1           # write moves parquet

# absolute shards dir (your correct path)
SHARDS_DIR="/rds/general/user/dbk24/home/irp-dbk24/code_and_analysis/data/optimisation_development/city_week_shards"
shopt -s nullglob
shards=( "$SHARDS_DIR"/*.parquet )
TOTAL=${#shards[@]}
(( TOTAL == 0 )) && { echo "No shards in $SHARDS_DIR"; exit 1; }

# script to run (must be in $PBS_O_WORKDIR)
SCRIPT="$PBS_O_WORKDIR/step5_running_optimisation_full.py"
if [[ ! -f "$SCRIPT" ]]; then
  echo "Cannot find $SCRIPT"
  echo "Python files in $PBS_O_WORKDIR:"; ls -1 "$PBS_O_WORKDIR"/*.py || true
  exit 1
fi

# derive ranks from PBS, cap by shard count
if [[ -n "${PBS_NODEFILE:-}" && -f "$PBS_NODEFILE" ]]; then
  CORES=$(wc -l < "$PBS_NODEFILE")
else
  CORES=${PBS_NP:-}; CORES=${CORES:-128}
fi
NP=$CORES
(( NP > TOTAL )) && NP=$TOTAL
(( NP < 1 )) && NP=1

HOST=""; [[ -f "$PBS_NODEFILE" ]] && HOST="--hostfile $PBS_NODEFILE"
MAP="--bind-to core --map-by slot:pe=1"

echo "=== $JOBNAME ($JOBID) starting on $(hostname) at $(date)"
echo "Using SHARDS_DIR=$SHARDS_DIR"
echo "TOTAL shards=$TOTAL, CORES=$CORES, NP (ranks)=$NP"

mpiexec --report-bindings $MAP $HOST -n "$NP" \
  -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS \
  -x VECLIB_MAXIMUM_THREADS -x RAYON_NUM_THREADS -x POLARS_MAX_THREADS \
  -x PYTHONUNBUFFERED -x GROUP_TARGET_MB -x FLUSH_EVERY -x WRITE_MOVES \
  python -u "$SCRIPT" --solver highs --policy p2

echo "=== $JOBNAME ($JOBID) finished at $(date)"


# RUN EXAMPLES
# # LP (HiGHS) with stricter policy 2
# mpiexec -n 3 python step5_run_optimisation_from_shards_min.py --solver highs --policy 2

# Greedy with relaxed policy 1
# mpiexec -n 3 python step5_run_optimisation_from_shards_min.py --solver greedy --policy p1

# Continuous SLSQP with policy 2
# mpiexec -n 3 python step5_run_optimisation_from_shards_min.py --solver slsqp --policy policy_2


# ARCHIVE SCRIPT

# #!/bin/bash
# #PBS -l select=1:ncpus=128:mpiprocs=82:mem=2400gb
# #PBS -l walltime=10:00:00
# #PBS -N step5_running_optimisation_full

# set -euo pipefail
# trap 'echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) failed on $(hostname) at $(date) ==="' ERR

# cd "$PBS_O_WORKDIR"
# JOBNAME=${PBS_JOBNAME}
# JOBID=${PBS_JOBID}
# exec 1>${PBS_JOBNAME}.o${PBS_JOBID}
# exec 2>${PBS_JOBNAME}.e${PBS_JOBID}

# module purge
# module load tools/prod
# module load miniforge/3
# eval "$(~/miniforge3/bin/conda shell.bash hook)"
# conda activate irpenv_4

# # one thread per rank
# export OMP_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export VECLIB_MAXIMUM_THREADS=1
# export RAYON_NUM_THREADS=1
# export POLARS_MAX_THREADS=1
# export PYTHONUNBUFFERED=1

# # runner knobs
# export GROUP_TARGET_MB=256     # ~256MB micro-batches
# export FLUSH_EVERY=15          # write every 15 batches
# export WRITE_MOVES=1

# NP="${PBS_NP:-}"
# if [[ -z "$NP" && -f "$PBS_NODEFILE" ]]; then NP=$(wc -l < "$PBS_NODEFILE"); fi
# # don’t overshoot shard count
# TOTAL=$(ls -1 optimisation_development/city_week_shards/*.parquet | wc -l)
# if [[ "$NP" -gt "$TOTAL" ]]; then NP="$TOTAL"; fi

# MAP="--bind-to core --map-by slot:pe=1"
# HOST=""
# [[ -f "$PBS_NODEFILE" ]] && HOST="--hostfile $PBS_NODEFILE"

# echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) starting on $(hostname) at $(date); NP=$NP ==="
# mpiexec $MAP $HOST -n "$NP" \
#   -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS \
#   -x VECLIB_MAXIMUM_THREADS -x RAYON_NUM_THREADS -x POLARS_MAX_THREADS \
#   -x PYTHONUNBUFFERED -x GROUP_TARGET_MB -x FLUSH_EVERY \
#   -x WRITE_MOVES -x EMIT_OPTIMISED \
#   python step5_running_optimisation_full.py --solver greedy --policy p2

# echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="
