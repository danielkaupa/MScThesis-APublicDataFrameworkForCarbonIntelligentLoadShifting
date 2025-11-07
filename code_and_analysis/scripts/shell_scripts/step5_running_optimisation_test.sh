#!/bin/bash
#PBS -l select=1:ncpus=36:mpiprocs=3:mem=1300gb
#PBS -l walltime=6:30:00
#PBS -N step5_running_optimisation_test
set -euo pipefail

cd "$PBS_O_WORKDIR"
JOBNAME=${PBS_JOBNAME}; JOBID=${PBS_JOBID}
exec 1>${PBS_JOBNAME}.o${PBS_JOBID}
exec 2>${PBS_JOBNAME}.e${PBS_JOBID}
trap 'echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) failed on $(hostname) at $(date) ==="' ERR

module purge
module load tools/prod
module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_4

# single-thread everything; greedy is single-threaded
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export RAYON_NUM_THREADS=1
export POLARS_MAX_THREADS=1
export PYTHONUNBUFFERED=1

# no local pool for greedy (or set to 1)
export LOCAL_WORKERS=1

HOST=""; [[ -f "$PBS_NODEFILE" ]] && HOST="--hostfile $PBS_NODEFILE"

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) starting on $(hostname) at $(date) ==="

# 3 ranks × PE=1 → 3 cores in use (one per group); this is what actually speeds greedy
mpiexec --report-bindings \
  --bind-to core \
  --map-by slot:pe=1 \
  -n 3 $HOST \
  -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x VECLIB_MAXIMUM_THREADS \
  -x RAYON_NUM_THREADS -x POLARS_MAX_THREADS -x PYTHONUNBUFFERED -x LOCAL_WORKERS \
  python step5_running_optimisation_test.py

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="



# #PBS -l walltime=6:30:00
# #PBS -N step5_running_optimisation_test
# set -euo pipefail

# cd "$PBS_O_WORKDIR"
# JOBNAME=${PBS_JOBNAME}
# JOBID=${PBS_JOBID}
# exec 1>${PBS_JOBNAME}.o${PBS_JOBID}
# exec 2>${PBS_JOBNAME}.e${PBS_JOBID}

# trap 'echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) failed on $(hostname) at $(date) ==="' ERR

# module purge
# module load tools/prod
# # module load GCC/14.2.0
# # module load OpenMPI
# # Do NOT load system OpenMPI if your conda env provides mpi4py+openmpi
# module load miniforge/3
# eval "$(~/miniforge3/bin/conda shell.bash hook)"
# conda activate irpenv_4

# # One thread per library to avoid oversubscription (greedy is single-threaded)
# export OMP_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export VECLIB_MAXIMUM_THREADS=1
# export PYTHONUNBUFFERED=1

# # Match your internal pool size to the cores bound to each rank
# export LOCAL_WORKERS=12   # each rank gets 12 cores

# # If PBS provides a hostfile, use it (harmless on single node)
# HOST=""
# [[ -f "$PBS_NODEFILE" ]] && HOST="--hostfile $PBS_NODEFILE"

# echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) starting on $(hostname) at $(date) ==="

# # Place 2 ranks on the node, bind each rank to 12 distinct cores
# mpiexec --report-bindings \
#   --bind-to core \
#   --map-by slot:pe=12 \
#   -n 2 $HOST \
#   -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x VECLIB_MAXIMUM_THREADS \
#   -x PYTHONUNBUFFERED -x LOCAL_WORKERS \
#   python step5_running_optimisation_test.py

# echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="



# ------- ARCHIVE - OLD USES
# #!/bin/bash
# #PBS -l select=1:ncpus=6:mpiprocs=6:mem=800gb
# #PBS -l walltime=2:00:00
# #PBS -N step5_running_optimisation_test
# set -euo pipefail

# cd "$PBS_O_WORKDIR"
# JOBNAME=${PBS_JOBNAME}
# JOBID=${PBS_JOBID}
# exec 1>${PBS_JOBNAME}.o${PBS_JOBID}
# exec 2>${PBS_JOBNAME}.e${PBS_JOBID}

# trap 'echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) failed on $(hostname) at $(date) ==="' ERR

# module purge
# module load tools/prod
# module load GCC/14.2.0
# module load OpenMPI
# module load miniforge/3

# # one thread per rank
# #export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# #export PYTHONUNBUFFERED=1  # flush prints promptly

# eval "$(~/miniforge3/bin/conda shell.bash hook)"
# conda activate irpenv_4

# NP="${PBS_NP:-}"
# if [[ -z "$NP" && -f "$PBS_NODEFILE" ]]; then NP=$(wc -l < "$PBS_NODEFILE"); fi
# NP=${NP:-64}

# # Prefer PBS layout; use hostfile if present
# MAP="--bind-to core --map-by slot"
# HOST=""
# [[ -f "$PBS_NODEFILE" ]] && HOST="--hostfile $PBS_NODEFILE"

# echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) starting on $(hostname) at $(date); NP=$NP ==="

# mpiexec $MAP $HOST -n "$NP" python step5_running_optimisation_test.py

# echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="


# #  -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x VECLIB_MAXIMUM_THREADS -x PYTHONUNBUFFERED \


# #!/bin/bash
# #PBS -l select=1:ncpus=8:mpiprocs=6:mem=800gb
# #PBS -l walltime=8:00:00
# #PBS -N step5_running_optimisation_test
# set -euo pipefail

# cd "$PBS_O_WORKDIR"
# JOBNAME=${PBS_JOBNAME}
# JOBID=${PBS_JOBID}
# exec 1>${PBS_JOBNAME}.o${PBS_JOBID}
# exec 2>${PBS_JOBNAME}.e${PBS_JOBID}

# trap 'echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) failed on $(hostname) at $(date) ==="' ERR

# module purge
# module load tools/prod
# module load GCC/14.2.0
# module load OpenMPI
# module load miniforge/3

# # one thread per rank
# export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# export PYTHONUNBUFFERED=1  # flush prints promptly

# eval "$(~/miniforge3/bin/conda shell.bash hook)"
# conda activate irpenv_4

# NP="${PBS_NP:-}"
# if [[ -z "$NP" && -f "$PBS_NODEFILE" ]]; then NP=$(wc -l < "$PBS_NODEFILE"); fi
# NP=${NP:-64}

# # Prefer PBS layout; use hostfile if present
# MAP="--bind-to core --map-by slot"
# HOST=""
# [[ -f "$PBS_NODEFILE" ]] && HOST="--hostfile $PBS_NODEFILE"

# echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) starting on $(hostname) at $(date); NP=$NP ==="

# mpiexec $MAP $HOST -n "$NP" \
#   -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS -x MKL_NUM_THREADS -x VECLIB_MAXIMUM_THREADS -x PYTHONUNBUFFERED \
#   python step5_running_optimisation_test.py

# echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="
