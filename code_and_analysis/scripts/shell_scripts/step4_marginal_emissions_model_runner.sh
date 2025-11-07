#!/bin/bash
#PBS -l select=1:ncpus=32:mpiprocs=32:mem=800gb
#PBS -l walltime=2:00:00
#PBS -N step4_marginal_emissions_model_runner
set -euo pipefail

cd "$PBS_O_WORKDIR"
JOBNAME=${PBS_JOBNAME}
JOBID=${PBS_JOBID}
exec 1>${JOBNAME}.o${JOBID}
exec 2>${JOBNAME}.e${JOBID}

module purge

# (Optional) clear lmod cache once; LMOD_DISABLE_CACHE avoids future reuse
CACHE="$HOME/.cache/lmod"
if [ -d "$CACHE" ]; then
  ts=$(date +%Y%m%d-%H%M%S)
  mv "$CACHE" "$HOME/.cache/lmod_bak_${ts}"
fi
export LMOD_DISABLE_CACHE=1


module load tools/prod
# module load GCC/14.20
# module load OpenMPI
# module load XZ
# module load zlib
# module load bzip2
module load miniforge/3

eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate irpenv_4

# keep each rank single-threaded to avoid oversubscription
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# pick the script no matter where you submit from
if [[ -f "scripts/step4_marginal_emissions_model_runner.py" ]]; then
  SCRIPT_PATH="scripts/step4_marginal_emissions_model_runner.py"
elif [[ -f "step4_marginal_emissions_model_runner.py" ]]; then
  SCRIPT_PATH="step4_marginal_emissions_model_runner.py"
else
  echo "Cannot find step4_marginal_emissions_model_runner.py in '$PWD' or '$PWD/scripts'"; exit 1
fi

# ranks: prefer PBS_NP; fallback to nodefile line count; else 32
NP=${PBS_NP:-$( [ -f "$PBS_NODEFILE" ] && wc -l < "$PBS_NODEFILE" || echo 32 )}

echo "[DEBUG] PWD=$PWD"
echo "[DEBUG] Using SCRIPT_PATH=$SCRIPT_PATH"
echo "[DEBUG] NP=$NP"
echo "[DEBUG] Nodefile summary:"; (uniq -c "$PBS_NODEFILE" || true)
which mpirun; mpirun --version

mpirun --hostfile "$PBS_NODEFILE" -np "$NP" --bind-to core --map-by core \
  python -u "$SCRIPT_PATH"

echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="




# ────────────────────────────────────────────────────────────────────────────
# ARCHIVE
# ────────────────────────────────────────────────────────────────────────────

# Original
# ────────────────────────────────────────────────────────────────────────────
# #!/bin/bash
# #PBS -l select=1:ncpus=64:mpiprocs=64:mem=640gb
# #PBS -l walltime=18:00:00
# #PBS -N step4_reproducingR_analysis
# set -euo pipefail

# cd "$PBS_O_WORKDIR"
# JOBNAME=${PBS_JOBNAME}
# JOBID=${PBS_JOBID}
# exec 1>${JOBNAME}.o${JOBID}
# exec 2>${JOBNAME}.e${JOBID}

# module purge

# # (Optional) clear lmod cache once; LMOD_DISABLE_CACHE avoids future reuse
# CACHE="$HOME/.cache/lmod"
# if [ -d "$CACHE" ]; then
#   ts=$(date +%Y%m%d-%H%M%S)
#   mv "$CACHE" "$HOME/.cache/lmod_bak_${ts}"
# fi
# export LMOD_DISABLE_CACHE=1


# module load tools/prod
# module load OpenMPI
# module load GCC
# module load XZ
# module load zlib
# module load bzip2
# module load miniforge/3

# eval "$(~/miniforge3/bin/conda shell.bash hook)"
# conda activate irpenv_4

# # keep each rank single-threaded to avoid oversubscription
# export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# # pick the script no matter where you submit from
# if [[ -f "scripts/step4_reproducingR_analysis.py" ]]; then
#   SCRIPT_PATH="scripts/step4_reproducingR_analysis.py"
# elif [[ -f "step4_reproducingR_analysis.py" ]]; then
#   SCRIPT_PATH="step4_reproducingR_analysis.py"
# else
#   echo "Cannot find step4_reproducingR_analysis.py in '$PWD' or '$PWD/scripts'"; exit 1
# fi

# # ranks: prefer PBS_NP; fallback to nodefile line count; else 32
# NP=${PBS_NP:-$( [ -f "$PBS_NODEFILE" ] && wc -l < "$PBS_NODEFILE" || echo 32 )}

# echo "[DEBUG] PWD=$PWD"
# echo "[DEBUG] Using SCRIPT_PATH=$SCRIPT_PATH"
# echo "[DEBUG] NP=$NP"
# echo "[DEBUG] Nodefile summary:"; (uniq -c "$PBS_NODEFILE" || true)
# which mpirun; mpirun --version

# mpirun --hostfile "$PBS_NODEFILE" -np "$NP" --bind-to core --map-by core \
#   python -u "$SCRIPT_PATH"

# echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="


# RECURSION
# ────────────────────────────────────────────────────────────────────────────
# #!/bin/bash
# #PBS -l select=1:ncpus=40:mpiprocs=40:mem=900gb
# #PBS -l walltime=4:00:00
# #PBS -N step4_reproducingR_analysis
# set -euo pipefail

# cd "$PBS_O_WORKDIR"
# JOBNAME=${PBS_JOBNAME}
# JOBID=${PBS_JOBID}

# # capture stdout/stderr to files named like step4_reproducingR_analysis.o12345
# exec 1>${JOBNAME}.o${JOBID}
# exec 2>${JOBNAME}.e${JOBID}

# ############################
# # Self-resubmit parameters #
# ############################
# # If your Python touches this file when **everything** is done, we stop.
# DONE_FILE="${PBS_O_WORKDIR}/.step4_DONE"

# # Alternatively, if your Python prints this exact line, we stop.
# # Change if you prefer a different phrase.
# SENTINEL="${SENTINEL:-ALL_COMBINATIONS_EXHAUSTED}"

# # Prevent infinite resubmits
# COUNT_FILE="${PBS_O_WORKDIR}/.step4_resubmits"
# MAX_RESUBMITS="${MAX_RESUBMITS:-500}"

# # Path to this job script for requeueing
# JOB_SCRIPT="${PBS_O_WORKDIR}/step4_reproducingR_analysis_med_process.sh"

# # Read current resubmit count (0 if file not present)
# RESUBMITS=$( (cat "$COUNT_FILE" 2>/dev/null) || echo 0 )

# echo "[INFO] $(date) :: Starting job ${JOBNAME} (ID ${JOBID})"
# echo "[INFO] Resubmit count so far: ${RESUBMITS}/${MAX_RESUBMITS}"

# #############
# # Modules   #
# #############
# module purge

# # (Optional) clear lmod cache once; LMOD_DISABLE_CACHE avoids future reuse
# CACHE="$HOME/.cache/lmod"
# if [ -d "$CACHE" ]; then
#   ts=$(date +%Y%m%d-%H%M%S)
#   mv "$CACHE" "$HOME/.cache/lmod_bak_${ts}"
# fi
# export LMOD_DISABLE_CACHE=1

# module load tools/prod
# module load OpenMPI
# module load GCC
# module load XZ
# module load zlib
# module load bzip2
# module load miniforge/3

# eval "$(~/miniforge3/bin/conda shell.bash hook)"
# conda activate irpenv_4

# # keep each rank single-threaded to avoid oversubscription
# export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# # pick the script no matter where you submit from
# if [[ -f "scripts/step4_reproducingR_analysis.py" ]]; then
#   SCRIPT_PATH="scripts/step4_reproducingR_analysis.py"
# elif [[ -f "step4_reproducingR_analysis.py" ]]; then
#   SCRIPT_PATH="step4_reproducingR_analysis.py"
# else
#   echo "Cannot find step4_reproducingR_analysis.py in '$PWD' or '$PWD/scripts'"; exit 1
# fi

# # ranks: prefer PBS_NP; fallback to nodefile line count; else 32
# NP=${PBS_NP:-$( [ -f "$PBS_NODEFILE" ] && wc -l < "$PBS_NODEFILE" || echo 32 )}

# echo "[DEBUG] PWD=$PWD"
# echo "[DEBUG] Using SCRIPT_PATH=$SCRIPT_PATH"
# echo "[DEBUG] NP=$NP"
# echo "[DEBUG] Nodefile summary:"; (uniq -c "$PBS_NODEFILE" || true)
# which mpirun; mpirun --version

# #################
# # Run the work  #
# #################
# mpirun --hostfile "$PBS_NODEFILE" -np "$NP" --bind-to core --map-by core \
#   python -u "$SCRIPT_PATH"

# echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="

# ##########################################
# # Decide whether to stop or re-submit    #
# ##########################################

# stop_now=false

# # Condition 1: DONE flag exists
# if [[ -f "$DONE_FILE" ]]; then
#   echo "[INFO] Detected DONE file at '$DONE_FILE' — stopping."
#   stop_now=true
# fi

# # Condition 2: Sentinel string seen in stdout
# if [[ "$stop_now" = false ]]; then
#   if grep -q -- "$SENTINEL" "${JOBNAME}.o${JOBID}"; then
#     echo "[INFO] Found sentinel '$SENTINEL' in ${JOBNAME}.o${JOBID} — stopping."
#     stop_now=true
#   fi
# fi

# # Condition 3: Hit max resubmits
# if [[ "$stop_now" = false ]]; then
#   if [[ "$RESUBMITS" -ge "$MAX_RESUBMITS" ]]; then
#     echo "[WARN] Max resubmits (${MAX_RESUBMITS}) reached — stopping."
#     stop_now=true
#   fi
# fi

# # Stop or requeue
# if [[ "$stop_now" = true ]]; then
#   echo "[INFO] All done. No resubmission."
#   exit 0
# else
#   # Increment and persist resubmit count
#   NEXT=$((RESUBMITS + 1))
#   echo "$NEXT" > "$COUNT_FILE"

#   echo "[INFO] Re-submitting ${JOB_SCRIPT} (attempt ${NEXT}/${MAX_RESUBMITS})"
#   # Re-submit with the same job name; inherit environment by default
#   NEWID=$(qsub -N "$JOBNAME" "$JOB_SCRIPT")
#   echo "[INFO] Submitted new job id: ${NEWID}"
# fi

# LARGE MEMORY
# ────────────────────────────────────────────────────────────────────────────
# #!/bin/bash
# #PBS -l select=1:ncpus=64:mpiprocs=64:mem=1400gb
# #PBS -l walltime=2:00:00
# #PBS -N step4_reproducingR_analysis
# set -euo pipefail

# cd "$PBS_O_WORKDIR"
# JOBNAME=${PBS_JOBNAME}
# JOBID=${PBS_JOBID}
# exec 1>${JOBNAME}.o${JOBID}
# exec 2>${JOBNAME}.e${JOBID}

# module purge

# # (Optional) clear lmod cache once; LMOD_DISABLE_CACHE avoids future reuse
# CACHE="$HOME/.cache/lmod"
# if [ -d "$CACHE" ]; then
#   ts=$(date +%Y%m%d-%H%M%S)
#   mv "$CACHE" "$HOME/.cache/lmod_bak_${ts}"
# fi
# export LMOD_DISABLE_CACHE=1


# module load tools/prod
# module load OpenMPI
# module load GCC
# module load XZ
# module load zlib
# module load bzip2
# module load miniforge/3

# eval "$(~/miniforge3/bin/conda shell.bash hook)"
# conda activate irpenv_4

# # keep each rank single-threaded to avoid oversubscription
# export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# # pick the script no matter where you submit from
# if [[ -f "scripts/step4_reproducingR_analysis.py" ]]; then
#   SCRIPT_PATH="scripts/step4_reproducingR_analysis.py"
# elif [[ -f "step4_reproducingR_analysis.py" ]]; then
#   SCRIPT_PATH="step4_reproducingR_analysis.py"
# else
#   echo "Cannot find step4_reproducingR_analysis.py in '$PWD' or '$PWD/scripts'"; exit 1
# fi

# # ranks: prefer PBS_NP; fallback to nodefile line count; else 32
# NP=${PBS_NP:-$( [ -f "$PBS_NODEFILE" ] && wc -l < "$PBS_NODEFILE" || echo 32 )}

# echo "[DEBUG] PWD=$PWD"
# echo "[DEBUG] Using SCRIPT_PATH=$SCRIPT_PATH"
# echo "[DEBUG] NP=$NP"
# echo "[DEBUG] Nodefile summary:"; (uniq -c "$PBS_NODEFILE" || true)
# which mpirun; mpirun --version

# mpirun --hostfile "$PBS_NODEFILE" -np "$NP" --bind-to core --map-by core \
#   python -u "$SCRIPT_PATH"

# echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="


# LOW MEMORY & PROCESSING
# ────────────────────────────────────────────────────────────────────────────
# #!/bin/bash
# #PBS -l select=1:ncpus=6:mpiprocs=6:mem=360gb
# #PBS -l walltime=2:00:00
# #PBS -N step4_reproducingR_analysis
# set -euo pipefail

# cd "$PBS_O_WORKDIR"
# JOBNAME=${PBS_JOBNAME}
# JOBID=${PBS_JOBID}
# exec 1>${JOBNAME}.o${JOBID}
# exec 2>${JOBNAME}.e${JOBID}

# module purge

# # (Optional) clear lmod cache once; LMOD_DISABLE_CACHE avoids future reuse
# CACHE="$HOME/.cache/lmod"
# if [ -d "$CACHE" ]; then
#   ts=$(date +%Y%m%d-%H%M%S)
#   mv "$CACHE" "$HOME/.cache/lmod_bak_${ts}"
# fi
# export LMOD_DISABLE_CACHE=1


# module load tools/prod
# module load OpenMPI
# module load GCC
# module load XZ
# module load zlib
# module load bzip2
# module load miniforge/3

# eval "$(~/miniforge3/bin/conda shell.bash hook)"
# conda activate irpenv_4

# # keep each rank single-threaded to avoid oversubscription
# export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# # pick the script no matter where you submit from
# if [[ -f "scripts/step4_reproducingR_analysis.py" ]]; then
#   SCRIPT_PATH="scripts/step4_reproducingR_analysis.py"
# elif [[ -f "step4_reproducingR_analysis.py" ]]; then
#   SCRIPT_PATH="step4_reproducingR_analysis.py"
# else
#   echo "Cannot find step4_reproducingR_analysis.py in '$PWD' or '$PWD/scripts'"; exit 1
# fi

# # ranks: prefer PBS_NP; fallback to nodefile line count; else 32
# NP=${PBS_NP:-$( [ -f "$PBS_NODEFILE" ] && wc -l < "$PBS_NODEFILE" || echo 32 )}

# echo "[DEBUG] PWD=$PWD"
# echo "[DEBUG] Using SCRIPT_PATH=$SCRIPT_PATH"
# echo "[DEBUG] NP=$NP"
# echo "[DEBUG] Nodefile summary:"; (uniq -c "$PBS_NODEFILE" || true)
# which mpirun; mpirun --version

# mpirun --hostfile "$PBS_NODEFILE" -np "$NP" --bind-to core --map-by core \
#   python -u "$SCRIPT_PATH"

# echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="


# MISC
# ────────────────────────────────────────────────────────────────────────────
# #!/bin/bash
# #PBS -l select=1:ncpus=64:mpiprocs=32:mem=1400gb
# #PBS -l walltime=2:00:00
# #PBS -N step4_reproducingR_analysis
# set -euo pipefail

# cd "$PBS_O_WORKDIR"
# JOBNAME=${PBS_JOBNAME}
# JOBID=${PBS_JOBID}
# exec 1>${JOBNAME}.o${JOBID}
# exec 2>${JOBNAME}.e${JOBID}

# module purge

# # (Optional) clear lmod cache once; LMOD_DISABLE_CACHE avoids future reuse
# CACHE="$HOME/.cache/lmod"
# if [ -d "$CACHE" ]; then
#   ts=$(date +%Y%m%d-%H%M%S)
#   mv "$CACHE" "$HOME/.cache/lmod_bak_${ts}"
# fi
# export LMOD_DISABLE_CACHE=1


# module load tools/prod
# module load OpenMPI
# module load GCC
# module load XZ
# module load zlib
# module load bzip2
# module load miniforge/3

# eval "$(~/miniforge3/bin/conda shell.bash hook)"
# conda activate irpenv_4

# # keep each rank single-threaded to avoid oversubscription
# export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# # pick the script no matter where you submit from
# if [[ -f "scripts/step4_reproducingR_analysis.py" ]]; then
#   SCRIPT_PATH="scripts/step4_reproducingR_analysis.py"
# elif [[ -f "step4_reproducingR_analysis.py" ]]; then
#   SCRIPT_PATH="step4_reproducingR_analysis.py"
# else
#   echo "Cannot find step4_reproducingR_analysis.py in '$PWD' or '$PWD/scripts'"; exit 1
# fi

# # ranks: prefer PBS_NP; fallback to nodefile line count; else 32
# NP=${PBS_NP:-$( [ -f "$PBS_NODEFILE" ] && wc -l < "$PBS_NODEFILE" || echo 32 )}

# echo "[DEBUG] PWD=$PWD"
# echo "[DEBUG] Using SCRIPT_PATH=$SCRIPT_PATH"
# echo "[DEBUG] NP=$NP"
# echo "[DEBUG] Nodefile summary:"; (uniq -c "$PBS_NODEFILE" || true)
# which mpirun; mpirun --version

# mpirun --hostfile "$PBS_NODEFILE" -np "$NP" --bind-to core --map-by core \
#   python -u "$SCRIPT_PATH"

# echo "=== Job $PBS_JOBNAME (ID $PBS_JOBID) finished on $(hostname) at $(date) ==="
