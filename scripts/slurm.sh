#!/usr/bin/env bash
#SBATCH --job-name=nanogpt-noise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# Partition/account/GPU type are supplied with sbatch arguments by the operator.
# Start with a single job; explicitly opt into an array after hardware acceptance.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
bash scripts/run.sh "${SLURM_ARRAY_TASK_ID:-0}"
