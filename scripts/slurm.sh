#!/usr/bin/env bash
#SBATCH --job-name=nanogpt-noise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Partition/account/GPU type are supplied with sbatch arguments by the operator.
# Four GPUs run four independent experiment jobs from one named scope.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${SCOPE:?Export SCOPE as full, last1, last2, last3, last6 or last12 before sbatch}"
# Split the requested host threads across the four independent Python workers.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
bash scripts/run_4gpu.sh scope "$SCOPE"
