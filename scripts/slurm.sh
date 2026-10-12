#!/usr/bin/env bash
#SBATCH --job-name=nanogpt-noise
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
# Partition/account/GPU type are supplied with sbatch arguments by the operator.
# Four GPUs run four independent experiment jobs. START_JOB..END_JOB is inclusive.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${START_JOB:?Export START_JOB before sbatch}"
: "${END_JOB:?Export END_JOB before sbatch}"
# Split the requested host threads across the four independent Python workers.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
bash scripts/run_4gpu.sh queue "$START_JOB" "$END_JOB"
