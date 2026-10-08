#!/usr/bin/env bash
set -euo pipefail
: "${DATA_DIR:?Set DATA_DIR to the directory containing train.bin, val.bin, meta.json}"
: "${ARTIFACT_DIR:?Set ARTIFACT_DIR to frozen artifacts}"
: "${RUN_DIR:?Set RUN_DIR to the experiment output root}"
JOB_INDEX="${1:?Pass a job index from 0 to 839}"
exec torchrun --standalone --nproc-per-node="${GPUS_PER_JOB:-1}" \
  -m nanogpt_noise.node --job-index "$JOB_INDEX" \
  --data "$DATA_DIR" --artifacts "$ARTIFACT_DIR" --output "$RUN_DIR" \
  --micro-batch "${MICRO_BATCH:-4}"
