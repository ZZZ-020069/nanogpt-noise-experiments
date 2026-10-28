#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
INPUT_ROOT="${INPUT_ROOT:-/data/260010171/inputs-v1.8}"
PILOT_OUT="${PILOT_OUT:-$REPO_ROOT/joint-pilot-$(date +%Y%m%d-%H%M%S)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "${PYTHON:-python}" scripts/joint_pilot.py \
  --data "${DATA_DIR:-$INPUT_ROOT/openwebtext_subset}" \
  --artifacts "${ARTIFACT_DIR:-$INPUT_ROOT/frozen-artifacts}" \
  --output "$PILOT_OUT" --gpus "${CUDA_VISIBLE_DEVICES:-0,1}" \
  --micro-batch "${MICRO_BATCH:-4}" "$@"
