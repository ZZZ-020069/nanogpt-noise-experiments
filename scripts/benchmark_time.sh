#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$REPO_ROOT"
INPUT_ROOT="${INPUT_ROOT:-/data/260010171/inputs-v1.7}"
BENCH_OUT="${BENCH_OUT:-$REPO_ROOT/timing-$(date +%Y%m%d-%H%M%S)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "${PYTHON:-python}" scripts/benchmark_time.py \
  --data "${DATA_DIR:-$INPUT_ROOT/openwebtext_subset}" \
  --artifacts "${ARTIFACT_DIR:-$INPUT_ROOT/frozen-artifacts}" \
  --output "$BENCH_OUT" --gpus "${CUDA_VISIBLE_DEVICES:-0,1}" \
  --steps "${BENCH_STEPS:-20}" --micro "${MICRO_BATCH:-4}" \
  --minutes "${BENCH_MINUTES:-30}" --scope "${SCOPE:-last1}"
