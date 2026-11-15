#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${INPUT_ROOT:?Set the verified legacy input root}"
: "${SELECTION_ROOT:?Set a fresh hardware benchmark output root}"
PYTHON="${PYTHON:-python}"
GPUS="${GPUS:-0,1}"
"$PYTHON" -c 'import sys; from scripts.select_lm_head_micro import gpu_inventory; print(gpu_inventory(sys.argv[1].split(",")))' "$GPUS"
CUDA_VISIBLE_DEVICES="${GPUS%%,*}" "$PYTHON" scripts/benchmark_lm_head.py \
  --input-root "$INPUT_ROOT" --run-root "$SELECTION_ROOT/statistics" --prepare-statistics
exec "$PYTHON" scripts/select_lm_head_micro.py --input-root "$INPUT_ROOT" \
  --selection-root "$SELECTION_ROOT" --devices "$GPUS" --benchmark-only \
  --statistics "$SELECTION_ROOT/statistics/K-benchmark.pt" "$@"
