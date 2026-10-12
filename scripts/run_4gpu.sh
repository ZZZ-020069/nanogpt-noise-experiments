#!/usr/bin/env bash
# Four independent one-GPU workers; never parallelize candidates within one job.
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: bash scripts/run_4gpu.sh smoke [scope optimizer noise_node]' \
    '       bash scripts/run_4gpu.sh node JOB_INDEX' \
    '       bash scripts/run_4gpu.sh queue START_JOB END_JOB' \
    'Set DATA_DIR, ARTIFACT_DIR and RUN_DIR first (paths may contain spaces).' \
    'smoke: one selected GPU, eight steps, not a formal result.' \
    'node: one selected GPU, one resumable eight-candidate job.' \
    'queue: exactly four visible GPUs dynamically process the inclusive job interval.' \
    'Optional: GPU_SLOT=0 PYTHON=python MICRO_BATCH=4 OMP_NUM_THREADS=1.'
}

if [[ "${1:-}" == '--help' || "${1:-}" == '-h' ]]; then
  usage
  exit 0
fi
MODE="${1:-}"
case "$MODE" in
  smoke)
    if (( $# != 1 && $# != 4 )); then usage >&2; exit 2; fi
    SCOPE="${2:-full}"
    OPTIMIZER="${3:-muon}"
    NOISE_NODE="${4:-10}"
    case "$SCOPE" in full|last1|last2|last3|last6|last12) ;; *) usage >&2; exit 2;; esac
    case "$OPTIMIZER" in adamw|adam|sgd|muon) ;; *) usage >&2; exit 2;; esac
    if [[ ! "$NOISE_NODE" =~ ^[0-9]{1,2}$ ]] || (( 10#$NOISE_NODE > 34 )); then
      usage >&2; exit 2
    fi
    NOISE_NODE=$((10#$NOISE_NODE))
    ;;
  node)
    if (( $# != 2 )) || [[ ! "$2" =~ ^[0-9]{1,3}$ ]] || (( 10#$2 > 839 )); then
      usage >&2; exit 2
    fi
    JOB_INDEX=$((10#$2))
    ;;
  queue)
    if (( $# != 3 )) || [[ ! "$2" =~ ^[0-9]{1,3}$ ]] || \
       [[ ! "$3" =~ ^[0-9]{1,3}$ ]] || (( 10#$2 > 10#$3 )) || (( 10#$3 > 839 )); then
      usage >&2; exit 2
    fi
    START_JOB=$((10#$2))
    END_JOB=$((10#$3))
    ;;
  *) usage >&2; exit 2 ;;
esac

: "${DATA_DIR:?Set DATA_DIR to the directory with train.bin, val.bin and meta.json}"
: "${ARTIFACT_DIR:?Set ARTIFACT_DIR to the frozen initial.pt, windows.pt and identity.json directory}"
: "${RUN_DIR:?Set RUN_DIR to a writable experiment output root}"
for variable in DATA_DIR ARTIFACT_DIR RUN_DIR; do
  if [[ "${!variable}" != /* ]]; then
    printf -v "$variable" '%s/%s' "$PWD" "${!variable}"
  fi
done
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
MICRO_BATCH="${MICRO_BATCH:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1

# Preserve scheduler-provided physical IDs/UUIDs when isolating child processes.
VISIBLE_GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
IFS=',' read -r -a GPU_TOKENS <<< "$VISIBLE_GPU_LIST"
COMMON=(--data "$DATA_DIR" --artifacts "$ARTIFACT_DIR" --micro-batch "$MICRO_BATCH")

if [[ "$MODE" == queue ]]; then
  if (( ${#GPU_TOKENS[@]} != 4 )); then
    printf 'queue requires exactly four CUDA_VISIBLE_DEVICES tokens; got %s\n' \
      "${#GPU_TOKENS[@]}" >&2
    exit 2
  fi
  "$PYTHON" -m nanogpt_noise.preflight "${COMMON[@]}" \
    --expected-gpus 4 --training-world 1
  printf 'FOUR-GPU POOL: jobs %s..%s; one independent job per GPU. Output: %s\n' \
    "$START_JOB" "$END_JOB" "$RUN_DIR"
  exec "$PYTHON" -m nanogpt_noise.pool "${COMMON[@]}" --output "$RUN_DIR" \
    --start "$START_JOB" --end "$END_JOB" --gpus "$VISIBLE_GPU_LIST"
fi

GPU_SLOT="${GPU_SLOT:-0}"
if [[ ! "$GPU_SLOT" =~ ^[0-9]+$ ]] || (( GPU_SLOT >= ${#GPU_TOKENS[@]} )); then
  printf 'GPU_SLOT must select one of the %s CUDA_VISIBLE_DEVICES tokens\n' \
    "${#GPU_TOKENS[@]}" >&2
  exit 2
fi
SELECTED_GPU="${GPU_TOKENS[$GPU_SLOT]}"
printf 'Selected GPU slot %s (CUDA_VISIBLE_DEVICES=%s)\n' "$GPU_SLOT" "$SELECTED_GPU"
CUDA_VISIBLE_DEVICES="$SELECTED_GPU" "$PYTHON" -m nanogpt_noise.preflight \
  "${COMMON[@]}" --expected-gpus 1 --training-world 1

if [[ "$MODE" == smoke ]]; then
  OUTPUT="$RUN_DIR/smoke/1gpu/$SCOPE-$OPTIMIZER-node$NOISE_NODE"
  printf 'ONE-GPU SMOKE: %s (8 steps, not formal training)\n' "$OUTPUT"
  CUDA_VISIBLE_DEVICES="$SELECTED_GPU" exec "$PYTHON" -m nanogpt_noise.train \
    "${COMMON[@]}" --output "$OUTPUT" --scope "$SCOPE" \
    --optimizer "$OPTIMIZER" --node "$NOISE_NODE" --exponent 0 --smoke-steps 8
else
  printf 'ONE-GPU FORMAL: job %s, eight sequential full-budget candidates. Output: %s\n' \
    "$JOB_INDEX" "$RUN_DIR"
  CUDA_VISIBLE_DEVICES="$SELECTED_GPU" exec "$PYTHON" -m nanogpt_noise.node \
    "${COMMON[@]}" --output "$RUN_DIR" --job-index "$JOB_INDEX"
fi
