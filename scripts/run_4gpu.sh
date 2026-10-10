#!/usr/bin/env bash
# One node, four GPUs sharing each candidate; never four independent candidates.
set -euo pipefail

usage() {
  printf '%s\n' \
    'Usage: bash scripts/run_4gpu.sh smoke [scope optimizer noise_node]' \
    '       bash scripts/run_4gpu.sh node JOB_INDEX' \
    'Set DATA_DIR, ARTIFACT_DIR and RUN_DIR first (paths may contain spaces).' \
    'smoke defaults: full muon 10, eight steps; not a formal result.' \
    'node: JOB_INDEX=0..839; eight full-budget LR candidates, resumable.' \
    'Optional: PYTHON=python MICRO_BATCH=4 OMP_NUM_THREADS=1.' \
    'Allocate four GPUs first. With more visible GPUs, select four using CUDA_VISIBLE_DEVICES.'
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
  *) usage >&2; exit 2 ;;
esac

: "${DATA_DIR:?Set DATA_DIR to the directory with train.bin, val.bin and meta.json}"
: "${ARTIFACT_DIR:?Set ARTIFACT_DIR to the frozen initial.pt, windows.pt and identity.json directory}"
: "${RUN_DIR:?Set RUN_DIR to a writable experiment output root}"
# Resolve relative data/output paths against the caller's directory before cd.
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

"$PYTHON" -m nanogpt_noise.preflight --data "$DATA_DIR" \
  --artifacts "$ARTIFACT_DIR" --micro-batch "$MICRO_BATCH"

LAUNCH=("$PYTHON" -m torch.distributed.run --standalone --nnodes=1 \
  --nproc-per-node=4 --max-restarts=0)
COMMON=(--data "$DATA_DIR" --artifacts "$ARTIFACT_DIR" --micro-batch "$MICRO_BATCH")
if [[ "$MODE" == smoke ]]; then
  OUTPUT="$RUN_DIR/smoke/4gpu/$SCOPE-$OPTIMIZER-node$NOISE_NODE"
  printf 'Four-GPU SMOKE: %s (8 steps, not formal training)\n' "$OUTPUT"
  exec "${LAUNCH[@]}" -m nanogpt_noise.train "${COMMON[@]}" \
    --output "$OUTPUT" --scope "$SCOPE" --optimizer "$OPTIMIZER" \
    --node "$NOISE_NODE" --exponent 0 --smoke-steps 8
else
  printf 'Four-GPU FORMAL: job %s, eight full-budget candidates. Output: %s\n' "$JOB_INDEX" "$RUN_DIR"
  exec "${LAUNCH[@]}" -m nanogpt_noise.node "${COMMON[@]}" \
    --output "$RUN_DIR" --job-index "$JOB_INDEX"
fi
