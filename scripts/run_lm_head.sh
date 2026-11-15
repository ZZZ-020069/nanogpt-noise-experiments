#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python}"
action="${1:-help}"
if [[ "$action" == help ]]; then
  cat <<'EOF'
LM Head-only / LH-QP-20261114-draft2 (independent from last-k)
Usage: bash scripts/run_lm_head.sh ACTION
Actions: check smoke clean grid resume pause status report plan curvature-fit curvature-check
Required: INPUT_ROOT (registered immutable inputs), LH_RUN_ROOT (new output tree)
Options: OBJECTIVE=G|T|K, GPUS=0, MICRO_BATCH=16, PYTHON=python
Final H100 micro=16: use scripts/run_lm_head_h100.sh. G/T/K share one locked micro per root.
K requires K_STATISTICS=/path/to/K-approved.pt.
resume requires LH_PHASE=clean|grid. plan uses LH_PHASE=clean by default.
Curvature: LH_CURVATURE_DIR; check also LH_FIT_DIR.
Optional: LH_CHECKPOINT, LH_TOKENS=32768, LH_BLOCKS=5, LH_CHUNK_TOKENS=64.
Each action runs only the explicitly selected stage. No automatic stage chaining.
EOF
  exit 0
fi
: "${LH_RUN_ROOT:?Set LH_RUN_ROOT to a new independent output tree}"
case "$action" in
  pause|status|report)
    exec "$PYTHON" -m nanogpt_noise.lm_head "$action" --run-root "$LH_RUN_ROOT" ;;
esac
: "${INPUT_ROOT:?Set INPUT_ROOT to the registered v1.8 input bundle}"
OBJECTIVE="${OBJECTIVE:-G}"
GPUS="${GPUS:-0}"
MICRO_BATCH="${MICRO_BATCH:-16}"
args=(--input-root "$INPUT_ROOT" --run-root "$LH_RUN_ROOT"
      --objective "$OBJECTIVE" --micro-batch "$MICRO_BATCH")
if [[ "$OBJECTIVE" == K ]]; then
  : "${K_STATISTICS:?K requires an approved held-out statistics artifact}"
  args+=(--statistics "$K_STATISTICS")
fi
case "$action" in
  check) exec "$PYTHON" -m nanogpt_noise.lm_head check "${args[@]}" ;;
  smoke)
    export CUDA_VISIBLE_DEVICES="${GPUS%%,*}"
    exec "$PYTHON" -m nanogpt_noise.lm_head smoke "${args[@]}" ;;
  clean|grid)
    exec "$PYTHON" -m nanogpt_noise.lm_head run "${args[@]}" --phase "$action" --devices "$GPUS" ;;
  resume)
    : "${LH_PHASE:?Set LH_PHASE=clean or grid to resume the intended stage}"
    exec "$PYTHON" -m nanogpt_noise.lm_head run "${args[@]}" --phase "$LH_PHASE" --devices "$GPUS" --resume ;;
  plan)
    exec "$PYTHON" -m nanogpt_noise.lm_head run "${args[@]}" --phase "${LH_PHASE:-clean}" --dry-run ;;
  curvature-fit|curvature-check)
    : "${LH_CURVATURE_DIR:?Set LH_CURVATURE_DIR to a new diagnostic output directory}"
    more=()
    if [[ "$action" == curvature-check ]]; then
      : "${LH_FIT_DIR:?Set LH_FIT_DIR to the frozen fit directory}"
      more+=(--fit-dir "$LH_FIT_DIR")
    fi
    default_blocks=5
    if [[ -n "${LH_CHECKPOINT:-}" ]]; then
      more+=(--checkpoint "$LH_CHECKPOINT")
      default_blocks=2
    fi
    export CUDA_VISIBLE_DEVICES="${GPUS%%,*}"
    exec "$PYTHON" -m nanogpt_noise.lm_head curvature --input-root "$INPUT_ROOT" \
      --output "$LH_CURVATURE_DIR" --mode "${action#curvature-}" \
      --tokens "${LH_TOKENS:-32768}" --blocks "${LH_BLOCKS:-$default_blocks}" \
      --chunk-tokens "${LH_CHUNK_TOKENS:-64}" --micro-batch "$MICRO_BATCH" "${more[@]}" ;;
  *) printf 'Unknown action: %s\n' "$action" >&2; exit 2 ;;
esac
