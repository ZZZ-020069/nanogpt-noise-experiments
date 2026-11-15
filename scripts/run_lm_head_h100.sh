#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
action="${1:-help}"
if [[ $# -gt 1 ]]; then
  printf 'Use environment settings and one ACTION; extra arguments are not accepted.\n' >&2
  exit 2
fi
if [[ "$action" == help ]]; then
  cat <<'EOF'
H100 LM Head-only: final user-selected micro=16, accumulation=32, global batch=512x1024.
Usage: bash scripts/run_lm_head_h100.sh ACTION
Actions: check plan smoke smoke-all formal clean grid resume pause status report curvature-fit curvature-check
formal = clean (four optimizers, eight LR candidates each, 3815 steps each).
Required: INPUT_ROOT (registered v1.8 inputs), LH_RUN_ROOT (fresh formal output).
Options: PYTHON=python, GPUS=0,1,2,3, OBJECTIVE=G|T|K (default G).
SwanLab defaults on: set SWANLAB_API_KEY in the task environment.
Default project: nanogpt-lm-head; NANOGPT_SWANLAB=0 explicitly disables uploads.
K and smoke-all require formal approved K_STATISTICS.
resume requires LH_PHASE=clean|grid. MICRO_BATCH must be unset or exactly 16.
Smoke never starts formal training. Grid is a separate explicit action.
See LM_HEAD_RUN.md for the complete H100 commands.
EOF
  exit 0
fi
if [[ -n "${MICRO_BATCH:-}" && "$MICRO_BATCH" != 16 ]]; then
  printf 'H100 protocol is fixed at MICRO_BATCH=16; unset the stale value or set it to 16.\n' >&2
  exit 2
fi
export MICRO_BATCH=16
export GPUS="${GPUS:-0,1,2,3}"
export OBJECTIVE="${OBJECTIVE:-G}"
export NANOGPT_SWANLAB="${NANOGPT_SWANLAB:-1}"
export SWANLAB_PROJ_NAME="${SWANLAB_PROJ_NAME:-nanogpt-lm-head}"
if [[ "$action" == smoke-all ]]; then
  : "${K_STATISTICS:?smoke-all requires formal K-approved.pt; G/T smoke can run separately}"
fi
case "$action" in
  check|smoke|smoke-all|formal|clean|grid|resume)
    export RUN_BASE="${LH_RUN_ROOT:?Set LH_RUN_ROOT to a fresh output tree}"
    if [[ "$NANOGPT_SWANLAB" == 1 ]]; then
      : "${SWANLAB_API_KEY:?Set SWANLAB_API_KEY in the task environment for SwanLab}"
      "${PYTHON:-python}" -m nanogpt_noise.lm_head.telemetry
    elif [[ "$NANOGPT_SWANLAB" != 0 ]]; then
      printf 'NANOGPT_SWANLAB must be 0 or 1.\n' >&2
      exit 2
    fi
    ;;
esac
case "$action" in
  formal) action=clean ;;
  smoke-all)
    : "${K_STATISTICS:?smoke-all requires formal K-approved.pt; G/T smoke can run separately}"
    for objective in G T K; do
      OBJECTIVE="$objective" bash scripts/run_lm_head.sh smoke
    done
    exit 0 ;;
  curvature-fit|curvature-check) export OBJECTIVE=G ;;
  check|plan|smoke|clean|grid|resume|pause|status|report) ;;
  *) printf 'Unknown H100 action: %s\n' "$action" >&2; exit 2 ;;
esac
exec bash scripts/run_lm_head.sh "$action"
