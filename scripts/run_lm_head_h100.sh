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
Actions: check plan prepare-k smoke smoke-all formal clean clean-all formal-all grid resume pause status report curvature-fit curvature-check
prepare-k = run initial curvature-fit and held-out curvature-check, then auto-fill K_STATISTICS in LH_ENV_FILE.
formal = clean (four optimizers, eight LR candidates each, 3815 steps each).
clean-all (or formal-all) = sequentially run G -> T -> K clean without manual chaining, then generate report.
Required: INPUT_ROOT (registered v1.8 inputs), LH_RUN_ROOT (fresh formal output).
Options: PYTHON=python, GPUS=0,1,2,3, OBJECTIVE=G|T|K (default G).
SwanLab defaults on: set SWANLAB_API_KEY in the task environment.
Default project: nanogpt-lm-head; NANOGPT_SWANLAB=0 explicitly disables uploads.
K, smoke-all, clean-all, and formal-all require formal approved K_STATISTICS.
resume requires LH_PHASE=clean|grid|clean-all. MICRO_BATCH must be unset or exactly 16.
Smoke never starts formal training. Grid is a separate explicit per-objective action.
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
if [[ "$action" == smoke-all || "$action" == clean-all || "$action" == formal-all || ( "$action" == resume && ( "${LH_PHASE:-}" == clean-all || "${LH_PHASE:-}" == formal-all ) ) ]]; then
  : "${K_STATISTICS:?$action requires formal K-approved.pt; G/T can run separately}"
fi
smoke_proof_guard() {
  if [[ -d "$LH_RUN_ROOT" ]]; then
    for objective in G T K; do
      if [[ ! -f "$LH_RUN_ROOT/smoke/$objective/passed.json" ]]; then
        printf 'Missing %s smoke proof (%s/smoke/%s/passed.json); run smoke-all first.\n' \
          "$objective" "$LH_RUN_ROOT" "$objective" >&2
        exit 2
      fi
    done
  fi
}
print_prepare_k_summary() {
  local fit_dir="$1" check_dir="$2"
  if [[ -f "$fit_dir/report.json" && -f "$check_dir/report.json" ]]; then
    "${PYTHON:-python}" - "$fit_dir/report.json" "$check_dir/report.json" "$check_dir/K-approved.pt" "${LH_ENV_FILE:-}" <<'PY'
import json, sys
from pathlib import Path

fit = json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
chk = json.loads(Path(sys.argv[2]).read_text(encoding='utf8'))
k_path = Path(sys.argv[3])
env_file = sys.argv[4]
print('\n=== PREPARE-K SUMMARY ===')
print(f"fit  : passed={fit.get('passed')} tokens={fit.get('tokens')} stability={fit.get('stability_passed')} numeric={fit.get('numeric_passed')} geometry={fit.get('geometry_passed')}")
print(f"check: passed={chk.get('passed')} tokens={chk.get('tokens')} trace_rel_err={chk.get('trace_relative_error')} compat={chk.get('compatibility', {}).get('passed')} fixed_grad={chk.get('fixed_gradient_checks_passed')}")
print(f"K-approved.pt: {k_path.resolve()} (exists={k_path.is_file()})")
if env_file and Path(env_file).is_file():
    lines = [l.strip() for l in Path(env_file).read_text(encoding='utf8').splitlines() if 'K_STATISTICS=' in l and not l.strip().startswith('#')]
    print(f".env ({env_file}): {lines[-1] if lines else 'K_STATISTICS not active'}")
PY
  fi
}
print_smoke_summary() {
  if [[ -f "$LH_RUN_ROOT/smoke/G/passed.json" && -f "$LH_RUN_ROOT/smoke/T/passed.json" && -f "$LH_RUN_ROOT/smoke/K/passed.json" ]]; then
    "${PYTHON:-python}" - "$LH_RUN_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
print('\n=== SMOKE-ALL SUMMARY ===')
for obj in ('G', 'T', 'K'):
    proof = json.loads((root / 'smoke' / obj / 'passed.json').read_text(encoding='utf8'))
    perfs = []
    for opt in ('sgd', 'adam', 'adamw', 'muon'):
        p = root / 'smoke' / obj / opt / 'performance.json'
        if p.is_file():
            d = json.loads(p.read_text(encoding='utf8'))
            perfs.append(f"{opt}:{d.get('wall_seconds', 0):.1f}s/{d.get('peak_allocated_bytes', 0)/(1024**3):.1f}GiB")
    print(f"{obj}: passed={proof.get('passed')} | {'  '.join(perfs)}")
print(f"SwanLab project: {__import__('os').environ.get('SWANLAB_PROJ_NAME', 'nanogpt-lm-head')} (verify G/T/K x 4 optimizers reached step 32)")
PY
  fi
}
print_clean_summary() {
  if [[ -f "$LH_RUN_ROOT/comparison.csv" ]]; then
    "${PYTHON:-python}" - "$LH_RUN_ROOT" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
print('\n=== CLEAN-ALL SUMMARY ===')
for obj in ('G', 'T', 'K'):
    s = root / obj / 'summary.json'
    done = json.loads(s.read_text(encoding='utf8')).get('completed_jobs') if s.is_file() else None
    print(f"{obj}: completed_jobs={done}")
print(f"comparison.csv ({root / 'comparison.csv'}):")
print((root / 'comparison.csv').read_text(encoding='utf8').strip())
PY
  fi
}
case "$action" in
  check|smoke|smoke-all|formal|clean|clean-all|formal-all|grid|resume)
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
  prepare-k)
    unset LH_CHECKPOINT
    export OBJECTIVE=G
    tokens="${LH_TOKENS:-32768}"
    blocks="${LH_BLOCKS:-5}"
    curv_root="${LH_CURVATURE_ROOT:-${LH_RUN_ROOT:-/data/260010171/results/lm-head-m16}-curvature}"
    fit_dir="${LH_FIT_DIR:-$curv_root/init-fit-$tokens}"
    check_dir="${LH_CURVATURE_DIR:-$curv_root/init-check-$tokens}"
    LH_TOKENS="$tokens" LH_BLOCKS="$blocks" LH_CURVATURE_DIR="$fit_dir" bash scripts/run_lm_head.sh curvature-fit
    LH_TOKENS="$tokens" LH_BLOCKS="$blocks" LH_FIT_DIR="$fit_dir" LH_CURVATURE_DIR="$check_dir" bash scripts/run_lm_head.sh curvature-check
    print_prepare_k_summary "$fit_dir" "$check_dir"
    exit 0 ;;
  smoke-all)
    for objective in G T K; do
      OBJECTIVE="$objective" bash scripts/run_lm_head.sh smoke
    done
    print_smoke_summary
    exit 0 ;;
  clean-all|formal-all)
    smoke_proof_guard
    for objective in G T K; do
      OBJECTIVE="$objective" bash scripts/run_lm_head.sh clean
    done
    bash scripts/run_lm_head.sh report
    print_clean_summary
    exit 0 ;;
  resume)
    if [[ "${LH_PHASE:-}" == clean-all || "${LH_PHASE:-}" == formal-all ]]; then
      smoke_proof_guard
      for objective in G T K; do
        OBJECTIVE="$objective" LH_PHASE=clean bash scripts/run_lm_head.sh resume
      done
      bash scripts/run_lm_head.sh report
      print_clean_summary
      exit 0
    fi ;;
  curvature-fit|curvature-check) export OBJECTIVE=G ;;
  check|plan|smoke|clean|grid|pause|status|report) ;;
  *) printf 'Unknown H100 action: %s\n' "$action" >&2; exit 2 ;;
esac
exec bash scripts/run_lm_head.sh "$action"