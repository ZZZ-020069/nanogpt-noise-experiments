#!/usr/bin/env bash
# Operator-facing wrapper for the confirmed H100 server layout.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$REPO_ROOT"

INPUT_ROOT="${INPUT_ROOT:-/data/260010171/inputs-v1.8}"
DATA_DIR="${DATA_DIR:-$INPUT_ROOT/openwebtext_subset}"
ARTIFACT_DIR="${ARTIFACT_DIR:-$INPUT_ROOT/frozen-artifacts}"
RUN_BASE="${RUN_BASE:-/data/260010171/results/nanogpt-v1.8}"
PYTHON="${PYTHON:-python}"
GIT_BIN="${GIT_BIN:-git}"
SMOKE_SCOPE="${2:-last1}"
SMOKE_STATUS="$RUN_BASE/smoke/1gpu/$SMOKE_SCOPE-muon-node10/status.json"
SMOKE_COMMIT="$RUN_BASE/smoke/$SMOKE_SCOPE/operator_commit.txt"
SMOKE_INPUT="$RUN_BASE/smoke/$SMOKE_SCOPE/input_id.txt"
export DATA_DIR ARTIFACT_DIR PYTHON

usage() {
  printf '%s\n' \
    'Usage: bash scripts/h100.sh check' \
    '       bash scripts/h100.sh smoke [SCOPE]' \
    '       bash scripts/h100.sh scope SCOPE' \
    '       bash scripts/h100.sh finalize SCOPE' \
    '       bash scripts/h100.sh status [SCOPE]' \
    '' \
    'SCOPE: full, last1, last2, last3, last6, or last12.' \
    'Defaults:' \
    '  INPUT_ROOT=/data/260010171/inputs-v1.8' \
    '  RUN_BASE=/data/260010171/results/nanogpt-v1.8' \
    '' \
    'smoke defaults to last1, tests all four optimizers on one H100; scope requires four GPUs.' \
    'Activate the intended Python/CUDA environment before using this script.'
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 2
}

configure_git_safe_directory() {
  local index="${GIT_CONFIG_COUNT:-0}"
  local key_name value_name
  [[ "$index" =~ ^[0-9]+$ ]] || die 'GIT_CONFIG_COUNT must be a non-negative integer'
  key_name="GIT_CONFIG_KEY_${index}"
  value_name="GIT_CONFIG_VALUE_${index}"
  export "$key_name=safe.directory"
  export "$value_name=$REPO_ROOT"
  export GIT_CONFIG_COUNT=$((index + 1))
}

# The platform can mount a repository owned by a different host user. Trust only
# this script-derived repository path, for this process tree, without modifying
# the runtime user's global Git configuration.
configure_git_safe_directory

validate_scope() {
  case "${1:-}" in
    full|last1|last2|last3|last6|last12) ;;
    *) die 'scope must be one of: full last1 last2 last3 last6 last12' ;;
  esac
}

check_code() {
  local dirty
  dirty="$("$GIT_BIN" status --porcelain --untracked-files=normal)"
  if [[ -n "$dirty" ]]; then
    printf '%s\n' "$dirty" >&2
    die 'code worktree is not clean; do not run a formal experiment from modified files'
  fi
  CODE_COMMIT="$("$GIT_BIN" rev-parse HEAD)"
  export CODE_COMMIT
  printf 'Code commit: %s\n' "$CODE_COMMIT"
}

check_files() {
  local missing=0
  local path
  for path in \
    "$DATA_DIR/train.bin" \
    "$DATA_DIR/val.bin" \
    "$DATA_DIR/meta.json" \
    "$ARTIFACT_DIR/initial.pt" \
    "$ARTIFACT_DIR/windows.pt" \
    "$ARTIFACT_DIR/identity.json"; do
    if [[ ! -f "$path" ]]; then
      printf 'Missing input: %s\n' "$path" >&2
      missing=1
    fi
  done
  (( missing == 0 )) || die 'one or more registered input files are missing'
}

verify_inputs() {
  local identity expected
  if [[ -f "$INPUT_ROOT/SHA256SUMS" ]]; then
    printf 'Checking input-repository SHA-256 manifest...\n'
    (cd -- "$INPUT_ROOT" && sha256sum -c SHA256SUMS)
  else
    printf 'WARNING: %s is absent; continuing with the project identity verifier.\n' \
      "$INPUT_ROOT/SHA256SUMS" >&2
  fi
  if [[ -f "$INPUT_ROOT/registration.json" ]]; then
    identity="$("$PYTHON" -m nanogpt_noise.input_bundle verify --input-root "$INPUT_ROOT")"
    expected="$identity"
    printf 'Input registration: %s\n' "$INPUT_ROOT/registration.json"
  else
    die 'registration.json missing; run bash scripts/run_acp.sh inputs with a fresh INPUT_ROOT'
  fi
  if [[ "$identity" != "$expected" ]]; then
    printf 'Expected input ID: %s\nActual output: %s\n' \
      "$expected" "$identity" >&2
    die 'registered input identity verification failed'
  fi
  INPUT_ID="$identity"
  export INPUT_ID
  printf 'Input identity: %s\n' "$identity"
}

show_storage() {
  mkdir -p -- "$RUN_BASE"
  printf 'DATA_DIR=%s\nARTIFACT_DIR=%s\nRUN_BASE=%s\n' \
    "$DATA_DIR" "$ARTIFACT_DIR" "$RUN_BASE"
  df -B1 "$RUN_BASE"
  du -sh "$RUN_BASE" 2>/dev/null || true
  printf '%s\n' 'Storage is reported only; this script does not enforce a disk threshold.'
}

run_check() {
  local scope
  check_code
  check_files
  verify_inputs
  "$PYTHON" -m unittest discover -s tests -v
  printf '%s\n' 'Static storage estimates:'
  for scope in full last1 last2 last3 last6 last12; do
    "$PYTHON" -m nanogpt_noise.scope estimate --scope "$scope"
  done
  show_storage
  printf '%s\n' 'CHECK PASSED'
}

validate_smoke() {
  [[ -f "$SMOKE_STATUS" ]] || die "smoke status is missing: $SMOKE_STATUS"
  grep -Eq '"status"[[:space:]]*:[[:space:]]*"completed"' "$SMOKE_STATUS" || \
    die 'smoke status is not completed'
  grep -Eq '"mode"[[:space:]]*:[[:space:]]*"smoke"' "$SMOKE_STATUS" || \
    die 'smoke output is not marked as smoke'
  grep -Eq '"step"[[:space:]]*:[[:space:]]*8([^0-9]|$)' "$SMOKE_STATUS" || \
    die 'smoke output did not finish step 8'
  "$PYTHON" -m nanogpt_noise.smoke --runs "$RUN_BASE" --scope "$SMOKE_SCOPE" \
    --input-id "$INPUT_ID" --micro-batch "${MICRO_BATCH:-4}"
}

record_smoke_identity() {
  mkdir -p -- "$(dirname -- "$SMOKE_COMMIT")"
  printf '%s\n' "$CODE_COMMIT" > "$SMOKE_COMMIT"
  printf '%s\n' "$INPUT_ID" > "$SMOKE_INPUT"
}

require_matching_smoke() {
  local smoke_commit smoke_input
  validate_smoke
  [[ -f "$SMOKE_COMMIT" ]] || die "smoke commit marker is missing: $SMOKE_COMMIT"
  smoke_commit="$(tr -d '\r\n' < "$SMOKE_COMMIT")"
  if [[ "$smoke_commit" != "$CODE_COMMIT" ]]; then
    die "smoke used commit $smoke_commit but current code is $CODE_COMMIT; rerun smoke"
  fi
  [[ -f "$SMOKE_INPUT" ]] || die "smoke input marker is missing: $SMOKE_INPUT"
  smoke_input="$(tr -d '\r\n' < "$SMOKE_INPUT")"
  if [[ "$smoke_input" != "$INPUT_ID" ]]; then
    die "smoke used input $smoke_input but current input is $INPUT_ID; rerun smoke"
  fi
}

command="${1:-}"
case "$command" in
  --help|-h|help)
    usage
    ;;
  check)
    (( $# == 1 )) || die 'check takes no additional arguments'
    run_check
    ;;
  smoke)
    (( $# <= 2 )) || die 'smoke accepts at most one scope'
    validate_scope "$SMOKE_SCOPE"
    run_check
    export RUN_DIR="$RUN_BASE"
    for optimizer in adamw adam sgd muon; do
      bash scripts/run_4gpu.sh smoke "$SMOKE_SCOPE" "$optimizer" 10
    done
    validate_smoke
    record_smoke_identity
    printf 'SMOKE PASSED: %s\n' "$SMOKE_STATUS"
    ;;
  scope)
    (( $# == 2 )) || die 'scope requires exactly one scope name'
    validate_scope "$2"
    check_code
    check_files
    verify_inputs
    require_matching_smoke
    export RUN_DIR="$RUN_BASE/$2"
    mkdir -p -- "$RUN_DIR"
    show_storage
    printf 'STARTING SCOPE: %s -> %s\n' "$2" "$RUN_DIR"
    exec bash scripts/run_4gpu.sh scope "$2"
    ;;
  finalize)
    (( $# == 2 )) || die 'finalize requires exactly one scope name'
    validate_scope "$2"
    check_code
    check_files
    verify_inputs
    require_matching_smoke
    RUN_DIR="$RUN_BASE/$2"
    [[ -d "$RUN_DIR" ]] || die "scope directory is missing: $RUN_DIR"
    "$PYTHON" -m nanogpt_noise.scope finalize --runs "$RUN_DIR" --scope "$2"
    printf 'FINALIZED SCOPE: %s\n' "$RUN_DIR"
    ;;
  status)
    (( $# <= 2 )) || die 'status accepts at most one scope name'
    if (( $# == 2 )); then validate_scope "$2"; fi
    if [[ -f "$SMOKE_STATUS" ]]; then
      printf '%s\n' 'Smoke status:'
      "$PYTHON" -m json.tool "$SMOKE_STATUS"
    else
      printf 'Smoke status not found: %s\n' "$SMOKE_STATUS"
    fi
    if (( $# == 2 )); then
      scope_dir="$RUN_BASE/$2"
      printf 'Scope directory: %s\n' "$scope_dir"
      [[ -d "$scope_dir" ]] && du -sh "$scope_dir" || true
    fi
    [[ -d "$RUN_BASE" ]] && df -B1 "$RUN_BASE" || true
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
