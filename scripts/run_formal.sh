#!/usr/bin/env bash
# Public one-command entrypoint for one complete four-H100 scope.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if (( $# > 1 )); then
  printf 'Usage: bash scripts/run_formal.sh [full|last1|last2|last3|last6|last12]\n' >&2
  exit 2
fi

scope="${1:-full}"
result_root="${RUN_BASE:-/data/260010171/results/nanogpt-v1.5}"

# set -e guarantees that finalization is never attempted after a failed pool.
bash "$REPO_ROOT/scripts/h100.sh" scope "$scope"
bash "$REPO_ROOT/scripts/h100.sh" finalize "$scope"

printf '\nFORMAL SCOPE FINALIZED: %s/%s\n' "$result_root" "$scope"
printf '%s\n' \
  'Transfer this scope, verify scope_manifest.json at the receiver, and only then clean the server copy.' \
  'Do not start the next scope before that handoff is complete.'
