#!/usr/bin/env bash
# Cooperative pause: no Python installation, GPU allocation, or PID guessing.
set -euo pipefail
scope="${1:-last1}"
(( $# <= 1 )) || exit 2
case "$scope" in full|last1|last2|last3|last6|last12) ;; *) exit 2;; esac
root="${RUN_BASE:-/data/260010171/results/nanogpt-v1.8}/$scope"
[[ "$root" == /* ]] || { echo 'RUN_BASE must be absolute' >&2; exit 2; }
[[ -f "$root/scope_run.json" && ! -f "$root/scope_manifest.json" ]] || {
  echo 'No unfinished scope found at the selected RUN_BASE.' >&2; exit 2;
}
# Existing requests are immutable/idempotent. Presence is the control signal.
timeout="${PAUSE_WAIT_SECONDS:-300}"
[[ "$timeout" =~ ^[0-9]+$ ]] || exit 2
touch -- "$root/pause.request"
echo "PAUSE REQUESTED: $root; waiting for workers to save and exit"
deadline=$((SECONDS + timeout))
while [[ -e "$root/launcher/.lock" ]] || find "$root" -name .lock -print -quit | grep -q .; do
  if (( SECONDS >= deadline )); then
    echo 'Still waiting: request retained. Check worker logs; do not delete locks or force-kill a healthy save.' >&2
    exit 3
  fi
  sleep 1
done
echo 'SCOPE QUIESCENT: no worker locks remain. Resume with bash scripts/run_acp.sh resume '"$scope"
