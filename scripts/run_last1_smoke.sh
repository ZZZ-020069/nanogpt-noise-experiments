#!/usr/bin/env bash
# Explicit current-stage smoke, using an already installed CUDA environment.
set -euo pipefail
(( $# == 0 )) || { echo 'This script accepts no arguments.' >&2; exit 2; }
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec bash "$REPO_ROOT/scripts/run_smoke.sh" last1
