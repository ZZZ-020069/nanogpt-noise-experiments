#!/usr/bin/env bash
# Launch/resume exactly last1; finalize it and stop. Never chain scopes.
set -euo pipefail
(( $# == 0 )) || { echo 'This script accepts no arguments.' >&2; exit 2; }
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec bash "$REPO_ROOT/scripts/run_formal.sh" last1
