#!/usr/bin/env bash
# Public one-command entrypoint for the one-H100 smoke test.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
exec bash "$REPO_ROOT/scripts/h100.sh" smoke
