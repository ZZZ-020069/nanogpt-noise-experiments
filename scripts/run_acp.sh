#!/usr/bin/env bash
# ACP public entrypoint: prepare an isolated runtime, then dispatch existing scripts.
set -euo pipefail
trap 'printf "ACP FAILED at line %s (exit %s). Training is not certified; retain the full log.\n" "$LINENO" "$?" >&2' ERR
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd -- "$REPO_ROOT"
mode="${1:-}"
case "$mode" in
  prepare|inputs) (( $# == 1 )) ;;
  smoke|formal|pause|resume) (( $# <= 2 )); case "${2:-last1}" in full|last1|last2|last3|last6|last12) ;; *) exit 2;; esac ;;
  *) echo 'Usage: bash scripts/run_acp.sh prepare|inputs|smoke|formal|pause|resume [SCOPE]' >&2; exit 2 ;;
esac
if [[ "$mode" == pause ]]; then
  exec bash scripts/pause.sh "${2:-last1}"
fi
if [[ "$mode" == resume ]]; then
  export NANOGPT_RESUME_PAUSED=1
  mode=formal
fi

# Applies to pip's build subprocesses and experiment metadata as well as Git.
index="${GIT_CONFIG_COUNT:-0}"
[[ "$index" =~ ^[0-9]+$ ]]
export "GIT_CONFIG_KEY_${index}=safe.directory"
export "GIT_CONFIG_VALUE_${index}=$REPO_ROOT"
export GIT_CONFIG_COUNT=$((index + 1))
GIT_BIN="${GIT_BIN:-git}"
dirty="$("$GIT_BIN" status --porcelain --untracked-files=normal)"
[[ -z "$dirty" ]] || {
  echo 'ERROR: repository has local changes; resolve them before running ACP.' >&2; exit 2;
}
commit="$("$GIT_BIN" rev-parse HEAD)"
printf 'ACP code commit: %s\n' "$commit"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

runtime_ok() {
  "$1" -c 'import sys; assert sys.version_info >= (3,11), sys.version; import torch,numpy,matplotlib; assert (2,6) <= tuple(map(int,torch.__version__.split("+")[0].split(".")[:2])) < (3,0); assert torch.version.cuda is not None, "CPU-only PyTorch"'
}

if [[ -n "${PYTHON:-}" ]]; then
  # An explicit environment is an operator decision: never silently replace it.
  runtime_ok "$PYTHON" || {
    echo 'ERROR: explicit PYTHON requires Python >=3.11, CUDA PyTorch >=2.6,<3, NumPy and matplotlib.' >&2
    exit 2
  }
elif runtime_ok python; then
  PYTHON="$(command -v python)"
else
  # Keep packages off the tightly budgeted results disk. This is reusable when
  # the platform preserves HOME; ENV_PREFIX can name a persistent system disk.
  ENV_PREFIX="${ENV_PREFIX:-${HOME:?}/.local/share/nanogpt-noise/env-py311}"
  [[ "$ENV_PREFIX" == /* ]] || { echo 'ENV_PREFIX must be absolute' >&2; exit 2; }
  PYTHON="$ENV_PREFIX/bin/python"
  if ! runtime_ok "$PYTHON"; then
    CONDA_BIN="${CONDA_BIN:-$(command -v conda || true)}"
    if [[ -z "$CONDA_BIN" && -x /root/miniconda3/bin/conda ]]; then
      CONDA_BIN=/root/miniconda3/bin/conda
    fi
    [[ -n "$CONDA_BIN" ]] || {
      echo 'ERROR: no compatible runtime or Conda. Provide PYTHON=/absolute/path/to/compatible/python.' >&2; exit 2;
    }
    printf 'Preparing isolated runtime: %s (requires network and system-disk space)\n' "$ENV_PREFIX"
    if [[ ! -x "$PYTHON" ]]; then
      "$CONDA_BIN" create --yes --prefix "$ENV_PREFIX" --override-channels -c conda-forge python=3.11 pip
    fi
    "$PYTHON" -m pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
    "$PYTHON" -m pip install --no-cache-dir numpy==2.1.3 matplotlib==3.9.4 'setuptools>=68'
    runtime_ok "$PYTHON"
  fi
fi
export PYTHON
INPUT_ROOT="${INPUT_ROOT:-/data/260010171/inputs-v1.8}"
export INPUT_ROOT
printf 'ACP Python: %s\n' "$PYTHON"
"$PYTHON" -c 'import sys,torch,numpy; print("Python:",sys.version); print("torch:",torch.__version__,"CUDA build:",torch.version.cuda,"numpy:",numpy.__version__)'
# Dependencies were verified above. Do not let editable installation replace
# CUDA PyTorch, or pull CPU wheels from a platform mirror.
"$PYTHON" -m pip install --no-deps --no-cache-dir -e "$REPO_ROOT"
printf 'ACP ENVIRONMENT READY\n'
case "$mode" in
  prepare) ;;
  inputs)
    "$PYTHON" -m pip install --no-cache-dir -r requirements-prepare.txt
    "$PYTHON" -m nanogpt_noise.prepare --input-root "$INPUT_ROOT" --restart-incomplete
    printf 'INPUT BUNDLE REGISTERED: %s\n' "$INPUT_ROOT"
    ;;
  smoke) bash scripts/run_smoke.sh "${2:-last1}" ;;
  formal) exec bash scripts/run_formal.sh "${2:-last1}" ;;
esac
