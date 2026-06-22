#!/usr/bin/env bash
# VAS-50 — Run the kvpress baseline calibration and record the per-run number.
# Thin wrapper: activates the venv from 01-setup, then runs 02-run-baseline.py,
# also capturing a coarse `time` of the whole process for cross-checking the
# in-script wall-clock. All extra args pass through to the Python script.
#
# Usage:  bash scripts/eviction/kvpress/02-run-baseline.sh [--context N --ratio R --press snapkv ...]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${KVPRESS_VENV:-.venv-kvpress}"

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[02-baseline] no venv at $VENV — run 01-setup-kvpress-env.sh first" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# default to the SnapKV/RULER calibration the ticket calls for
ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
  ARGS=(--press snapkv --context 4096 --ratio 0.5 --samples 1)
fi

echo "[02-baseline] python $("$VENV/bin/python" --version 2>&1)"
echo "[02-baseline] args: ${ARGS[*]}"
time python "$HERE/02-run-baseline.py" "${ARGS[@]}"
