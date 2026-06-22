#!/usr/bin/env bash
# VAS-50 — Create the isolated kvpress venv and install the pinned stack.
# This is the one step that actually mutates the machine (creates a venv,
# downloads wheels). It does NOT download the model — that happens lazily on
# the first 02-run-baseline.sh. Idempotent: re-running reuses the venv.
#
# Usage:  bash scripts/eviction/kvpress/01-setup-kvpress-env.sh
# Env:    KVPRESS_VENV   venv path (default .venv-kvpress)
#         KVPRESS_PY     python interpreter to seed the venv (default: auto)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="$HERE/requirements.txt"
VENV="${KVPRESS_VENV:-.venv-kvpress}"
REQ_PYTHON_MM="3.11"

log()  { printf '\033[1m[01-setup]\033[0m %s\n' "$*"; }
die()  { printf '\033[31m[01-setup] %s\033[0m\n' "$*" >&2; exit 1; }

# --- pick a python >= 3.11 ---------------------------------------------------
pick_python() {
  if [[ -n "${KVPRESS_PY:-}" ]]; then echo "$KVPRESS_PY"; return; fi
  for py in python3.12 python3.11; do
    command -v "$py" >/dev/null 2>&1 && { echo "$py"; return; }
  done
  # pyenv fallback
  if command -v pyenv >/dev/null 2>&1; then
    local latest
    latest="$(pyenv versions --bare 2>/dev/null | grep -E '^3\.(11|12)\.' | tail -1 || true)"
    if [[ -n "$latest" ]]; then echo "$(pyenv root)/versions/$latest/bin/python"; return; fi
  fi
  echo ""   # none found
}

PY="$(pick_python)"
if [[ -z "$PY" ]]; then
  cat >&2 <<EOF
[01-setup] No Python >= ${REQ_PYTHON_MM} found. The system 3.9 is too old for kvpress.
           Install one first, e.g. with pyenv (already present on the M4 Air):
             pyenv install ${REQ_PYTHON_MM}
           then re-run this script (it will auto-detect it).
EOF
  exit 1
fi
log "seeding venv with: $PY ($("$PY" --version 2>&1))"

# --- create / reuse venv -----------------------------------------------------
if [[ -d "$VENV" ]]; then
  log "venv already exists at $VENV — reusing"
else
  log "creating venv at $VENV"
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"

log "upgrading pip/wheel"
python -m pip install --upgrade pip wheel >/dev/null

log "installing pinned stack from requirements.txt (this pulls torch — a few minutes)"
python -m pip install -r "$REQ"

log "verifying import + MPS"
python - <<'PYEOF'
import torch, transformers, kvpress
print(f"  torch        {torch.__version__}")
print(f"  transformers {transformers.__version__}")
print(f"  kvpress      {getattr(kvpress,'__version__','?')}")
print(f"  mps available {torch.backends.mps.is_available()} / built {torch.backends.mps.is_built()}")
assert torch.backends.mps.is_available(), "MPS not available — check torch build / macOS"
PYEOF

log "done. Next: bash scripts/eviction/kvpress/02-run-baseline.sh"
