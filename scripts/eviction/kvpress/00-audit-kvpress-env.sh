#!/usr/bin/env bash
# VAS-50 — Audit the local environment for the kvpress baseline (Track A,
# the FP16-vs-q4 quality oracle). Read-only: installs nothing, mutates nothing.
# Run this first to see how far the M4 Air is from a working kvpress run, then
# run 01-setup-kvpress-env.sh to close the gaps.
#
# Usage:  bash scripts/eviction/kvpress/00-audit-kvpress-env.sh
# Exit:   0 always (this is a report, not a gate)

set -uo pipefail

# --- floors the kvpress + transformers + MPS stack expects -------------------
# kvpress targets recent transformers (>=4.44) and torch>=2.3 for a usable MPS
# backend; Apple's MPS attention paths matured through torch 2.3/2.4. The
# project-standard test model is Qwen2.5-3B-Instruct.
REQ_PYTHON="3.11"     # kvpress + recent transformers; system 3.9 is too old
REQ_TORCH="2.3"       # MPS backend stability for attention + SDPA
REQ_TRANSFORMERS="4.44"
MODEL="Qwen/Qwen2.5-3B-Instruct"

note()  { printf '  %s\n' "$*"; }
ok()    { printf '  \033[32m✓\033[0m %s\n' "$*"; }
miss()  { printf '  \033[31m✗\033[0m %s\n' "$*"; }
warn()  { printf '  \033[33m!\033[0m %s\n' "$*"; }
hdr()   { printf '\n\033[1m%s\033[0m\n' "$*"; }

ver() { command -v "$1" >/dev/null 2>&1 && "$@" 2>/dev/null | head -1; }

hdr "Host"
note "$(uname -msr)"
if [[ "$(uname -s)" == "Darwin" ]]; then
  note "macOS $(sw_vers -productVersion 2>/dev/null) ($(sw_vers -buildVersion 2>/dev/null))"
  if [[ "$(uname -m)" == "arm64" ]]; then
    ok "Apple Silicon (arm64) — MPS backend available; matches M4 Air target"
  else
    warn "non-arm64 host: no MPS, numbers won't match the M4 Air reference"
  fi
  mem_gb="$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))"
  [[ "$mem_gb" -gt 0 ]] && note "unified memory: ${mem_gb} GB (Qwen2.5-3B FP16 ≈ 6 GB weights + KV)"
fi

hdr "Python (kvpress needs >= ${REQ_PYTHON}; system 3.9 is too old)"
for py in python3.12 python3.11 python3; do
  if v="$(ver "$py" --version)"; then
    note "$py — $v"
  fi
done
if command -v pyenv >/dev/null 2>&1; then
  ok "pyenv present — $(pyenv --version 2>/dev/null)"
  note "01-setup will use 'pyenv install ${REQ_PYTHON}' if no 3.11/3.12 found"
else
  warn "pyenv not found — install a Python >= ${REQ_PYTHON} some other way"
fi

hdr "Existing kvpress venv"
VENV="${KVPRESS_VENV:-.venv-kvpress}"
if [[ -d "$VENV" ]]; then
  ok "venv exists at $VENV"
  if [[ -x "$VENV/bin/python" ]]; then
    note "$("$VENV/bin/python" --version 2>&1)"
  fi
else
  miss "no venv at $VENV (01-setup-kvpress-env.sh creates it)"
fi

hdr "Python packages (probed via venv if present, else active python)"
PYBIN="python3"
[[ -x "$VENV/bin/python" ]] && PYBIN="$VENV/bin/python"
note "probing with: $PYBIN"
"$PYBIN" - "$REQ_TORCH" "$REQ_TRANSFORMERS" <<'PYEOF' 2>/dev/null || warn "probe python could not import the stack (expected before 01-setup)"
import importlib, sys
req_torch, req_tf = sys.argv[1], sys.argv[2]
def show(mod):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception as e:
        return None
for mod in ("torch", "transformers", "kvpress", "datasets", "accelerate"):
    v = show(mod)
    mark = "\033[32m✓\033[0m" if v else "\033[31m✗\033[0m"
    print(f"  {mark} {mod} — {v or 'NOT INSTALLED'}")
try:
    import torch
    print(f"    torch.backends.mps.is_available() = {torch.backends.mps.is_available()}")
    print(f"    torch.backends.mps.is_built()     = {torch.backends.mps.is_built()}")
except Exception:
    pass
PYEOF

hdr "Hugging Face model cache (target: ${MODEL})"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
note "HF_HOME = $HF_HOME"
if [[ -d "$HF_HOME/hub" ]] && ls "$HF_HOME/hub" 2>/dev/null | grep -qi "Qwen2.5-3B"; then
  ok "Qwen2.5-3B appears cached — no ~6 GB download needed"
else
  warn "Qwen2.5-3B not cached — first run downloads ~6 GB (FP16 safetensors)"
fi

hdr "Summary"
note "When all of python>=${REQ_PYTHON}, torch>=${REQ_TORCH} (MPS), transformers,"
note "kvpress, datasets are ✓ and the model is cached, run:"
note "  bash scripts/eviction/kvpress/02-run-baseline.sh"
note "That produces the single calibration number VAS-50 asks for."
