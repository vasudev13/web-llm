#!/usr/bin/env bash
# VAS-87 — Install the prerequisites the TVM + MLC-LLM WebGPU build needs on
# Apple Silicon macOS. Idempotent: safe to re-run; skips anything already
# present. Installs Homebrew formulae, rustup, emsdk, and a Python 3.11 env.
#
# This script is NOT run automatically by the scheduled task — it mutates the
# host (brew installs, ~/emsdk, ~/.cargo, a pyenv version). Run it by hand:
#
#     bash scripts/eviction/01-bootstrap-prereqs.sh
#
# After it finishes, open a NEW shell (or source the printed env lines) and
# re-run 00-audit-toolchain.sh — the verdict should flip to all-green.

set -euo pipefail

EMSDK_VERSION="3.1.56"   # emscripten with stable WebGPU (Dawn)
PYTHON_VERSION="3.11.9"  # mlc-llm build target
EMSDK_DIR="${EMSDK_DIR:-$HOME/emsdk}"

log() { printf '\033[1m[bootstrap]\033[0m %s\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# --- 1. Homebrew formulae ----------------------------------------------------
if ! have brew; then
  echo "Homebrew is required. Install from https://brew.sh first." >&2
  exit 1
fi
log "Installing brew formulae (cmake, ninja, llvm@17, git-lfs, pyenv)…"
brew install cmake ninja llvm@17 git-lfs pyenv || true
brew list rust >/dev/null 2>&1 && log "brew 'rust' present (rustup preferred; harmless)"

LLVM_PREFIX="$(brew --prefix llvm@17)"
log "LLVM prefix: ${LLVM_PREFIX}"
log "  → add to your shell rc:  export PATH=\"${LLVM_PREFIX}/bin:\$PATH\""

# --- 2. Rust via rustup ------------------------------------------------------
if ! have rustc; then
  log "Installing rustup…"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
log "rustc: $(rustc --version 2>/dev/null || echo 'open a new shell to pick up ~/.cargo/env')"

# --- 3. Emscripten (emsdk) ---------------------------------------------------
if [[ ! -d "$EMSDK_DIR" ]]; then
  log "Cloning emsdk to ${EMSDK_DIR}…"
  git clone https://github.com/emscripten-core/emsdk.git "$EMSDK_DIR"
fi
log "Installing + activating emscripten ${EMSDK_VERSION}…"
( cd "$EMSDK_DIR" && ./emsdk install "$EMSDK_VERSION" && ./emsdk activate "$EMSDK_VERSION" )
log "  → source emscripten in each build shell:  source ${EMSDK_DIR}/emsdk_env.sh"

# --- 4. Python 3.11 env (mlc-llm build + kvpress) ----------------------------
if have pyenv; then
  log "Installing Python ${PYTHON_VERSION} via pyenv…"
  pyenv install -s "$PYTHON_VERSION"
  log "  → activate for this project:  pyenv local ${PYTHON_VERSION}"
else
  log "pyenv not found — install Python ${PYTHON_VERSION%.*}+ another way (conda/python.org)."
fi

cat <<EOF

============================================================================
Bootstrap done. Add these to your shell rc (~/.zshrc), then open a new shell:

  export PATH="${LLVM_PREFIX}/bin:\$PATH"
  source "\$HOME/.cargo/env"
  source "${EMSDK_DIR}/emsdk_env.sh"

Then verify:   bash scripts/eviction/00-audit-toolchain.sh
Next:          bash scripts/eviction/02-build-tvm.sh
============================================================================
EOF
