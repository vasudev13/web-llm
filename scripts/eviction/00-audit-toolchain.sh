#!/usr/bin/env bash
# VAS-87 — Audit the local toolchain for the TVM + MLC-LLM "build-from-source"
# eviction workflow. Read-only: installs nothing, mutates nothing. Run this
# first to see how far the machine is from a working build, then run
# 01-bootstrap-prereqs.sh to close the gaps.
#
# Usage:  bash scripts/eviction/00-audit-toolchain.sh
# Exit:   0 always (this is a report, not a gate)

set -uo pipefail

# --- minimum versions the relax/mlc-llm WebGPU toolchain expects -------------
# Sourced from scripts/prep_deps.sh (emcc + tvm/web make) and the mlc-llm
# "Build from source" + emsdk WebGPU docs. Pins are conservative floors.
REQ_PYTHON="3.11"     # mlc-llm wheels/build target 3.11–3.12; system 3.9 is too old
REQ_CMAKE="3.24"
REQ_EMSDK="3.1.56"    # emscripten with stable WebGPU (Dawn) bindings
REQ_NODE="20"         # .nvmrc pins v24.11.1; 20+ is the practical floor

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
  [[ "$(uname -m)" == "arm64" ]] && ok "Apple Silicon (arm64) — matches M4 Air target" \
    || warn "non-arm64 host: numbers won't match the M4 Air reference"
fi

hdr "Core build tools"
for t in git clang cmake ninja make pkg-config; do
  v="$(ver "$t" --version)"; [[ -n "$v" ]] && ok "$t — $v" || miss "$t — NOT FOUND"
done

hdr "LLVM (TVM codegen needs llvm-config on PATH)"
if v="$(ver llvm-config --version)"; then
  ok "llvm-config — $v"
  note "llvm prefix: $(llvm-config --prefix 2>/dev/null)"
else
  miss "llvm-config — NOT FOUND  (brew install llvm@17, then export PATH)"
fi

hdr "Rust (xgrammar + some relax components)"
if v="$(ver rustc --version)"; then ok "rustc — $v"; else miss "rustc — NOT FOUND (https://rustup.rs)"; fi

hdr "Emscripten (WASM/WebGPU codegen — used by tvm/web make)"
if v="$(ver emcc --version)"; then
  ok "emcc — $v"
  note "needs >= ${REQ_EMSDK} for stable WebGPU"
else
  miss "emcc — NOT FOUND (install emsdk; see 01-bootstrap-prereqs.sh)"
fi

hdr "Python (mlc-llm build + kvpress oracle)"
if v="$(ver python3 --version)"; then
  ok "python3 — $v"
  pyv="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
  awk -v a="$pyv" -v b="$REQ_PYTHON" 'BEGIN{split(a,x,".");split(b,y,".");
    if (x[1]<y[1]||(x[1]==y[1]&&x[2]<y[2])) exit 1}' \
    && ok "python ${pyv} >= ${REQ_PYTHON}" \
    || warn "python ${pyv} < ${REQ_PYTHON} — mlc-llm wants 3.11+; use a pyenv/conda env"
else
  miss "python3 — NOT FOUND"
fi

hdr "Node / npm (tvm/web build + web-llm fork)"
for t in node npm; do v="$(ver "$t" --version)"; [[ -n "$v" ]] && ok "$t — $v" || miss "$t — NOT FOUND"; done
[[ -f .nvmrc ]] && note ".nvmrc pins $(cat .nvmrc)"

hdr "Package managers"
for t in brew pyenv conda rustup; do v="$(ver "$t" --version)"; [[ -n "$v" ]] && ok "$t — $v" || note "$t — not present"; done

hdr "Repo wiring"
[[ -f scripts/prep_deps.sh ]] && ok "scripts/prep_deps.sh present (clones mlc-ai/relax → 3rdparty/tvm-unity)" \
  || warn "scripts/prep_deps.sh missing"
[[ -n "${TVM_SOURCE_DIR:-}" ]] && ok "TVM_SOURCE_DIR=${TVM_SOURCE_DIR}" || note "TVM_SOURCE_DIR unset (prep_deps.sh defaults to 3rdparty/tvm-unity)"
[[ -n "${MLC_LLM_HOME:-}" ]] && ok "MLC_LLM_HOME=${MLC_LLM_HOME}" || note "MLC_LLM_HOME unset (needed for serve_mlc_llm_dist.sh)"
[[ -e tvm_home ]] && ok "tvm_home symlink present" || note "tvm_home symlink absent (created by prep_deps.sh)"

hdr "Verdict"
missing=0
for t in cmake ninja llvm-config rustc emcc; do command -v "$t" >/dev/null 2>&1 || missing=$((missing+1)); done
if [[ $missing -eq 0 ]]; then
  ok "all heavy prerequisites present — proceed to 02-build-tvm.sh"
else
  warn "${missing} core prerequisite(s) missing — run 01-bootstrap-prereqs.sh first"
fi
exit 0
