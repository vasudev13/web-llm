#!/usr/bin/env bash
# VAS-87 — Build the relax ("tvm-unity") fork from source with the WebGPU/WASM
# target, then build its browser runtime (tvm/web). This is the C++ layer that
# owns the KV cache (src/runtime/relax_vm/paged_kv_cache.cc) — the file any
# eviction policy must eventually touch.
#
# Mirrors scripts/prep_deps.sh (clone mlc-ai/relax → 3rdparty/tvm-unity, then
# `make` in tvm/web) but adds an explicit native-libtvm build first, which the
# Python mlc-llm package in 03 links against.
#
# Prereqs: run 01-bootstrap-prereqs.sh and source the env lines it prints
# (LLVM on PATH, emsdk_env.sh, ~/.cargo/env). Verify with 00-audit-toolchain.sh.
#
#     bash scripts/eviction/02-build-tvm.sh
#
# Time/space: first build ~30–60 min on an M4 Air, ~5–8 GB.

set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Pin TVM source to the relax fork web-llm's prep_deps.sh uses.
TVM_SOURCE_DIR="${TVM_SOURCE_DIR:-$REPO_ROOT/3rdparty/tvm-unity}"

# --- toolchain sanity --------------------------------------------------------
emcc --version            >/dev/null || { echo "emcc missing — source emsdk_env.sh"; exit 1; }
command -v cmake          >/dev/null || { echo "cmake missing — run 01-bootstrap";   exit 1; }
command -v ninja          >/dev/null || { echo "ninja missing — run 01-bootstrap";   exit 1; }
command -v llvm-config    >/dev/null || { echo "llvm-config missing — put llvm@17/bin on PATH"; exit 1; }

# --- 1. clone relax (tvm-unity) ----------------------------------------------
if [[ ! -d "$TVM_SOURCE_DIR" ]]; then
  echo "Cloning mlc-ai/relax → $TVM_SOURCE_DIR"
  git clone https://github.com/mlc-ai/relax "$TVM_SOURCE_DIR" --recursive
else
  echo "Reusing existing $TVM_SOURCE_DIR (run 'git submodule update --init --recursive' if stale)"
fi
export TVM_SOURCE_DIR

# --- 2. native libtvm (host) -------------------------------------------------
# mlc-llm's Python build (step 03) and any C++ edits to paged_kv_cache.cc need
# a host libtvm. Config flags follow the mlc-llm "Build from source" docs.
mkdir -p "$TVM_SOURCE_DIR/build"
cat > "$TVM_SOURCE_DIR/build/config.cmake" <<CFG
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(USE_LLVM "$(command -v llvm-config) --ignore-libllvm --link-static")
set(HIDE_PRIVATE_SYMBOLS ON)
set(USE_METAL ON)         # Apple GPU backend for local sanity checks
set(USE_CUDA OFF)
set(USE_VULKAN OFF)
set(USE_OPENCL OFF)
CFG
( cd "$TVM_SOURCE_DIR/build" && cmake .. -G Ninja && ninja )
echo "Native libtvm built: $TVM_SOURCE_DIR/build"

# --- 3. browser runtime (tvm/web → WASM) -------------------------------------
# Same path prep_deps.sh drives; produces dist/wasm + the tvmjs runtime.
( cd "$TVM_SOURCE_DIR/web" && make && npm install && npm run build )

# --- 4. wire the repo to this TVM (matches prep_deps.sh tail) -----------------
cd "$REPO_ROOT"
rm -rf tvm_home
ln -s "$TVM_SOURCE_DIR" tvm_home

set +x
cat <<EOF

============================================================================
TVM build complete.
  TVM_SOURCE_DIR = $TVM_SOURCE_DIR
  native libtvm  = $TVM_SOURCE_DIR/build
  web runtime    = $TVM_SOURCE_DIR/web/dist
  tvm_home       -> $TVM_SOURCE_DIR  (symlink)

Export for the next steps:
  export TVM_SOURCE_DIR="$TVM_SOURCE_DIR"

Next: bash scripts/eviction/03-build-mlc-llm.sh
============================================================================
EOF
