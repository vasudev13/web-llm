#!/usr/bin/env bash
# VAS-87 — Build MLC-LLM from source against the TVM built in 02-build-tvm.sh.
# MLC-LLM holds the Python model definitions (KV-cache wiring, attention) that
# compile down to the .wasm/.wgsl artifacts web-llm loads — so any eviction
# logic at the model-definition level lives here, and the model-library
# compiler (`mlc_llm compile`) used in 04 ships with this package.
#
# Prereqs: 02-build-tvm.sh done; TVM_SOURCE_DIR exported; a Python 3.11 env
# active (pyenv local 3.11.9 or a venv). Verify with 00-audit-toolchain.sh.
#
#     export TVM_SOURCE_DIR=.../3rdparty/tvm-unity
#     bash scripts/eviction/03-build-mlc-llm.sh
#
# Time/space: ~15–30 min, ~2–3 GB.

set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MLC_LLM_HOME="${MLC_LLM_HOME:-$REPO_ROOT/3rdparty/mlc-llm}"
TVM_SOURCE_DIR="${TVM_SOURCE_DIR:-$REPO_ROOT/3rdparty/tvm-unity}"

[[ -d "$TVM_SOURCE_DIR/build" ]] || { echo "TVM not built — run 02-build-tvm.sh"; exit 1; }
python3 -c 'import sys; assert sys.version_info[:2] >= (3,11)' \
  || { echo "Need Python 3.11+ (pyenv local 3.11.9)"; exit 1; }

# --- 1. clone mlc-llm --------------------------------------------------------
if [[ ! -d "$MLC_LLM_HOME" ]]; then
  git clone https://github.com/mlc-ai/mlc-llm "$MLC_LLM_HOME" --recursive
fi
export MLC_LLM_HOME TVM_SOURCE_DIR

# --- 2. native libmlc_llm ----------------------------------------------------
mkdir -p "$MLC_LLM_HOME/build"
cat > "$MLC_LLM_HOME/build/config.cmake" <<CFG
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(TVM_SOURCE_DIR "$TVM_SOURCE_DIR")
set(USE_METAL ON)
CFG
( cd "$MLC_LLM_HOME/build" && cmake .. -G Ninja && ninja )

# --- 3. install the Python packages (editable: edits to .py reflect live) ----
python3 -m pip install --upgrade pip
python3 -m pip install -e "$TVM_SOURCE_DIR/python"
python3 -m pip install -e "$MLC_LLM_HOME/python"

# --- 4. verify the compiler entrypoint is importable -------------------------
python3 - <<'PY'
import tvm, mlc_llm
print("tvm     :", tvm.__version__, "->", tvm.__file__)
print("mlc_llm :", getattr(mlc_llm, "__version__", "editable"), "->", mlc_llm.__file__)
PY
mlc_llm --help >/dev/null && echo "mlc_llm CLI OK"

set +x
cat <<EOF

============================================================================
MLC-LLM build complete.
  MLC_LLM_HOME = $MLC_LLM_HOME
Export for the next step:
  export MLC_LLM_HOME="$MLC_LLM_HOME"
  export TVM_SOURCE_DIR="$TVM_SOURCE_DIR"

Next: bash scripts/eviction/04-compile-model-lib.sh
============================================================================
EOF
