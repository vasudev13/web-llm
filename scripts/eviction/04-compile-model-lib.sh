#!/usr/bin/env bash
# VAS-87 — Compile a Qwen2.5-3B-q4f16 model library locally to a WebGPU/WASM
# artifact and stage it for the web-llm fork. This closes the loop: a model lib
# produced by *our* TVM+MLC-LLM build, not a prebuilt one — the precondition for
# the edit→compile→run proof that any eviction change actually reaches the
# browser.
#
# Prereqs: 02 + 03 done; TVM_SOURCE_DIR and MLC_LLM_HOME exported; Python 3.11
# env active; emsdk sourced (WASM target).
#
#     bash scripts/eviction/04-compile-model-lib.sh
#
# Time/space: weight convert + compile ~20–40 min, ~6 GB (weights + lib).

set -euxo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MLC_LLM_HOME="${MLC_LLM_HOME:-$REPO_ROOT/3rdparty/mlc-llm}"
DIST="${DIST:-$MLC_LLM_HOME/dist}"

MODEL_ID="Qwen2.5-3B-Instruct"
QUANT="q4f16_1"
HF_SRC="${HF_SRC:-Qwen/Qwen2.5-3B-Instruct}"   # HF weights to convert
MODEL_TAG="${MODEL_ID}-${QUANT}-MLC"

command -v mlc_llm >/dev/null || { echo "mlc_llm CLI missing — run 03-build-mlc-llm.sh"; exit 1; }
emcc --version >/dev/null     || { echo "emcc missing — source emsdk_env.sh";           exit 1; }

mkdir -p "$DIST/$MODEL_TAG" "$DIST/libs"

# --- 1. convert weights to MLC q4f16_1 ---------------------------------------
# (skip if already converted)
if [[ ! -f "$DIST/$MODEL_TAG/ndarray-cache.json" ]]; then
  mlc_llm convert_weight "$HF_SRC" \
    --quantization "$QUANT" \
    -o "$DIST/$MODEL_TAG"
fi

# --- 2. generate the model config (mlc-chat-config.json) ----------------------
if [[ ! -f "$DIST/$MODEL_TAG/mlc-chat-config.json" ]]; then
  mlc_llm gen_config "$HF_SRC" \
    --quantization "$QUANT" \
    --conv-template qwen2 \
    --context-window-size 32768 \
    -o "$DIST/$MODEL_TAG"
fi

# --- 3. compile the WebGPU/WASM model library --------------------------------
# The .wasm output is what web-llm loads. THIS is the artifact that changes when
# you edit paged_kv_cache.cc (via the relink in 02) or the MLC-LLM model def.
LIB_OUT="$DIST/libs/${MODEL_TAG}-webgpu.wasm"
mlc_llm compile "$DIST/$MODEL_TAG/mlc-chat-config.json" \
  --device webgpu \
  -o "$LIB_OUT"

set +x
cat <<EOF

============================================================================
Local model library compiled:
  weights : $DIST/$MODEL_TAG
  lib     : $LIB_OUT

--- Load the SELF-COMPILED lib in the web-llm fork --------------------------
1. Serve the dist dir (CORS):
     export MLC_LLM_HOME="$MLC_LLM_HOME"
     bash scripts/serve_mlc_llm_dist.sh          # http-server :8000

2. In the example app, register a custom appConfig model_list entry pointing
   model_url -> http://localhost:8000/$MODEL_TAG/
   model_lib -> http://localhost:8000/libs/$(basename "$LIB_OUT")
   model_id  -> "${MODEL_TAG}-local"
   (see docs/eviction/VAS-87-build-toolchain.md §6 for the exact snippet)

3. npm run build (root) + npm start (example) and confirm it chats.

--- Prove the edit->compile->run loop (acceptance criterion) ----------------
Make a trivial, observable change in
  3rdparty/tvm-unity/src/runtime/relax_vm/paged_kv_cache.cc
(e.g. LOG(INFO) on cache init), re-run 02 (native+web) and this script, reload
the browser, and confirm the marker appears. See guide §7.
============================================================================
EOF
