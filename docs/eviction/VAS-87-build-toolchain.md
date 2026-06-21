# VAS-87 — Build TVM + MLC-LLM from source & compile a model lib locally

**Milestone:** Phase 0 (Environment Setup) · **Priority:** Urgent
**Status:** In Progress — reproducible toolchain (guide + scripts) landed; the
multi-hour native compile + the `paged_kv_cache.cc` round-trip proof are the
remaining hands-on steps (need an interactive machine, see §8).

> Split out from [VAS-48](https://linear.app/vasudev13/issue/VAS-48). VAS-48
> built the **web-llm JavaScript fork** and ran a *prebuilt* model in Chrome —
> the outer layer. It does **not** let us change eviction. The KV cache and
> attention kernels live one layer down, in **Apache TVM**
> (`src/runtime/relax_vm/paged_kv_cache.cc`, C++) and **MLC-LLM** (Python model
> definitions), which compile into the `.wasm`/`.wgsl` artifacts web-llm loads.
> To *add* eviction we must edit that layer and **recompile model libraries
> ourselves** — which needs the full TVM + MLC-LLM build-from-source toolchain
> working locally. This ticket owns that toolchain.

This is the real foundation for all Phase 1 Track B work. The
[VAS-47 spike](./VAS-47-attention-score-extraction-spike.md) already concluded
its empirical de-risking (Q/K buffer access, 8B recompute cost) is **gated on
VAS-87** — so this unblocks VAS-56 (SnapKV), VAS-57 (PyramidKV), VAS-60 (H2O),
and VAS-52's TVM-side hook.

---

## 1. The layer cake (why a JS `npm install` is not enough)

```
┌─────────────────────────────────────────────────────────────┐
│ web-llm (TypeScript)   ← VAS-48 done. Loads opaque compiled   │
│   engine.ts, llm_chat.ts  globals: vm.builtin.attention_kv_   │
│                           cache_* . Cannot see/alter the KV   │
│                           cache from here (verified VAS-47).  │
├─────────────────────────────────────────────────────────────┤
│ @mlc-ai/web-runtime (tvmjs)  ← prebuilt npm pkg today; built  │
│   the WASM/WebGPU runtime      from source by step 02 (tvm/web)│
├─────────────────────────────────────────────────────────────┤
│ MLC-LLM (Python)       ← model defs: KV-cache wiring, attn.    │
│   nn.kv_cache, model/*  `mlc_llm compile` produces the .wasm.  │  ← step 03
├─────────────────────────────────────────────────────────────┤
│ Apache TVM / relax (C++)  ← THE KV CACHE LIVES HERE:           │
│   src/runtime/relax_vm/      paged_kv_cache.cc                 │  ← step 02
│   FlashInfer attention kernels                                │
└─────────────────────────────────────────────────────────────┘
```

Eviction logic lands as: **a WGSL kernel + a compaction hook in
`paged_kv_cache.cc`** (the VAS-47 route), compiled through MLC-LLM into a model
lib. None of that is reachable without building these two bottom layers from
source. That is exactly what this ticket establishes.

## 2. Source pins (matched to the fork, not guessed)

The repo's existing `scripts/prep_deps.sh` already encodes the canonical TVM
source, so we pin to it rather than upstream `apache/tvm`:

| Component        | Source                                   | Notes |
|------------------|------------------------------------------|-------|
| TVM ("tvm-unity")| `github.com/mlc-ai/relax` (recursive)    | exactly what `prep_deps.sh` clones → `3rdparty/tvm-unity` |
| MLC-LLM          | `github.com/mlc-ai/mlc-llm` (recursive)  | provides `mlc_llm` compiler CLI |
| Browser runtime  | `@mlc-ai/web-runtime ^0.25.0-dev0`       | from `package.json`; built from `tvm/web` in step 02 |
| Tokenizers       | `@mlc-ai/web-tokenizers ^0.1.6`          | unchanged, prebuilt |
| Model (target)   | `Qwen2.5-3B-Instruct` → `q4f16_1`        | project-standard test model (~2.5 GB VRAM) |

**Toolchain version floors** (enforced by `00-audit-toolchain.sh`):

| Tool        | Floor    | Why |
|-------------|----------|-----|
| LLVM        | 17       | TVM host codegen (`llvm-config`) |
| Emscripten  | 3.1.56   | WASM + stable WebGPU (Dawn) — used by `tvm/web make` |
| CMake       | 3.24     | TVM/MLC build |
| Ninja       | any      | build generator |
| Rust        | stable   | xgrammar / some relax components |
| Python      | 3.11     | mlc-llm build target (system 3.9 is too old) |
| Node        | 20+      | `.nvmrc` pins v24.11.1 |

## 3. Scripts (run in order)

All under `scripts/eviction/`, each idempotent and re-runnable:

| # | Script | What it does | Mutates host? |
|---|--------|--------------|---------------|
| 00 | `00-audit-toolchain.sh` | Read-only audit vs the floors above; prints a verdict. | No |
| 01 | `01-bootstrap-prereqs.sh` | Install cmake/ninja/llvm@17 (brew), rustup, emsdk 3.1.56, Python 3.11 (pyenv). | **Yes** |
| 02 | `02-build-tvm.sh` | Clone `mlc-ai/relax`, build native libtvm (Metal) + `tvm/web` WASM runtime, symlink `tvm_home`. | repo + `3rdparty/` |
| 03 | `03-build-mlc-llm.sh` | Clone `mlc-ai/mlc-llm`, build libmlc_llm, `pip install -e` tvm + mlc_llm. | `3rdparty/` + active py env |
| 04 | `04-compile-model-lib.sh` | `convert_weight` + `gen_config` + `compile` Qwen2.5-3B-q4f16 → `*-webgpu.wasm`. | `3rdparty/mlc-llm/dist/` |

```bash
bash scripts/eviction/00-audit-toolchain.sh      # see the gaps
bash scripts/eviction/01-bootstrap-prereqs.sh    # close them (one-time)
#   then add the printed env lines to ~/.zshrc and open a NEW shell
export TVM_SOURCE_DIR="$PWD/3rdparty/tvm-unity"
bash scripts/eviction/02-build-tvm.sh            # ~30–60 min
export MLC_LLM_HOME="$PWD/3rdparty/mlc-llm"
bash scripts/eviction/03-build-mlc-llm.sh        # ~15–30 min
bash scripts/eviction/04-compile-model-lib.sh    # ~20–40 min
```

## 4. Required environment (every build shell)

```bash
export PATH="$(brew --prefix llvm@17)/bin:$PATH"   # llvm-config
source "$HOME/.cargo/env"                           # rustc/cargo
source "$HOME/emsdk/emsdk_env.sh"                    # emcc (WASM/WebGPU)
pyenv local 3.11.9                                   # mlc-llm build python
export TVM_SOURCE_DIR="$PWD/3rdparty/tvm-unity"
export MLC_LLM_HOME="$PWD/3rdparty/mlc-llm"
```

## 5. Verified baseline on the reference machine (M4 Air)

Captured by `00-audit-toolchain.sh` on the dev machine (macOS 15.6, arm64) at
the time this ticket landed — i.e. the *starting* state before bootstrap:

```
✓ git, clang 17, make, pkg-config, node v24.13.1, npm 11.8.0, brew, pyenv 2.6.26
✗ cmake        — NOT FOUND   → brew (step 01)
✗ ninja        — NOT FOUND   → brew (step 01)
✗ llvm-config  — NOT FOUND   → brew install llvm@17 (step 01)
✗ rustc        — NOT FOUND   → rustup (step 01)
✗ emcc         — NOT FOUND   → emsdk 3.1.56 (step 01)
! python 3.9   — too old     → pyenv 3.11.9 (step 01)
```

Verdict: **5 core prerequisites missing**; `pyenv` already present, so the
Python 3.11 env is one `pyenv install` away. Re-run 00 after step 01 — it should
flip to all-green before attempting step 02.

## 6. Loading the self-compiled lib in the web-llm fork

After step 04, serve the dist and register a custom model in the example's
`appConfig` (this is what makes web-llm load *our* artifact, not a CDN one):

```bash
export MLC_LLM_HOME="$PWD/3rdparty/mlc-llm"
bash scripts/serve_mlc_llm_dist.sh          # http-server on :8000 with CORS
```

```ts
// examples/get-started/src/get_started.ts
const appConfig: webllm.AppConfig = {
  model_list: [{
    model:     "http://localhost:8000/Qwen2.5-3B-Instruct-q4f16_1-MLC/",
    model_id:  "Qwen2.5-3B-Instruct-q4f16_1-MLC-local",
    model_lib: "http://localhost:8000/libs/Qwen2.5-3B-Instruct-q4f16_1-MLC-webgpu.wasm",
  }],
};
const engine = await webllm.CreateMLCEngine(
  "Qwen2.5-3B-Instruct-q4f16_1-MLC-local",
  { appConfig },
);
```

Then `npm run build` at repo root and `npm install && npm start` in the example
(per the project-standard run path) and confirm it chats.

## 7. Acceptance criteria → how each is met

| Criterion | Status | Evidence / how |
|-----------|--------|----------------|
| Build steps documented for reproducibility | ✅ Done | this guide + 5 committed, syntax-checked scripts; versions pinned in §2 |
| Toolchain versions recorded | ✅ Done | §2 floors + §5 verified machine baseline (script-captured) |
| Compile a model lib from source & run in browser (not prebuilt) | ⏳ Scripted, not yet executed | `02→03→04` + §6 loader; needs the multi-hour interactive run (§8) |
| Trivial `paged_kv_cache.cc` change propagates to the running model | ⏳ Procedure defined | §7 below; gates VAS-56/57/60 |

### The edit→compile→run proof (the loop that matters)

1. Add an observable marker in
   `3rdparty/tvm-unity/src/runtime/relax_vm/paged_kv_cache.cc` — e.g.
   `LOG(INFO) << "[VAS-87] PagedKVCache init, num_layers=" << num_layers;`
   in the cache constructor.
2. Re-run `02-build-tvm.sh` (rebuilds native libtvm **and** the `tvm/web` WASM
   runtime) then `04-compile-model-lib.sh` (recompiles the model lib).
3. Reload the browser with the self-compiled lib (§6) and confirm the marker
   shows up (console for a JS-surfaced log, or a counter wired into the demo).

Once that marker round-trips, the full **C++ → WASM → browser** edit loop is
proven and the real eviction kernels (VAS-56 WGSL + compaction hook) can land.

## 8. Why this ticket is "In Progress", not "Done"

Steps 02–04 are a **multi-hour, multi-GB, system-mutating** build (emsdk + LLVM
+ two recursive source trees + a 3B weight conversion). The scheduled
(unattended) run that produced this deliverable deliberately did **not** kick
off a host-wide install/compile of that size. What's committed is the
reproducible, version-pinned, audited path so the remaining work is a clean
`01 → 02 → 03 → 04 → §7-proof` execution on an interactive machine. Recommend
moving to Done only after the `paged_kv_cache.cc` marker round-trips in Chrome.

---

*Refs:* `scripts/prep_deps.sh` (canonical TVM source) ·
[VAS-47 spike](./VAS-47-attention-score-extraction-spike.md) (eviction route,
gated on this) · [VAS-48](https://linear.app/vasudev13/issue/VAS-48) (parent).
