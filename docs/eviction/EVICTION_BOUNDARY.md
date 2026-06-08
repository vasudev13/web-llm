# KV-Cache Eviction: Runtime / JS Boundary (VAS-52)

Scaffold for a **pluggable, attention-aware** KV-cache eviction framework
(StreamingLLM / H2O / SnapKV / PyramidKV) for long-context browser inference.

This document draws the boundary the issue asks for: **what lives in TVM, what lives
in MLC-LLM, and what lives in web-llm (TypeScript)** — and the concrete hook points a
policy attaches to. It is the design contract; later issues implement the pieces.

> ⚠️ Scope correction (Jun 2026 repo check): the KV cache is **not** in the web-llm JS
> layer. It is managed in Apache TVM's `src/runtime/relax_vm/paged_kv_cache.cc`, driven
> by MLC-LLM Python model definitions, and surfaced to web-llm only as opaque compiled
> globals (`vm.builtin.attention_kv_cache_*`). Any eviction logic must touch
> TVM / MLC-LLM, not just the web-llm TypeScript package.

## Layer responsibilities

| Layer | Repo | Responsibility for eviction |
| --- | --- | --- |
| **web-llm (TS)** | `mlc-ai/web-llm` | Declarative only. Selects a policy + budget (`EvictionConfig`), exposes it in `ChatConfig`/the demo UI, and forwards it to the runtime. Never manipulates pages directly. `src/eviction_policy.ts`. |
| **MLC-LLM (Python)** | `mlc-ai/mlc-llm` | Model definitions wire the policy through at compile time and bind the runtime control hooks. Decides which attention call returns scores (for H2O/SnapKV). |
| **TVM (C++/WGSL)** | `apache/tvm` | The actual KV cache. `PagedKVCache` page-table manipulation, sink/window retention, and score-based selection kernels live here. This is where eviction physically happens. |

## Hook points in `paged_kv_cache.cc`

The policies attach to the `PagedKVCacheObj` (`src/runtime/relax_vm/paged_kv_cache.cc`).
Precedent to mirror: **`EnableSlidingWindowForSeq(seq_id, window_size, sink_size)`**
(apache/tvm#16729) — the existing static sliding-window + attention-sink path. It is the
template for how a retention policy plugs into the page table, and it is the
differentiation baseline (VAS-86).

Attach points, by policy phase:

1. **Sequence setup** — `AddSequence` / `EnableSlidingWindowForSeq`.
   StreamingLLM (VAS-53) reuses this almost verbatim: `window = budget - sink`,
   `sink = sinkTokens`. No new kernel. The TS side (`StreamingLLMEvictionPolicy`) is
   implemented and emits a normalized `{budget, sinkTokens, windowSize}` config for this
   hook; binding it to `EnableSlidingWindowForSeq` in the runtime is the remaining piece
   and depends on the local toolchain (VAS-87).

2. **Prefill-time selection** — after the prefill attention compute, before pages are
   committed. SnapKV (VAS-56) / PyramidKV (VAS-57) pool attention over the observation
   window and select which earlier pages to retain. Needs a **pooled-select WGSL
   kernel** and an attention-score read path (de-risked in VAS-47).

3. **Decode-time eviction** — invoked every `evictionInterval` decode steps. H2O (VAS-60)
   accumulates per-token attention mass and drops the lowest heavy-hitters, guarding the
   `sinkTokens` prefix and the recent `windowSize`. Reuses the score read path.

4. **Score extraction** — the cross-cutting dependency for H2O/SnapKV/PyramidKV: getting
   attention scores out of the MLC-compiled FlashAttention kernel. Tracked as a spike
   (VAS-47) because the fused kernel does not currently surface per-token scores.

`NoOp` (default) attaches **nothing** — none of the above fire, so the engine runs
unchanged. This is the VAS-52 safety guarantee.

## `EvictionConfig` → hook mapping

`EvictionConfig` (see `src/eviction_policy.ts`) is the single declarative shape passed
from web-llm down to the runtime hooks:

| Field | Consumed by | Hook point |
| --- | --- | --- |
| `kind` | dispatch | selects which hooks bind (or none, for `NoOp`) |
| `budget` (ratio or absolute) | all | retained-page count |
| `sinkTokens` | StreamingLLM/H2O/SnapKV/PyramidKV | sequence setup (sink) |
| `windowSize` | StreamingLLM/H2O/SnapKV | sequence setup / recent guard |
| `observationWindow` | SnapKV/PyramidKV | prefill-time selection |
| `evictionInterval` | H2O | decode-time eviction cadence |
| `pyramidAlpha` | PyramidKV | per-layer budget at selection |

The ablation axes (VAS-65) map 1:1 onto `sinkTokens`, `observationWindow`,
`pyramidAlpha`, and `evictionInterval`, so sweeps need no code changes.

## Status of this scaffold (VAS-52)

- [x] `EvictionPolicy` + `EvictionConfig` defined, runtime/JS boundary drawn (this doc).
- [x] No-op policy; optional `ChatConfig.eviction_config` defaulting to no-op → engine
  runs unchanged. Public API exports added in `src/index.ts`.
- [x] Hook points in `paged_kv_cache.cc` documented (above).
- [ ] TVM/MLC-LLM-side hook binding — depends on the local eviction toolchain
  (VAS-87) and the attention-score spike (VAS-47). Out of scope for the TS scaffold.

## StreamingLLM TS policy (VAS-53)

- [x] `StreamingLLMEvictionPolicy` implemented (sink + sliding window, no new kernels).
  Resolves `budget` (ratio or absolute) → absolute token count, defaults `sinkTokens` to
  4, derives `windowSize = budget - sink` (or honors an explicit window), and emits a
  normalized `{kind, budget, sinkTokens, windowSize}` config for the sequence-setup hook.
- [x] `createEvictionPolicy()` factory + `resolveBudgetTokens()` helper added; exported
  from `src/index.ts`. Unit tests in `tests/eviction_policy.test.ts`.
- [ ] Runtime binding to `EnableSlidingWindowForSeq` + sparse position IDs (plan §3.5
  Option 1) — depends on VAS-87. Until then the policy resolves config but no eviction
  physically fires; the engine still runs full-cache.

> Note: per VAS-53, StreamingLLM is the **differentiation baseline** (it maps onto a
> primitive TVM already ships), not a novel contribution — it is the bar SnapKV/PyramidKV
> must beat at equal budget (VAS-86) and the end-to-end validation of the VAS-52 plumbing.

## SnapKV TS policy + pooled-select kernel (VAS-56)

SnapKV is the **★ primary attention-aware policy** for the EMNLP demo — the headline
result the StreamingLLM baseline must be beaten by. It attaches at **hook point 2
(prefill-time selection)** and adds **zero decode-time dispatches**: one-shot selection
at the end of prefill, fixed budget thereafter. At batch=1 in the browser the per-token
dispatch cost dominates (H3), so a policy with no per-step eviction is predicted to beat
rolling H2O in-browser (H1/H3) — this is the reason SnapKV is the recommended route.

- [x] `SnapKVEvictionPolicy` implemented (TS declarative layer). Resolves `budget`
  (ratio or absolute) → absolute token count; defaults `sinkTokens` → 4,
  `observationWindow` → 32, `poolingKernelSize` → 7 (SnapKV paper defaults); validates
  `poolingKernelSize` is a positive **odd** integer and that
  `sinkTokens + observationWindow < budget` (room to select earlier tokens). Emits a
  normalized `{kind, budget, sinkTokens, observationWindow, poolingKernelSize}` config.
- [x] New `EvictionConfig.poolingKernelSize` field (ablation axis, VAS-65) + defaults
  `DEFAULT_OBSERVATION_WINDOW`, `DEFAULT_POOLING_KERNEL_SIZE`. `createEvictionPolicy()`
  dispatches `snapkv → SnapKVEvictionPolicy`. Exported from `src/index.ts`; unit tests in
  `tests/eviction_policy.test.ts`.
- [x] `pooled_select` WGSL kernel **design reference** written —
  `docs/eviction/snapkv_pooled_select.wgsl`. Three stages: (1) score earlier keys with
  the observation-window queries only — the last `obs_window` rows of the attention
  matrix, `obs_window << seq_len`, so the full matrix is never materialized; (2)
  symmetric avg-pool of width `poolingKernelSize`; (3) top-K → retained page list for the
  PagedKVCache compaction hook. Sink prefix forced to the top via an `+inf` sentinel;
  retained K keep their prefill RoPE (sparse position IDs, plan §3.5 Option 1).
- [ ] Runtime binding of the kernel into a compiled model lib + the prefill-end
  compaction hook in `paged_kv_cache.cc`. Gated on the attention-score read path (VAS-47)
  and the local eviction toolchain (VAS-87). Until then the policy resolves config but no
  eviction physically fires; the engine still runs full-cache.

> PyramidKV (VAS-57) is the per-layer-budget variant of SnapKV — it reuses this kernel
> with a per-layer `select_count` (steeper budget in lower layers via `pyramidAlpha`), so
> `SnapKVEvictionPolicy` is written with a `protected` config to be subclassable.
