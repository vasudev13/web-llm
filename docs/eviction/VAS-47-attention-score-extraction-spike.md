# VAS-47 — Spike: attention-score extraction from the MLC-compiled FlashAttention kernel

**Status:** Decision recorded (go/no-go) — empirical de-risking pending local toolchain (VAS-87).
**Priority:** Urgent · **Milestone:** Phase 0 · **Risk:** Highest (critical path for H2O/SnapKV/PyramidKV).

> **TL;DR — GO, via recomputation, not extraction.**
> Do **not** try to read attention scores out of the fused FlashAttention kernel. Instead
> **recompute** the small `[observation_window, seq_len]` score block from `Q_obs · Kᵀ` in a
> dedicated WGSL kernel at the end of prefill (the SnapKV route, VAS-56). Score-free proxies
> (key-L2 / KeyDiff) are the fallback if recomputation proves too costly at 8B. Exact
> full-attention H2O is deferred to a research stretch.

---

## 1. The question

H2O, SnapKV, and PyramidKV are all **attention-aware**: they decide which KV positions to keep
by ranking each past key by the attention mass it has received. That requires per-token (really
per-key) attention scores. The spike asks: **can we get those scores in the browser, and if so,
where does the work physically land — TVM, MLC-LLM, or a separate WGSL kernel?**

## 2. Why this is hard (verified against the live repo, Jun 2026)

Two architectural facts, both confirmed in this codebase, make naïve score extraction infeasible:

1. **The KV cache and attention kernels are not in web-llm.** The web-llm TypeScript layer only
   holds **opaque compiled handles**. Every cache operation is a `vm.builtin.attention_kv_cache_*`
   `PackedFunc` resolved from the compiled WASM at load time — e.g.
   `src/llm_chat.ts:415` binds `vm.builtin.attention_kv_cache_enable_sliding_window_for_seq`,
   and `src/llm_chat.ts:236` / `:1336` reference `create_tir_paged_kv_cache`. There is **no
   attention-score buffer surfaced to JS** anywhere in `src/`. The cache lives in Apache TVM's
   `src/runtime/relax_vm/paged_kv_cache.cc`, driven by MLC-LLM model definitions, and compiles to
   the `.wasm`/`.wgsl` web-llm loads. (This matches `docs/eviction/EVICTION_BOUNDARY.md`.)

2. **FlashAttention never materializes the attention matrix.** The MLC-compiled attention path is
   FlashAttention-class (fused softmax×V with online running max/sum). It **deliberately never
   writes the full `[seq, seq]` matrix to global memory** — that fusion is the entire point of the
   design. So there is no `scores` tensor to tap, and forcing one would mean either (a) a
   non-fused scoring pass, or (b) modifying the fused kernel to spill scores — both expensive and
   both landing inside TVM/FlashInfer, not web-llm.

**Conclusion of §2:** "extract the scores the model already computed" is a dead end. The scores
the demo needs are never persisted, and the layer we own (web-llm TS) cannot see them.

## 3. Options considered

| # | Route | Where it lands | Cost | Accuracy | Verdict |
|---|-------|----------------|------|----------|---------|
| **A** | **SnapKV via custom WGSL pooled-select** — recompute `Q_obs·Kᵀ` for the last `obs_window` queries only, once at prefill end | New WGSL kernel + TVM compaction hook | Bounded one-shot prefill cost `O(obs_window · seq_len · head_dim)`; **zero per-decode dispatch** | Real attention-aware scores (observation-window approximation, per SnapKV paper) | ✅ **PRIMARY** |
| **B** | **Score-free proxies** — key-L2-norm / KeyDiff ranking; no attention scores at all | WGSL kernel over `K` only | Cheapest; no `Q` needed | Lower; sidesteps the score problem | 🟡 **FALLBACK** |
| **C** | **StreamingLLM / attention-sink** — no scores, static sink + sliding window | Reuses TVM `enable_sliding_window_for_seq` (already bound, `llm_chat.ts:415`) | ~Free | N/A (position-based, not attention-aware) | ⚪ **BASELINE to beat** (VAS-53/VAS-86), not novel |
| **D** | **Exact full-attention H2O** — running cumulative attention mass per key per head | Non-fused scoring pass or fused-kernel modification in TVM/FlashInfer | High (defeats FlashAttention fusion; per-step dispatch — H3) | Highest | 🔴 **DEFERRED** (research stretch) |

## 4. The decision — Option A (recompute, don't extract)

**The key reframe: SnapKV does not need the full attention matrix.** It needs only the **last
`obs_window` rows** of it — the scores the observation-window queries place on earlier keys. That
is an `[obs_window, seq_len]` block with `obs_window << seq_len` (paper default `obs_window = 32`).
Computing it directly from `Q_obs · Kᵀ` is cheap and **completely sidesteps** the fused-kernel
problem: we never touch FlashAttention's internals, we just read the same `Q` and `K` buffers it
reads and do a small, separate matmul.

This is exactly what the VAS-56 design-reference kernel does
(`docs/eviction/snapkv_pooled_select.wgsl`):

- **Bindings** are `q_obs : [num_heads, obs_window, head_dim]` and `k_all : [num_heads, seq_len,
  head_dim]` — i.e. it reads **Q and K, not scores**. (Confirmed in the kernel source.)
- **Step 1** `score_obs_window`: for each candidate key `j`, accumulate
  `Σ_i exp(Q_obs[i]·K[j] / √head_dim)` over the `obs_window` queries. Exact softmax normalization is
  skipped — we only **rank** keys, never use magnitudes — so no online-max bookkeeping is needed.
- **Step 2** `avg_pool`: symmetric avg-pool of width `poolingKernelSize` (odd) so coherent spans
  win over lone spikes (SnapKV §3.2). The sink prefix is forced to the top with a `+inf` sentinel.
- **Step 3** (separate small dispatch): top-`(budget − sinkTokens − obs_window)` over `pooled` per
  head → retained position list → PagedKVCache page-table compaction at the prefill-end hook.
  Retained K keep their prefill RoPE (sparse position IDs, plan §3.5 Option 1) — no re-rope.

**Why this also wins on systems grounds (H1/H3):** selection is **one-shot at prefill end**; after
that the retained set is fixed and there is **zero per-decode-step eviction dispatch**. At batch=1
in the browser, per-token dispatch overhead dominates (H3), so a one-shot policy is predicted to
beat rolling H2O (Option D) in-browser — which is *why* SnapKV is the recommended route, not just a
convenient one.

### Answers to the acceptance criteria

- **Go/no-go on per-head score extraction, with the chosen route:** **GO** — per-head scores are
  obtained by **recomputation** (`Q_obs·Kᵀ` over the observation window), **not** by extracting
  them from the fused kernel. Per-head granularity falls out naturally because the kernel iterates
  per KV head (GQA: KV heads, not query heads).
- **Where the work lands:** a **separate WGSL kernel** (the pooled-select kernel) hosted in a
  locally compiled model lib, plus a **prefill-end compaction hook in TVM**
  (`paged_kv_cache.cc`, mirroring `EnableSlidingWindowForSeq` / apache/tvm#16729). **Nothing**
  lands in the web-llm TS layer beyond the existing declarative `EvictionConfig`. **No change to
  the fused FlashAttention kernel is required** for Options A/B/C.
- **Decision recorded:** **SnapKV-WGSL primary** (Option A), **score-free proxies fallback**
  (Option B), **exact full-attention H2O deferred** (Option D). StreamingLLM (Option C) is the
  baseline to beat (VAS-53 / VAS-86), not a contribution.

## 5. Residual risk & what remains to fully close the spike

The decision is sound on architecture, but two items need the **local eviction toolchain
(VAS-87)** to be empirically de-risked before SnapKV (VAS-56) can move from "config resolves" to
"eviction physically fires":

1. **Q/K buffer access at prefill end.** The kernel assumes `Q_obs` and `K_all` are reachable as
   storage buffers at the prefill-end hook. Confirm the PagedKVCache lays K out as
   `[num_heads, seq_len, head_dim]` contiguously (or add a gather) and that the observation-window
   Q is still resident (it is the prefill tail) when the hook fires. **Owner: VAS-87 + VAS-56.**
2. **Recompute cost at 8B / long context.** `O(obs_window · seq_len · head_dim · num_kv_heads)` is
   bounded but not free at `seq_len = 32K`. Measure it against the prefill it rides on; if it is a
   material fraction, fall back to Option B (key-L2/KeyDiff, no `Q`, no matmul). **Owner: VAS-59
   systems eval.**

Neither residual risk threatens the **go** decision — both have a defined fallback (Option B), and
the recomputation route is the consensus approach in the on-device eviction literature precisely
because it avoids fused-kernel surgery.

## 6. Cross-references

- `docs/eviction/EVICTION_BOUNDARY.md` — layer responsibilities + hook points (VAS-52/53/56).
- `docs/eviction/snapkv_pooled_select.wgsl` — the design-reference kernel this spike validates (VAS-56).
- `src/llm_chat.ts:415` — `enable_sliding_window_for_seq` binding (proof the TS layer sees only opaque handles; the StreamingLLM/Option-C path).
- Prior art: apache/tvm#16729 (sliding-window+sink baseline), NVIDIA kvpress (correctness oracle), SnapKV paper §3.2 (pooling).
- Unblocks: **VAS-56** (SnapKV impl), **VAS-57** (PyramidKV), and informs **VAS-60** (H2O, deferred).
- Gated on: **VAS-87** (local TVM/MLC-LLM build) for empirical de-risking of §5.
