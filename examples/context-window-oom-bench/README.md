# Context-Window / OOM-Cliff Benchmark (VAS-49)

Reproduces the **4K context cap**, measures the **OOM cliff**, and exercises the
**PagedKVCache instrumentation** added in `src/llm_chat.ts`. It sweeps
`context_window_size` across a 3B and a 7B model, pushes prompt + generation
toward the configured window, and records per-run telemetry plus an estimated
peak-VRAM breakdown. Each run is labelled **PASS / CONTEXT_CAP /
OOM_DEVICE_LOST / OTHER_ERROR** so the OOM cliff falls straight out of the
table.

This is the motivating measurement for the _KV-Cache Eviction for Browser LLM
Inference_ project: it quantifies the wall and gives an early read on whether
**7B @ 16K is viable on an M4 Air**, or whether to pivot to 3B (Risk #4).

> **WebGPU required.** Empirical OOM numbers can only be produced in a browser
> with WebGPU (e.g. Chrome on the M4 Air). They **cannot** be reproduced in a
> headless / CI container — that's why this repo ships the harness + an
> analytical estimate, and you fill the empirical table below after running.

## Run

> **Uses the local instrumented source.** This example reads
> `engine.getKVCacheMetrics()`, which only exists in this branch's source — it
> is **not** in the published npm package. The example aliases `@mlc-ai/web-llm`
> directly to `../../src/index.ts` (Parcel compiles the TypeScript), so no
> separate build step is needed — but the root `node_modules` must be installed
> so the transitive deps (tvmjs, tokenizers, …) resolve.

```bash
# 1. From the repo root: install deps (provides tvmjs etc.). No build needed.
cd /path/to/web-llm
npm install

# 2. Run the benchmark example (resolves @mlc-ai/web-llm to ../../src)
cd examples/context-window-oom-bench
npm install
npm start          # serves on http://localhost:8888
```

**Verify instrumentation is live** (the earlier blank-VRAM runs happened because
a stale npm package was being served): the console should show
`PagedKVCache allocation: ...` and `PagedKVCache memory estimate: ...` lines at
model load, and the first result row should have non-empty `kvCacheMB` /
`estPeakVRAMMB`. If you instead see `getKVCacheMetrics is not a function`, a
stale build is cached — fix with:

```bash
# in examples/context-window-oom-bench
rm -rf node_modules .parcel-cache dist
npm install
npm start
```

`npm start` already clears `.parcel-cache` on each launch, so editing root
`src/` and restarting is enough to pick up changes.

Open the page in Chrome and **open the console**. You'll see:

- Model download / load progress.
- The PagedKVCache instrumentation logs per load, e.g.
  `PagedKVCache allocation: maxTotalSeqLen=16384, pages=1024 x pageSize=16, ...`
  and `PagedKVCache memory estimate: params=..MB, tempBuffers=..MB, kvCache=..MB, estPeakVRAM=..MB`.
- A live HTML results table, and on completion a `console.table`, an **OOM cliff
  summary**, and **CSV** + **JSON** dumps you can paste into the template below.

### URL modes (no code editing)

| URL                             | What it does                                                                   |
| ------------------------------- | ------------------------------------------------------------------------------ |
| `…/?coverage`                   | **Full coverage (recommended)** — runs the entire VAS-49 matrix in one launch. |
| `http://localhost:8888`         | **Full sweep** — both models, `[2048, 4096, 8192, 10240]` (minutes).           |
| `…/?smoke`                      | **Smoke** — 3B only, `[2048, 4096]` (~1 min sanity check).                     |
| `…/?fast`                       | **Fast cliff-finder** — `max_tokens=8`, sweeps up through 16384.               |
| `…/?ctx=11264,12288`            | **Explicit context list** — probe exactly these sizes.                         |
| `…/?fast&ctx=11264,12288,16384` | Fast mode over a custom list — the safe way to pin the exact OOM cliff.        |

**`?coverage` — the one-shot that addresses every acceptance gap.** Open
`http://localhost:8888/?coverage` and it runs three phases **per model** (3B and
7B), labelling each row with a `phase` column:

1. **`cap-4k`** — explicit 4K context-cap reproduction (full decode); expects
   `finish_reason="length"` / `CONTEXT_CAP` at `ctx=4096` (issue #752).
2. **`vram-perf`** — VRAM + throughput curve over the known-safe sizes
   (`2048, 4096, 8192, 10240`), full decode.
3. **`cliff`** — a fast (`max_tokens=8`) auto-escalating climb up
   `8192 … 65536` that **stops each model at its first OOM**, so the OOM cliff
   is found for **both** 3B and 7B. Nothing past a model's cliff is attempted.

Every run also records the real-memory probe (`measuredMemMB`), so the estimated
peak VRAM is validated against observed memory. Combined with crash-recovery
(below), if a phase-3 climb crashes the machine, just reload `?coverage` — it
records that size as the cliff, skips it, and continues.

**Why the fast cliff probe is safe at high context:** the OOM cliff is hit at
KV-cache **allocation / prefill** time, not during long decode — so `max_tokens=8`
reproduces the crash in seconds, without generating hundreds of tokens. (`?fast`
exposes this mode standalone; `?coverage` uses it for phase 3.)

The active mode, phase, and context list are printed to the console and status
line at startup.

### Other config

Edit the config block at the top of `src/context_window_oom_bench.ts`:

- `ALL_MODELS` — defaults to `Llama-3.2-3B-Instruct-q4f16_1-MLC` (3B) and
  `Qwen2.5-7B-Instruct-q4f16_1-MLC` (7B). The full sweep uses both; smoke mode
  uses just the first.
- `CONTEXT_SIZES` — full sweep is `[2048, 4096, 8192, 10240]` (capped below the
  12288/16384 sizes that crashed a 16 GB M4 Air); smoke mode is `[2048, 4096]`.
  Raise the cap only if you have memory headroom and accept the crash risk.
- `PROMPT_FILL_FRACTION` — how much of the window the prompt fills before decode
  reaches the cap (default `0.85`).

> **Runtime warning:** large windows (16K/32K) require a very large prefill and
> can take **minutes** per run. Trim `CONTEXT_SIZES` (or use `?smoke`) while
> iterating.

## Crash recovery (important for the OOM cliff)

A high-context run can be violent enough to crash the browser tab — or even
hang/restart the whole machine — which takes the console logs with it. To make
results survivable:

- **Each result is saved to `localStorage` the instant it finishes.** This is
  on-disk, per-origin storage that survives a tab crash _and_ an OS reboot.
- **Reloading the page recovers everything and resumes** the sweep where it left
  off (already-done `(model, context)` pairs are skipped, not repeated).
- **The run currently in flight is marked before it starts.** If that run hard-
  crashes the tab/machine (so it never records a result), reloading detects the
  leftover marker and records that context size as `OOM_DEVICE_LOST` — which
  pinpoints the OOM cliff — then skips it so you don't crash on the same size
  again.
- **Download CSV / Download JSON** buttons export whatever has been collected so
  far, at any time. **Clear saved results** wipes the saved state to start fresh.

So if it dies: just **reopen `http://localhost:8888`**, click **Download JSON**,
and you have your data — including the size that killed it.

## What each run records

`finish_reason`, `prompt_tokens`, `completion_tokens`, peak KV length
(`prompt + completion`), `prefill_tokens_per_s`, `decode_tokens_per_s`,
`e2e_latency_s` (all from `usage` / `usage.extra`), plus the instrumentation
fields from `engine.getKVCacheMetrics()`: `numPages`, `paramBytes`,
`maxTempFuncBytes`, `kvCacheBytes`, `estimatedTotalVRAMBytes`.

**Buffer-limit analysis (is the cliff a per-buffer cap or total memory?).**
The harness also queries the device's `maxStorageBufferBindingSize` once via
`engine.getMaxStorageBufferBindingSize()` and compares it to the largest single
KV storage buffer (`largestKVBufferBytes` from the metrics — one layer's K/V
tensor, which scales with context). Columns:

- `largestKVBufferMB` — biggest single KV allocation at this context size.
- `maxStorageBufferMB` — the device's per-buffer cap.
- `bufferLimitHeadroomPct` — `largestKVBuffer / maxStorageBuffer * 100`.
- `exceedsBufferLimit` — `true` if a single buffer is over the cap.
- `measuredMemMB` — **real** observed memory via
  `performance.measureUserAgentSpecificMemory()` (Chrome + cross-origin
  isolation only; blank otherwise), to validate the _estimated_ peak VRAM.

If a crash occurs while estimated peak VRAM is far below physical memory but
`bufferLimitHeadroomPct` is near/over 100%, the OOM cliff is a **per-buffer
limit**, not total-memory exhaustion — precisely the wall KV paging/eviction is
meant to remove. The final console "OOM cliff summary" prints this diagnosis.

**Outcome classification:**

| Outcome           | Meaning                                                                          |
| ----------------- | -------------------------------------------------------------------------------- |
| `PASS`            | Generation finished naturally (`finish_reason = "stop"`).                        |
| `CONTEXT_CAP`     | Hit the window: `finish_reason = "length"`, or `ContextWindowSizeExceededError`. |
| `OOM_DEVICE_LOST` | WebGPU `device.lost` / `DeviceLostError` / allocation failure (the cliff).       |
| `OTHER_ERROR`     | Anything else (logged in the `error` column).                                    |

## PagedKVCache instrumentation

This example reads `engine.getKVCacheMetrics(modelId)` — added for VAS-49 in
`src/llm_chat.ts` (`KVCacheMetrics`), surfaced through `src/engine.ts`,
`src/types.ts`, and the web-worker engine. It reports the paged allocation
geometry (pages × pageSize, `maxTotalSeqLen`, `prefillChunkSize`) and an
estimated peak-VRAM breakdown computed with the same formula as
`utils/vram_requirements`. The same numbers are also `log.info`-ed at load.

> These are **estimates**, not GPU queries — WebGPU exposes no memory-usage API.
> KV bytes are scaled to the configured context window when a compiled context
> length is available in metadata (`scaledToConfiguredContext`), otherwise the
> compile-time `kv_cache_bytes` is reported unscaled.

## Analytical estimate (early read — confirm empirically)

KV-cache bytes scale linearly with context length:

```
kvBytesPerToken = 2 (K,V) · num_kv_heads · head_dim · num_layers · dtype_bytes
estPeakVRAM(ctx) = paramBytes + maxTempFuncBytes + kvBytesPerToken · ctx
```

Assumptions (q4f16, KV cache fp16 = 2 bytes/elem):

| Model        | layers | kv_heads | head_dim | KV/token | params(≈) |
| ------------ | -----: | -------: | -------: | -------: | --------: |
| Llama-3.2-3B |     28 |        8 |      128 | ~112 KiB |   ~1.9 GB |
| Qwen2.5-7B   |     28 |        4 |      128 |  ~56 KiB |   ~4.5 GB |

Estimated **KV cache** / **total peak VRAM** (params + ~0.4–0.6 GB temp + KV):

| ctx   | 3B KV    | 3B total (≈) | 7B KV    | 7B total (≈) |
| ----- | -------- | ------------ | -------- | ------------ |
| 2048  | ~0.22 GB | ~2.5 GB      | ~0.11 GB | ~5.2 GB      |
| 4096  | ~0.45 GB | ~2.8 GB      | ~0.22 GB | ~5.3 GB      |
| 8192  | ~0.90 GB | ~3.2 GB      | ~0.45 GB | ~5.6 GB      |
| 16384 | ~1.79 GB | ~4.1 GB      | ~0.90 GB | ~6.0 GB      |
| 32768 | ~3.58 GB | ~5.9 GB      | ~1.79 GB | ~6.9 GB      |

**7B @ 16K feasibility (M4 Air, 16 GB unified memory) — pre-run estimate.** The
~6 GB total estimate sits under 16 GB unified memory, so 7B @ 16K looked
feasible on a total-memory basis. **This was partly borne out and partly
overturned by measurement** — see the Empirical results below. Memory was not
the blocker (neither total memory nor the per-buffer cap), but _usability_ was:
7B is already ~5 tok/s with multi-minute latency by 10K, so 16K is functionally
unusable. The measured numbers below supersede this estimate.

## Empirical results (`?coverage` run)

Device: **MacBook Air M4, 16 GB unified memory** · Browser: **Chrome (WebGPU)** ·
`maxStorageBufferBindingSize`: **1024 MB** (1 GB).

> Measured against the local instrumented build (`kvScaled: true`). The
> `?coverage` run produced three phases per model: `cap-4k`, `vram-perf` (full
> decode), and `cliff` (fast, `max_tokens=8`).

### Headline: the crash is a _sustained-decode_ failure, not a KV-allocation cliff

The fast `cliff` phase **allocates and prefills the full context window** (e.g.
65536 → 4096 pages, ~44.6K prompt tokens) but only decodes a handful of tokens.
Result: **no OOM at any size**, for either model:

- **3B** ran clean through **65536** (est. peak ~8.9 GB, 128 MB largest buffer).
- **7B** ran clean through **49152** (est. peak ~8.0 GB); only **65536** failed,
  and with `"A valid external Instance reference no longer exists"` (engine
  teardown after the prior heavy run) — not a clean device-lost OOM.

This **overturns the earlier conclusion.** The hard machine crashes we saw
before happened only in **full-decode** runs at 12288/16384 — i.e. generating
thousands of tokens over many minutes. Since static allocation + prefill of a
_much larger_ context (65536) does **not** crash, the failure is tied to
**sustained decode duration / load**, not KV-cache size. The earlier
"OOM cliff at ~12K" and "cumulative allocation" diagnoses were **wrong** — the
likely real cause is prolonged 100% GPU load (thermal / OS watchdog / a
per-step growth over a long run), consistent with the multi-hour `e2e` we once
observed.

> ⚠️ Implication for methodology: the fast cliff-finder does **not** reproduce
> this crash, because the crash needs long decode. It is still the right tool
> for an _allocation_ cliff (and proves there isn't one up to 64K), but the
> sustained-decode failure must be characterised with full-decode runs.

### `cliff` phase — allocation/prefill only (`max_tokens=8`), all PASS

| ctx   | 3B est VRAM (MB) | 3B decode tok/s | 7B est VRAM (MB) | 7B decode tok/s |
| ----- | ---------------- | --------------- | ---------------- | --------------- |
| 8192  | 2669             | 10.8            | 5774             | 5.8             |
| 12288 | 3117             | 8.2             | 5998             | 4.4             |
| 16384 | 3565             | 6.5             | 6222             | 3.5             |
| 24576 | 4461             | 4.7             | 6670             | —               |
| 32768 | 5357             | 3.6             | 7118             | 2.0             |
| 49152 | 7149             | 2.5             | 8014             | 1.4             |
| 65536 | 8941             | 1.9             | (teardown error) | —               |

Every `largestKVBufferMB` stayed ≤ 128 MB vs the 1024 MB cap (`exceedsBufferLimit:
false` throughout) — **the per-buffer limit is never the constraint.**

### `vram-perf` phase — full decode (the usable-throughput picture)

| ctx   | 3B decode tok/s | 3B e2e (s) | 7B decode tok/s | 7B e2e (s) |
| ----- | --------------- | ---------- | --------------- | ---------- |
| 2048  | 21.9            | 7.5        | 9.5             | 66         |
| 4096  | 16.1            | 16.2       | 7.9             | 116        |
| 8192  | 11.0            | 38.2       | 5.7             | 185        |
| 10240 | 9.4             | 52.7       | 5.0             | 206        |

### `cap-4k` phase — ⚠️ did not reproduce the cap

Both models returned `finish_reason="stop"` at ctx=4096 (the model answered in
~600 tokens and stopped naturally, never reaching the window). This phase has
been **fixed** to overfill the prompt past the window (`fillFraction 1.2`) so the
next `?coverage` run yields a true `CONTEXT_CAP` / `finish_reason="length"`.

### Verdict (revised)

- **Allocation is not the wall.** Both models allocate + prefill far past 16K
  (3B to 64K, 7B to 48K) with peak VRAM ≤ ~9 GB and KV buffers ≪ the 1 GB cap.
  So 7B@16K is **memory-feasible** — confirmed, not just estimated.
- **Sustained decode is the wall.** Long generations are what destabilised the
  machine earlier; throughput also makes long context impractical (3B ~9 tok/s
  at 10K, 7B ~5 tok/s; both fall to 1–3 tok/s by 32–64K).
- **7B remains impractical for interactive use** (≤5 tok/s by 8–10K) → **3B
  pivot stands** on usability grounds, even though the memory cliff we first
  reported doesn't exist as such.
- **For KV-cache eviction**, the takeaway shifts: the win is **decode-time
  efficiency / stability over long generations**, not dodging an allocation OOM
  at ~12K.

### Remaining to finalise

1. **Re-run `?coverage`** to get a true `cap-4k` reproduction (fix landed) and
   `measuredMemMB` (was unavailable — needs Chrome cross-origin isolation).
2. **Characterise the sustained-decode crash** deliberately: a full-decode run
   at 12288/16384 with logging, to confirm it's load/thermal vs. a memory leak.
