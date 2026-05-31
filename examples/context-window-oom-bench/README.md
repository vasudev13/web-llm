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

| URL                             | What it does                                                            |
| ------------------------------- | ----------------------------------------------------------------------- |
| `http://localhost:8888`         | **Full sweep** — both models, `[2048, 4096, 8192, 10240]` (minutes).    |
| `…/?smoke`                      | **Smoke** — 3B only, `[2048, 4096]` (~1 min sanity check).              |
| `…/?fast`                       | **Fast cliff-finder** — `max_tokens=8`, sweeps up through 16384.        |
| `…/?ctx=11264,12288`            | **Explicit context list** — probe exactly these sizes.                  |
| `…/?fast&ctx=11264,12288,16384` | Fast mode over a custom list — the safe way to pin the exact OOM cliff. |

**Why `?fast` is safe for probing high context:** the OOM cliff is hit at
KV-cache **allocation / prefill** time, not during long decode. So you don't
need to generate hundreds of tokens to trigger it — `max_tokens=8` makes each
run finish in seconds while still reproducing the crash. This lets you pin the
exact cliff (e.g. binary-search 11264) and find **7B's** cliff (which the slow
default sweep never reached) without multi-minute runs.

The active mode and the context list are printed to the console and status line
at startup.

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

## Empirical results

Device: **MacBook Air M4, 16 GB unified memory** · Browser: **Chrome (WebGPU)** ·
`maxStorageBufferBindingSize`: **1024 MB** (1 GB).

> Measured against the local instrumented build (KV bytes derived from
> architecture; `kvScaled: true`). Earlier runs with blank VRAM columns used the
> published npm package and are superseded.

### Llama-3.2-3B-Instruct-q4f16_1-MLC

| ctx   | est peak VRAM (MB) | kvCache (MB) | largest KV buf (MB) | finish | decode tok/s | e2e (s) | outcome          |
| ----- | ------------------ | ------------ | ------------------- | ------ | ------------ | ------- | ---------------- |
| 2048  | 1997               | 224          | 4                   | stop   | 21.7         | 7.6     | PASS             |
| 4096  | 2221               | 448          | 8                   | stop   | 16.5         | 16.1    | PASS             |
| 8192  | 2669               | 896          | 16                  | stop   | 11.0         | 38.2    | PASS             |
| 10240 | 2893               | 1120         | 20                  | stop   | 9.4          | 52.5    | PASS             |
| 12288 | ~3117 (est)        | ~1344 (est)  | ~24 (est)           | —      | —            | —       | OOM (hard crash) |
| 16384 | ~3564 (est)        | ~1792 (est)  | ~32 (est)           | —      | —            | —       | OOM (hard crash) |

OOM cliff (3B): between **10240 (stable)** and **12288 (crash)**. Crucially, the
crash happens at only **~3 GB estimated VRAM on a 16 GB machine**, and the
largest single KV buffer (~24 MB) is **far** under the 1024 MB per-buffer cap —
so the cliff is **neither** total-memory exhaustion **nor** a single-buffer
limit. It is most consistent with cumulative WebGPU allocation pressure /
fragmentation across the many paged buffers tipping the browser/OS over on
unified memory. Decode throughput also halves from 2K→8K (21.7 → 11.0 tok/s).

### Qwen2.5-7B-Instruct-q4f16_1-MLC

| ctx   | est peak VRAM (MB) | kvCache (MB) | largest KV buf (MB) | finish | decode tok/s | e2e (s) | outcome     |
| ----- | ------------------ | ------------ | ------------------- | ------ | ------------ | ------- | ----------- |
| 2048  | 5438               | 112          | 2                   | length | 10.7         | 70      | CONTEXT_CAP |
| 4096  | 5550               | 224          | 4                   | stop   | 8.2          | 106     | PASS        |
| 8192  | 5774               | 448          | 8                   | stop   | 5.7          | 175     | PASS        |
| 10240 | 5886               | 560          | 10                  | stop   | 5.0          | 182     | PASS        |

OOM cliff (7B): **no OOM observed up to 10240** — 7B is not memory-limited in
this range (~5.9 GB peak on 16 GB). It is instead **performance-limited**:
decode falls to ~5 tok/s at 8K–10K and end-to-end latency runs **3+ minutes per
request**. (An earlier 10240 run logged a multi-hour e2e, consistent with the
machine thrashing/sleeping under sustained load.)

### Verdict — 7B@16K feasibility (Risk #4)

- **Memory:** 7B@16K is _not_ blocked by total memory (~6 GB peak ≪ 16 GB) nor by
  the per-buffer cap (largest KV buffer ~16 MB ≪ 1024 MB at 16K).
- **Usability:** 7B is already at ~5 tok/s / multi-minute latency by 10K, so 16K
  would be functionally unusable even if it loads. **7B is not viable for
  interactive use on the M4 Air** — a **3B pivot is the practical choice**.
- **3B reality:** even 3B hard-crashes the machine at 12288, so the usable
  ceiling today is **~10K context**. This is the baseline KV-cache eviction
  needs to beat, and the crash mechanism (cumulative allocation, not a single
  oversized buffer) is exactly what paged eviction addresses.

### Open items / remaining tests

These refine the findings above; run them with the fast cliff-finder so they
are cheap and crash-safe (`?fast`, `max_tokens=8`):

1. **7B OOM cliff not yet found.** 7B was only swept to 10240 (no OOM); its
   actual memory cliff is unknown. Run `?fast&ctx=12288,16384,24576,32768` for
   7B to find it. (7B is already perf-unusable by 10K, but the issue explicitly
   asks for the cliff of _both_ models.)
2. **Exact 3B cliff.** Currently bracketed 10240–12288. Run
   `?fast&ctx=11264,11776` to pin it.
3. **Validate estimated VRAM against real memory.** All peak-VRAM figures are
   _computed_. The new `measuredMemMB` column captures real memory when Chrome
   is cross-origin isolated; confirm the estimates track the observed numbers,
   which underpins the "crash is not total-memory exhaustion" conclusion.
4. **Explicit 4K-cap repro (issue #752).** Confirm a clean
   `finish_reason="length"` at exactly `context_window_size=4096` in normal
   (non-fast) mode — the canonical "4K context cap" reproduction.
