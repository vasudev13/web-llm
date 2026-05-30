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

```bash
cd examples/context-window-oom-bench
npm install
npm start          # serves on http://localhost:8888
```

Open the page in Chrome and **open the console**. You'll see:

- Model download / load progress.
- The PagedKVCache instrumentation logs per load, e.g.
  `PagedKVCache allocation: maxTotalSeqLen=16384, pages=1024 x pageSize=16, ...`
  and `PagedKVCache memory estimate: params=..MB, tempBuffers=..MB, kvCache=..MB, estPeakVRAM=..MB`.
- A live HTML results table, and on completion a `console.table`, an **OOM cliff
  summary**, and **CSV** + **JSON** dumps you can paste into the template below.

### Smoke mode (fast check, no code editing)

For a quick "does this work?" run (~1 min) instead of the full sweep, append
`?smoke` to the URL:

- `http://localhost:8888` → **full sweep**: both models, all 5 context sizes up
  to 32K (many minutes).
- `http://localhost:8888/?smoke` → **smoke mode**: just the 3B model and
  `[2048, 4096]`.

The active mode is printed to the console and the status line at startup.

### Other config

Edit the config block at the top of `src/context_window_oom_bench.ts`:

- `ALL_MODELS` — defaults to `Llama-3.2-3B-Instruct-q4f16_1-MLC` (3B) and
  `Qwen2.5-7B-Instruct-q4f16_1-MLC` (7B). The full sweep uses both; smoke mode
  uses just the first.
- `CONTEXT_SIZES` — full sweep is `[2048, 4096, 8192, 16384, 32768]`; smoke mode
  is `[2048, 4096]`.
- `PROMPT_FILL_FRACTION` — how much of the window the prompt fills before decode
  reaches the cap (default `0.85`).

> **Runtime warning:** large windows (16K/32K) require a very large prefill and
> can take **minutes** per run. Trim `CONTEXT_SIZES` (or use `?smoke`) while
> iterating.

## What each run records

`finish_reason`, `prompt_tokens`, `completion_tokens`, peak KV length
(`prompt + completion`), `prefill_tokens_per_s`, `decode_tokens_per_s`,
`e2e_latency_s` (all from `usage` / `usage.extra`), plus the instrumentation
fields from `engine.getKVCacheMetrics()`: `numPages`, `paramBytes`,
`maxTempFuncBytes`, `kvCacheBytes`, `estimatedTotalVRAMBytes`.

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

**7B @ 16K feasibility (M4 Air, 16 GB unified memory):** the ~6 GB total
estimate sits comfortably under 16 GB unified memory, so 7B @ 16K is **likely
feasible** from a total-memory standpoint — **no 3B pivot indicated yet**. The
real risks to confirm empirically are (a) WebGPU per-buffer limits
(`maxStorageBufferBindingSize` / `maxBufferSize`, queryable via
`engine.getMaxStorageBufferBindingSize()`), which can bite before total memory
does, especially at 32K, and (b) browser/OS memory reservation. Treat the table
above as estimates and replace it with measured numbers below.

## Empirical results (fill after running on the M4 Air)

Device: \_\_\_\_ (e.g. MacBook Air M4, 16 GB) · Browser: \_\_\_\_ ·
`maxStorageBufferBindingSize`: \_\_\_\_

### Llama-3.2-3B-Instruct-q4f16_1-MLC

| ctx   | peak VRAM (est, MB) | finish_reason | peak KV len | decode tok/s | outcome |
| ----- | ------------------- | ------------- | ----------- | ------------ | ------- |
| 2048  |                     |               |             |              |         |
| 4096  |                     |               |             |              |         |
| 8192  |                     |               |             |              |         |
| 16384 |                     |               |             |              |         |
| 32768 |                     |               |             |              |         |

OOM cliff (3B): \_\_\_\_

### Qwen2.5-7B-Instruct-q4f16_1-MLC

| ctx   | peak VRAM (est, MB) | finish_reason | peak KV len | decode tok/s | outcome |
| ----- | ------------------- | ------------- | ----------- | ------------ | ------- |
| 2048  |                     |               |             |              |         |
| 4096  |                     |               |             |              |         |
| 8192  |                     |               |             |              |         |
| 16384 |                     |               |             |              |         |
| 32768 |                     |               |             |              |         |

OOM cliff (7B): \_\_\_\_

**7B @ 16K viable?** \_\_\_\_ · **3B pivot needed?** \_\_\_\_
