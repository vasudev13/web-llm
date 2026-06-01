import * as webllm from "@mlc-ai/web-llm";
import type { KVCacheMetrics } from "@mlc-ai/web-llm";

/**
 * Context-window / OOM-cliff benchmark (Linear VAS-49).
 *
 * Sweeps `context_window_size` across a 3B and a 7B model, pushes the prompt +
 * generation toward the configured window, and records per-run telemetry plus
 * the PagedKVCache instrumentation (peak-VRAM estimate). Each run is classified
 * as PASS / CONTEXT_CAP / OOM_DEVICE_LOST / OTHER_ERROR so the OOM cliff can be
 * read directly from the summary table.
 *
 * Must be run in Chrome with WebGPU (e.g. on the M4 Air) -- empirical OOM
 * cannot be reproduced in a headless/CI environment.
 */

// ---------------------------------------------------------------------------
// Configuration -- tweak these for your sweep.
// ---------------------------------------------------------------------------
interface ModelSpec {
  label: string;
  modelId: string;
  sizeClass: string;
}

// Smoke mode: append `?smoke` to the URL (e.g. http://localhost:8888/?smoke)
// to run a fast subset -- one small model and two small context windows --
// for a quick "does this work?" check (~1 min) instead of the full sweep
// (both models, up to 32K, many minutes). No code editing required.
const params =
  typeof location !== "undefined"
    ? new URLSearchParams(location.search)
    : new URLSearchParams();
const SMOKE = params.has("smoke");

// Fast cliff-finder: append `?fast` to the URL. Uses tiny max_tokens so each
// run is seconds, not minutes -- the OOM cliff is hit at allocation/prefill
// time, NOT during long decode, so we don't need to generate hundreds of
// tokens to trigger it. This makes it safe and cheap to probe higher context
// sizes and pin the exact cliff for both 3B and 7B. Combine with `?ctx=` to
// override the sweep, e.g. `?fast&ctx=11264,12288,16384`.
const FAST = params.has("fast");

// Coverage mode: append `?coverage` to run the COMPLETE test matrix in one
// launch, addressing every VAS-49 acceptance gap automatically:
//   Phase 1 (cap-4k)   -- explicit 4K context-cap reproduction (full decode,
//                         expects finish_reason="length" at ctx=4096).
//   Phase 2 (vram-perf) -- VRAM + throughput curve over safe context sizes.
//   Phase 3 (cliff)    -- fast (max_tokens=8) auto-escalating climb per model
//                         that STOPS that model on its first OOM, so the OOM
//                         cliff is found for BOTH 3B and 7B without manual URLs.
// Each run still records the real-memory probe, so estimated VRAM is validated.
const COVERAGE = params.has("coverage");

// Optional explicit context list, e.g. `?ctx=4096,8192,11264`.
const CTX_OVERRIDE = (params.get("ctx") ?? "")
  .split(",")
  .map((s) => parseInt(s.trim(), 10))
  .filter((n) => Number.isFinite(n) && n > 0);

// Context ladder for the coverage cliff-finder (Phase 3). The climb stops per
// model at the first OOM, so going high here is safe -- nothing past the cliff
// actually runs for that model.
const COVERAGE_CLIFF_LADDER = [
  8192, 10240, 11264, 12288, 14336, 16384, 20480, 24576, 32768, 40960, 49152,
  65536,
];
// Context sizes for the VRAM/perf curve (Phase 2) -- known-safe, full decode.
const COVERAGE_CURVE = [2048, 4096, 8192, 10240];

const ALL_MODELS: ModelSpec[] = [
  {
    label: "Llama-3.2-3B (q4f16)",
    modelId: "Llama-3.2-3B-Instruct-q4f16_1-MLC",
    sizeClass: "3B",
  },
  {
    label: "Qwen2.5-7B (q4f16)",
    modelId: "Qwen2.5-7B-Instruct-q4f16_1-MLC",
    sizeClass: "7B",
  },
];

// Full sweep uses both models; smoke mode uses just the 3B.
const MODELS: ModelSpec[] = SMOKE ? [ALL_MODELS[0]] : ALL_MODELS;

// Context windows to sweep. Capped at 10240 by default: on a 16 GB M4 Air,
// Llama-3.2-3B hard-crashed the machine (not just the tab) at both 16384 and
// 12288, while 8192 was stable -- so the cliff is between 8192 and 12288. The
// default stays at/under 10240 to probe that gap without rebooting your Mac.
// Raise these only if you have headroom and accept the crash risk -- the
// crash-recovery logic will still record the size that dies. Smoke mode uses
// just two small windows for speed. `?ctx=` overrides this list; `?fast` adds
// the historical 4K cap + the known cliff-bracket probes.
const CONTEXT_SIZES =
  CTX_OVERRIDE.length > 0
    ? CTX_OVERRIDE
    : SMOKE
      ? [2048, 4096]
      : FAST
        ? [2048, 4096, 8192, 10240, 11264, 12288, 16384]
        : [2048, 4096, 8192, 10240];

// max_tokens per request. In FAST mode we only need enough decode to confirm
// the run survives prefill/allocation (the cliff is at allocation time), so a
// handful of tokens keeps each run to seconds.
const MAX_TOKENS = FAST ? 8 : undefined; // undefined => fill toward the window

// Fraction of the window filled by the prompt; the remainder is generated so
// decode reaches the cap quickly while still stressing KV memory.
const PROMPT_FILL_FRACTION = 0.85;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
type Outcome = "PASS" | "CONTEXT_CAP" | "OOM_DEVICE_LOST" | "OTHER_ERROR";

interface RunResult {
  model: string;
  sizeClass: string;
  contextSize: number;
  outcome: Outcome;
  // Which coverage phase produced this row (coverage mode only).
  phase?: string;
  finishReason?: string;
  promptTokens?: number;
  completionTokens?: number;
  peakKVLength?: number;
  prefillTokPerSec?: number;
  decodeTokPerSec?: number;
  e2eLatencyS?: number;
  numPages?: number;
  paramMB?: number;
  tempMB?: number;
  kvCacheMB?: number;
  estPeakVRAMMB?: number;
  kvScaled?: boolean;
  // Buffer-limit analysis: is the OOM cliff a per-buffer cap, not total memory?
  largestKVBufferMB?: number;
  maxStorageBufferMB?: number;
  bufferLimitHeadroomPct?: number; // largestKVBuffer / maxStorageBuffer * 100
  exceedsBufferLimit?: boolean;
  // Real browser memory after load+generate, if the API is available (Chrome,
  // cross-origin-isolated only). Validates the estimated peak VRAM.
  measuredMemMB?: number;
  error?: string;
}

// Device WebGPU limit, queried once and reused for every run.
let maxStorageBufferBytes: number | undefined;

// Whether we've already warned that the memory-measurement API is unavailable.
let warnedNoMemAPI = false;

/**
 * Best-effort real memory reading via `performance.measureUserAgentSpecificMemory()`
 * (Chrome only, requires the page to be cross-origin isolated). Returns total
 * bytes as MB, or undefined if unavailable. Note: this measures JS/renderer
 * memory and may not capture all GPU allocations, but it is a real observed
 * number to sanity-check the estimated peak VRAM against.
 */
async function measureMemoryMB(): Promise<number | undefined> {
  const perf = performance as unknown as {
    measureUserAgentSpecificMemory?: () => Promise<{ bytes: number }>;
  };
  if (typeof perf.measureUserAgentSpecificMemory !== "function") {
    if (!warnedNoMemAPI) {
      warnedNoMemAPI = true;
      console.warn(
        "[oom-bench] performance.measureUserAgentSpecificMemory() unavailable " +
          "(needs Chrome + cross-origin isolation). Skipping real memory probe; " +
          "estimated peak VRAM is still reported.",
      );
    }
    return undefined;
  }
  try {
    const sample = await perf.measureUserAgentSpecificMemory();
    return sample.bytes / 1024 / 1024;
  } catch (err) {
    console.warn("[oom-bench] measureUserAgentSpecificMemory failed:", err);
    return undefined;
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function setLabel(id: string, text: string) {
  const label = document.getElementById(id);
  if (label != null) {
    label.innerText = text;
  }
}

function setStatus(text: string) {
  console.log(`[oom-bench] ${text}`);
  setLabel("status-label", text);
}

// ---------------------------------------------------------------------------
// Crash-proof persistence. A high-context OOM can take the whole tab down,
// taking the console logs with it. We persist each result to localStorage the
// moment it completes (written to disk per-origin, survives a tab crash) and
// restore + display them on reload, so a crash never erases what already ran.
// ---------------------------------------------------------------------------
const STORAGE_KEY = "oom-bench-results";
// Marker for the run currently in flight. If a hard tab crash (a high-context
// OOM can kill the whole tab, not just throw) prevents the run from recording a
// result, this marker is still on disk on reload -- that pinpoints the context
// size that crashed, which IS the OOM cliff.
const PENDING_KEY = "oom-bench-pending";

function loadSavedResults(): RunResult[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as RunResult[]) : [];
  } catch {
    return [];
  }
}

function setPending(spec: ModelSpec, ctx: number) {
  try {
    localStorage.setItem(
      PENDING_KEY,
      JSON.stringify({
        model: spec.label,
        sizeClass: spec.sizeClass,
        contextSize: ctx,
      }),
    );
  } catch {
    /* ignore */
  }
}

function clearPending() {
  try {
    localStorage.removeItem(PENDING_KEY);
  } catch {
    /* ignore */
  }
}

/** A leftover pending marker means the previous run hard-crashed the tab. */
function takeCrashedPending(): RunResult | undefined {
  try {
    const raw = localStorage.getItem(PENDING_KEY);
    if (!raw) return undefined;
    localStorage.removeItem(PENDING_KEY);
    const p = JSON.parse(raw) as {
      model: string;
      sizeClass: string;
      contextSize: number;
    };
    return {
      model: p.model,
      sizeClass: p.sizeClass,
      contextSize: p.contextSize,
      outcome: "OOM_DEVICE_LOST",
      error: "Tab crashed during this run (recovered on reload) -- OOM cliff.",
    };
  } catch {
    return undefined;
  }
}

function saveResults(rows: RunResult[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(rows));
  } catch (e) {
    console.warn("[oom-bench] could not persist results:", e);
  }
}

function clearSavedResults() {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    /* ignore */
  }
}

/** Trigger a browser download of `contents` as a file named `filename`. */
function downloadFile(filename: string, contents: string, mime: string) {
  const blob = new Blob([contents], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** Build a prompt of roughly `approxTokens` tokens (~4 chars/token heuristic). */
function buildFillerPrompt(approxTokens: number): string {
  const base =
    "The quick brown fox jumps over the lazy dog while counting numbers " +
    "like one two three four five six seven eight nine ten. ";
  const approxTokensPerBase = 30; // rough token count of `base`
  const reps = Math.max(1, Math.ceil(approxTokens / approxTokensPerBase));
  return (
    "Please read the following text carefully, then continue writing an " +
    "essay that keeps going for as long as possible without stopping.\n\n" +
    base.repeat(reps) +
    "\n\nNow continue the essay in your own words and keep writing."
  );
}

function classifyError(err: unknown): { outcome: Outcome; msg: string } {
  const name = (err as { name?: string })?.name ?? "";
  const msg = (err as { message?: string })?.message ?? String(err);
  if (
    name === "ContextWindowSizeExceededError" ||
    /context window size/i.test(msg)
  ) {
    return { outcome: "CONTEXT_CAP", msg };
  }
  if (
    name === "DeviceLostError" ||
    /device (was )?lost|out of memory|\bOOM\b|insufficient memory/i.test(msg)
  ) {
    return { outcome: "OOM_DEVICE_LOST", msg };
  }
  return { outcome: "OTHER_ERROR", msg };
}

function fillMetrics(r: RunResult, m: KVCacheMetrics) {
  const toMB = (b: number) => b / 1024 / 1024;
  r.numPages = m.numPages;
  r.paramMB = toMB(m.paramBytes);
  r.tempMB = toMB(m.maxTempFuncBytes);
  r.kvCacheMB = toMB(m.kvCacheBytes);
  r.estPeakVRAMMB = toMB(m.estimatedTotalVRAMBytes);
  r.kvScaled = m.scaledToConfiguredContext;

  // Buffer-limit analysis: compare the largest single KV buffer against the
  // device's maxStorageBufferBindingSize. If the largest buffer approaches or
  // exceeds the limit, the cliff is a per-buffer cap (which paging/eviction
  // can fix), not total-memory exhaustion.
  if (m.largestKVBufferBytes > 0) {
    r.largestKVBufferMB = toMB(m.largestKVBufferBytes);
    if (maxStorageBufferBytes && maxStorageBufferBytes > 0) {
      r.maxStorageBufferMB = toMB(maxStorageBufferBytes);
      r.bufferLimitHeadroomPct =
        (m.largestKVBufferBytes / maxStorageBufferBytes) * 100;
      r.exceedsBufferLimit = m.largestKVBufferBytes > maxStorageBufferBytes;
    }
  }
}

// ---------------------------------------------------------------------------
// Engine lifecycle: keep one engine, reload per context size. After a
// device-lost/OOM the GPU device is gone and the engine self-unloads, so we
// drop the reference and recreate a fresh engine on the next run.
// ---------------------------------------------------------------------------
let engine: webllm.MLCEngineInterface | undefined;

const initProgressCallback = (report: webllm.InitProgressReport) => {
  setLabel("init-label", report.text);
};

async function loadEngine(modelId: string, ctx: number): Promise<void> {
  const chatOpts: webllm.ChatOptions = { context_window_size: ctx };
  if (engine === undefined) {
    engine = await webllm.CreateMLCEngine(
      modelId,
      { initProgressCallback, logLevel: "INFO" },
      chatOpts,
    );
  } else {
    await engine.reload(modelId, chatOpts);
  }
}

async function runOne(
  spec: ModelSpec,
  ctx: number,
  opts: { maxTokens?: number; phase?: string; fillFraction?: number } = {},
): Promise<RunResult> {
  // Per-run max_tokens: explicit override (coverage) > FAST global > fill window.
  const maxTokens = opts.maxTokens ?? MAX_TOKENS;
  const fillFraction = opts.fillFraction ?? PROMPT_FILL_FRACTION;
  const result: RunResult = {
    model: spec.label,
    sizeClass: spec.sizeClass,
    contextSize: ctx,
    outcome: "OTHER_ERROR",
    phase: opts.phase,
  };

  // 1. Load / reload at the target context window (allocation-time OOM lives here).
  try {
    setStatus(`Loading ${spec.label} @ context_window_size=${ctx} ...`);
    await loadEngine(spec.modelId, ctx);
  } catch (err) {
    const c = classifyError(err);
    result.outcome = c.outcome;
    result.error = c.msg;
    engine = undefined; // device likely lost / engine self-unloaded
    return result;
  }

  // 2. Snapshot PagedKVCache instrumentation (peak-VRAM estimate).
  // Query the device's max storage buffer size once -- this is the cap a single
  // KV buffer must stay under, and the likely cause of the OOM cliff.
  if (maxStorageBufferBytes === undefined) {
    try {
      maxStorageBufferBytes = await engine!.getMaxStorageBufferBindingSize();
      console.log(
        `[oom-bench] device maxStorageBufferBindingSize = ` +
          `${(maxStorageBufferBytes / 1024 / 1024).toFixed(0)} MB`,
      );
    } catch (err) {
      console.warn("[oom-bench] getMaxStorageBufferBindingSize failed:", err);
      maxStorageBufferBytes = 0; // don't retry every run
    }
  }
  try {
    const m = await engine!.getKVCacheMetrics(spec.modelId);
    if (m) fillMetrics(result, m);
  } catch (err) {
    console.warn("[oom-bench] getKVCacheMetrics failed:", err);
  }

  // 3. Drive generation toward the cap.
  try {
    setStatus(`Generating ${spec.label} @ context_window_size=${ctx} ...`);
    const prompt = buildFillerPrompt(Math.floor(ctx * fillFraction));
    const reply = await engine!.chat.completions.create({
      messages: [{ role: "user", content: prompt }],
      temperature: 0,
      // A small max_tokens is enough to find the cliff -- it is hit at
      // allocation/prefill time, not during decode. Otherwise fill the window.
      max_tokens: maxTokens ?? ctx,
    });
    const choice = reply.choices[0];
    const usage = reply.usage;
    const extra = usage?.extra as
      | {
          prefill_tokens_per_s?: number;
          decode_tokens_per_s?: number;
          e2e_latency_s?: number;
        }
      | undefined;
    result.finishReason = choice?.finish_reason ?? undefined;
    result.promptTokens = usage?.prompt_tokens;
    result.completionTokens = usage?.completion_tokens;
    result.peakKVLength =
      (usage?.prompt_tokens ?? 0) + (usage?.completion_tokens ?? 0);
    result.prefillTokPerSec = extra?.prefill_tokens_per_s;
    result.decodeTokPerSec = extra?.decode_tokens_per_s;
    result.e2eLatencyS = extra?.e2e_latency_s;
    // When the run is capped early (small max_tokens), "length" is expected and
    // not a context-cap finding; only treat it as CONTEXT_CAP for full-decode
    // runs that fill the whole window.
    const filledWindow = maxTokens === undefined;
    result.outcome =
      filledWindow && result.finishReason === "length" ? "CONTEXT_CAP" : "PASS";
  } catch (err) {
    const c = classifyError(err);
    result.outcome = c.outcome;
    result.error = c.msg;
    if (c.outcome === "OOM_DEVICE_LOST") {
      engine = undefined;
    }
  }

  // 4. Real browser memory probe (validates the estimated peak VRAM). Only
  // available in Chrome when the page is cross-origin isolated; otherwise
  // silently skipped.
  const measuredMemMB = await measureMemoryMB();
  if (measuredMemMB !== undefined) result.measuredMemMB = measuredMemMB;

  return result;
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------
const fmt = (n: number | undefined, digits = 1) =>
  n === undefined || Number.isNaN(n) ? "" : n.toFixed(digits);

function toCSV(rows: RunResult[]): string {
  const header = [
    "model",
    "sizeClass",
    "contextSize",
    "outcome",
    "finishReason",
    "promptTokens",
    "completionTokens",
    "peakKVLength",
    "prefillTokPerSec",
    "decodeTokPerSec",
    "e2eLatencyS",
    "numPages",
    "paramMB",
    "tempMB",
    "kvCacheMB",
    "estPeakVRAMMB",
    "kvScaled",
    "largestKVBufferMB",
    "maxStorageBufferMB",
    "bufferLimitHeadroomPct",
    "exceedsBufferLimit",
    "measuredMemMB",
    "error",
  ];
  const lines = rows.map((r) =>
    [
      r.model,
      r.sizeClass,
      r.contextSize,
      r.outcome,
      r.finishReason ?? "",
      r.promptTokens ?? "",
      r.completionTokens ?? "",
      r.peakKVLength ?? "",
      fmt(r.prefillTokPerSec, 2),
      fmt(r.decodeTokPerSec, 2),
      fmt(r.e2eLatencyS, 2),
      r.numPages ?? "",
      fmt(r.paramMB),
      fmt(r.tempMB),
      fmt(r.kvCacheMB),
      fmt(r.estPeakVRAMMB),
      r.kvScaled ?? "",
      fmt(r.largestKVBufferMB),
      fmt(r.maxStorageBufferMB),
      fmt(r.bufferLimitHeadroomPct, 1),
      r.exceedsBufferLimit ?? "",
      fmt(r.measuredMemMB),
      (r.error ?? "").replace(/[\r\n,]+/g, " "),
    ].join(","),
  );
  return [header.join(","), ...lines].join("\n");
}

function renderTable(rows: RunResult[]) {
  const cols: Array<[string, (r: RunResult) => string]> = [
    ["Model", (r) => r.model],
    ["Ctx", (r) => String(r.contextSize)],
    ["Outcome", (r) => r.outcome],
    ["finish", (r) => r.finishReason ?? ""],
    ["peakKV", (r) => String(r.peakKVLength ?? "")],
    ["pages", (r) => String(r.numPages ?? "")],
    ["kvCacheMB", (r) => fmt(r.kvCacheMB)],
    ["estPeakVRAM(MB)", (r) => fmt(r.estPeakVRAMMB)],
    ["maxKVbuf(MB)", (r) => fmt(r.largestKVBufferMB)],
    ["bufLimit%", (r) => fmt(r.bufferLimitHeadroomPct, 1)],
    ["measMem(MB)", (r) => fmt(r.measuredMemMB)],
    ["decode tok/s", (r) => fmt(r.decodeTokPerSec, 1)],
  ];
  const thead = "<tr>" + cols.map(([h]) => `<th>${h}</th>`).join("") + "</tr>";
  const tbody = rows
    .map(
      (r) =>
        "<tr>" + cols.map(([, f]) => `<td>${f(r)}</td>`).join("") + "</tr>",
    )
    .join("");
  const el = document.getElementById("results-table");
  if (el) {
    el.innerHTML = `<table border="1" cellpadding="4" cellspacing="0">${thead}${tbody}</table>`;
  }
}

/** Wire up the Download CSV / Download JSON / Clear buttons. */
function setupButtons(getRows: () => RunResult[]) {
  const bind = (id: string, fn: () => void) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("click", fn);
  };
  bind("download-csv", () =>
    downloadFile("oom-bench-results.csv", toCSV(getRows()), "text/csv"),
  );
  bind("download-json", () =>
    downloadFile(
      "oom-bench-results.json",
      JSON.stringify(getRows(), null, 2),
      "application/json",
    ),
  );
  bind("clear-results", () => {
    clearSavedResults();
    location.reload();
  });
}

function summarizeCliff(rows: RunResult[]) {
  console.log("\n===== OOM cliff summary =====");
  for (const spec of MODELS) {
    const modelRows = rows
      .filter((r) => r.model === spec.label)
      .sort((a, b) => a.contextSize - b.contextSize);
    const firstOOM = modelRows.find((r) => r.outcome === "OOM_DEVICE_LOST");
    const lastOK = [...modelRows]
      .reverse()
      .find((r) => r.outcome === "PASS" || r.outcome === "CONTEXT_CAP");
    const cliff = firstOOM
      ? `OOM cliff at context_window_size=${firstOOM.contextSize} ` +
        `(largest non-OOM window: ${lastOK ? lastOK.contextSize : "none"})`
      : `No OOM observed up to ${
          modelRows[modelRows.length - 1]?.contextSize ?? "?"
        }`;
    console.log(`${spec.label} [${spec.sizeClass}]: ${cliff}`);

    // Buffer-limit vs total-memory diagnosis at the largest stable window.
    const lastWithBuf = [...modelRows]
      .reverse()
      .find((r) => r.largestKVBufferMB !== undefined);
    if (lastWithBuf?.maxStorageBufferMB) {
      const pct = lastWithBuf.bufferLimitHeadroomPct ?? 0;
      console.log(
        `  buffer analysis @ ctx=${lastWithBuf.contextSize}: ` +
          `largest KV buffer ${fmt(lastWithBuf.largestKVBufferMB)}MB vs ` +
          `device max ${fmt(lastWithBuf.maxStorageBufferMB)}MB (${fmt(pct, 1)}% of cap). ` +
          (lastWithBuf.estPeakVRAMMB
            ? `est. peak VRAM ${fmt(lastWithBuf.estPeakVRAMMB)}MB. `
            : "") +
          (firstOOM
            ? "If peak VRAM at the cliff is far below physical memory, the cliff " +
              "is most likely a per-buffer cap (maxStorageBufferBindingSize), " +
              "not total-memory exhaustion -- exactly what KV paging/eviction targets."
            : ""),
      );
    }
  }
  console.log("=============================\n");
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  // Restore any results from a previous (possibly crashed) run so they are
  // never lost, and keep them downloadable via the buttons at all times.
  const results: RunResult[] = loadSavedResults();
  // If the previous run hard-crashed the tab/machine, recover that context size
  // as the OOM cliff so the sweep doesn't just retry and crash again.
  const crashed = takeCrashedPending();
  if (
    crashed &&
    !results.some(
      (r) => r.contextSize === crashed.contextSize && r.model === crashed.model,
    )
  ) {
    results.push(crashed);
    saveResults(results);
    console.warn(
      `[oom-bench] recovered a hard crash at ${crashed.model}@${crashed.contextSize} ` +
        "-- recording it as OOM_DEVICE_LOST (likely the cliff).",
    );
  }
  setupButtons(() => results);
  if (results.length > 0) {
    console.log(
      `[oom-bench] restored ${results.length} result(s) from a previous run. ` +
        "Use the Download buttons to export, or Clear to start fresh.",
    );
    renderTable(results);
  }

  // Skip (model, ctx, phase) keys we already have a result for, so reloading
  // after a crash resumes instead of repeating.
  const keyOf = (r: { model: string; contextSize: number; phase?: string }) =>
    `${r.model}@${r.contextSize}#${r.phase ?? ""}`;
  const done = new Set(results.map(keyOf));

  // Run one (model, ctx) and record it durably (pending marker + persist +
  // render). Returns the result so callers can branch on the outcome.
  const runAndRecord = async (
    spec: ModelSpec,
    ctx: number,
    opts: { maxTokens?: number; phase?: string; fillFraction?: number } = {},
  ): Promise<RunResult | undefined> => {
    if (
      done.has(
        keyOf({ model: spec.label, contextSize: ctx, phase: opts.phase }),
      )
    ) {
      console.log(
        `[oom-bench] skipping already-done ${spec.label}@${ctx} (${opts.phase ?? "-"})`,
      );
      return results.find(
        (r) =>
          keyOf(r) ===
          keyOf({ model: spec.label, contextSize: ctx, phase: opts.phase }),
      );
    }
    setPending(spec, ctx); // mark in-flight; recovered if the tab/OS crashes
    const result = await runOne(spec, ctx, opts);
    clearPending();
    results.push(result);
    done.add(keyOf(result));
    saveResults(results); // persist immediately -- survives a tab/OS crash
    console.log("[oom-bench] result:", result);
    renderTable(results);
    return result;
  };

  const freeEngine = async () => {
    try {
      await engine?.unload();
    } catch {
      /* ignore */
    }
    engine = undefined;
  };

  if (COVERAGE) {
    await runCoverage(runAndRecord, freeEngine);
  } else {
    const mode = SMOKE ? "SMOKE" : FAST ? "FAST" : "FULL";
    setStatus(
      `${mode} sweep: ${MODELS.length} model(s) x ` +
        `${CONTEXT_SIZES.length} context size(s) [${CONTEXT_SIZES.join(", ")}]` +
        `${FAST ? ` (max_tokens=${MAX_TOKENS}, cliff-finder)` : ""}`,
    );
    for (const spec of MODELS) {
      for (const ctx of CONTEXT_SIZES) {
        await runAndRecord(spec, ctx);
      }
      await freeEngine(); // start each model clean
    }
  }

  setStatus(
    "Sweep complete. Use the Download buttons above, or copy the CSV / JSON " +
      "from the console.",
  );
  console.table(results);
  summarizeCliff(results);
  console.log("\n===== CSV =====\n" + toCSV(results));
  console.log("\n===== JSON =====\n" + JSON.stringify(results, null, 2));
}

/**
 * Coverage mode: run the COMPLETE VAS-49 matrix in one launch.
 *
 * Phase 1 (cap-4k): explicit 4K context-cap reproduction with full decode --
 *   expects finish_reason="length" / outcome CONTEXT_CAP at ctx=4096.
 * Phase 2 (vram-perf): VRAM + throughput curve over known-safe sizes (full
 *   decode), per model.
 * Phase 3 (cliff): fast (max_tokens=8) auto-escalating climb that stops each
 *   model at its FIRST OOM -- finding the OOM cliff for BOTH 3B and 7B without
 *   manual URL juggling. Nothing past a model's cliff is ever attempted.
 */
async function runCoverage(
  runAndRecord: (
    spec: ModelSpec,
    ctx: number,
    opts?: { maxTokens?: number; phase?: string; fillFraction?: number },
  ) => Promise<RunResult | undefined>,
  freeEngine: () => Promise<void>,
): Promise<void> {
  for (const spec of ALL_MODELS) {
    // Phase 1: explicit 4K cap repro (full decode).
    setStatus(`COVERAGE ${spec.label}: Phase 1 -- 4K context-cap repro`);
    // Overfill the window (prompt > ctx) so generation genuinely hits the cap
    // and returns finish_reason="length" / CONTEXT_CAP, rather than answering
    // briefly and stopping. fillFraction 1.2 puts the prompt past 4096.
    await runAndRecord(spec, 4096, { phase: "cap-4k", fillFraction: 1.2 });

    // Phase 2: VRAM + perf curve at safe sizes (full decode).
    for (const ctx of COVERAGE_CURVE) {
      setStatus(`COVERAGE ${spec.label}: Phase 2 -- VRAM/perf @ ${ctx}`);
      await runAndRecord(spec, ctx, { phase: "vram-perf" });
    }

    // Phase 3: fast cliff climb; stop this model at the first OOM.
    for (const ctx of COVERAGE_CLIFF_LADDER) {
      setStatus(`COVERAGE ${spec.label}: Phase 3 -- cliff probe @ ${ctx}`);
      const r = await runAndRecord(spec, ctx, { maxTokens: 8, phase: "cliff" });
      if (r?.outcome === "OOM_DEVICE_LOST") {
        console.log(
          `[oom-bench] ${spec.label}: OOM cliff found at ctx=${ctx}; ` +
            "stopping this model's climb.",
        );
        break;
      }
    }
    await freeEngine(); // clean slate before the next model
  }
}

main();
