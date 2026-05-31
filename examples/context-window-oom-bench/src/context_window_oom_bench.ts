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
const SMOKE =
  typeof location !== "undefined" &&
  new URLSearchParams(location.search).has("smoke");

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

// Context windows to sweep. Capped at 12288 by default: on a 16 GB M4 Air,
// Llama-3.2-3B at 16384 hard-crashed the machine (not just the tab), so the
// default deliberately stays below that to bracket the cliff (8192 stable,
// 12288 probe) without rebooting your Mac. Raise these only if you have headroom
// and accept the crash risk -- the crash-recovery logic will still record the
// size that dies. Smoke mode uses just two small windows for speed.
const CONTEXT_SIZES = SMOKE ? [2048, 4096] : [2048, 4096, 8192, 12288];

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
  error?: string;
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

async function runOne(spec: ModelSpec, ctx: number): Promise<RunResult> {
  const result: RunResult = {
    model: spec.label,
    sizeClass: spec.sizeClass,
    contextSize: ctx,
    outcome: "OTHER_ERROR",
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
  try {
    const m = await engine!.getKVCacheMetrics(spec.modelId);
    if (m) fillMetrics(result, m);
  } catch (err) {
    console.warn("[oom-bench] getKVCacheMetrics failed:", err);
  }

  // 3. Drive generation toward the cap.
  try {
    setStatus(`Generating ${spec.label} @ context_window_size=${ctx} ...`);
    const prompt = buildFillerPrompt(Math.floor(ctx * PROMPT_FILL_FRACTION));
    const reply = await engine!.chat.completions.create({
      messages: [{ role: "user", content: prompt }],
      temperature: 0,
      max_tokens: ctx, // large enough that only the context cap should stop us
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
    result.outcome = result.finishReason === "length" ? "CONTEXT_CAP" : "PASS";
  } catch (err) {
    const c = classifyError(err);
    result.outcome = c.outcome;
    result.error = c.msg;
    if (c.outcome === "OOM_DEVICE_LOST") {
      engine = undefined;
    }
  }

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
    ["prefill tok/s", (r) => fmt(r.prefillTokPerSec, 1)],
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

  // Skip (model, ctx) pairs we already have a result for, so reloading after a
  // crash resumes the sweep instead of repeating it.
  const done = new Set(results.map((r) => `${r.model}@${r.contextSize}`));

  setStatus(
    `${SMOKE ? "SMOKE" : "FULL"} sweep: ${MODELS.length} model(s) x ` +
      `${CONTEXT_SIZES.length} context size(s)` +
      `${SMOKE ? " (append/remove ?smoke in the URL to switch)" : ""}`,
  );
  for (const spec of MODELS) {
    for (const ctx of CONTEXT_SIZES) {
      if (done.has(`${spec.label}@${ctx}`)) {
        console.log(`[oom-bench] skipping already-done ${spec.label}@${ctx}`);
        continue;
      }
      setPending(spec, ctx); // mark in-flight; recovered if the tab/OS crashes
      const result = await runOne(spec, ctx);
      clearPending();
      results.push(result);
      saveResults(results); // persist immediately -- survives a tab/OS crash
      console.log("[oom-bench] result:", result);
      renderTable(results);
    }
    // Free the model between size classes so the next one starts clean.
    try {
      await engine?.unload();
    } catch {
      /* ignore */
    }
    engine = undefined;
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

main();
