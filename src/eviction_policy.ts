/**
 * KV-cache eviction policy abstraction (Phase 0 scaffold — VAS-52).
 *
 * This file defines the *web-llm (TypeScript) side* of a pluggable, attention-aware
 * KV-cache eviction framework. It is deliberately the thin, declarative layer: the
 * actual page-table manipulation, sink/window logic, and score-based selection live
 * in the TVM / MLC-LLM runtime (`paged_kv_cache.cc`). See
 * `docs/eviction/EVICTION_BOUNDARY.md` for the full runtime/JS boundary writeup and
 * the concrete hook points in `paged_kv_cache.cc` that each policy attaches to.
 *
 * Design goals:
 *  - The abstraction is declarative. web-llm describes *what* policy to run and *with
 *    what budget*; the TVM runtime decides *how* to evict against PagedKVCache.
 *  - The no-op policy (`NoOpEvictionPolicy`) is the default. When selected, the engine
 *    must behave bit-for-bit identically to an unmodified web-llm — no hooks fire.
 *  - Every concrete policy from the project (StreamingLLM, H2O, SnapKV, PyramidKV) maps
 *    onto a single `EvictionConfig` shape so the demo UI can switch policies without
 *    recompiling the model lib.
 */

/**
 * The family of eviction policy. These mirror the four attention-aware policies the
 * project implements plus the two baselines (no-op = full cache; sliding-window+sink is
 * already shipped by TVM and used as the differentiation baseline, apache/tvm#16729).
 */
export enum EvictionPolicyKind {
  /** Full KV cache, no eviction. Engine runs unchanged. Default. */
  NoOp = "no-op",
  /** Attention sink + sliding window. Maps onto TVM's existing
   *  `EnableSlidingWindowForSeq(seq_id, window, sink)` — no new kernels (VAS-53). */
  StreamingLLM = "streaming-llm",
  /** Rolling heavy-hitter eviction driven by accumulated attention mass (VAS-60). */
  H2O = "h2o",
  /** Prefill-time pooled-attention selection with a fixed observation window (VAS-56). */
  SnapKV = "snapkv",
  /** Per-layer budget variant of SnapKV (more budget to lower layers) (VAS-57). */
  PyramidKV = "pyramid-kv",
}

/**
 * Tunable knobs for a policy. Fields are a union across all policy families; a given
 * policy reads only the subset relevant to it (documented per field). Unset fields fall
 * back to runtime defaults chosen in the TVM/MLC-LLM layer.
 *
 * Keep this in sync with the ablation axes in VAS-65 (sink tokens, SnapKV obs window,
 * PyramidKV alpha, H2O interval) so ablations are expressible without code changes.
 */
export interface EvictionConfig {
  kind: EvictionPolicyKind;

  /**
   * Target KV budget. Either an absolute number of retained tokens (`>= 1`) or a
   * retention *ratio* in `(0, 1]` of the full context. Ignored by `NoOp`.
   * Mutually exclusive interpretation is resolved by magnitude: `<= 1` → ratio.
   */
  budget?: number;

  /**
   * Number of always-retained attention-sink tokens at the start of the sequence.
   * Used by StreamingLLM, H2O, SnapKV, PyramidKV. Ablation axis (VAS-65).
   */
  sinkTokens?: number;

  /**
   * Sliding-window size (most recent tokens always retained). Used by StreamingLLM
   * (window = budget - sink) and as the "recent" guard for H2O/SnapKV.
   */
  windowSize?: number;

  /**
   * Observation window for prefill-time selection (SnapKV / PyramidKV): how many of the
   * most-recent prompt tokens vote on which earlier tokens to keep. Ablation axis.
   */
  observationWindow?: number;

  /**
   * Average-pooling kernel size applied to the pooled observation-window scores before
   * top-K (SnapKV / PyramidKV). Pooling clusters contiguous important positions so the
   * selection keeps coherent spans rather than isolated tokens (SnapKV paper §3.2).
   * Must be a positive odd integer so the window is symmetric about each position.
   */
  poolingKernelSize?: number;

  /**
   * H2O eviction interval — how many decode steps between heavy-hitter recomputations.
   * Trades dispatch overhead (H3) against selection freshness. Ablation axis.
   */
  evictionInterval?: number;

  /**
   * PyramidKV layer-budget decay factor `alpha` in `(0, 1]`. `1` = uniform (== SnapKV);
   * smaller = steeper pyramid (more budget to lower layers). Ablation axis.
   */
  pyramidAlpha?: number;
}

/**
 * The eviction policy contract surfaced to web-llm.
 *
 * This is intentionally minimal and declarative. A policy does NOT manipulate the KV
 * cache from JS — it cannot, because the cache lives behind opaque compiled globals
 * (`vm.builtin.attention_kv_cache_*`). Instead a policy:
 *   1. validates and normalizes its config, and
 *   2. emits the `EvictionConfig` that the TVM/MLC-LLM runtime hooks consume at the
 *      documented attach points (see EVICTION_BOUNDARY.md).
 */
export interface EvictionPolicy {
  readonly kind: EvictionPolicyKind;
  /**
   * Validate & normalize the config (fill defaults, resolve ratio↔absolute budget,
   * range-check ablation knobs). Throws on invalid input. Returns the config that the
   * runtime hooks should receive.
   */
  resolve(contextWindowSize: number): EvictionConfig;
}

/**
 * No-op policy: full KV cache, no eviction, no runtime hooks fire. This is the default
 * and the safety guarantee for VAS-52 — with it selected the engine runs unchanged.
 */
export class NoOpEvictionPolicy implements EvictionPolicy {
  readonly kind = EvictionPolicyKind.NoOp;

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  resolve(_contextWindowSize: number): EvictionConfig {
    return { kind: EvictionPolicyKind.NoOp };
  }
}

/** Default number of always-retained attention-sink tokens (StreamingLLM paper: 4). */
export const DEFAULT_SINK_TOKENS = 4;

/**
 * Default observation window for SnapKV/PyramidKV prefill-time selection (SnapKV paper
 * default: 32). The last `observationWindow` prompt tokens vote on which earlier tokens
 * to keep; these recent tokens are themselves always retained.
 */
export const DEFAULT_OBSERVATION_WINDOW = 32;

/**
 * Default average-pooling kernel size for SnapKV/PyramidKV selection (SnapKV paper: 7).
 * Must be a positive odd integer (symmetric window about each scored position).
 */
export const DEFAULT_POOLING_KERNEL_SIZE = 7;

/**
 * Resolve a `budget` field (ratio in `(0, 1]` or absolute token count `>= 1`) against a
 * concrete context window into an absolute retained-token count. Shared by every
 * budgeted policy so ratio↔absolute resolution is identical everywhere.
 *
 * `budget` unset → full context (`contextWindowSize`). `<= 1` → treated as a ratio.
 * Throws on non-finite, non-positive, or out-of-range values.
 */
export function resolveBudgetTokens(
  budget: number | undefined,
  contextWindowSize: number,
): number {
  if (!Number.isFinite(contextWindowSize) || contextWindowSize < 1) {
    throw new Error(
      `EvictionConfig: contextWindowSize must be a positive integer, got ${contextWindowSize}.`,
    );
  }
  if (budget === undefined) {
    return Math.floor(contextWindowSize);
  }
  if (!Number.isFinite(budget) || budget <= 0) {
    throw new Error(
      `EvictionConfig.budget must be a positive number (ratio in (0,1] or absolute >= 1), got ${budget}.`,
    );
  }
  // Magnitude rule (matches the EvictionConfig.budget contract): <= 1 is a ratio.
  const absolute = budget <= 1 ? budget * contextWindowSize : budget;
  return Math.min(contextWindowSize, Math.max(1, Math.floor(absolute)));
}

/**
 * StreamingLLM policy (VAS-53): retain the first `sinkTokens` attention-sink tokens plus
 * a sliding window of the most-recent tokens, evict the middle. This is the cheapest
 * policy and adds **no new kernels** — it maps directly onto TVM's existing
 * `EnableSlidingWindowForSeq(seq_id, window_size, sink_size)` (apache/tvm#16729). It is
 * the differentiation baseline that the attention-aware policies (SnapKV/PyramidKV/H2O)
 * must beat at equal budget, and it validates the VAS-52 policy-hook plumbing end-to-end.
 *
 * Resolution semantics:
 *  - `budget` resolves to an absolute retained-token count (see `resolveBudgetTokens`).
 *  - `sinkTokens` defaults to `DEFAULT_SINK_TOKENS` (4).
 *  - `windowSize`, if unset, is derived as `budget - sinkTokens` so that
 *    `sink + window == budget`. If set explicitly it is honored (and budget is treated
 *    as `sink + window` for the runtime hook).
 *
 * Position IDs: retained K already carries correct RoPE; only newly appended tokens get
 * fresh positions (plan §3.5 Option 1 — sparse position IDs). The runtime hook owns this;
 * the TS layer only emits the normalized config.
 */
export class StreamingLLMEvictionPolicy implements EvictionPolicy {
  readonly kind = EvictionPolicyKind.StreamingLLM;

  constructor(private readonly config: EvictionConfig) {
    if (config.kind !== EvictionPolicyKind.StreamingLLM) {
      throw new Error(
        `StreamingLLMEvictionPolicy got config.kind="${config.kind}", expected "${EvictionPolicyKind.StreamingLLM}".`,
      );
    }
  }

  resolve(contextWindowSize: number): EvictionConfig {
    const budget = resolveBudgetTokens(this.config.budget, contextWindowSize);

    const sinkTokens = this.config.sinkTokens ?? DEFAULT_SINK_TOKENS;
    if (!Number.isInteger(sinkTokens) || sinkTokens < 0) {
      throw new Error(
        `StreamingLLM: sinkTokens must be a non-negative integer, got ${sinkTokens}.`,
      );
    }
    if (sinkTokens >= budget) {
      throw new Error(
        `StreamingLLM: sinkTokens (${sinkTokens}) must be smaller than the resolved budget (${budget}); no room for a sliding window.`,
      );
    }

    // Derive the window from the budget when not given; otherwise honor the explicit one.
    const windowSize = this.config.windowSize ?? budget - sinkTokens;
    if (!Number.isInteger(windowSize) || windowSize < 1) {
      throw new Error(
        `StreamingLLM: windowSize must be a positive integer, got ${windowSize}.`,
      );
    }

    return {
      kind: EvictionPolicyKind.StreamingLLM,
      budget: sinkTokens + windowSize,
      sinkTokens,
      windowSize,
    };
  }
}

/**
 * SnapKV policy (VAS-56) — **★ the primary attention-aware policy for the EMNLP demo.**
 *
 * One-shot selection at the *end of prefill*: pool attention scores from the last
 * `observationWindow` prompt tokens (the "observation window") over the earlier prompt
 * tokens, average-pool the pooled scores with a `poolingKernelSize` kernel so coherent
 * spans win over isolated spikes, then keep the top-K earlier positions per head within
 * the budget. The `sinkTokens` prefix and the observation window itself are always
 * retained. After prefill the retained set is **fixed** — there is **zero per-decode-step
 * eviction**, hence no per-token dispatch overhead. At batch=1 in the browser that
 * dispatch cost dominates (H3), which is exactly why SnapKV is predicted to beat the
 * rolling H2O policy in-browser (H1/H3).
 *
 * Like every policy here this is the *declarative* TS layer only: it validates and
 * normalizes the config the runtime hook consumes. The physical selection runs in a
 * **`pooled_select` WGSL kernel** at the prefill-end hook (see
 * `docs/eviction/snapkv_pooled_select.wgsl` and EVICTION_BOUNDARY.md §SnapKV). The
 * kernel reads pooled observation-window scores **without materializing the full
 * attention matrix** — the score read path is de-risked in VAS-47.
 *
 * Budget bookkeeping: `budget` resolves to the total retained-token count. The sink
 * prefix and the observation window are always kept, so the kernel selects
 * `budget - sinkTokens - observationWindow` *earlier* positions by pooled score. The
 * resolved config therefore requires `sinkTokens + observationWindow < budget` so there
 * is room for at least one selected token.
 */
export class SnapKVEvictionPolicy implements EvictionPolicy {
  readonly kind = EvictionPolicyKind.SnapKV;

  constructor(protected readonly config: EvictionConfig) {
    if (config.kind !== this.kind) {
      throw new Error(
        `${this.constructor.name} got config.kind="${config.kind}", expected "${this.kind}".`,
      );
    }
  }

  resolve(contextWindowSize: number): EvictionConfig {
    const budget = resolveBudgetTokens(this.config.budget, contextWindowSize);

    const sinkTokens = this.config.sinkTokens ?? DEFAULT_SINK_TOKENS;
    if (!Number.isInteger(sinkTokens) || sinkTokens < 0) {
      throw new Error(
        `${this.kind}: sinkTokens must be a non-negative integer, got ${sinkTokens}.`,
      );
    }

    const observationWindow =
      this.config.observationWindow ?? DEFAULT_OBSERVATION_WINDOW;
    if (!Number.isInteger(observationWindow) || observationWindow < 1) {
      throw new Error(
        `${this.kind}: observationWindow must be a positive integer, got ${observationWindow}.`,
      );
    }

    const poolingKernelSize =
      this.config.poolingKernelSize ?? DEFAULT_POOLING_KERNEL_SIZE;
    if (
      !Number.isInteger(poolingKernelSize) ||
      poolingKernelSize < 1 ||
      poolingKernelSize % 2 === 0
    ) {
      throw new Error(
        `${this.kind}: poolingKernelSize must be a positive odd integer, got ${poolingKernelSize}.`,
      );
    }

    // The sink prefix and the observation window are always retained, so there must be
    // room left in the budget to select at least one earlier token by pooled score.
    if (sinkTokens + observationWindow >= budget) {
      throw new Error(
        `${this.kind}: sinkTokens (${sinkTokens}) + observationWindow (${observationWindow}) ` +
          `must be smaller than the resolved budget (${budget}); no room to select earlier tokens.`,
      );
    }

    return {
      kind: this.kind,
      budget,
      sinkTokens,
      observationWindow,
      poolingKernelSize,
    };
  }
}

/** The default eviction config: full cache, engine unchanged. */
export const DEFAULT_EVICTION_CONFIG: EvictionConfig = {
  kind: EvictionPolicyKind.NoOp,
};

/**
 * Construct the `EvictionPolicy` for a config. Central dispatch point — concrete
 * attention-aware policies register here as they land (SnapKV is live as of VAS-56;
 * PyramidKV/H2O in VAS-57/60). Unset config → no-op (engine runs unchanged).
 */
export function createEvictionPolicy(config?: EvictionConfig): EvictionPolicy {
  if (config === undefined || config.kind === EvictionPolicyKind.NoOp) {
    return new NoOpEvictionPolicy();
  }
  switch (config.kind) {
    case EvictionPolicyKind.StreamingLLM:
      return new StreamingLLMEvictionPolicy(config);
    case EvictionPolicyKind.SnapKV:
      return new SnapKVEvictionPolicy(config);
    case EvictionPolicyKind.H2O:
    case EvictionPolicyKind.PyramidKV:
      throw new Error(
        `Eviction policy "${config.kind}" is not implemented yet (tracked in VAS-57/60).`,
      );
    default:
      throw new Error(
        `Unknown eviction policy kind: ${(config as EvictionConfig).kind}.`,
      );
  }
}

/**
 * Returns true when the config requires no runtime hooks (so the engine can take the
 * fast, unmodified path). Any explicit non-no-op kind (StreamingLLM live as of VAS-53,
 * SnapKV as of VAS-56; PyramidKV/H2O land in VAS-57/60) requires hooks → returns false.
 */
export function isNoOpEviction(config?: EvictionConfig): boolean {
  return config === undefined || config.kind === EvictionPolicyKind.NoOp;
}
