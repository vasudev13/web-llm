import {
  EvictionConfig,
  EvictionPolicyKind,
  NoOpEvictionPolicy,
  StreamingLLMEvictionPolicy,
  SnapKVEvictionPolicy,
  PyramidKVEvictionPolicy,
  DEFAULT_SINK_TOKENS,
  DEFAULT_OBSERVATION_WINDOW,
  DEFAULT_POOLING_KERNEL_SIZE,
  DEFAULT_PYRAMID_ALPHA,
  isNoOpEviction,
  resolveBudgetTokens,
  computePyramidLayerBudgets,
  createEvictionPolicy,
} from "../src/eviction_policy";
import { describe, expect, test } from "@jest/globals";

describe("resolveBudgetTokens", () => {
  test("undefined budget → full context", () => {
    expect(resolveBudgetTokens(undefined, 4096)).toBe(4096);
  });

  test("ratio (<= 1) resolves against context window", () => {
    expect(resolveBudgetTokens(0.25, 4096)).toBe(1024);
    expect(resolveBudgetTokens(1, 4096)).toBe(4096);
  });

  test("absolute (> 1) is taken as-is, floored, capped at context", () => {
    expect(resolveBudgetTokens(1000, 4096)).toBe(1000);
    expect(resolveBudgetTokens(1000.9, 4096)).toBe(1000);
    expect(resolveBudgetTokens(99999, 4096)).toBe(4096);
  });

  test("rejects non-positive or non-finite budget", () => {
    expect(() => resolveBudgetTokens(0, 4096)).toThrow();
    expect(() => resolveBudgetTokens(-1, 4096)).toThrow();
    expect(() => resolveBudgetTokens(NaN, 4096)).toThrow();
  });

  test("rejects invalid context window", () => {
    expect(() => resolveBudgetTokens(0.5, 0)).toThrow();
  });
});

describe("NoOpEvictionPolicy", () => {
  test("resolves to no-op and reads as no-op", () => {
    const policy = new NoOpEvictionPolicy();
    expect(policy.kind).toBe(EvictionPolicyKind.NoOp);
    expect(policy.resolve(4096)).toEqual({ kind: EvictionPolicyKind.NoOp });
    expect(isNoOpEviction(policy.resolve(4096))).toBe(true);
  });
});

describe("StreamingLLMEvictionPolicy", () => {
  test("derives window = budget - sink with default sink tokens", () => {
    const config: EvictionConfig = {
      kind: EvictionPolicyKind.StreamingLLM,
      budget: 1024,
    };
    const resolved = new StreamingLLMEvictionPolicy(config).resolve(4096);
    expect(resolved).toEqual({
      kind: EvictionPolicyKind.StreamingLLM,
      budget: 1024,
      sinkTokens: DEFAULT_SINK_TOKENS,
      windowSize: 1024 - DEFAULT_SINK_TOKENS,
    });
    expect(isNoOpEviction(resolved)).toBe(false);
  });

  test("ratio budget resolves against context window", () => {
    const config: EvictionConfig = {
      kind: EvictionPolicyKind.StreamingLLM,
      budget: 0.25,
      sinkTokens: 4,
    };
    const resolved = new StreamingLLMEvictionPolicy(config).resolve(8192);
    // 0.25 * 8192 = 2048 budget
    expect(resolved.budget).toBe(2048);
    expect(resolved.windowSize).toBe(2048 - 4);
  });

  test("explicit windowSize is honored; budget = sink + window", () => {
    const config: EvictionConfig = {
      kind: EvictionPolicyKind.StreamingLLM,
      budget: 4096,
      sinkTokens: 8,
      windowSize: 500,
    };
    const resolved = new StreamingLLMEvictionPolicy(config).resolve(4096);
    expect(resolved.sinkTokens).toBe(8);
    expect(resolved.windowSize).toBe(500);
    expect(resolved.budget).toBe(508);
  });

  test("rejects sinkTokens >= budget (no room for a window)", () => {
    const config: EvictionConfig = {
      kind: EvictionPolicyKind.StreamingLLM,
      budget: 4,
      sinkTokens: 4,
    };
    expect(() =>
      new StreamingLLMEvictionPolicy(config).resolve(4096),
    ).toThrow();
  });

  test("rejects negative / non-integer sink tokens", () => {
    expect(() =>
      new StreamingLLMEvictionPolicy({
        kind: EvictionPolicyKind.StreamingLLM,
        budget: 1024,
        sinkTokens: -1,
      }).resolve(4096),
    ).toThrow();
  });

  test("rejects a mismatched config kind", () => {
    expect(
      () =>
        new StreamingLLMEvictionPolicy({
          kind: EvictionPolicyKind.NoOp,
        }),
    ).toThrow();
  });
});

describe("createEvictionPolicy", () => {
  test("undefined / no-op → NoOpEvictionPolicy", () => {
    expect(createEvictionPolicy()).toBeInstanceOf(NoOpEvictionPolicy);
    expect(
      createEvictionPolicy({ kind: EvictionPolicyKind.NoOp }),
    ).toBeInstanceOf(NoOpEvictionPolicy);
  });

  test("streaming-llm → StreamingLLMEvictionPolicy", () => {
    expect(
      createEvictionPolicy({
        kind: EvictionPolicyKind.StreamingLLM,
        budget: 1024,
      }),
    ).toBeInstanceOf(StreamingLLMEvictionPolicy);
  });

  test("snapkv → SnapKVEvictionPolicy", () => {
    expect(
      createEvictionPolicy({
        kind: EvictionPolicyKind.SnapKV,
        budget: 1024,
      }),
    ).toBeInstanceOf(SnapKVEvictionPolicy);
  });

  test("pyramid-kv → PyramidKVEvictionPolicy", () => {
    expect(
      createEvictionPolicy({
        kind: EvictionPolicyKind.PyramidKV,
        budget: 1024,
      }),
    ).toBeInstanceOf(PyramidKVEvictionPolicy);
  });

  test("not-yet-implemented policies throw with a tracking pointer", () => {
    expect(() =>
      createEvictionPolicy({ kind: EvictionPolicyKind.H2O }),
    ).toThrow(/not implemented/);
  });
});

describe("SnapKVEvictionPolicy", () => {
  test("fills defaults: sink, observation window, pooling kernel", () => {
    const config: EvictionConfig = {
      kind: EvictionPolicyKind.SnapKV,
      budget: 1024,
    };
    const resolved = new SnapKVEvictionPolicy(config).resolve(4096);
    expect(resolved).toEqual({
      kind: EvictionPolicyKind.SnapKV,
      budget: 1024,
      sinkTokens: DEFAULT_SINK_TOKENS,
      observationWindow: DEFAULT_OBSERVATION_WINDOW,
      poolingKernelSize: DEFAULT_POOLING_KERNEL_SIZE,
    });
    expect(isNoOpEviction(resolved)).toBe(false);
  });

  test("ratio budget resolves against context window", () => {
    const config: EvictionConfig = {
      kind: EvictionPolicyKind.SnapKV,
      budget: 0.25,
    };
    const resolved = new SnapKVEvictionPolicy(config).resolve(8192);
    // 0.25 * 8192 = 2048
    expect(resolved.budget).toBe(2048);
  });

  test("honors explicit observation window and pooling kernel", () => {
    const config: EvictionConfig = {
      kind: EvictionPolicyKind.SnapKV,
      budget: 2048,
      sinkTokens: 8,
      observationWindow: 64,
      poolingKernelSize: 13,
    };
    const resolved = new SnapKVEvictionPolicy(config).resolve(8192);
    expect(resolved.sinkTokens).toBe(8);
    expect(resolved.observationWindow).toBe(64);
    expect(resolved.poolingKernelSize).toBe(13);
  });

  test("rejects sink + observation window >= budget (no room to select)", () => {
    const config: EvictionConfig = {
      kind: EvictionPolicyKind.SnapKV,
      budget: 32,
      sinkTokens: 4,
      observationWindow: 32,
    };
    expect(() => new SnapKVEvictionPolicy(config).resolve(4096)).toThrow();
  });

  test("rejects an even pooling kernel size", () => {
    expect(() =>
      new SnapKVEvictionPolicy({
        kind: EvictionPolicyKind.SnapKV,
        budget: 1024,
        poolingKernelSize: 8,
      }).resolve(4096),
    ).toThrow(/odd/);
  });

  test("rejects a non-positive observation window", () => {
    expect(() =>
      new SnapKVEvictionPolicy({
        kind: EvictionPolicyKind.SnapKV,
        budget: 1024,
        observationWindow: 0,
      }).resolve(4096),
    ).toThrow();
  });

  test("rejects a mismatched config kind", () => {
    expect(
      () =>
        new SnapKVEvictionPolicy({
          kind: EvictionPolicyKind.StreamingLLM,
        }),
    ).toThrow();
  });
});

describe("computePyramidLayerBudgets", () => {
  test("per-layer counts sum to exactly the total selectable budget", () => {
    for (const alpha of [0.5, 0.7, 0.9, 1]) {
      const budgets = computePyramidLayerBudgets(1000, 28, alpha);
      expect(budgets).toHaveLength(28);
      expect(budgets.reduce((a, b) => a + b, 0)).toBe(1000);
      expect(budgets.every((b) => Number.isInteger(b) && b >= 0)).toBe(true);
    }
  });

  test("alpha < 1 → non-increasing budget up the stack (lower layers keep more)", () => {
    const budgets = computePyramidLayerBudgets(1000, 28, 0.7);
    for (let l = 1; l < budgets.length; l++) {
      expect(budgets[l]).toBeLessThanOrEqual(budgets[l - 1]);
    }
    // First layer strictly larger than the last when there is budget to differentiate.
    expect(budgets[0]).toBeGreaterThan(budgets[budgets.length - 1]);
  });

  test("alpha === 1 → uniform split (degenerates to flat SnapKV budget)", () => {
    const budgets = computePyramidLayerBudgets(120, 4, 1);
    expect(budgets).toEqual([30, 30, 30, 30]);
  });

  test("uniform split with remainder is distributed deterministically to lower layers", () => {
    // 10 across 4 uniform layers → 2 each + 2 leftover to the two lowest layers.
    expect(computePyramidLayerBudgets(10, 4, 1)).toEqual([3, 3, 2, 2]);
  });

  test("single layer gets the whole budget", () => {
    expect(computePyramidLayerBudgets(512, 1, 0.5)).toEqual([512]);
  });

  test("zero selectable budget → all layers zero", () => {
    expect(computePyramidLayerBudgets(0, 4, 0.7)).toEqual([0, 0, 0, 0]);
  });

  test("rejects invalid layer count, budget, or alpha", () => {
    expect(() => computePyramidLayerBudgets(100, 0, 0.7)).toThrow();
    expect(() => computePyramidLayerBudgets(100, 2.5, 0.7)).toThrow();
    expect(() => computePyramidLayerBudgets(-1, 4, 0.7)).toThrow();
    expect(() => computePyramidLayerBudgets(100, 4, 0)).toThrow();
    expect(() => computePyramidLayerBudgets(100, 4, 1.5)).toThrow();
  });
});

describe("PyramidKVEvictionPolicy", () => {
  test("inherits SnapKV defaults and fills pyramidAlpha", () => {
    const resolved = new PyramidKVEvictionPolicy({
      kind: EvictionPolicyKind.PyramidKV,
      budget: 1024,
    }).resolve(4096);
    expect(resolved).toEqual({
      kind: EvictionPolicyKind.PyramidKV,
      budget: 1024,
      sinkTokens: DEFAULT_SINK_TOKENS,
      observationWindow: DEFAULT_OBSERVATION_WINDOW,
      poolingKernelSize: DEFAULT_POOLING_KERNEL_SIZE,
      pyramidAlpha: DEFAULT_PYRAMID_ALPHA,
    });
    expect(isNoOpEviction(resolved)).toBe(false);
  });

  test("is a SnapKV subclass (reuses pooled-select selection)", () => {
    const policy = new PyramidKVEvictionPolicy({
      kind: EvictionPolicyKind.PyramidKV,
      budget: 1024,
    });
    expect(policy).toBeInstanceOf(SnapKVEvictionPolicy);
    expect(policy.kind).toBe(EvictionPolicyKind.PyramidKV);
  });

  test("honors an explicit pyramidAlpha", () => {
    const resolved = new PyramidKVEvictionPolicy({
      kind: EvictionPolicyKind.PyramidKV,
      budget: 2048,
      pyramidAlpha: 0.5,
    }).resolve(8192);
    expect(resolved.pyramidAlpha).toBe(0.5);
  });

  test("rejects a pyramidAlpha outside (0, 1]", () => {
    for (const bad of [0, -0.1, 1.5, NaN]) {
      expect(() =>
        new PyramidKVEvictionPolicy({
          kind: EvictionPolicyKind.PyramidKV,
          budget: 1024,
          pyramidAlpha: bad,
        }).resolve(4096),
      ).toThrow(/pyramidAlpha/);
    }
  });

  test("inherits SnapKV's no-room-to-select guard", () => {
    expect(() =>
      new PyramidKVEvictionPolicy({
        kind: EvictionPolicyKind.PyramidKV,
        budget: 32,
        sinkTokens: 4,
        observationWindow: 32,
      }).resolve(4096),
    ).toThrow();
  });

  test("rejects a mismatched config kind", () => {
    expect(
      () =>
        new PyramidKVEvictionPolicy({
          kind: EvictionPolicyKind.SnapKV,
        }),
    ).toThrow();
  });

  describe("resolveLayerBudgets", () => {
    test("per-layer totals = sink + obs + pyramid share; selectable sums correctly", () => {
      const policy = new PyramidKVEvictionPolicy({
        kind: EvictionPolicyKind.PyramidKV,
        budget: 1024,
        sinkTokens: 4,
        observationWindow: 32,
        pyramidAlpha: 0.7,
      });
      const layerBudgets = policy.resolveLayerBudgets(4096, 28);
      const alwaysRetained = 4 + 32;
      expect(layerBudgets).toHaveLength(28);
      // Each layer keeps at least the always-retained prefix + window.
      expect(layerBudgets.every((b) => b >= alwaysRetained)).toBe(true);
      // Selectable shares (budget minus always-retained) sum to budget - alwaysRetained.
      const selectableSum = layerBudgets.reduce(
        (acc, b) => acc + (b - alwaysRetained),
        0,
      );
      expect(selectableSum).toBe(1024 - alwaysRetained);
      // Pyramid shape: lower layers keep at least as much as higher layers.
      for (let l = 1; l < layerBudgets.length; l++) {
        expect(layerBudgets[l]).toBeLessThanOrEqual(layerBudgets[l - 1]);
      }
    });

    test("alpha === 1 → every layer gets the same flat SnapKV budget", () => {
      const policy = new PyramidKVEvictionPolicy({
        kind: EvictionPolicyKind.PyramidKV,
        budget: 1024,
        sinkTokens: 4,
        observationWindow: 32,
        pyramidAlpha: 1,
      });
      const layerBudgets = policy.resolveLayerBudgets(4096, 8);
      const selectable = 1024 - (4 + 32);
      const base = Math.floor(selectable / 8);
      // Uniform up to the largest-remainder distribution of selectable % 8 tokens.
      expect(
        Math.max(...layerBudgets) - Math.min(...layerBudgets),
      ).toBeLessThanOrEqual(1);
      expect(layerBudgets[layerBudgets.length - 1]).toBe(4 + 32 + base);
    });
  });
});
