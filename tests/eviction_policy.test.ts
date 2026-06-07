import {
  EvictionConfig,
  EvictionPolicyKind,
  NoOpEvictionPolicy,
  StreamingLLMEvictionPolicy,
  DEFAULT_SINK_TOKENS,
  isNoOpEviction,
  resolveBudgetTokens,
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

  test("not-yet-implemented policies throw with a tracking pointer", () => {
    expect(() =>
      createEvictionPolicy({ kind: EvictionPolicyKind.SnapKV }),
    ).toThrow(/not implemented/);
  });
});
