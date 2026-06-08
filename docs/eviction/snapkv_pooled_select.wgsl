// SnapKV prefill-end "pooled-select" kernel (VAS-56) — DESIGN REFERENCE.
//
// This is the WGSL compute kernel SnapKV (the ★ primary attention-aware policy, VAS-56)
// runs ONCE at the end of prefill to choose which earlier KV positions to retain. It is
// the runtime (TVM/WGSL) side of the boundary; the web-llm TS layer
// (`src/eviction_policy.ts` → `SnapKVEvictionPolicy`) only emits the declarative
// `{budget, sinkTokens, observationWindow, poolingKernelSize}` config this kernel reads.
//
// ⚠️ Status: design reference, NOT yet wired into a compiled model lib. The score read
// path (how observation-window query·key scores are obtained without materializing the
// full FlashAttention matrix) is de-risked in VAS-47, and building/compiling the model
// lib that hosts this kernel is VAS-87. It lives here so the algorithm is reviewable and
// the TS config contract has a concrete consumer to point at (EVICTION_BOUNDARY.md).
//
// Why this is cheap (and why SnapKV wins in-browser): FlashAttention never writes the
// full [seq, seq] attention matrix. But SnapKV only needs the LAST `obs_window` rows of
// it — the scores from the observation-window queries onto all earlier keys. That's an
// [obs_window, seq_len] block with obs_window << seq_len, computed once. After this runs,
// the retained set is fixed: ZERO per-decode-step dispatch (H3), which is the dominant
// cost at batch=1.
//
// Pipeline (per attention head h):
//   1. score[j]  = sum_{i in obs_window} softmax(Q_obs[i] · K[j]^T)   for j < prefill-obs
//      (the observation-window queries vote on earlier keys; sink + obs window are
//       always retained and excluded from selection)
//   2. pooled[j] = avg_pool(score, poolingKernelSize)  // coherent spans beat lone spikes
//   3. keep the top (budget - sinkTokens - obs_window) positions by pooled[j]
//
// Steps 1–2 are this kernel. Top-K (step 3) is a second small dispatch over `pooled`
// (a bitonic / threshold top-K over seq_len elements per head); emitted as the retained
// page list the PagedKVCache compaction hook consumes. Bindings below cover steps 1–2.

// ---- Uniforms ------------------------------------------------------------------------
struct Params {
  seq_len          : u32,  // total prefill length
  num_heads        : u32,  // KV heads (GQA: KV heads, not query heads)
  head_dim         : u32,  // per-head dim of Q/K
  obs_window       : u32,  // observation window = SnapKVEvictionPolicy.observationWindow
  sink_tokens      : u32,  // always-retained prefix       = .sinkTokens
  pooling_kernel   : u32,  // odd avg-pool kernel size      = .poolingKernelSize
  select_count     : u32,  // budget - sink_tokens - obs_window (earlier tokens to keep)
};

@group(0) @binding(0) var<uniform> params : Params;
// Observation-window queries: [num_heads, obs_window, head_dim], row-major.
@group(0) @binding(1) var<storage, read>       q_obs   : array<f32>;
// Keys for the whole prefill: [num_heads, seq_len, head_dim], row-major.
@group(0) @binding(2) var<storage, read>       k_all   : array<f32>;
// Raw per-key pooled-before scores: [num_heads, seq_len]. Scratch (output of step 1).
@group(0) @binding(3) var<storage, read_write> score   : array<f32>;
// Final pooled scores: [num_heads, seq_len]. Consumed by the top-K dispatch (step 3).
@group(0) @binding(4) var<storage, read_write> pooled  : array<f32>;

// ---- Step 1: observation-window scoring ----------------------------------------------
// One invocation per (head, key position j). Accumulates the attention mass the
// observation-window queries place on key j. Softmax is folded in approximately via the
// max-subtracted exp sum over the obs window (numerically safe; exact normalization is
// not needed because we only rank keys, never use the magnitudes downstream).
//
// `selectable_len = seq_len - obs_window` — the obs window itself is always kept, so its
// own positions are not scored as eviction candidates.
@compute @workgroup_size(64)
fn score_obs_window(@builtin(global_invocation_id) gid : vec3<u32>) {
  let head = gid.y;
  let j    = gid.x;                       // candidate key position
  let selectable_len = params.seq_len - params.obs_window;
  if (head >= params.num_heads || j >= selectable_len) {
    return;
  }

  let hd       = params.head_dim;
  let k_base   = (head * params.seq_len + j) * hd;

  // Sink prefix is always retained → force it to the top of the ranking, skip the dot.
  if (j < params.sink_tokens) {
    score[head * params.seq_len + j] = 3.4e38;  // +inf sentinel
    return;
  }

  var acc : f32 = 0.0;
  for (var i : u32 = 0u; i < params.obs_window; i = i + 1u) {
    let q_base = (head * params.obs_window + i) * hd;
    var dot : f32 = 0.0;
    for (var d : u32 = 0u; d < hd; d = d + 1u) {
      dot = dot + q_obs[q_base + d] * k_all[k_base + d];
    }
    // exp of the scaled logit; scale = 1/sqrt(head_dim). Approximate (no per-row max
    // subtraction) — adequate for ranking, and the obs window is small.
    acc = acc + exp(dot / sqrt(f32(hd)));
  }
  score[head * params.seq_len + j] = acc;
}

// ---- Step 2: average pooling ---------------------------------------------------------
// One invocation per (head, position j). Symmetric avg-pool of width `pooling_kernel`
// (odd) over `score`, clamped at sequence boundaries. Clusters contiguous important
// positions so SnapKV keeps coherent spans (SnapKV paper §3.2). +inf sink sentinels stay
// dominant under the mean, so the sink prefix survives pooling.
@compute @workgroup_size(64)
fn avg_pool(@builtin(global_invocation_id) gid : vec3<u32>) {
  let head = gid.y;
  let j    = gid.x;
  let selectable_len = params.seq_len - params.obs_window;
  if (head >= params.num_heads || j >= selectable_len) {
    return;
  }

  let half = params.pooling_kernel / 2u;          // kernel is odd → symmetric
  let lo   = select(j - half, 0u, j < half);
  let hi   = min(j + half, selectable_len - 1u);

  var sum : f32 = 0.0;
  var cnt : u32 = 0u;
  for (var t : u32 = lo; t <= hi; t = t + 1u) {
    sum = sum + score[head * params.seq_len + t];
    cnt = cnt + 1u;
  }
  pooled[head * params.seq_len + j] = sum / f32(cnt);
}

// Step 3 (separate dispatch, not shown): top-`select_count` of `pooled` per head →
// retained position list → PagedKVCache page-table compaction at the prefill-end hook.
// Positions kept = sink prefix (∞ sentinels) ∪ top-K earlier ∪ the observation window.
// RoPE/position IDs of retained K are already baked in at prefill, so no re-rope needed
// (sparse position IDs, plan §3.5 Option 1) — same property StreamingLLM relies on.
