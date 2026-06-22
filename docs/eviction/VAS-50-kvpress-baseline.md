# VAS-50 — kvpress baseline run on M4 Air (MPS) + calibrate per-run timing

**Milestone:** Phase 0 (Environment Setup) · **Priority:** High
**Status:** In Progress — runnable, version-pinned harness landed (audit + setup +
calibration run + result schema). The one remaining step is the hands-on
execution: `01-setup` (pulls torch, a few min) → first `02-run` (downloads
~6 GB Qwen2.5-3B, then the MPS run). An unattended job shouldn't trigger that
multi-GB download/install, so this stays In Progress until the number is
recorded (see §5).

> **Why this ticket matters / where it sits.** This is the entry point to
> **Track A** — the NVIDIA **kvpress** quality oracle (LongBench + RULER, FP16
> vs q4). kvpress already implements all four policy families
> (StreamingLLM, H2O via ObservedAttention, SnapKV, PyramidKV), so it is the
> **correctness oracle**, not a competitor: the browser port (Track B) is later
> validated to match these numbers (VAS-58). Critically, Track A is
> **independent of the stuck build-from-source toolchain (VAS-87)** — it runs in
> pure Python on MPS and can proceed in parallel. VAS-50 → VAS-51 (full-cache
> baselines) → VAS-54 (FP16 sweep) → VAS-55 (q4 sweep, the novel H4 finding).

---

## 1. What VAS-50 actually asks for

From the ticket acceptance:

- [ ] kvpress installed and runs on M4 Air (MPS)
- [ ] One clean SnapKV/RULER run on Qwen2.5-3B completes
- [ ] Measured per-run wall-clock number recorded

The plan assumes **30–60 min/run** and a full grid of **~80 runs = 2–3 days**.
The whole point of this ticket is to replace that *assumption* with a *measured*
number, so the Phase 1 sweep schedule (VAS-54/55) is grounded.

## 2. The harness (this run)

Everything lives under `scripts/eviction/kvpress/`, numbered like the VAS-87
build scripts so the run order is obvious:

| File | Role | Mutates machine? |
|------|------|------------------|
| `00-audit-kvpress-env.sh` | read-only env probe (python, torch, MPS, model cache) | no |
| `requirements.txt` | conservative version floors (torch 2.3+, transformers 4.44+, kvpress 0.2+) | no |
| `01-setup-kvpress-env.sh` | create `.venv-kvpress`, `pip install` the stack | venv + wheels only |
| `02-run-baseline.py` | load Qwen2.5-3B on MPS → SnapKV → RULER-NIAH → time it → JSON | downloads model on first run |
| `02-run-baseline.sh` | venv wrapper + coarse `time` cross-check | — |
| `results/baseline-*.json` | recorded calibration number (created by the run) | — |

**Design choices that matter:**

- **Isolated venv, never the system interpreter.** The M4 Air's `python3` is
  3.9.6 (pyenv shim) — too old for kvpress. `01-setup` auto-picks a 3.11/3.12
  (or tells you to `pyenv install 3.11`) and builds `.venv-kvpress` from it.
- **RULER with a synthetic fallback.** `02-run-baseline.py` tries to pull a real
  RULER needle-in-a-haystack sample (`simonjegou/ruler`, the mirror kvpress
  uses); if the dataset is unavailable offline it builds a self-contained NIAH
  prompt so the calibration *always* completes. Either way it checks the needle
  was actually retrieved — a free correctness sanity check on the SnapKV path.
- **Calibration, not coverage.** `--samples 1` by default. We want a trustworthy
  per-run number, not the sweep (that's VAS-54/55). The script extrapolates a
  full-grid estimate (`mean_run_s × 64 runs`) and prints it next to the plan's
  30–60 min/run assumption.

## 3. kvpress API surface used

kvpress registers a custom transformers pipeline, `kv-press-text-generation`,
driven by a `Press` object:

```python
from transformers import pipeline
from kvpress import SnapKVPress
pipe  = pipeline("kv-press-text-generation", model="Qwen/Qwen2.5-3B-Instruct",
                 device="mps", torch_dtype=torch.float16)
press = SnapKVPress(compression_ratio=0.5)
answer = pipe(context, question=question, press=press)["answer"]
```

The harness wraps exactly this. `--press` selects `SnapKVPress` (default),
`StreamingLLMPress`, `KnormPress`, or `ExpectedAttentionPress` — the same four
families that map onto the browser policies (VAS-53/56/57/60), so this single
script also doubles as the smoke test for each policy's oracle side.

## 4. How to run it (hands-on, ~one sitting)

```bash
# 0. see the gaps (read-only)
bash scripts/eviction/kvpress/00-audit-kvpress-env.sh

# 1. one-time: build the venv + install the stack (pulls torch; a few minutes)
bash scripts/eviction/kvpress/01-setup-kvpress-env.sh

# 2. the calibration run — SnapKV / RULER-NIAH / Qwen2.5-3B on MPS
#    first invocation also downloads ~6 GB of Qwen weights
bash scripts/eviction/kvpress/02-run-baseline.sh
#    → prints "CALIBRATION: mean per-run = …s" and writes results/baseline-*.json
```

To sanity-check a second policy without re-downloading:
`bash scripts/eviction/kvpress/02-run-baseline.sh --press streaming --ratio 0.25`.

## 5. Measured baseline so far (`00-audit`, this run, on the M4 Air)

```
Host            Darwin 24.6.0 arm64 · macOS 15.6 (24G84)
                Apple Silicon — MPS available ✓
Unified memory  16 GB
python3         3.9.6 (pyenv shim) — too old; pyenv 2.6.26 present ✓
venv            none yet (.venv-kvpress)
packages        torch / transformers / kvpress / datasets — NOT installed
model cache     Qwen2.5-3B not cached — first run downloads ~6 GB
```

**Hardware flag for VAS-51 (full-cache 3B + 7B baselines):** only **16 GB**
unified memory. Qwen2.5-3B FP16 (~6 GB weights + KV) is comfortable; **7B FP16
(~14 GB) will be extremely tight or OOM** alongside the OS and the eval harness.
Expect to run 7B at a lower precision or to lean on q4 there — worth confirming
early, before VAS-54's 2-model sweep assumes 7B FP16 fits. This matches the
Risk #4 concern already flagged in VAS-49.

**The per-run wall-clock number is intentionally not filled in here** — it
requires the hands-on `01`→`02` execution (torch install + 6 GB model download +
the MPS run), which an unattended scheduled run should not kick off. Once
recorded, paste the `calibration` block from `results/baseline-*.json` below and
tick the §1 boxes.

```
(measured mean per-run: ____ s  →  ____ min/run vs plan's 30–60)
(full-grid estimate: ____ h for 64 runs)
```

## 6. Acceptance mapping

| Acceptance criterion | Status | Evidence |
|---|---|---|
| kvpress installed + runs on MPS | ⏳ scripted | `01-setup` + `00-audit` confirm MPS available; install pending hands-on run |
| One clean SnapKV/RULER run on Qwen2.5-3B | ⏳ scripted | `02-run-baseline.py` (SnapKVPress + RULER-NIAH + retrieval check) |
| Measured per-run wall-clock recorded | ⏳ pending | `results/baseline-*.json` `calibration.mean_run_s` (run §4 step 2) |

Move to **Done** once `results/baseline-*.json` exists with a real
`mean_run_s` and the §5 number is pasted in.
