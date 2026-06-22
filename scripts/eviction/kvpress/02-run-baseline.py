#!/usr/bin/env python3
"""VAS-50 — kvpress baseline calibration run on Apple Silicon / MPS.

Goal of the ticket: confirm kvpress runs end-to-end on the M4 Air and record
ONE measured per-run wall-clock number, so the full-grid schedule (plan assumes
30–60 min/run, ~80 runs = 2–3 days) can be calibrated against reality.

What this does, concretely:
  1. Loads Qwen2.5-3B-Instruct (project-standard test model) on MPS, FP16.
  2. Builds a RULER-style needle-in-a-haystack (NIAH) prompt at a target context
     length — uses the real `RULER` dataset via `datasets` if available, else a
     self-contained synthetic NIAH (so the calibration always completes offline).
  3. Runs it through kvpress's SnapKV press at a compression ratio.
  4. Times prefill+decode wall-clock, checks the needle was retrieved (sanity),
     extrapolates a full-grid estimate, and writes results/baseline-<ts>.json.

This is a CALIBRATION run, not the full sweep (that's VAS-54/55). Keep --samples
small (default 1) — the point is a trustworthy per-run number, not coverage.

Usage:
  scripts/eviction/kvpress/02-run-baseline.sh            # wrapper (recommended)
  python 02-run-baseline.py --context 4096 --ratio 0.5 --press snapkv --samples 1
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL_DEFAULT = "Qwen/Qwen2.5-3B-Instruct"
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"


def log(msg: str) -> None:
    print(f"[02-baseline] {msg}", flush=True)


def build_synthetic_niah(tokenizer, target_tokens: int, seed: int = 0):
    """Self-contained needle-in-a-haystack: hide a magic number in filler text.

    Returns (context, question, needle_answer). Sized so context ≈ target_tokens.
    """
    rng = random.Random(seed)
    magic = rng.randint(100000, 999999)
    needle = f"The special access code for the Vasudev13 lab is {magic}."
    filler = (
        "The grass is green and the sky is blue. Researchers gathered data on "
        "long-context inference throughout the long afternoon. "
    )
    # grow filler until we hit the token budget, then insert the needle ~mid-way
    fill_tokens = tokenizer(filler, add_special_tokens=False)["input_ids"]
    per = max(1, len(fill_tokens))
    reps = max(4, target_tokens // per)
    haystack = (filler * reps).split(". ")
    insert_at = len(haystack) // 2
    haystack.insert(insert_at, needle)
    context = ". ".join(haystack)
    question = "What is the special access code for the Vasudev13 lab?"
    return context, question, str(magic)


def try_ruler_sample(tokenizer, target_tokens: int):
    """Best-effort: pull one real RULER NIAH sample. Returns None if unavailable."""
    try:
        from datasets import load_dataset  # noqa: WPS433

        # kvpress mirrors RULER under simonjegou/ruler with length configs.
        cfg = "4096" if target_tokens <= 4096 else "8192"
        ds = load_dataset("simonjegou/ruler", cfg, split="test", streaming=True)
        for row in ds:
            if "niah" in str(row.get("task", "")).lower():
                ctx = row.get("context") or row.get("input") or ""
                q = row.get("question") or row.get("query") or ""
                ans = row.get("answer") or (row.get("outputs") or [""])[0]
                if ctx and q and ans:
                    return ctx, q, str(ans)
        return None
    except Exception as exc:  # offline / dataset moved / schema drift
        log(f"RULER dataset unavailable ({type(exc).__name__}: {exc}); using synthetic NIAH")
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("KVPRESS_MODEL", MODEL_DEFAULT))
    ap.add_argument("--context", type=int, default=4096, help="target context tokens")
    ap.add_argument("--ratio", type=float, default=0.5, help="kvpress compression ratio")
    ap.add_argument("--press", default="snapkv", choices=["snapkv", "streaming", "knorm", "expected"])
    ap.add_argument("--samples", type=int, default=1, help="keep small — this is calibration")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    import torch  # imported here so --help works without the heavy stack
    from transformers import pipeline
    import kvpress

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        log("MPS not available — falling back to cpu (numbers won't match M4 Air)")
        device = "cpu"

    press_map = {
        "snapkv": "SnapKVPress",
        "streaming": "StreamingLLMPress",
        "knorm": "KnormPress",
        "expected": "ExpectedAttentionPress",
    }
    PressCls = getattr(kvpress, press_map[args.press])
    press = PressCls(compression_ratio=args.ratio)

    log(f"model={args.model} device={device} press={args.press} ratio={args.ratio} "
        f"context~{args.context} samples={args.samples}")

    t_load0 = time.perf_counter()
    pipe = pipeline(
        "kv-press-text-generation",
        model=args.model,
        device=device,
        torch_dtype=torch.float16,
    )
    load_s = time.perf_counter() - t_load0
    log(f"model loaded in {load_s:.1f}s")
    tokenizer = pipe.tokenizer

    per_sample = []
    retrieved_hits = 0
    for i in range(args.samples):
        sample = try_ruler_sample(tokenizer, args.context)
        source = "ruler"
        if sample is None:
            sample = build_synthetic_niah(tokenizer, args.context, seed=i)
            source = "synthetic-niah"
        context, question, answer = sample
        ctx_tokens = len(tokenizer(context, add_special_tokens=False)["input_ids"])

        t0 = time.perf_counter()
        out = pipe(context, question=question, press=press, max_new_tokens=args.max_new_tokens)
        dt = time.perf_counter() - t0
        pred = (out.get("answer") or "").strip()
        hit = answer.lower() in pred.lower()
        retrieved_hits += int(hit)
        per_sample.append({
            "i": i, "source": source, "ctx_tokens": ctx_tokens,
            "wall_s": round(dt, 2), "retrieved": hit, "answer": answer,
            "pred_head": pred[:80],
        })
        log(f"sample {i}: {ctx_tokens} ctx tok, {dt:.2f}s, retrieved={hit} ({source})")

    wall = [s["wall_s"] for s in per_sample]
    mean_s = sum(wall) / len(wall)
    # full grid from plan §: 4 methods × 4 ratios × 2 models × (LongBench+RULER)
    grid_runs = 4 * 4 * 2 * 2
    est_hours = mean_s * grid_runs / 3600.0

    result = {
        "ticket": "VAS-50",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch": torch.__version__,
            "device": device,
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        },
        "config": {
            "model": args.model, "press": args.press, "ratio": args.ratio,
            "target_context": args.context, "max_new_tokens": args.max_new_tokens,
            "samples": args.samples,
        },
        "model_load_s": round(load_s, 1),
        "per_sample": per_sample,
        "calibration": {
            "mean_run_s": round(mean_s, 2),
            "retrieval_accuracy": round(retrieved_hits / len(per_sample), 3),
            "assumed_full_grid_runs": grid_runs,
            "est_full_grid_hours": round(est_hours, 1),
            "plan_assumption_min_per_run": "30-60",
            "measured_min_per_run": round(mean_s / 60.0, 2),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"baseline-{args.press}-{args.context}-{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2))

    log("─" * 60)
    log(f"CALIBRATION: mean per-run = {mean_s:.2f}s ({mean_s/60:.2f} min)")
    log(f"retrieval sanity = {retrieved_hits}/{len(per_sample)}")
    log(f"full-grid estimate ({grid_runs} runs) ≈ {est_hours:.1f} h")
    log(f"plan assumed 30–60 min/run → measured {mean_s/60:.2f} min/run")
    log(f"wrote {out_path}")
    log("─" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
