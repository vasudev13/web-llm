# VAS-85 — Cross-origin isolation for measured VRAM (`measureUserAgentSpecificMemory`)

**Status:** Mechanism landed + header-verified; empirical measured-vs-estimated capture pending a hands-on Chrome session.
**Priority:** Medium · **Milestone:** Phase 1 · **Surfaced by:** VAS-49 (peak-VRAM numbers are *estimated*, not measured).
**Branch base:** `claude/4k-context-oom-cliff-SjWWx` (the VAS-49 harness is not yet on `main`).

> **TL;DR.** The OOM-cliff harness already records a real-memory probe
> (`performance.measureUserAgentSpecificMemory()`), but it comes back **blank**
> because the Parcel dev server does not make the page **cross-origin isolated**.
> This ticket adds a zero-dependency server (`scripts/serve-isolated.mjs` +
> `npm run start:isolated`) that sets **COOP `same-origin` + COEP
> `credentialless`**, flipping `crossOriginIsolated` to `true` so the probe
> returns data. The harness now logs isolation status up front and records a
> `crossOriginIsolated` provenance flag + a measured/estimated ratio per run.

---

## 1. The problem

For the MLSys systems story, reviewers expect **measured** memory, not a formula.
The harness's peak-VRAM column is computed from compiled metadata (params + KV +
temp buffers); it scales correctly and matches buffer geometry, but "we computed
this from metadata" is a soft spot in a systems claim.

`performance.measureUserAgentSpecificMemory()` is Chrome's real-memory API, but it
**only resolves on a cross-origin-isolated page**. Isolation requires two response
headers the default Parcel dev server (`npm start`) does not send, so
`measureUserAgentSpecificMemory` is `undefined` and the `measMem(MB)` column is
empty.

## 2. The fix — a cross-origin-isolated static server

`scripts/serve-isolated.mjs` serves the Parcel **build** output (`lib/`) with:

```
Cross-Origin-Opener-Policy:   same-origin
Cross-Origin-Embedder-Policy: credentialless     # COEP=require-corp to override
```

Both are needed for `crossOriginIsolated === true`. The headers are applied to
**every** response (document *and* subresources) — a subresource served without
COEP would otherwise break isolation.

```bash
# from examples/context-window-oom-bench
npm run start:isolated      # build + serve isolated on :8888
# or, if lib/ is already built:
npm run serve:isolated
# knobs:
PORT=9000 npm run serve:isolated
COEP=require-corp npm run serve:isolated
```

Then open `http://localhost:8888/` in Chrome with the DevTools console open:
`crossOriginIsolated` should be `true` and the `measMem(MB)` column non-blank.

### Why COEP `credentialless`, not `require-corp`

WebLLM fetches model weights **cross-origin** from the MLC/HuggingFace CDN. Under
COEP `require-corp`, every cross-origin subresource must carry a
`Cross-Origin-Resource-Policy` header or the fetch is **blocked** — and the CDN
does not send one. COEP `credentialless` keeps the page isolated while still
allowing those no-CORS cross-origin fetches (issued without credentials), which is
exactly the model-download path. Use `require-corp` only if you mirror the weights
same-origin. This is the model-download CORS fallout the ticket asked to document.

## 3. Harness wiring (this branch)

* **Startup banner** (`main()`): logs `crossOriginIsolated` and, when false, tells
  you to use `npm run start:isolated`. No more silent blank column.
* **Per-run provenance:** `RunResult.crossOriginIsolated` is recorded and added to
  the CSV, so a blank `measuredMemMB` is distinguishable from a genuine zero.
* **Validation log:** when both numbers exist, the run logs
  `measured/estimated = <m>MB / <e>MB = <ratio>x` to cross-check the estimate.
* The "API unavailable" warning now reports the live `crossOriginIsolated` value
  and names the fix.

## 4. ⚠️ Honest caveat — what `measureUserAgentSpecificMemory()` actually measures

It reports **renderer-process memory** (JS heap, DOM, workers, and some
GPU/staging accounting) — **not** a pure GPU-VRAM figure. In Chrome, large WebGPU
device buffers are largely tracked in the **GPU process**, which this API does not
fully attribute to the renderer. So:

* Treat the measured number as a **cross-check / lower bound** on the estimate, not
  an exact match. The validation ratio is expected to be **< 1** for VRAM-dominated
  runs (most KV/param bytes live GPU-side).
* For a true GPU-memory figure you still need `chrome://gpu`, `chrome://histograms`
  (`Memory.GPU.*`), or OS GPU tooling. The estimate-from-metadata remains the
  primary VRAM number; the measured probe corroborates direction/scale.

The paper (VAS-67) should report the metadata estimate as the headline VRAM figure
and cite the measured renderer memory as an independent sanity check, stating this
limitation rather than claiming `measureUserAgentSpecificMemory()` measures VRAM.

## 5. Acceptance mapping

| Acceptance criterion | State |
|---|---|
| Benchmark page is cross-origin isolated; measured memory available | **Mechanism done + header-verified** (curl confirms COOP `same-origin` + COEP `credentialless` on document & subresources; `require-corp` override works). Live `crossOriginIsolated===true` + non-blank probe is a one-line check in a hands-on Chrome session. |
| Measured vs estimated VRAM comparison recorded | **Wired** — per-run ratio logged + `crossOriginIsolated`/`measuredMemMB` in CSV. Numbers populate on the next browser run; see the §4 caveat on interpreting the ratio. |
| Ready to report real memory figures (VAS-67) | **Guidance written** (§4): estimate is the headline, measured probe corroborates, with the GPU-process limitation stated. |
| Document model-download CORS fallout from COEP | **Done** (§2 — COEP `credentialless` rationale). |

## 6. Remaining (hands-on, hardware-gated)

The unattended run verified the server headers but did **not** trigger a Parcel
build (pulls the example's dev deps) or a multi-GB model download. To fully close:

1. `cd examples/context-window-oom-bench && npm run start:isolated`
2. Open `http://localhost:8888/?smoke` in Chrome; confirm console prints
   `crossOriginIsolated=true` and a `measured/estimated = …x` line.
3. Paste one measured/estimated pair into this doc §4 and move VAS-85 → Done.

## 7. Cross-references

* `examples/context-window-oom-bench/` — the VAS-49 harness this extends.
* `scripts/serve-isolated.mjs` — the isolated server (this branch).
* VAS-49 — origin of the estimate-vs-measured gap. · VAS-64 — best done before the
  Phase 3 sweep so the sweep captures measured numbers. · VAS-67 — MLSys paper
  memory figures.
* Spec: `WindowOrWorkerGlobalScope.crossOriginIsolated`, `performance.measureUserAgentSpecificMemory()` (MDN).
