/**
 * Cross-origin-isolated static server for the context-window / OOM-cliff
 * benchmark (Linear VAS-85).
 *
 * `performance.measureUserAgentSpecificMemory()` -- the API the harness uses to
 * record a *real* memory number alongside the estimated peak VRAM -- only
 * resolves when the page is **cross-origin isolated** (`crossOriginIsolated ===
 * true`). That requires two response headers the default Parcel dev server does
 * NOT send:
 *
 *   Cross-Origin-Opener-Policy:   same-origin
 *   Cross-Origin-Embedder-Policy: credentialless   (or require-corp)
 *
 * This zero-dependency server serves the Parcel *build* output with those
 * headers so the bench page becomes cross-origin isolated and the measured
 * memory probe returns data instead of coming back blank.
 *
 * COEP mode -- why `credentialless` is the default:
 *   WebLLM fetches model weights cross-origin from the MLC/HuggingFace CDN.
 *   Under COEP `require-corp`, every cross-origin subresource must send a
 *   `Cross-Origin-Resource-Policy` header or the fetch is blocked -- and the CDN
 *   does not. COEP `credentialless` keeps the page cross-origin isolated while
 *   still allowing those no-CORS/no-CORP cross-origin fetches (they are sent
 *   without credentials), which is exactly the model-download case. Override
 *   with `COEP=require-corp` if you mirror the weights same-origin.
 *
 * Usage (from examples/context-window-oom-bench):
 *   npm run build                 # produce lib/ first
 *   node scripts/serve-isolated.mjs            # serves ./lib on :8888
 *   PORT=9000 node scripts/serve-isolated.mjs  # custom port
 *   COEP=require-corp node scripts/serve-isolated.mjs
 *   node scripts/serve-isolated.mjs ./some-dir # custom root
 *
 * Or via the npm wrapper: `npm run start:isolated` (build + serve).
 */
import http from "node:http";
import { createReadStream, existsSync, readdirSync, statSync } from "node:fs";
import { extname, join, normalize, resolve } from "node:path";

const ROOT = resolve(process.argv[2] ?? "lib");
const PORT = parseInt(process.env.PORT ?? "8888", 10);
const COEP = process.env.COEP === "require-corp" ? "require-corp" : "credentialless";

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".wasm": "application/wasm",
  ".map": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
  ".png": "image/png",
  ".woff2": "font/woff2",
};

// Parcel names the output after the entry HTML (not index.html), so resolve a
// sensible default document for `/`.
function defaultDoc() {
  if (existsSync(join(ROOT, "index.html"))) return "index.html";
  if (!existsSync(ROOT)) return "index.html";
  const html = readdirSync(ROOT).filter((f) => f.endsWith(".html"));
  return html[0] ?? "index.html";
}

// The two headers that make the page cross-origin isolated, applied to every
// response so subresources are covered too.
function setIsolationHeaders(res) {
  res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
  res.setHeader("Cross-Origin-Embedder-Policy", COEP);
}

const server = http.createServer((req, res) => {
  setIsolationHeaders(res);

  // Resolve + contain the request path to ROOT (no path traversal).
  let rel = decodeURIComponent((req.url ?? "/").split("?")[0]);
  if (rel === "/" || rel === "") rel = defaultDoc();
  const filePath = normalize(join(ROOT, rel));
  if (!filePath.startsWith(ROOT)) {
    res.statusCode = 403;
    res.end("Forbidden");
    return;
  }

  if (!existsSync(filePath) || !statSync(filePath).isFile()) {
    res.statusCode = 404;
    res.setHeader("Content-Type", "text/plain; charset=utf-8");
    res.end(`Not found: ${rel}\nServing root: ${ROOT}`);
    return;
  }

  res.statusCode = 200;
  res.setHeader("Content-Type", MIME[extname(filePath)] ?? "application/octet-stream");
  createReadStream(filePath).pipe(res);
});

server.listen(PORT, () => {
  if (!existsSync(ROOT)) {
    console.warn(
      `[serve-isolated] WARNING: root ${ROOT} does not exist yet. ` +
        "Run `npm run build` first (or `npm run start:isolated`).",
    );
  }
  console.log(`[serve-isolated] root          : ${ROOT}`);
  console.log(`[serve-isolated] COOP          : same-origin`);
  console.log(`[serve-isolated] COEP          : ${COEP}`);
  console.log(`[serve-isolated] cross-origin isolated page at http://localhost:${PORT}/`);
  console.log(
    "[serve-isolated] open DevTools console; expect `crossOriginIsolated === true` " +
      "and a non-blank measMem(MB) column.",
  );
});
