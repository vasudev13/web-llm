/**
 * @mlc-ai/web-xgrammar and @mlc-ai/web-tokenizers ship sourcemaps that list
 * ../src/*_binding.js, but npm packages only include `lib/`. Parcel still
 * tries to open those paths during packaging. Minimal stubs satisfy ENOENT.
 */
/* eslint-disable @typescript-eslint/no-require-imports -- Node CJS pre-bundler */
const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..", "..", "..");
const stubs = [
  path.join(
    root,
    "node_modules",
    "@mlc-ai",
    "web-xgrammar",
    "src",
    "xgrammar_binding.js",
  ),
  path.join(
    root,
    "node_modules",
    "@mlc-ai",
    "web-tokenizers",
    "src",
    "tokenizers_binding.js",
  ),
];

const banner =
  "// Auto-generated stub for Parcel (see examples/context-window-oom-bench/scripts/ensure-vendor-source-stubs.js)\n";

for (const file of stubs) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  if (!fs.existsSync(file)) {
    fs.writeFileSync(file, banner + "export {};\n", "utf8");
  }
}
