/**
 * Stub for Node's `ws` package when bundling @mlc-ai/web-runtime for the browser.
 * The real code path only runs when `typeof WebSocket === "undefined"` (Node).
 */
function WsStub() {}
module.exports = WsStub;
module.exports.default = WsStub;
