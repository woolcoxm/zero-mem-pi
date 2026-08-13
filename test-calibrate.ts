/**
 * Zero-Mem v0.6 test — answer-level calibration (deterministic, zero-LLM).
 * Checks: code-fence balance, JSON type expectation, entity coverage, and
 * verbatim-memory over-reliance detection. Non-destructive (emits warnings only).
 */
import { calibrate, makeExtractor, type Hit } from "./core.ts";

let pass = 0, fail = 0;
const check = (ok: boolean, label: string, extra = "") => { console.log(`  ${ok ? "✓ PASS" : "✗ FAIL"} — ${label}${extra ? "  " + extra : ""}`); ok ? pass++ : fail++; };
const extract = makeExtractor(null);

console.log("=".repeat(70));
console.log("ZERO-MEM v0.6 — answer-level calibration");
console.log("=".repeat(70));

// 1. balanced fences → ok
let r = calibrate("Here is code:\n```js\nconsole.log(1)\n```\nDone.");
check(r.ok && r.signals.fences === 2, "balanced code fences → ok", `(fences=${r.signals.fences})`);

// 2. unbalanced fence → warning
r = calibrate("Look:\n```py\nprint(1)\nOops no close");
check(!r.ok && r.warnings.some((w) => /unbalanced/.test(w)), "unbalanced fence → warning", `(${r.warnings[0]})`);

// 3a. expects json, valid → ok
r = calibrate('```json\n{"a":1}\n```', { expects: "json" });
check(r.ok && r.signals.jsonParses === true, "expects json + valid → ok");

// 3b. expects json, invalid → warning
r = calibrate("{not json}", { expects: "json" });
check(!r.ok && r.signals.jsonParses === false, "expects json + invalid → warning", `(${r.warnings[0]})`);

// 4a. entity coverage: response ignores all query entities → warning
r = calibrate("Sure, here is something unrelated about apples.", { query: "how do I configure the llama_server port", extract });
check(!r.ok && r.signals.entityCoverage === 0, "zero entity coverage → warning", `(${r.warnings[0]})`);

// 4b. entity coverage: response mentions a query entity → ok
r = calibrate("To change the llama_server port, edit server.json.", { query: "how do I configure the llama_server port", extract });
check(r.signals.entityCoverage > 0, "non-zero entity coverage → ok", `(coverage=${r.signals.entityCoverage.toFixed(2)})`);

// 5. verbatim memory reproduction → warning (memory is not authoritative)
const fakeHit: Hit = { unit: { id: "u9", sessionId: "s", cwd: "x", role: "assistant", text: "The api key is stored in the auth.json file under the agent directory.", timestamp: Date.now() - 3_600_000, entities: [], tokens: [], fp: "f", embedding: null }, score: 0.9, reason: "graph" };
r = calibrate("The api key is stored in the auth.json file under the agent directory.", { hits: [fakeHit] });
check(!r.ok && r.signals.verbatimMemory === true, "verbatim memory reproduction → warning", `(${r.warnings[0]})`);

// 6. clean answer with relevant entities and no verbatim copy → ok
r = calibrate("Your key lives in ~/.pi/agent/auth.json.", { query: "whats my api key", hits: [fakeHit], extract });
check(r.ok, "clean relevant answer → ok", `(coverage=${r.signals.entityCoverage.toFixed(2)})`);

console.log("\n" + "=".repeat(70));
console.log(`RESULTS: ${pass} passed, ${fail} failed`);
console.log("=".repeat(70));
if (fail) process.exit(1);
