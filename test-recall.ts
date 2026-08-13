/**
 * Zero-Mem v0.8 regression test — session-scoped recent-exclusion.
 *
 * Bug: recentExcludeMs (default 2 min) dropped freshly-stored facts across ALL
 * sessions, so a fact the user just told the agent (e.g. "my name is mark") was
 * unrecoverable in any new conversation opened within 2 minutes — it never even
 * entered the retrieval candidate pool, so neither auto-injection nor the
 * recall_memory tool could surface it.
 *
 * Fix: time-exclusion now applies only to the CURRENT session (whose recent
 * turns are almost certainly still in the model's context window and are also
 * caught by activeContext fingerprints); units from OTHER sessions are always
 * recallable, no matter how recent.
 *
 * BM25-only (no embedder) on purpose: the bug lives in the pool filter, not in
 * the semantic scorer, so this runs fast with no model download.
 */
import { MemoryStore, makeExtractor, retrieve } from "./core.ts";
import { rmSync } from "node:fs";

const path = "C:/Users/Robot/projects/zero-mem-pi/.zm-recall-test.json";
try { rmSync(path); } catch {}
const extract = makeExtractor(null);
const store = new MemoryStore(path, extract); // no embedder -> BM25 fallback
const cwd = "C:/proj/app";
const now = Date.now();
const add = (sessionId: string, role: "user" | "assistant", text: string, agoMs: number) =>
  store.add({ sessionId, cwd, role, text, timestamp: now - agoMs });

// A fact told 5 seconds ago in session A.
add("session-A", "user", "my name is mark", 5_000);
// Older filler (10 min) so the candidate pool isn't trivially empty.
add("session-A", "assistant", "we were debugging the llama.cpp router earlier", 600_000);

let pass = 0, fail = 0;
const check = (ok: boolean, label: string, extra = "") => {
  console.log(`  ${ok ? "✓ PASS" : "✗ FAIL"} — ${label}${extra ? "  " + extra : ""}`);
  ok ? pass++ : fail++;
};

console.log("=".repeat(70));
console.log("ZERO-MEM v0.8 — session-scoped recent-exclusion");
console.log("=".repeat(70));

// 1. NEW session, <2min-old fact from a PRIOR session -> MUST be recallable.
const cross = await retrieve("whats my name", store, { cwd, sessionId: "session-B" });
const crossHit = cross.find((h) => /my name is mark/.test(h.unit.text));
check(!!crossHit, "new session recalls <2min-old fact from prior session",
  crossHit ? `(score ${crossHit.score.toFixed(2)}, ${crossHit.reason})` : "(no hit)");

// 2. SAME session, <2min-old fact -> still excluded (it's already in-context).
const same = await retrieve("whats my name", store, { cwd, sessionId: "session-A" });
const sameHit = same.find((h) => /my name is mark/.test(h.unit.text));
check(!sameHit, "same session excludes its own <2min-old fact (already in-context)");

// 3. Caller omits sessionId -> must NOT silently time-drop fresh facts (safe
//    default; relies on activeContext fingerprints for in-context dedup instead).
const none = await retrieve("whats my name", store, { cwd });
const noneHit = none.find((h) => /my name is mark/.test(h.unit.text));
check(!!noneHit, "omitted sessionId still recalls fresh fact (no blind drop)");

try { rmSync(path); } catch {}
console.log(`RESULTS: ${pass} passed, ${fail} failed`);
