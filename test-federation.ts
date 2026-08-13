/**
 * Zero-Mem v0.9 regression test — cross-project federation.
 *
 * When project scoping (scopeToProject) leaves a query with NO relevant hits in
 * the current project, retrieve() reaches across to OTHER projects (penalized)
 * rather than returning nothing. The instant the current project has a real
 * answer, scoping is respected again and nothing leaks across.
 *
 * BM25-only (no embedder) on purpose: federation lives in the pool/score layer,
 * not the semantic scorer, so this runs fast with no model download.
 */
import { MemoryStore, makeExtractor, retrieve } from "./core.ts";
import { rmSync } from "node:fs";

const path = "C:/Users/Robot/projects/zero-mem-pi/.zm-fed-test.json";
try { rmSync(path); } catch {}
const extract = makeExtractor(null);
const store = new MemoryStore(path, extract); // no embedder -> BM25 fallback
const cwdA = "C:/proj/appA";
const cwdB = "C:/proj/appB";
const now = Date.now();
const add = (cwd: string, role: "user" | "assistant", text: string) =>
  store.add({ sessionId: "s-other", cwd, role, text, timestamp: now });

// Project A: only an unrelated fact. Project B: the real answer.
add(cwdA, "user", "we use rust for the backend");
add(cwdB, "assistant", "the api key is KEY-Alpha-117, keep it secret.");
store.ensureIndex();

const q = "whats the api key";
// Query from a NEW session (so v0.8 recent-exclusion never interferes).
const noFed = await retrieve(q, store, { cwd: cwdA, sessionId: "cur", topK: 3, federate: false });
const fed   = await retrieve(q, store, { cwd: cwdA, sessionId: "cur", topK: 3, federate: true });

let pass = 0, fail = 0;
const check = (name: string, cond: boolean, extra = "") => {
  if (cond) { pass++; console.log(`  \u2713 PASS \u2014 ${name} ${extra}`); }
  else { fail++; console.log(`  \u2717 FAIL \u2014 ${name} ${extra}`); }
};

console.log("\n--- thin in-project (only an unrelated fact) ---");
check("federation OFF: no cross-project leak", noFed.every((h) => !h.unit.text.includes("KEY-Alpha-117")), `(returned ${noFed.length})`);
check("federation ON: cross-project hit surfaced", fed.some((h) => h.unit.text.includes("KEY-Alpha-117")), `(returned ${fed.length})`);
check("federated hit is tagged cross-project", fed.some((h) => h.reason.includes("cross-project")));

console.log("\n--- rich in-project (now has the real answer) ---");
add(cwdA, "assistant", "the api key is KEY-Beta-999."); // A now answers the query itself
store.ensureIndex();
const rich = await retrieve(q, store, { cwd: cwdA, sessionId: "cur", topK: 3, federate: true });
check("rich in-project: own answer ranked first", rich.length > 0 && rich[0].unit.text.includes("KEY-Beta-999"), `(top: ${rich[0]?.reason})`);
check("rich in-project: no needless cross-project leak", !rich.some((h) => h.reason.includes("cross-project")));

console.log(`\nRESULTS: ${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
