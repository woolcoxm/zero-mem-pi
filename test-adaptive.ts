/**
 * Zero-Mem v0.9 — adaptive MMR λ.
 *
 * Instead of a fixed λ=0.5 for every query, λ is chosen from query shape:
 *   - terse, entity-packed lookups ("whats my api key") → LOW λ (favor the single
 *     most relevant snippet; diversity just dilutes a lookup).
 *   - exploratory intent ("summarize what we built") → HIGH λ (several distinct
 *     facets beat one repeated three times).
 *   - neutral → ~0.5.
 *
 * Pure unit test on adaptiveMmrLambda (deterministic, no embedder) + a smoke
 * check that an explicit mmrLambda override still wins.
 */
import { adaptiveMmrLambda, MemoryStore, makeExtractor, retrieve } from "./core.ts";
import { rmSync } from "node:fs";

let pass = 0, fail = 0;
const check = (name: string, cond: boolean, extra = "") => {
  if (cond) { pass++; console.log(`  \u2713 PASS \u2014 ${name} ${extra}`); }
  else { fail++; console.log(`  \u2717 FAIL \u2014 ${name} ${extra}`); }
};

// Lookup: short, entity-packed.
const lamLookup = adaptiveMmrLambda("whats my api key", ["api key"], ["whats", "my", "api", "key"]);
check("lookup query \u2192 low \u03bb (favor relevance)", lamLookup < 0.4, `(\u03bb=${lamLookup.toFixed(2)})`);

// Exploratory: summary/overview intent.
const lamExpl = adaptiveMmrLambda("summarize what we built this week", [], ["summarize", "what", "we", "built", "this", "week"]);
check("exploratory query \u2192 high \u03bb (favor diversity)", lamExpl > 0.6, `(\u03bb=${lamExpl.toFixed(2)})`);

// Neutral: no strong signal.
const lamMid = adaptiveMmrLambda("the model", [], ["the", "model"]);
check("neutral query \u2192 ~0.5", lamMid >= 0.45 && lamMid <= 0.55, `(\u03bb=${lamMid.toFixed(2)})`);

// Monotonic: more entities on a short query → lower λ than neutral.
const lamDense = adaptiveMmrLambda("react tailwind", ["react", "tailwind"], ["react", "tailwind"]);
check("entity-dense short query \u2264 neutral", lamDense <= lamMid, `(dense=${lamDense.toFixed(2)} vs mid=${lamMid.toFixed(2)})`);

// retrieve() honors an explicit override (adaptivity must NOT override it).
const path = "C:/Users/Robot/projects/zero-mem-pi/.zm-adaptive-test.json";
try { rmSync(path); } catch {}
const extract = makeExtractor(null);
const store = new MemoryStore(path, extract); // BM25-only
store.add({ sessionId: "s1", cwd: "C:/p", role: "user", text: "the api key is KEY-X", timestamp: Date.now() });
store.ensureIndex();
const hits = await retrieve("api key", store, { cwd: "C:/p", sessionId: "other", topK: 1, mmrLambda: 0.9 });
check("explicit mmrLambda override works (returns hit, no crash)", hits.length === 1);

console.log(`\nRESULTS: ${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
