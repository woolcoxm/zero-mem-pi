/**
 * Zero-Mem v0.9 — HNSW incremental insert.
 *
 * Bug being fixed: at scale (>hnswThreshold) the HNSW index is rebuilt only when
 * embedded-count grows +50%. Between rebuilds, units added AFTER the last build
 * are NOT in the index, so retrieve()'s semantic path zeroes them (they're not
 * HNSW candidates) — i.e. freshly-captured facts are invisible to semantic
 * search until a rebuild. This test proves (1) HNSWIndex.add() makes a new
 * vector searchable immediately, and (2) MemoryStore folds newly-embedded units
 * into the live index via embedAll(), with no spurious rebuild.
 *
 * Uses a fake embedder (deterministic per-text vectors) — no model download.
 */
import { MemoryStore, makeExtractor, HNSWIndex } from "./core.ts";
import { rmSync } from "node:fs";

let pass = 0, fail = 0;
const check = (name: string, cond: boolean, extra = "") => {
  if (cond) { pass++; console.log(`  \u2713 PASS \u2014 ${name} ${extra}`); }
  else { fail++; console.log(`  \u2717 FAIL \u2014 ${name} ${extra}`); }
};

const DIM = 32;
const seedRng = (s: number) => () => { s = (s * 1664525 + 1013904223) | 0; return (s >>> 0) / 4294967296; };
const vec = (i: number) => { const r = seedRng(i + 7); const v = Array.from({ length: DIM }, r); const n = Math.hypot(...v) || 1; return v.map((x) => x / n); };

console.log("--- Part 1: HNSWIndex.add() makes a new vector searchable ---");
const base = Array.from({ length: 300 }, (_, i) => vec(i));
const h = new HNSWIndex({ M: 16, efConstruction: 200 });
h.build(base);
h.unitIndex = base.map((_, i) => i);
check("built index size == base", h.size === base.length && h.unitIndex.length === base.length, `(size ${h.size})`);

const newIdx = 9999;
const newVec = vec(newIdx);
check("before add: new vector's unitIdx is NOT yet returned", !h.searchUnitIndices(newVec, 50, 300).includes(newIdx));
h.add(newVec, newIdx);
check("after add: new vector is searchable (its unitIdx returned)", h.searchUnitIndices(newVec, 50, 300).includes(newIdx));
check("after add: index grew by exactly one", h.size === base.length + 1 && h.unitIndex.length === base.length + 1, `(size ${h.size})`);

console.log("\n--- Part 2: MemoryStore folds newly-embedded units into the live index ---");
const path = "C:/Users/Robot/projects/zero-mem-pi/.zm-incr-test.json";
try { rmSync(path); } catch {}
const store = new MemoryStore(path, makeExtractor(null));
store.hnswThreshold = 8; store.hnswEnabled = true;
const DIM2 = 16;
const txtVec = (txt: string) => { const r = seedRng([...txt].reduce((a, c) => a + c.charCodeAt(0), 3) + 1); const v = Array.from({ length: DIM2 }, r); const n = Math.hypot(...v) || 1; return v.map((x) => x / n); };
store.embedder = { ready: true, init: async () => {}, embed: async (t: string) => txtVec(t) } as any;

for (let i = 0; i < 10; i++) store.add({ sessionId: "s", cwd: "C:/p", role: "user", text: `seed number ${i}`, timestamp: Date.now() + i });
await store.embedAll();              // embed the 10 seeds (hnsw null → no incremental yet)
const bp = store.ensureHnsw(); if (bp) await bp;   // build over the 10 seeds
check("store built HNSW over seeds", !!store.hnsw, `(hnsw size ${store.hnsw?.size ?? 0})`);
const builtFor0 = store.hnswBuiltFor;

const newTexts = ["brand new fact alpha", "brand new fact beta", "brand new fact gamma", "brand new fact delta", "brand new fact epsilon"];
for (const t of newTexts) store.add({ sessionId: "s", cwd: "C:/p", role: "user", text: t, timestamp: Date.now() });
await store.embedAll();              // should fold the 5 new units into the LIVE index

let reachable = 0;
for (let i = 0; i < store.units.length; i++) {
  const u = store.units[i];
  if (!newTexts.includes(u.text) || !u.embedding) continue;
  if (store.hnsw!.searchUnitIndices(u.embedding, 20, 300).includes(i)) reachable++;
}
check("newly-embedded units are searchable via the live HNSW", reachable === newTexts.length, `(${reachable}/${newTexts.length} reachable)`);
check("incremental adds did NOT trigger a spurious rebuild", store.hnswBuiltFor === builtFor0, `(hnswBuiltFor ${store.hnswBuiltFor})`);

console.log(`\nRESULTS: ${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
