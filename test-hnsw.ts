/**
 * Zero-Mem v0.6 test — HNSW index correctness + retrieve() integration.
 * 1. HNSW recall@10 vs exact brute force on dim-384 vectors (≥ 0.85 at ef=200).
 * 2. retrieve() returns ≥85% the same hits with HNSW on vs off (forced via low threshold).
 * 3. Below hnswThreshold, the store builds no index (exact brute force preserved).
 *
 * NOTE: at N≤5000 brute-force cosine is faster AND exact, so the default
 * hnswThreshold (10000) keeps HNSW dormant until it actually pays off.
 */
import { HNSWIndex, MemoryStore, makeExtractor, retrieve } from "./core.ts";
import { rmSync, mkdirSync } from "node:fs";

const DIM = 384;
function randUnit(): number[] {
  const v: number[] = [];
  for (let i = 0; i < DIM; i++) v.push((Math.random() - 0.5) * 2);
  let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n);
  return n ? v.map((x) => x / n) : v;
}
function dot(a: number[], b: number[]): number { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function bruteTopK(q: number[], vecs: number[][], k: number): number[] {
  return vecs.map((v, i) => ({ i, c: dot(q, v) })).sort((a, b) => b.c - a.c).slice(0, k).map((x) => x.i);
}
let pass = 0, fail = 0;
const check = (ok: boolean, label: string, extra = "") => { console.log(`  ${ok ? "✓ PASS" : "✗ FAIL"} — ${label}${extra ? "  " + extra : ""}`); ok ? pass++ : fail++; };

console.log("=".repeat(70));
console.log("ZERO-MEM v0.6 — HNSW recall + integration");
console.log("=".repeat(70));

// 1. recall@10 vs brute force
const N = 3000, K = 10;
const vecs: number[][] = [];
for (let i = 0; i < N; i++) vecs.push(randUnit());
const idx = new HNSWIndex({ M: 16, efConstruction: 100 });
const tb = performance.now(); idx.build(vecs); idx.unitIndex = vecs.map((_, i) => i);
console.log(`built HNSW over ${N} dim-${DIM} vectors in ${(performance.now() - tb).toFixed(0)} ms`);
let hits = 0, tot = 0;
for (let t = 0; t < 60; t++) {
  const q = randUnit(); const bf = new Set(bruteTopK(q, vecs, K));
  const ap = idx.searchUnitIndices(q, K, 200);
  for (const id of ap) { tot++; if (bf.has(id)) hits++; }
}
const recall = hits / tot;
check(recall >= 0.85, `recall@${K} ≥ 0.85 at ef=200`, `(${(recall * 100).toFixed(1)}%)`);

// 2. retrieve() overlap: HNSW on vs off (force index with a tiny threshold)
const cwd = "C:/proj/x";
try { rmSync("/tmp/zhnsw", { recursive: true, force: true }); } catch {}
mkdirSync("/tmp/zhnsw", { recursive: true });
const p = new MemoryStore("/tmp/zhnsw/store.json", makeExtractor(null));
const parity: number[][] = [];
for (let i = 0; i < 60; i++) {
  const v = randUnit();
  parity.push(v);
  const u = p.add({ sessionId: "s", sessionName: "t", cwd, role: "assistant", text: `doc ${i} alpha beta gamma widgets number ${i}`, timestamp: Date.now() - 300_000 - i * 1000 });
  u.embedding = v;
}
const queryVec = parity[3];
p.embedder = { ready: true, dim: DIM, init: async () => {}, embed: async () => queryVec } as any;
p.ensureIndex();
const off = (await retrieve("doc", p, { cwd, topK: 8, recentExcludeMs: 0, useHnsw: false })).map((h) => h.unit.id);
p.hnsw = null; p.hnswThreshold = 2; // force build on next retrieve
const on = (await retrieve("doc", p, { cwd, topK: 8, recentExcludeMs: 0, useHnsw: true })).map((h) => h.unit.id);
const overlap = on.filter((id) => off.includes(id)).length / Math.max(1, off.length);
check(overlap >= 0.85, "retrieve() overlap: HNSW on vs off ≥ 0.85", `(${(overlap * 100).toFixed(0)}% of ${off.length} hits matched)`);

// 3. below threshold → no index (exact path)
const small = new MemoryStore("/tmp/zhnsw/small.json", makeExtractor(null));
small.units = p.units.slice(); small.dirty = true; small.ensureIndex();
small.hnswThreshold = 10000;
small.buildHnswIfNeeded();
check(small.hnsw === null, "below threshold: no HNSW built (exact brute force)", `(${small.units.length} units < ${small.hnswThreshold})`);

console.log("\n" + "=".repeat(70));
console.log(`RESULTS: ${pass} passed, ${fail} failed`);
console.log("=".repeat(70));
if (fail) process.exit(1);
