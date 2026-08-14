/** v0.12 — incremental-HNSW recall drift curve + rebuild restoration. Maps recall
 *  vs % growth so we can pick a tighter rebuild gate than the current +50%. */
import { HNSWIndex } from "./core.ts";
const DIM = 64, N = 5000, TOTAL = 7000, NQ = 150, K = 10, EF = 200, STEP = 500;
let _s = 12345;
const vec = (i: number) => { let s = (i + 7) * 9301 + 49297; const v: number[] = []; for (let d = 0; d < DIM; d++) { s = (s * 1103515245 + 12345) & 0x7fffffff; v.push(s / 0x7fffffff); } const n = Math.hypot(...v) || 1; return v.map((x) => x / n); };
const dot = (a: number[], b: number[]) => { let d = 0; for (let i = 0; i < a.length; i++) d += a[i] * b[i]; return d; };
const recall = (idx: HNSWIndex, unitIdx: number[], queries: number[][]) => {
  let hits = 0, tot = 0;
  for (const q of queries) {
    const pool = unitIdx.map((u) => vec(u));
    const truth = new Set(pool.map((v, i) => ({ i, d: dot(q, v) })).sort((a, b) => b.d - a.d).slice(0, K).map((x) => x.i));
    for (const g of idx.searchUnitIndices(q, K, EF)) { if (truth.has(g)) hits++; tot++; }
  }
  return tot ? hits / tot : 0;
};
const idx = new HNSWIndex({ M: 16, efConstruction: 200 });
const base = Array.from({ length: N }, (_, i) => i);
idx.build(base.map((i) => vec(i))); idx.unitIndex = base;
const queries = Array.from({ length: NQ }, (_, i) => vec(100000 + i));
let unitIdx = [...base];
console.log(`after build (N=${N}, +0%):  recall@${K} ${(recall(idx, unitIdx, queries) * 100).toFixed(1)}%`);
for (let start = N; start < TOTAL; start += STEP) {
  for (let i = start; i < start + STEP; i++) { idx.add(vec(i), i); unitIdx.push(i); }
  const pct = ((unitIdx.length - N) / N * 100);
  console.log(`after +${STEP} (total ${unitIdx.length}, +${pct.toFixed(0)}%): recall@${K} ${(recall(idx, unitIdx, queries) * 100).toFixed(1)}%`);
}
// full rebuild restores?
const t = Date.now();
const idx2 = new HNSWIndex({ M: 16, efConstruction: 200 });
idx2.build(unitIdx.map((u) => vec(u))); idx2.unitIndex = unitIdx;
console.log(`\nfull rebuild over ${unitIdx.length} in ${((Date.now() - t) / 1000).toFixed(1)}s: recall@${K} ${(recall(idx2, unitIdx, queries) * 100).toFixed(1)}% (restored?)`);
