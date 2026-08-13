/**
 * Zero-Mem v0.7 test — MMR diversity pruning.
 * Builds three semantic clusters; queries the first cluster. Without MMR the
 * top-K is several near-duplicate snippets (high pairwise similarity); with MMR
 * the same K is more diverse (lower pairwise similarity) while still relevant.
 */
import { MemoryStore, makeExtractor, retrieve } from "./core.ts";
import { rmSync, mkdirSync } from "node:fs";

const DIM = 64;
function near(base: number[], jitter: number): number[] {
  const v = base.map((x) => x + (Math.random() - 0.5) * jitter);
  let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n);
  return n ? v.map((x) => x / n) : v;
}
function randBase(): number[] {
  const v: number[] = []; for (let i = 0; i < DIM; i++) v.push((Math.random() - 0.5) * 2);
  let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n);
  return n ? v.map((x) => x / n) : v;
}
function dot(a: number[], b: number[]): number { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
function avgPairwise(units: { embedding: number[] }[]): number {
  if (units.length < 2) return 0;
  let sum = 0, n = 0;
  for (let i = 0; i < units.length; i++) for (let j = i + 1; j < units.length; j++) { sum += dot(units[i].embedding, units[j].embedding); n++; }
  return sum / n;
}

let pass = 0, fail = 0;
const check = (ok: boolean, label: string, extra = "") => { console.log(`  ${ok ? "✓ PASS" : "✗ FAIL"} — ${label}${extra ? "  " + extra : ""}`); ok ? pass++ : fail++; };

console.log("=".repeat(70));
console.log("ZERO-MEM v0.7 — MMR diversity pruning");
console.log("=".repeat(70));

const cwd = "C:/proj/x";
try { rmSync("/tmp/zmmr", { recursive: true, force: true }); } catch {}
mkdirSync("/tmp/zmmr", { recursive: true });
const store = new MemoryStore("/tmp/zmmr/store.json", makeExtractor(null));

const aBase = randBase(), bBase = randBase(), cBase = randBase();
// cluster A (4 near aBase), B (4 near bBase), C (4 near cBase)
const clusters = [aBase, bBase, cBase];
for (let ci = 0; ci < 3; ci++) {
  for (let i = 0; i < 4; i++) {
    const u = store.add({ sessionId: "s", sessionName: "t", cwd, role: "assistant", text: `cluster${ci} doc ${i} relevant detail number ${i}`, timestamp: Date.now() - 300_000 - (ci * 4 + i) * 1000 });
    u.embedding = near(clusters[ci], 0.05);
  }
}
// query embedding = aBase (so cluster A is most relevant)
store.embedder = { ready: true, dim: DIM, init: async () => {}, embed: async () => aBase } as any;
store.ensureIndex();

const K = 3;
const noMmr = await retrieve("relevant detail", store, { cwd, topK: K, recentExcludeMs: 0, useHnsw: false, mmr: false });
const withMmr = await retrieve("relevant detail", store, { cwd, topK: K, recentExcludeMs: 0, useHnsw: false, mmr: true, mmrLambda: 0.7 });

const noAvg = avgPairwise(noMmr.map((h) => h.unit));
const mmrAvg = avgPairwise(withMmr.map((h) => h.unit));
console.log(`top-${K} without MMR: avg pairwise sim = ${noAvg.toFixed(3)}`);
console.log(`top-${K} with    MMR: avg pairwise sim = ${mmrAvg.toFixed(3)}`);
check(mmrAvg < noAvg, "MMR reduces redundancy (lower pairwise similarity)", `(Δ=${(noAvg - mmrAvg).toFixed(3)})`);

// MMR should still include at least one cluster-A (relevant) unit
const hasRelevant = withMmr.some((h) => dot(h.unit.embedding!, aBase) > 0.9);
check(hasRelevant, "MMR still surfaces a top-relevant (cluster-A) unit");

// mmr:false path unchanged; mmr:true returns exactly K
check(noMmr.length === K && withMmr.length === K, "both return exactly topK", `(${noMmr.length}/${withMmr.length})`);

console.log("\n" + "=".repeat(70));
console.log(`RESULTS: ${pass} passed, ${fail} failed`);
console.log("=".repeat(70));
if (fail) process.exit(1);
