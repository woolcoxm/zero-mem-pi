/**
 * Zero-Mem v0.5 test — int8 quantized embedding sidecar + retention.
 * Verifies: round-trip cosine fidelity, .bin size, reload fidelity,
 * retention trimming, end-to-end retrieval, and legacy-store migration.
 */
import { MemoryStore, makeExtractor, retrieve, quantize, dequantize } from "./core.ts";
import { existsSync, rmSync, statSync, writeFileSync, mkdirSync, readFileSync } from "node:fs";

const extract = makeExtractor(null);
const cwd = "C:/proj/app";

function randUnitVector(dim: number): number[] {
  const v = Array.from({ length: dim }, () => (Math.random() - 0.5) * 2);
  let n = 0; for (const x of v) n += x * x; n = Math.sqrt(n);
  return n ? v.map((x) => x / n) : v;
}
function cos(a: number[], b: number[]): number {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na && nb ? d / (Math.sqrt(na) * Math.sqrt(nb)) : 0;
}
let pass = 0, fail = 0;
const check = (ok: boolean, label: string, extra = "") => {
  console.log(`  ${ok ? "✓ PASS" : "✗ FAIL"} — ${label}${extra ? "  " + extra : ""}`);
  ok ? pass++ : fail++;
};

console.log("=".repeat(70));
console.log("ZERO-MEM v0.5 — int8 quantized sidecar + retention");
console.log("=".repeat(70));

// 1. quantize round-trip fidelity
const DIM = 384;
let maxDrift = 0;
for (let t = 0; t < 500; t++) {
  const a = randUnitVector(DIM), b = randUnitVector(DIM);
  const full = cos(a, b);
  const qa = dequantize(quantize(a).scale, quantize(a).bytes);
  const qb = dequantize(quantize(b).scale, quantize(b).bytes);
  maxDrift = Math.max(maxDrift, Math.abs(full - cos(qa, qb)));
}
check(maxDrift < 0.005, "cosine drift across 500 pairs < 0.005", `(max ${maxDrift.toFixed(5)})`);

// 2. persist N units with synthetic embeddings, measure sidecar size
const dir = "/tmp/zeromem-storage-test";
const storePath = `${dir}/store.json`, embPath = `${dir}/store.emb.bin`;
try { rmSync(dir, { recursive: true, force: true }); } catch {}
const store = new MemoryStore(storePath, extract);
const N = 300;
const truth = new Map<string, number[]>();
for (let i = 0; i < N; i++) {
  const emb = randUnitVector(DIM);
  const unit = store.add({ sessionId: "s", sessionName: "t", cwd, role: "assistant", text: `unit ${i} widgets gadgets`, timestamp: Date.now() - i * 1000 });
  unit.embedding = emb;
  truth.set(unit.id, emb);
}
await store.persist();
const binKB = statSync(embPath).size / 1024;
const jsonKB = statSync(storePath).size / 1024;
const inlineEstKB = (N * DIM * 21) / 1024;
const ratio = (inlineEstKB / (binKB + jsonKB)).toFixed(1);
check(binKB + jsonKB < inlineEstKB / 2, "sidecar materially smaller than inline JSON",
  `(bin ${binKB.toFixed(1)} + json ${jsonKB.toFixed(1)} KB vs ~${inlineEstKB.toFixed(0)} KB inline → ${ratio}x)`);

// 3. reload + fidelity (self-cosine + cross-pair ranking)
const store2 = new MemoryStore(storePath, extract);
store2.load();
let worst = 0;
const byId = new Map(store2.units.map((u) => [u.id, u]));
for (const [id, orig] of truth) {
  const r = byId.get(id)?.embedding;
  if (r) worst = Math.max(worst, 1 - cos(orig, r));
}
let flips = 0, checks = 0;
const ids = [...truth.keys()];
for (let t = 0; t < 300; t++) {
  const i = ids[Math.floor(Math.random() * ids.length)];
  const j = ids[Math.floor(Math.random() * ids.length)];
  if (i === j) continue;
  const o = cos(truth.get(i)!, truth.get(j)!);
  const r = cos(byId.get(i)!.embedding!, byId.get(j)!.embedding!);
  checks++;
  if (Math.sign(o) !== Math.sign(r) && Math.abs(o - r) > 0.01) flips++;
}
check(worst < 0.05 && flips === 0, "embeddings survive the int8 round-trip", `(worst self-err ${worst.toFixed(5)}, ${flips}/${checks} sign flips)`);

// 4. retention cap
store2.maxUnits = 50;
store2.enforceRetention();
check(store2.units.length === 50, "retention trims to maxUnits", `(${store2.units.length} remain)`);

// 5. end-to-end retrieval after reload (BM25 path). recentExcludeMs:0 because the
// synthetic units above are all "recent" and would be (correctly) recency-guarded.
const hits = await retrieve("widgets", store2, { cwd, topK: 5, recentExcludeMs: 0 });
check(hits.length > 0, "retrieval functional after reload", `(${hits.length} hits)`);

// 6. legacy inline-embedding store migrates to sidecar on first persist
console.log("\n" + "=".repeat(70));
console.log("legacy migration: inline-embedding store.json → sidecar");
console.log("=".repeat(70));
const ldir = "/tmp/zeromem-legacy-test";
const lpath = `${ldir}/store.json`, lemb = `${ldir}/store.emb.bin`;
try { rmSync(ldir, { recursive: true, force: true }); } catch {}
mkdirSync(ldir, { recursive: true });
const lunits = [];
for (let i = 0; i < 5; i++) lunits.push({
  id: `u${i}`, sessionId: "s", cwd, role: "assistant", text: `legacy ${i}`,
  timestamp: Date.now() - i * 1000, entities: [], tokens: ["legacy", String(i)], fp: `f${i}`,
  embedding: randUnitVector(DIM),
});
writeFileSync(lpath, JSON.stringify({ units: lunits, scopeToProject: true, counter: 5 }));
const lm = new MemoryStore(lpath, extract);
lm.load();
const hadInline = lm.units.every((u) => Array.isArray(u.embedding) && u.embedding!.length === DIM);
await lm.persist();
const reJson = JSON.parse(readFileSync(lpath, "utf8"));
const stripped = reJson.units.every((u: any) => u.embedding === null);
check(hadInline && existsSync(lemb) && stripped, "legacy store migrated cleanly",
  `(inline@load=${hadInline}, sidecar@persist=${existsSync(lemb)}, json stripped=${stripped})`);

console.log("\n" + "=".repeat(70));
console.log(`RESULTS: ${pass} passed, ${fail} failed`);
console.log("=".repeat(70));
if (fail) process.exit(1);
