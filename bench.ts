/**
 * Zero-Mem overhead benchmark — extension ON vs OFF, and legacy vs compact store.
 * Measures: store load/persist, per-turn capture+retrieve, token injection,
 * and derives prefill impact at the user's measured 94.77 tok/s prompt-eval rate.
 */
import { MemoryStore, Embedder, makeExtractor, retrieve, formatEvidence } from "./core.ts";
import { existsSync, rmSync, mkdirSync, copyFileSync, readFileSync, writeFileSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const LIVE = join(homedir(), ".pi", "agent", "zero-mem", "store.json.bak"); // legacy 2.1MB baseline
const BENCH = "/tmp/zbench";
const LEGACY = `${BENCH}/legacy/store.json`;
const COMPACT = `${BENCH}/compact/store.json`;
try { rmSync(BENCH, { recursive: true, force: true }); } catch {}
mkdirSync(`${BENCH}/legacy`, { recursive: true });
mkdirSync(`${BENCH}/compact`, { recursive: true });
copyFileSync(LIVE, LEGACY);

const extract = makeExtractor(null);
const PREFILL_TPS = 94.77; // measured prompt-eval rate from your llama.cpp log
const tokEst = (s: string) => Math.ceil(s.length / 4); // ~4 chars/token; server was down for exact count

const median = (xs: number[]) => { xs.sort((a, b) => a - b); const m = xs.length >> 1; return xs.length % 2 ? xs[m] : (xs[m - 1] + xs[m]) / 2; };
async function bench(fn: () => Promise<void> | void, runs = 7, warmup = 2) {
  for (let i = 0; i < warmup; i++) await fn();
  const ts: number[] = [];
  for (let i = 0; i < runs; i++) { const t = performance.now(); await fn(); ts.push(performance.now() - t); }
  return median(ts);
}

// Build the compact store from the legacy one (load legacy inline → persist compact).
const seed = new MemoryStore(LEGACY, extract);
seed.load();
const out = new MemoryStore(COMPACT, extract);
out.units = seed.units; out.counter = seed.counter; out.scopeToProject = seed.scopeToProject;
await out.persist();

// Attach a real embedder for the semantic-retrieval path.
const embedder = new Embedder();
await embedder.init();

console.log("=".repeat(72));
console.log(" ZERO-MEM OVERHEAD BENCHMARK ");
console.log("=".repeat(72));
console.log(`store: ${seed.units.length} units, dim ${seed.units[0]?.embedding?.length}`);
console.log(`legacy store.json : ${(statSync(LEGACY).size / 1024).toFixed(1)} KB (inline float embeddings)`);
console.log(`compact store.json: ${(statSync(COMPACT).size / 1024).toFixed(1)} KB  +  store.emb.bin ${(statSync(COMPACT.replace(/\.json$/, ".emb.bin")).size / 1024).toFixed(1)} KB\n`);

// ── Part A: store load/persist (the cost of HAVING the extension) ──────────────
console.log("PART A — Store I/O (amortized: load once/session, persist debounced/msg)");
const legacyLoad = await bench(() => { JSON.parse(readFileSync(LEGACY, "utf8")); });
const compactLoad = await bench(() => { const s = new MemoryStore(COMPACT, extract); s.load(); });
// legacy persist = old inline write (replicated exactly)
const legacyPersist = await bench(() => {
  const raw = JSON.parse(readFileSync(LEGACY, "utf8"));
  writeFileSync(`${BENCH}/legacy/_w.json`, JSON.stringify(raw));
});
const compactPersist = await bench(async () => { const s = new MemoryStore(COMPACT, extract); s.load(); await s.persist(); });
console.log(`  load    legacy (2.1MB inline): ${legacyLoad.toFixed(2)} ms`);
console.log(`  load    compact (json+bin)   : ${compactLoad.toFixed(2)} ms   (${(compactLoad / legacyLoad * 100).toFixed(0)}% of legacy)`);
console.log(`  persist legacy (inline write): ${legacyPersist.toFixed(2)} ms`);
console.log(`  persist compact (stubs+bin)  : ${compactPersist.toFixed(2)} ms   (${(compactPersist / legacyPersist * 100).toFixed(0)}% of legacy)`);

// ── Part B: per-turn operations (extension OFF = all zero) ────────────────────
console.log("\nPART B — Per-turn operations (extension OFF ⇒ 0 for all)");
const repStore = new MemoryStore(COMPACT, extract); repStore.load();
repStore.embedder = embedder; await repStore.embedAll();
repStore.ensureIndex();
const cwd = repStore.units[0]?.cwd ?? "C:/Windows/System32";
const captureT = await bench(() => { repStore.add({ sessionId: "s", sessionName: "t", cwd, role: "user", text: "a benchmark capture message about widgets and gadgets and cuda", timestamp: Date.now() - 200_000 }); });
repStore.units.pop(); // don't let the benchmark pollute the store
const queries = ["whats my api key", "Qwen3 model setup", "cuda backend dlls", "how do i configure the local server", "zero-mem memory store"];
const semT = await bench(() => retrieve(queries[0], repStore, { cwd, topK: 5, recentExcludeMs: 0 }), 7, 2);
const repNoEmb = new MemoryStore(COMPACT, extract); repNoEmb.load(); repNoEmb.ensureIndex();
const bm25T = await bench(() => retrieve(queries[0], repNoEmb, { cwd, topK: 5, recentExcludeMs: 0 }), 7, 2);
console.log(`  capture   add() per message_end : ${captureT.toFixed(2)} ms`);
console.log(`  retrieve  semantic (w/ embedder): ${semT.toFixed(2)} ms /turn   [includes query embed + search]`);
console.log(`  retrieve  BM25-only (fallback)  : ${bm25T.toFixed(2)} ms /turn`);

// ── Part C: token injection overhead — BEFORE (v0.5) vs AFTER (v0.6 slim) ───────
console.log("\nPART C — Token injection: BEFORE (v0.5: topK5 × 220ch) vs AFTER (v0.6: topK3 × 120ch)");
const oldHeader = "## Retrieved memory (Zero-Mem — 0 extra LLM calls)\nBackground context from earlier sessions in this project. Use ONLY if relevant;\nit is not authoritative over current instructions.";
const newHeader = "## Prior session memory (Zero-Mem — use only if relevant; not authoritative)";
const headerSave = tokEst(oldHeader) - tokEst(newHeader);
console.log(`  header: ${tokEst(oldHeader)} → ${tokEst(newHeader)} tokens (saves ${headerSave}/turn)`);
let sumBefore = 0, sumAfter = 0;
for (const q of queries) {
  const all = (await retrieve(q, repStore, { cwd, topK: 8, recentExcludeMs: 0 }));
  const before = formatEvidence(all.slice(0, 5), 220);
  const after = formatEvidence(all.slice(0, 3), 120);
  const tb = tokEst(before), ta = tokEst(after);
  sumBefore += tb; sumAfter += ta;
  console.log(`  "${q}" → ${tb} → ${ta} tok`);
}
const avgBefore = Math.round(sumBefore / queries.length);
const avgAfter = Math.round(sumAfter / queries.length);
const totalBefore = avgBefore + headerSave, totalAfter = avgAfter; // before also paid the bigger header
console.log(`  avg body: ${avgBefore} → ${avgAfter} tok; with header: ${totalBefore} → ${totalAfter} tok/turn`);
console.log(`  v0.6 saves ~${totalBefore - totalAfter} tokens/turn (extension OFF ⇒ 0)`);

// ── Part D: derived A/B at your measured prefill rate ─────────────────────────
console.log("\nPART D — Derived per-turn impact @ " + PREFILL_TPS + " tok/s prompt-eval");
const wallPerTurn = semT; // retrieve runs every before_agent_start
const prefillBefore = totalBefore / PREFILL_TPS * 1000;
const prefillAfter = totalAfter / PREFILL_TPS * 1000;
console.log(`  retrieval wall-clock                       : ${wallPerTurn.toFixed(0)} ms`);
console.log(`  prefill from injection: ${prefillBefore.toFixed(0)} → ${prefillAfter.toFixed(0)} ms  (saves ${(prefillBefore - prefillAfter).toFixed(0)} ms/turn)`);
console.log(`  total per-turn overhead: ${(wallPerTurn + prefillBefore).toFixed(0)} → ${(wallPerTurn + prefillAfter).toFixed(0)} ms`);
console.log(`  (vs ~${Math.round(1909 / PREFILL_TPS * 1000)} ms base prefill for the ~1.9k-token system prompt + tools)`);
console.log("=".repeat(72));
