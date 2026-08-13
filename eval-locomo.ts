/**
 * Zero-Mem v0.9 — retrieval eval on the REAL LoCoMo10 benchmark.
 *
 * The Zero-Mem paper's headline metric is end-to-end F1/BLEU on LoCoMo using an
 * LLM reader. That needs a running LLM endpoint (none was up at eval time), and
 * Zero-Mem owns only the RETRIEVAL step anyway. So this harness measures the
 * retriever on the paper's actual dataset: for each QA, is the gold-evidence
 * utterance (by dia_id, e.g. "D1:3") surfaced in the top-K? recall@1/3/5 + MRR.
 *
 * Three configs, per QA:
 *   - BM25 only            : lexical baseline (Zero-Mem's fallback path).
 *   - + semantic (MiniLM)  : dense embeddings (Zero-Mem's default path).
 *   - HYBRID (RRF)         : reciprocal-rank fusion of BM25 + semantic — does
 *                            combining the two signals beat either alone?
 * Pure ranking (mmr off, minScore 0, brute force) so the comparison is the
 * retriever itself, not the injection post-processing.
 *
 * Dataset: snap-research/locomo · data/locomo10.json (10 long conversations,
 * ~199 QA each). Cached next to this file after first download (~2.8 MB).
 */
import { MemoryStore, Embedder, makeExtractor, retrieve } from "./core.ts";
import { existsSync, readFileSync, writeFileSync, rmSync } from "node:fs";

const CACHE = "C:/Users/Robot/projects/zero-mem-pi/.locomo10.json";
const URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json";
const TOPN = 10;            // retrieve depth for fusion; recall measured @1/3/5
const RRF_K = 60;           // standard reciprocal-rank-fusion constant

async function loadDataset() {
  if (existsSync(CACHE)) return JSON.parse(readFileSync(CACHE, "utf8"));
  console.log("[locomo] downloading dataset (~2.8 MB)…");
  const res = await fetch(URL);
  if (!res.ok) throw new Error(`download failed: ${res.status}`);
  const txt = await res.text();
  writeFileSync(CACHE, txt);
  return JSON.parse(txt);
}

const extract = makeExtractor(null);
const uttsOf = (conv: any) => {
  const sess = Object.keys(conv).filter((k) => /^session_\d+$/.test(k)).sort((a, b) => +a.split("_")[1] - +b.split("_")[1]);
  const out: { dia_id: string; text: string }[] = [];
  for (const sk of sess) for (const t of conv[sk]) if (t && t.text) out.push({ dia_id: t.dia_id, text: t.text });
  return out;
};
function buildStore(utts: { dia_id: string; text: string }[], embedder: Embedder | null) {
  const path = `${CACHE}.run-${Math.random().toString(36).slice(2)}.json`;
  const s = new MemoryStore(path, extract);
  if (embedder) s.embedder = embedder;
  const dia = new Map<string, string>();
  let i = 0;
  for (const u of utts) {
    const unit = s.add({ sessionId: "locomo", cwd: "C:/locomo", role: "user", text: u.text, timestamp: Date.now() + i++ });
    dia.set(unit.id, u.dia_id);
  }
  s.ensureIndex();
  return { store: s, dia };
}

const R_OPTS = { cwd: "C:/locomo", sessionId: "eval-query", scopeToProject: false, topK: TOPN, minScore: 0, mmr: false, useHnsw: false } as const;
const rankOf = (hits: any[], dia: Map<string, string>, gold: Set<string>) => { for (let i = 0; i < hits.length; i++) if (gold.has(dia.get(hits[i].unit.id) ?? "")) return i; return -1; };

type Acc = { r1: number; r3: number; r5: number; mrr: number; n: number };
const newAcc = (): Acc => ({ r1: 0, r3: 0, r5: 0, mrr: 0, n: 0 });
const bump = (a: Acc, rank: number) => { a.n++; if (rank < 0) return; if (rank < 1) a.r1++; if (rank < 3) a.r3++; if (rank < 5) a.r5++; a.mrr += 1 / (rank + 1); };
const add = (A: Acc, b: Acc) => { A.r1 += b.r1; A.r3 += b.r3; A.r5 += b.r5; A.mrr += b.mrr; A.n += b.n; };
const fmt = (a: Acc) => a.n ? `r@1 ${(a.r1 / a.n).toFixed(3)}  r@3 ${(a.r3 / a.n).toFixed(3)}  r@5 ${(a.r5 / a.n).toFixed(3)}  MRR ${(a.mrr / a.n).toFixed(3)}` : "(no qa)";

const data = await loadDataset();
const convs = (data as any[]);
console.log(`[locomo] ${convs.length} conversations, ${convs.reduce((s, c) => s + c.qa.length, 0)} QA total`);

const embedder = new Embedder(); await embedder.init();
console.log(`[locomo] embedder ready: ${embedder.ready}`);

const bm = newAcc(), sem = newAcc(), hyb = newAcc();
for (let ci = 0; ci < convs.length; ci++) {
  const utts = uttsOf(convs[ci].conversation);
  const qas: any[] = convs[ci].qa;
  const B = buildStore(utts, null);
  const S = buildStore(utts, embedder);
  await S.store.embedAll();
  for (const q of qas) {
    const gold = new Set<string>(q.evidence ?? []);
    const bh = await retrieve(q.question, B.store, R_OPTS);
    const sh = await retrieve(q.question, S.store, R_OPTS);
    bump(bm, rankOf(bh, B.dia, gold));
    bump(sem, rankOf(sh, S.dia, gold));
    // RRF fusion over dia_id
    const rrf = new Map<string, number>();
    bh.forEach((h, i) => { const d = B.dia.get(h.unit.id); if (d) rrf.set(d, (rrf.get(d) ?? 0) + 1 / (RRF_K + i)); });
    sh.forEach((h, i) => { const d = S.dia.get(h.unit.id); if (d) rrf.set(d, (rrf.get(d) ?? 0) + 1 / (RRF_K + i)); });
    const fused = [...rrf.entries()].sort((a, b) => b[1] - a[1]).map((x) => x[0]);
    let hr = -1; for (let i = 0; i < Math.min(TOPN, fused.length); i++) if (gold.has(fused[i])) { hr = i; break; }
    bump(hyb, hr);
  }
  try { rmSync((B.store as any).path ?? "", { force: true }); rmSync((S.store as any).path ?? "", { force: true }); } catch {}
  process.stdout.write(`  conv ${ci + 1}/${convs.length} (${convs[ci].sample_id}, ${utts.length} utts, ${qas.length} qa) done\n`);
}

const p = (a: number) => a.toFixed(3);
console.log("\n========================================================================");
console.log(" ZERO-MEM v0.9 — retrieval on LoCoMo10 (gold-evidence dia_id in top-K) ");
console.log("========================================================================");
console.log(`  BM25 only            ${fmt(bm)}`);
console.log(`  + semantic (MiniLM)  ${fmt(sem)}`);
console.log(`  HYBRID (RRF, k=${RRF_K})    ${fmt(hyb)}`);
console.log("========================================================================");
const n = bm.n || 1;
console.log(`  recall@5:  BM25 ${p(bm.r5 / n)}  ·  semantic ${p(sem.r5 / n)}  ·  hybrid ${p(hyb.r5 / n)}`);
console.log(`  MRR:       BM25 ${p(bm.mrr / n)}  ·  semantic ${p(sem.mrr / n)}  ·  hybrid ${p(hyb.mrr / n)}`);

let pass = 0, fail = 0;
const check = (name: string, cond: boolean, extra = "") => { if (cond) { pass++; console.log(`  \u2713 ${name} ${extra}`); } else { fail++; console.log(`  \u2717 ${name} ${extra}`); } };
check("BM25 recall@5 > 0.30 (retrieval works on real data)", bm.r5 / n > 0.30, `(actual ${p(bm.r5 / n)})`);
check("semantic recall@5 > 0 (dense path functional)", sem.r5 > 0);
check("BM25 recall@5 >= semantic recall@5 (lexical dominates LoCoMo factual lookups)", bm.r5 / n >= sem.r5 / n, `(BM25 ${p(bm.r5 / n)} vs semantic ${p(sem.r5 / n)})`);
console.log(`\nRESULTS: ${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
