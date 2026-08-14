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
 * Ranking has mmr off, minScore 0, brute force. NOTE (v0.13): the CORE config
 * runs the PRODUCTION pipeline — it includes co-occurrence bridges, evidence
 * closure (adjacent turns + shared entities CAN place gold in top-K), and
 * answer-type calibration (pass "calib" to disable). A closure-free ablation
 * ("retriever-only") is also reported so the retriever's own contribution is
 * separable from closure's.
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
const CALIB = process.argv[4] === "calib" ? { calibrateEvidence: true } : {};
// CORE production fusion under test (env-tunable): default max; FUSION=off ⇒ v0.8 dense-only; FUSION=weighted [+SEM_W].
const FENV = process.env.FUSION === "off" ? { hybrid: false }
  : process.env.FUSION === "weighted" ? { hybrid: true, fusion: "weighted" as const, semanticWeight: Number(process.env.SEM_W ?? 0.5) }
  : { hybrid: true, fusion: "coverage" as const };
const rankOf = (hits: any[], dia: Map<string, string>, gold: Set<string>) => { for (let i = 0; i < hits.length; i++) if (gold.has(dia.get(hits[i].unit.id) ?? "")) return i; return -1; };

type Acc = { r1: number; r3: number; r5: number; r10: number; mrr: number; n: number };
const newAcc = (): Acc => ({ r1: 0, r3: 0, r5: 0, r10: 0, mrr: 0, n: 0 });
const bump = (a: Acc, rank: number) => { a.n++; if (rank < 0) return; if (rank < 1) a.r1++; if (rank < 3) a.r3++; if (rank < 5) a.r5++; if (rank < 10) a.r10++; a.mrr += 1 / (rank + 1); };
const add = (A: Acc, b: Acc) => { A.r1 += b.r1; A.r3 += b.r3; A.r5 += b.r5; A.mrr += b.mrr; A.n += b.n; };
const fmt = (a: Acc) => a.n ? `r@1 ${(a.r1 / a.n).toFixed(3)}  r@3 ${(a.r3 / a.n).toFixed(3)}  r@5 ${(a.r5 / a.n).toFixed(3)}  r@10 ${(a.r10 / a.n).toFixed(3)}  MRR ${(a.mrr / a.n).toFixed(3)}` : "(no qa)";

const data = await loadDataset();
const convs = (data as any[]);
console.log(`[locomo] ${convs.length} conversations, ${convs.reduce((s, c) => s + c.qa.length, 0)} QA total`);

const embedder = new Embedder(process.argv[2], process.argv[3] as any); await embedder.init();
console.log(`[locomo] embedder ready: ${embedder.ready}`);

const bm = newAcc(), sem = newAcc(), hyb = newAcc(), core = newAcc(), coreNC = newAcc();
// v0.13: per-QA top-5 hit vectors for a PAIRED significance test (McNemar-style
// discordant-pair counts) — CORE vs BM25. The margin has historically been
// ~0.014 on n≈2000, within ~1 binomial SE, so ">= BM25" needs evidence, not
// just a point estimate. (The coverage router was also TUNED on this dataset,
// so treat even a significant margin as an upper bound until held out.)
const bmHit5: boolean[] = [], coreHit5: boolean[] = [];
// v0.14: held-out split — conversations 1–5 are the set the fusion was
// historically tuned on; 6–10 were never used for tuning decisions. Reporting
// both makes the tuned-on-test bias visible.
const bmT = newAcc(), coreT = newAcc(), bmH = newAcc(), coreH = newAcc();
for (let ci = 0; ci < convs.length; ci++) {
  const utts = uttsOf(convs[ci].conversation);
  const qas: any[] = convs[ci].qa;
  const B = buildStore(utts, null);
  const S = buildStore(utts, embedder);
  await S.store.embedAll();
  for (const q of qas) {
    const gold = new Set<string>(q.evidence ?? []);
    const bh = await retrieve(q.question, B.store, R_OPTS);
    const sh = await retrieve(q.question, S.store, { ...R_OPTS, hybrid: false }); // PURE semantic (v0.8) baseline
    const rbh = rankOf(bh, B.dia, gold), rsh = rankOf(sh, S.dia, gold);
    bump(bm, rbh); bump(sem, rsh);
    const isTune = ci < 5; // v0.14: convs 1–5 = historically tuned on; 6–10 = held out
    if (isTune) { bump(bmT, rbh); } else { bump(bmH, rbh); }
    const ch = await retrieve(q.question, S.store, { ...R_OPTS, ...FENV, ...CALIB });
    const rch = rankOf(ch, S.dia, gold);
    bump(core, rch);
    if (isTune) { bump(coreT, rch); } else { bump(coreH, rch); }
    const cn = await retrieve(q.question, S.store, { ...R_OPTS, ...FENV, useClosure: false }); // v0.13 ablation: retriever only
    bump(coreNC, rankOf(cn, S.dia, gold));
    bmHit5.push(rbh >= 0 && rbh < 5);
    coreHit5.push(rch >= 0 && rch < 5);
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
console.log(`  CORE hybrid (${FENV.fusion ?? "off"}${FENV.semanticWeight != null ? ",w=" + FENV.semanticWeight : ""})     ${fmt(core)}`);
console.log(`  CORE, no closure     ${fmt(coreNC)}   ← v0.13 ablation: retriever without closure neighbors`);
console.log(`  HYBRID (RRF, k=${RRF_K})    ${fmt(hyb)}`);
console.log("========================================================================");
const n = bm.n || 1;
console.log(`  recall@5:  BM25 ${p(bm.r5 / n)}  ·  semantic ${p(sem.r5 / n)}  ·  CORE ${p(core.r5 / n)}  ·  CORE-nc ${p(coreNC.r5 / n)}  ·  RRF ${p(hyb.r5 / n)}`);
console.log(`  MRR:       BM25 ${p(bm.mrr / n)}  ·  semantic ${p(sem.mrr / n)}  ·  CORE ${p(core.mrr / n)}  ·  CORE-nc ${p(coreNC.mrr / n)}  ·  RRF ${p(hyb.mrr / n)}`);

// v0.13: paired McNemar-style significance for CORE vs BM25 on top-5 hits.
let bOnly = 0, cOnly = 0; // discordant pairs: CORE hit/BM25 missed, and vice versa
for (let i = 0; i < bmHit5.length; i++) { if (coreHit5[i] && !bmHit5[i]) bOnly++; if (bmHit5[i] && !coreHit5[i]) cOnly++; }
const d = bOnly + cOnly;
// exact binomial two-sided p on the discordant pairs (normal approx with continuity correction)
const z = d > 0 ? (Math.abs(bOnly - cOnly) - 1) / Math.sqrt(d) : 0;
const pval = 2 * (1 - 0.5 * (1 + erf(Math.abs(z) / Math.SQRT2)));
function erf(x: number): number { // Abramowitz-Stegun 7.1.26
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  return s * (1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
}
console.log(`  CORE vs BM25 @5: CORE-only hits ${bOnly}, BM25-only hits ${cOnly}, p≈${pval < 0.0001 ? "<0.0001" : pval.toFixed(4)} ${pval < 0.05 ? "(significant)" : "(NOT significant — treat '≥ BM25' as a tie)"}`);
if (bOnly + cOnly < 20) console.log("  (too few discordant pairs for a meaningful test)");
// v0.14: held-out split — the honest number is the one on conversations the
// fusion was never tuned on.
{
  const pt = (a: Acc) => (a.n ? (a.r5 / a.n).toFixed(3) : "n/a");
  const pm = (a: Acc) => (a.n ? (a.mrr / a.n).toFixed(3) : "n/a");
  console.log(`  split           config   r@5     MRR    n`);
  console.log(`  convs 1-5 (tune) BM25    ${pt(bmT)}  ${pm(bmT)}  ${bmT.n}`);
  console.log(`  convs 1-5 (tune) CORE    ${pt(coreT)}  ${pm(coreT)}  ${coreT.n}`);
  console.log(`  convs 6-10 (held) BM25   ${pt(bmH)}  ${pm(bmH)}  ${bmH.n}`);
  console.log(`  convs 6-10 (held) CORE   ${pt(coreH)}  ${pm(coreH)}  ${coreH.n}`);
}

let pass = 0, fail = 0;
const check = (name: string, cond: boolean, extra = "") => { if (cond) { pass++; console.log(`  \u2713 ${name} ${extra}`); } else { fail++; console.log(`  \u2717 ${name} ${extra}`); } };
check("BM25 recall@5 > 0.30 (retrieval works on real data)", bm.r5 / n > 0.30, `(actual ${p(bm.r5 / n)})`);
check("semantic recall@5 > 0 (dense path functional)", sem.r5 > 0);
// v0.13: the fusion gate no longer hard-fails on a margin that's within noise,
// and the "BM25 >= semantic" belief-check was removed — it encoded an empirical
// property of one embedder (bge already halved the gap) and would flip to
// failing when a better embedder lands, with no bug existing.
check("CORE hybrid recall@5 within noise of BM25 or better", core.r5 / n >= bm.r5 / n - 2 * Math.sqrt(0.25 / n), `(CORE ${p(core.r5 / n)} vs BM25 ${p(bm.r5 / n)}${core.r5 / n >= bm.r5 / n ? "" : ", within 2 SE"})`);
console.log(`\nRESULTS: ${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
