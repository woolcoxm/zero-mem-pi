/**
 * Zero-Mem v0.10 — END-TO-END eval on real LoCoMo10 (the paper's metric).
 *
 * The LoCoMo paper scores final-answer quality with an LLM READER: given a
 * question + retrieved evidence, the LLM answers, and the answer is scored
 * (token-F1, exact-match, BLEU) against gold. eval-locomo.ts only measured
 * retrieval (is the gold utterance in top-K); this closes that gap — it's the
 * paper's actual metric, not a proxy.
 *
 * Pipeline per QA: retrieve top-K evidence (coverage fusion + bge) → build a
 * grounded prompt → OpenAI-compatible chat endpoint (temp 0, deterministic) →
 * score F1/EM/BLEU vs gold. If NO endpoint is reachable, it degrades to a
 * retrieval hit-rate report (gold dia_id in top-K) so the script is always useful.
 *
 * CLI (env doesn't propagate through npx in this shell, so use args):
 *   npx tsx eval-reader.ts [endpoint] [embedder] [sampleN] [topK]
 *   defaults: http://127.0.0.1:8080/v1  Xenova/bge-small-en-v1.5  100  5
 */
import { MemoryStore, Embedder, makeExtractor, retrieve, formatEvidence } from "./core.ts";
import { readFileSync, rmSync } from "node:fs";

const ENDPOINT = process.argv[2] ?? "http://127.0.0.1:8080/v1";
const EMB_MODEL = process.argv[3] ?? "Xenova/bge-small-en-v1.5";
const SAMPLE_N = Number(process.argv[4] ?? 100);
const TOPK = Number(process.argv[5] ?? 5);
const CACHE = "C:/Users/Robot/projects/zero-mem-pi/.locomo10.json";

// ── SQuAD-style scoring (deterministic) ─────────────────────────────────────
const norm = (s: string) => s.toLowerCase().replace(/\b(a|an|the)\b/g, "").replace(/[^\w\s]/g, " ").replace(/\s+/g, " ").trim();
const toks = (s: string) => norm(s).split(" ").filter(Boolean);
function f1(pred: string, gold: string) {
  const p = toks(pred), g = toks(gold);
  const gs = new Map<string, number>(); for (const t of g) gs.set(t, (gs.get(t) ?? 0) + 1);
  let common = 0; for (const t of p) { const c = gs.get(t) ?? 0; if (c > 0) { common++; gs.set(t, c - 1); } }
  if (!common) return 0;
  return (2 * (common / p.length) * (common / g.length)) / ((common / p.length) + (common / g.length));
}
const em = (pred: string, gold: string) => (norm(pred) === norm(gold) ? 1 : 0);
const bleu1 = (pred: string, gold: string) => { const p = toks(pred), g = toks(gold); if (!p.length) return 0; const gs = new Set(g); return p.filter((t) => gs.has(t)).length / p.length; };

// ── OpenAI-compatible reader (deterministic: temp 0) ────────────────────────
async function endpointUp(ep: string) { try { return (await fetch(ep + "/models")).ok; } catch { return false; } }
async function modelName(ep: string) { try { const j: any = await (await fetch(ep + "/models")).json(); return j.data?.[0]?.id ?? "model"; } catch { return "model"; } }
async function llmAnswer(ep: string, model: string, context: string, question: string) {
  const r = await fetch(ep + "/chat/completions", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages: [
        { role: "system", content: "Answer the user's question using ONLY the provided context. If the answer is not present, respond with exactly: Unanswerable. Be as concise as the gold answer would be." },
        { role: "user", content: `Context:\n${context}\n\nQuestion: ${question}\nAnswer:` },
      ],
      temperature: 0, max_tokens: 64,
    }),
  });
  const j: any = await r.json();
  return (j.choices?.[0]?.message?.content ?? "").trim();
}

// ── LoCoMo data + stores (reuse eval-locomo shape) ──────────────────────────
const extract = makeExtractor(null);
const data: any[] = JSON.parse(readFileSync(CACHE, "utf8"));
const uttsOf = (conv: any) => {
  const sess = Object.keys(conv).filter((k) => /^session_\d+$/.test(k)).sort((a, b) => +a.split("_")[1] - +b.split("_")[1]);
  const out: { dia_id: string; text: string }[] = [];
  for (const sk of sess) for (const t of conv[sk]) if (t && t.text) out.push({ dia_id: t.dia_id, text: t.text });
  return out;
};
function buildStore(utts: { dia_id: string; text: string }[], embedder: Embedder) {
  const path = `${CACHE}.rd-${Math.random().toString(36).slice(2)}.json`;
  const s = new MemoryStore(path, extract); s.embedder = embedder;
  const dia = new Map<string, string>(); let i = 0;
  for (const u of utts) { const unit = s.add({ sessionId: "locomo", cwd: "C:/locomo", role: "user", text: u.text, timestamp: Date.now() + i++ }); dia.set(unit.id, u.dia_id); }
  s.ensureIndex();
  return { store: s, dia };
}

// round-robin sample across the 10 conversations for category variety
const sampled: { ci: number; q: any }[] = [];
const qsByConv = data.map((c) => c.qa as any[]);
let idx = 0;
while (sampled.length < SAMPLE_N && qsByConv.some((q) => q.length)) {
  for (let ci = 0; ci < qsByConv.length && sampled.length < SAMPLE_N; ci++) {
    const q = qsByConv[ci].shift(); if (q) sampled.push({ ci, q });
  }
  if (idx++ > 1000) break;
}

const embedder = new Embedder(EMB_MODEL); await embedder.init();
console.log(`[reader] embedder: ${embedder.model} (ready=${embedder.ready})`);
const up = await endpointUp(ENDPOINT);
const model = up ? await modelName(ENDPOINT) : "";
console.log(`[reader] endpoint ${ENDPOINT} ${up ? `UP (model: ${model})` : "DOWN — running retrieval-only fallback"}`);
console.log(`[reader] ${sampled.length} QA sampled (round-robin), topK=${TOPK}\n`);

// build + embed stores only for touched conversations
const storeCache = new Map<number, { store: MemoryStore; dia: Map<string, string> }>();
const getStore = async (ci: number) => {
  if (!storeCache.has(ci)) { const s = buildStore(uttsOf(data[ci].conversation), embedder); await s.store.embedAll(); storeCache.set(ci, s); }
  return storeCache.get(ci)!;
};

type Acc = { f1: number; em: number; bleu: number; hit: number; n: number };
const acc = { f1: 0, em: 0, bleu: 0, hit: 0, n: 0 } as Acc;
const byCat = new Map<number, Acc>();
const cat = (c: number) => { if (!byCat.has(c)) byCat.set(c, { f1: 0, em: 0, bleu: 0, hit: 0, n: 0 }); return byCat.get(c)!; };

const R = { cwd: "C:/locomo", sessionId: "reader", scopeToProject: false, topK: TOPK, minScore: 0, mmr: false, useHnsw: false, hybrid: true, fusion: "coverage" as const };
for (let s = 0; s < sampled.length; s++) {
  const { ci, q } = sampled[s];
  const { store, dia } = await getStore(ci);
  const hits = await retrieve(q.question, store, R);
  const gold = new Set<string>(q.evidence ?? []);
  const hit = hits.some((h) => gold.has(dia.get(h.unit.id) ?? "")) ? 1 : 0;
  const a = acc, ca = cat(q.category); a.n++; ca.n++; a.hit += hit; ca.hit += hit;
  if (up) {
    const context = formatEvidence(hits, 200);
    let pred = ""; try { pred = await llmAnswer(ENDPOINT, model, context, q.question); } catch (e: any) { pred = `__err:${(e?.message ?? "").slice(0, 40)}`; }
    const f = f1(pred, q.answer ?? ""), e = em(pred, q.answer ?? ""), b = bleu1(pred, q.answer ?? "");
    a.f1 += f; a.em += e; a.bleu += b; ca.f1 += f; ca.em += e; ca.bleu += b;
  }
  if ((s + 1) % 10 === 0) process.stdout.write(`  ${s + 1}/${sampled.length}\n`);
}
for (const s of storeCache.values()) try { rmSync((s.store as any).path, { force: true }); } catch {}

const p = (x: number) => (x).toFixed(3);
console.log("========================================================================");
console.log(up ? ` END-TO-END on LoCoMo10 (LLM reader @ ${ENDPOINT}, temp=0) ` : ` RETRIEVAL fallback on LoCoMo10 (no endpoint) `);
console.log("========================================================================");
const n = acc.n || 1;
if (up) console.log(`  answer F1 ${p(acc.f1 / n)} · EM ${p(acc.em / n)} · BLEU-1 ${p(acc.bleu / n)} · (retrieval hit@${TOPK} ${p(acc.hit / n)})`);
else console.log(`  retrieval hit@${TOPK} ${p(acc.hit / n)} (no LLM → no answer scoring)`);
console.log("  --- by category ---");
const CN = { 1: "single-session", 2: "multi-session", 3: "temporal", 4: "open-domain", 5: "adversarial" };
for (const c of [...byCat.keys()].sort()) { const a = byCat.get(c)!; const m = a.n || 1; console.log(`    cat${c} ${CN[c as 1] ?? "?"}`.padEnd(28) + `n=${a.n}  ` + (up ? `F1 ${p(a.f1 / m)} EM ${p(a.em / m)}` : `hit ${p(a.hit / m)}`)); }
console.log("========================================================================");
if (!up) console.log("  (start an OpenAI-compatible endpoint and re-run for F1/EM/BLEU)");
