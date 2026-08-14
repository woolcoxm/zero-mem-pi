/**
 * Paired significance tests between two eval-reader.ts runs (DUMP=<tag> JSONL).
 * The QA sample is seeded, so run A and run B answer the SAME questions in the
 * SAME order — per-question diffs are paired.
 *
 *   node eval-reader-stats.mjs .reader-calibon.jsonl .reader-caliboff.jsonl calibon caliboff
 *
 * Reports: mean F1/EM/BLEU each side, mean per-QA F1 delta, a sign-flip
 * permutation test on the paired F1 diffs (20k perms, seeded ⇒ deterministic),
 * and exact two-sided binomial McNemar for EM and retrieval hit.
 */
import { readFileSync } from "node:fs";

const [pa, pb, la = "A", lb = "B"] = process.argv.slice(2);
if (!pa || !pb) { console.error("usage: node eval-reader-stats.mjs <a.jsonl> <b.jsonl> [labelA] [labelB]"); process.exit(1); }
const load = (p) => readFileSync(p, "utf8").trim().split("\n").map((l) => JSON.parse(l)).sort((x, y) => x.i - y.i);
const A = load(pa), B = load(pb);
if (A.length !== B.length) { console.error(`length mismatch: ${A.length} vs ${B.length} — not the same sample?`); process.exit(1); }
for (let k = 0; k < A.length; k++) if (A[k].q !== B[k].q) { console.error(`question mismatch at i=${A[k].i} — runs are not paired`); process.exit(1); }

const mean = (xs) => xs.reduce((s, x) => s + x, 0) / (xs.length || 1);
const f1a = mean(A.map((r) => r.f1)), f1b = mean(B.map((r) => r.f1));
const ema = mean(A.map((r) => r.em)), emb = mean(B.map((r) => r.em));
const bla = mean(A.map((r) => r.bleu)), blb = mean(B.map((r) => r.bleu));
const diffs = A.map((r, k) => r.f1 - B[k].f1);
const dbar = mean(diffs);

// sign-flip permutation test on paired F1 diffs (seeded mulberry32)
let seed = 1337;
const rnd = () => { seed |= 0; seed = (seed + 0x6D2B79F5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const PERMS = 20000;
let ge = 0;
for (let p = 0; p < PERMS; p++) {
  let m = 0;
  for (const d of diffs) m += rnd() < 0.5 ? d : -d;
  if (Math.abs(m / diffs.length) >= Math.abs(dbar) - 1e-12) ge++;
}
const pF1 = Math.min(1, (ge + 1) / (PERMS + 1));

// exact two-sided binomial McNemar on a binary metric
function mcnemar(key) {
  let b = 0, c = 0; // b: A yes/B no · c: B yes/A no
  for (let k = 0; k < A.length; k++) { if (A[k][key] && !B[k][key]) b++; if (B[k][key] && !A[k][key]) c++; }
  const n = b + c;
  if (!n) return { b, c, p: 1 };
  let tail = 0; // P(X ≤ min(b,c)) under Binom(n, .5)
  let choose = 1;
  const m = Math.min(b, c);
  for (let k = 0; k <= m; k++) { if (k > 0) choose = (choose * (n - k + 1)) / k; tail += choose * Math.pow(0.5, n); }
  return { b, c, p: Math.min(1, 2 * tail) };
}
const emM = mcnemar("em"), hitM = mcnemar("hit");

console.log(`paired reader runs: ${la} vs ${lb} · n=${A.length}`);
console.log(`  F1     ${la} ${f1a.toFixed(3)} · ${lb} ${f1b.toFixed(3)} · mean per-QA delta ${dbar >= 0 ? "+" : ""}${dbar.toFixed(4)} · permutation p=${pF1.toFixed(4)} ${pF1 < 0.05 ? "(significant)" : "(NOT significant)"}`);
console.log(`  EM     ${la} ${ema.toFixed(3)} · ${lb} ${emb.toFixed(3)} · McNemar b=${emM.b} c=${emM.c} p=${emM.p.toFixed(4)} ${emM.p < 0.05 ? "(significant)" : "(NOT significant)"}`);
console.log(`  BLEU-1 ${la} ${bla.toFixed(3)} · ${lb} ${blb.toFixed(3)}`);
console.log(`  hit    McNemar b=${hitM.b} c=${hitM.c} p=${hitM.p.toFixed(4)}`);
