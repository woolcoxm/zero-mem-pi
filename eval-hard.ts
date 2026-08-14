/**
 * Zero-Mem v0.14 — hard-paraphrase retrieval eval (replaces eval.ts's headline
 * role; eval.ts stays as the small historical ablation).
 *
 * Why: eval.ts measured 24 hand-written facts vs 40 trivially-separable
 * distractors — 0.98 there says little about real conversations. This eval
 * GENERATES 200 facts deterministically, each with:
 *   - a HARD NEGATIVE sibling: a second fact sharing nearly all tokens (same
 *     service/library) but differing in exactly one value (staging vs
 *     production, port A vs port B, owner X vs owner Y). Only the
 *     distinguishing term separates gold from the sibling.
 *   - two PARAPHRASE queries (synonym/template swaps, never verbatim).
 * Gold = the exact generating unit. Deterministic (seeded); zero LLM calls.
 */
import { MemoryStore, Embedder, makeExtractor, retrieve } from "./core.ts";

const SEED = 1337;
let s = SEED;
const rnd = () => { s = (s * 1664525 + 1013904223) | 0; return (s >>> 0) / 4294967296; };
const pick = <T>(a: T[]) => a[Math.floor(rnd() * a.length)];
const int = (lo: number, hi: number) => lo + Math.floor(rnd() * (hi - lo + 1));

const SERVICES = ["payments-api", "auth-gateway", "search-index", "notify-service", "billing-worker", "media-proxy", "queue-router", "cache-layer", "ingest-agent", "render-farm", "ledger-svc", "vault-sidecar"];
const PEOPLE = ["Dana", "Miguel", "Priya", "Tomoko", "Wren", "Ade", "Lena", "Kofi"];
const LIBS = ["pinia", "tokio", "sqlalchemy", "react-query", "celery", "zod", "gorilla/mux", "polars"];

type Fact = { text: string; queries: string[]; id: string };
const facts: Fact[] = [];

for (let i = 0; i < 200; i++) {
  const svc = SERVICES[i % SERVICES.length];
  const n = String(i).padStart(3, "0");
  switch (i % 4) {
    case 0: { // stage-scoped secret — sibling differs only in stage
      const stage = i % 8 < 4 ? "staging" : "production";
      facts.push({
        id: `sec-${n}`,
        text: `The ${svc} ${stage} database password is Pw${int(1000, 9999)}${pick(["x", "q", "z"])}, rotate it monthly.`,
        queries: [
          `what is the ${stage} db password for ${svc}`,
          `${svc} ${stage} database credentials — where do I find them`,
        ],
      });
      break;
    }
    case 1: { // env-scoped port — sibling same service, other env, other port
      const env = i % 8 < 4 ? "the dev cluster" : "production";
      facts.push({
        id: `port-${n}`,
        text: `${svc} listens on port ${int(2000, 9800)} in ${env}; the load balancer health-checks it every 10s.`,
        queries: [
          `which port does ${svc} run on in ${env.replace("the ", "")}`,
          `${svc} ${env.replace("the ", "")} port number`,
        ],
      });
      break;
    }
    case 2: { // phase-scoped owner — sibling same feature, other phase
      const phase = i % 8 < 4 ? "the migration" : "the rollback plan";
      facts.push({
        id: `own-${n}`,
        text: `${svc} ${phase} is owned by ${pick(PEOPLE)}; ask them before touching the config.`,
        queries: [
          `who is responsible for ${svc} ${phase}`,
          `${svc} ${phase} — who owns it`,
        ],
      });
      break;
    }
    default: { // pinned version — sibling same lib, other version/reason
      const lib = pick(LIBS), issue = pick(["memory leak", "race condition", "regex slowdown", "breaking API change"]);
      facts.push({
        id: `ver-${n}`,
        text: `We pinned ${lib} to v${int(0, 3)}.${int(0, 9)}.${int(0, 9)} because of the ${issue}.`,
        queries: [
          `what version did we pin ${lib} to because of the ${issue}`,
          `${lib} is frozen at which version due to the ${issue}`,
        ],
      });
    }
  }
}


// Build the store: facts + their hard siblings are all "distractors" to each
// other — every fact of the same case/type shares vocabulary with the others.
const extract = makeExtractor(null);
const store = new MemoryStore("C:/Users/Robot/projects/zero-mem-pi/.hard-eval.json", extract);
store.embedder = new Embedder();
await store.embedder.init();
const gold = new Map<string, string>(); // query-index → gold unit id
const queries: { q: string; goldId: string }[] = [];
{
  let t = 0;
  for (const f of facts) {
    const u = store.add({ sessionId: `s${Math.floor(t / 20)}`, cwd: "C:/proj", role: "assistant", text: f.text, timestamp: Date.now() - (400 - t) * 60_000 });
    for (const q of f.queries) queries.push({ q, goldId: u.id });
    t++;
  }
}
store.ensureIndex();
console.log(`dataset: ${facts.length} facts (each with same-type hard distractors), ${queries.length} paraphrase queries\n`);

const run = async (label: string, opts: any) => {
  let r1 = 0, r3 = 0, r5 = 0, mrr = 0;
  const misses: string[] = [];
  for (let i = 0; i < queries.length; i++) {
    const hits = await retrieve(queries[i].q, store, { cwd: "C:/proj", topK: 5, ...opts });
    const rank = hits.findIndex((h) => h.unit.id === queries[i].goldId);
    if (rank === 0) r1++; if (rank >= 0 && rank < 3) r3++; if (rank >= 0 && rank < 5) r5++;
    if (rank >= 0) mrr += 1 / (rank + 1);
    else if (misses.length < 3) misses.push(`"${queries[i].q}" → gold: ${store.units.find((u) => u.id === queries[i].goldId)?.text.slice(0, 70)}`);
  }
  const n = queries.length;
  console.log(`${label.padEnd(34)} r@1 ${(r1 / n).toFixed(3)}  r@3 ${(r3 / n).toFixed(3)}  r@5 ${(r5 / n).toFixed(3)}  MRR ${(mrr / n).toFixed(3)}`);
  for (const m of misses) console.log(`    miss: ${m}`);
  return { r5: r5 / n, mrr: mrr / n };
};

console.log("config                              recall@1  recall@3  recall@5   MRR");
console.log("-".repeat(80));
const bm = await run("BM25 only (no embeddings)", { hybrid: false, useHnsw: false });
// no-embedder path: simulate by hiding the embedder readiness
const core = await run("FULL (coverage + PPR graph)", {});
const noG = await run("FULL, graph=direct matches (no PPR)", { useBridges: false });

console.log();
let pass = 0, fail = 0;
const check = (name: string, cond: boolean, extra = "") => { if (cond) { pass++; console.log(`  ✓ ${name} ${extra}`); } else { fail++; console.log(`  ✗ ${name} ${extra}`); } };
check("FULL recall@5 ≥ 0.65 on hard negatives", core.r5 >= 0.65, `(actual ${core.r5.toFixed(3)})`);
check("FULL MRR ≥ 0.35", core.mrr >= 0.35, `(actual ${core.mrr.toFixed(3)})`);;
check("FULL ≥ BM25 baseline", core.r5 >= bm.r5, `(${core.r5.toFixed(3)} vs ${bm.r5.toFixed(3)})`);
check("PPR graph adds signal (or is neutral)", core.r5 >= noG.r5 - 0.01, `(${core.r5.toFixed(3)} vs graph-off ${noG.r5.toFixed(3)})`);
console.log(`\nRESULTS: ${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
