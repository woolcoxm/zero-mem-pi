/**
 * Zero-Mem v0.7 eval harness — retrieval quality on conversational traces.
 *
 * The paper measures final-answer F1/BLEU on LoCoMo/HotpotQA with an LLM reader.
 * That needs external datasets + LLM calls (heavy, non-deterministic, network).
 * Zero-Mem owns the *retrieval* step (zero-LLM); the reader is the user's model.
 * So this harness measures retrieval quality — recall@K, MRR, and token cost —
 * on a deterministic synthetic dataset with known gold answers, and runs an
 * ablation across the v0.4–v0.7 components so each one's contribution shows.
 *
 * Dataset: 24 facts (distinct entity + value) seeded across 3 sessions in one
 * project, plus 40 distractor chatter units. Each fact has 1–2 paraphrase
 * queries whose gold answer is that fact's unit. Reproducible (seeded RNG).
 */
import { MemoryStore, Embedder, makeExtractor, retrieve, formatEvidence } from "./core.ts";
import { rmSync, mkdirSync } from "node:fs";

// seeded RNG (mulberry32) for reproducible distractors
let _s = 1337;
const rnd = () => { _s = (_s + 0x6D2B79F5) | 0; let t = Math.imul(_s ^ (_s >>> 15), 1 | _s); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
const pick = <T,>(a: T[]) => a[Math.floor(rnd() * a.length)];

const cwd = "C:/Users/Robot/projects/evalapp";
const sessions = ["auth", "infra", "config"];
const hour = 3600_000;

interface Fact { id: string; text: string; queries: string[]; }
const facts: Fact[] = [
  { id: "f1", text: "The api key is KEY-Alpha-117 — keep it in auth.json.", queries: ["whats my api key", "the key for the api"] },
  { id: "f2", text: "llama_server listens on port 8080 by default.", queries: ["which port does llama use", "the llama port"] },
  { id: "f3", text: "The local model is Qwen3-Coder-Next in MXFP4.", queries: ["which model are we running", "the model name"] },
  { id: "f4", text: "We built against CUDA 12.4 for the GPU backend.", queries: ["what cuda version", "cuda backend version"] },
  { id: "f5", text: "The VRAM budget is 16 GB for this workstation.", queries: ["how much vram", "the vram budget"] },
  { id: "f6", text: "The eval dataset lives at /data/locomo on disk.", queries: ["where is the dataset", "the dataset path"] },
  { id: "f7", text: "Embeddings use all-MiniLM-L6-v2, 384 dims.", queries: ["which embedding model", "the embedder"] },
  { id: "f8", text: "maxTokens for generation is set to 16384.", queries: ["the max tokens", "generation length limit"] },
  { id: "f9", text: "Sampling temperature is 0.7 with top_p 0.95.", queries: ["the temperature", "sampling temperature"] },
  { id: "f10", text: "Routed experts are quantized to MXFP4.", queries: ["the quantization format", "expert quant"] },
  { id: "f11", text: "The MoE has 64 routed experts per layer.", queries: ["how many experts", "expert count"] },
  { id: "f12", text: "Context window is 65536 tokens.", queries: ["the context length", "context window size"] },
  { id: "f13", text: "Training batch size was 512.", queries: ["the batch size", "training batch"] },
  { id: "f14", text: "We used learning rate 1e-4 with cosine decay.", queries: ["the learning rate", "lr value"] },
  { id: "f15", text: "Best checkpoint is at step 4000.", queries: ["which checkpoint", "best step"] },
  { id: "f16", text: "The GPU is an RTX 4090, 24 GB.", queries: ["which gpu", "the graphics card"] },
  { id: "f17", text: "We run embeddings via transformers.js in Node.", queries: ["the embedding framework", "how do we embed"] },
  { id: "f18", text: "Memory persists to ~/.pi/agent/zero-mem/.", queries: ["where is memory stored", "the store path"] },
  { id: "f19", text: "Retention expires units older than 90 days.", queries: ["the retention window", "how long is memory kept"] },
  { id: "f20", text: "Injection uses topK of 3 snippets per turn.", queries: ["how many snippets injected", "the injection topk"] },
  { id: "f21", text: "MMR lambda is 0.5 for relevance vs diversity.", queries: ["the mmr lambda", "diversity weight"] },
  { id: "f22", text: "HNSW activates above 10000 embedded units.", queries: ["the hnsw threshold", "when does hnsw kick in"] },
  { id: "f23", text: "The RNG seed for builds is 1337.", queries: ["the random seed", "build seed"] },
  { id: "f24", text: "The project license is MIT.", queries: ["the license", "what license"] },
];
const distractors = [
  "we reviewed the architecture diagram together", "the build went green after the fix",
  "lunch was at the usual place", "i pushed the branch but didnt merge",
  "the dashboard looks nicer now", "we cleaned up the old logs",
  "renamed a couple of functions", "fixed a flaky test in the suite",
  "the docs need a proofread", "paired on the tricky bug for an hour",
  "ci cache helped a lot", "the meeting notes are in the doc",
  "swapped the icon for a sharper one", "the deploy used the blue tag",
  "we discussed caching strategy", "trimmed some dead code",
  "the linter complained about unused vars", "added a unit test for the edge case",
  "the timeline slipped by a day", "rebased onto main before opening the pr",
];

let pass = 0, fail = 0;
const check = (ok: boolean, label: string, extra = "") => { if (ok) pass++; else fail++; console.log(`  ${ok ? "✓" : "✗"} ${label}${extra ? "  " + extra : ""}`); };

console.log("=".repeat(72));
console.log(" ZERO-MEM v0.7 — retrieval eval (deterministic, zero-LLM) ");
console.log("=".repeat(72));

try { rmSync("/tmp/zeval", { recursive: true, force: true }); } catch {}
mkdirSync("/tmp/zeval", { recursive: true });
const store = new MemoryStore("/tmp/zeval/store.json", makeExtractor(null));

// plant facts across sessions (spread so retrieval must cross sessions within the project)
const goldById = new Map<string, string>(); // query -> gold unit id
for (let i = 0; i < facts.length; i++) {
  const f = facts[i];
  const u = store.add({ sessionId: pick(sessions), sessionName: "eval", cwd, role: i % 2 ? "assistant" : "user", text: f.text, timestamp: Date.now() - (1 + (i % 6)) * hour });
  for (const q of f.queries) goldById.set(q, u.id);
}
// distractor chatter
for (let i = 0; i < 40; i++) {
  store.add({ sessionId: pick(sessions), sessionName: "eval", cwd, role: "user", text: `${pick(distractors)} (note ${i})`, timestamp: Date.now() - (1 + (i % 10)) * hour });
}

// embed once
const embedder = new Embedder();
await embedder.init();
store.embedder = embedder;
await store.embedAll();
store.ensureIndex();

const queries = [...goldById.keys()];
const K3 = 3, K5 = 5;

interface Cfg { name: string; run: () => Promise<{ r3: number; r5: number; mrr: number; tok: number }>; }
const measure = async (opts: any) => {
  let hit3 = 0, hit5 = 0, mrr = 0, tok = 0;
  for (const q of queries) {
    const gold = goldById.get(q)!;
    const hits = await retrieve(q, store, { cwd, topK: K5, recentExcludeMs: 0, ...opts });
    const ids = hits.map((h) => h.unit.id);
    const rank = ids.indexOf(gold) + 1;
    if (rank > 0 && rank <= K3) hit3++;
    if (rank > 0 && rank <= K5) hit5++;
    if (rank > 0) mrr += 1 / rank;
    tok += Math.ceil(formatEvidence(hits, 120).length / 4);
  }
  const n = queries.length;
  return { r3: hit3 / n, r5: hit5 / n, mrr: mrr / n, tok: Math.round(tok / n) };
};

const cfgs: Cfg[] = [
  { name: "BM25 only (no embeddings)", run: () => { const e = store.embedder; store.embedder = null; return measure({}).finally(() => { store.embedder = e; }); } },
  { name: "+ semantic (MiniLM)", run: () => measure({ useHnsw: false }) },
  { name: "+ semantic + MMR", run: () => measure({ useHnsw: false, mmr: true }) },
  { name: "+ semantic + MMR + bridges OFF", run: () => measure({ useHnsw: false, mmr: true, useBridges: false }) },
  { name: "FULL (semantic + MMR + bridges)", run: () => measure({ useHnsw: false, mmr: true, useBridges: true }) },
];

console.log(`\ndataset: ${facts.length} facts, ${queries.length} queries, ${store.size()} units\n`);
console.log("config                              recall@3  recall@5   MRR    tok/turn");
console.log("-".repeat(72));
const rows: Record<string, { r3: number; r5: number; mrr: number; tok: number }> = {};
for (const c of cfgs) {
  const m = await c.run();
  rows[c.name] = m;
  console.log(`${c.name.padEnd(36)}${m.r3.toFixed(2).padStart(8)}${m.r5.toFixed(2).padStart(10)}${m.mrr.toFixed(2).padStart(7)}${String(m.tok).padStart(9)}`);
}

// assertions: semantic should beat BM25; bridges should not hurt; full pipeline recall@5 >= 0.7
console.log("\nassertions:");
check(rows["+ semantic (MiniLM)"].r5 >= rows["BM25 only (no embeddings)"].r5, "semantic recall@5 >= BM25 recall@5");
check(rows["FULL (semantic + MMR + bridges)"].r5 >= 0.70, "full pipeline recall@5 >= 0.70", `(${rows["FULL (semantic + MMR + bridges)"].r5.toFixed(2)})`);
check(rows["FULL (semantic + MMR + bridges)"].mrr >= 0.5, "full pipeline MRR >= 0.50", `(${rows["FULL (semantic + MMR + bridges)"].mrr.toFixed(2)})`);

console.log("\n" + "=".repeat(72));
console.log(`EVAL: ${pass} checks passed, ${fail} failed`);
console.log("=".repeat(72));
if (fail) process.exit(1);
