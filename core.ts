/**
 * Zero-Mem core — pure, pi-independent logic (testable in plain Node).
 * The pi wiring lives in index.ts and imports from here.
 *
 * Faithful reimplementation of Xiao et al., "Zero-Mem: Zero-Token Memory
 * Operations for LLM Agents", arXiv:2607.29377.
 */

import { join } from "node:path";
import { mkdirSync, readFileSync, writeFileSync, existsSync, unlinkSync } from "node:fs";

// ── Types ───────────────────────────────────────────────────────────────────
export type Role = "user" | "assistant" | "tool";

export interface TraceUnit {
  id: string;
  sessionId: string;
  sessionName?: string;
  cwd: string;
  role: Role;
  text: string;
  timestamp: number;
  entities: string[];
  tokens: string[];
  embedding?: number[] | null; // v0.2: dense semantic vector (meanings), null until embedded
  fp: string;                  // v0.3: content fingerprint (de-dup / context-aware exclusion)
}

export interface RetrieveOpts {
  cwd: string;
  sessionId?: string;        // v0.8: current session — recent-exclusion is scoped to it
  scopeToProject?: boolean;
  topK?: number;
  recentExcludeMs?: number;
  rho?: number;
  activeContext?: Set<string>; // v0.3: fingerprints of messages already in the model's window (excluded)
  minScore?: number;           // v0.3: relevance floor (default 0.15) — drop weak/tangential hits
  calibrateEvidence?: boolean; // v0.11: evidence calibration (paper Eq 15) — re-rank admissible evidence by answer-type compatibility (temporal→dates, quantity→numbers). Default ON (helps top-1/MRR, neutral elsewhere).
  useBridges?: boolean;        // v0.4: enable co-occurrence relational bridges (default true)
  useHnsw?: boolean;           // v0.6: HNSW ANN for semantic search at scale (default true; auto-gated by store.hnswThreshold)
  hnswEf?: number;             // v0.6: HNSW search ef (default 200 ~ recall 0.90 at dim 384; raise for more recall)
  mmr?: boolean;               // v0.7: maximal-marginal-relevance diversity selection (default true)
  mmrLambda?: number;          // v0.7: relevance vs diversity tradeoff, 0=relevance-only 1=diversity-only (default 0.5; v0.9: auto when omitted)
  federate?: boolean;          // v0.9: when project scoping yields nothing relevant, reach across to OTHER projects (penalized). Default true; only fires on an empty in-project result.
  federatePenalty?: number;    // v0.9: score multiplier for cross-project hits (default 0.7 — ranks them below a real in-project answer of equal strength).
  hybrid?: boolean;            // v0.9: fuse lexical (BM25) + dense (cosine) instead of dense-only when an embedder is loaded (default true).
  fusion?: "weighted" | "max" | "coverage"; // v0.9: hybrid strategy. "coverage" (default) blends by query lexical coverage; "max" best-of-both; "weighted" by semanticWeight.
  semanticWeight?: number;     // v0.9: dense share for "weighted" fusion, 0=BM25-only .. 1=semantic-only (default 0.5).
}

export interface Hit {
  unit: TraceUnit;
  score: number;
  reason: string;
}

// ── Text helpers ────────────────────────────────────────────────────────────
const STOPWORDS = new Set(
  ("a an the and or but if then else of to in on at by for with from into over " +
    "is are was were be been being this that these those it its as not no yes " +
    "do does did done have has had can could should would may might must will " +
    "i you he she we they me him her us them my your his our their what when " +
    "where why how which who whom there here about up down out so than very " +
    "just also only more most some any all each both few other such own same " +
    "get make use using used run runs running file code like need want one two")
    .split(" "),
);

export function tokenize(text: string): string[] {
  const out: string[] = [];
  const re = /[a-z0-9][a-z0-9_]{1,}/g;
  let m: RegExpExecArray | null;
  const lower = text.toLowerCase();
  while ((m = re.exec(lower)) !== null) {
    const t = m[0];
    if (t.length < 2 || STOPWORDS.has(t)) continue;
    out.push(t);
  }
  return out;
}

export function makeExtractor(nlp: any) {
  return function extractEntities(text: string): string[] {
    const found = new Set<string>();
    const add = (s: string) => {
      const t = s.trim().toLowerCase();
      if (t.length < 2 || t.length > 40 || STOPWORDS.has(t)) return;
      if (/^\d+$/.test(t)) return;
      found.add(t);
    };
    let m: RegExpExecArray | null;
    const pathRe = /([\w./-]+\/[\w./-]+)|([\w-]+\.(ts|js|tsx|jsx|py|rs|go|java|c|cpp|cc|h|hpp|md|json|yaml|yml|toml|sh|css|html|sql))/gi;
    while ((m = pathRe.exec(text)) !== null) add(m[0]);
    const tickRe = /`([^`\n]{2,40})`/g;
    while ((m = tickRe.exec(text)) !== null) add(m[1]);
    const quoteRe = /"([A-Za-z][A-Za-z0-9 _.\-/]{2,40})"/g;
    while ((m = quoteRe.exec(text)) !== null) add(m[1]);
    const camelRe = /\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b/g;
    while ((m = camelRe.exec(text)) !== null) add(m[0]);
    const snakeRe = /\b[a-z]+(?:_[a-z]+){1,}\b/g;
    while ((m = snakeRe.exec(text)) !== null) add(m[0]);
    const acroRe = /\b[A-Z][A-Z0-9]{2,}\b/g;
    while ((m = acroRe.exec(text)) !== null) add(m[0]);
    if (nlp) {
      try {
        const doc = nlp(text);
        for (const arr of [doc.people?.(), doc.organizations?.(), doc.places?.()]) {
          if (!arr) continue;
          for (const v of arr.out("array") as string[]) add(v);
        }
      } catch { /* regex entities still work */ }
    }
    return [...found].slice(0, 16);
  };
}

export function flattenContent(content: unknown): string {
  if (!content) return "";
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  const parts: string[] = [];
  for (const b of content as any[]) {
    if (!b) continue;
    if (typeof b === "string") { parts.push(b); continue; }
    if (b.type === "text" && typeof b.text === "string") parts.push(b.text);
    else if (b.type === "toolCall") parts.push(`[tool:${b.name}] ${safeJson(b.arguments)}`);
    else if (b.type === "toolResult") parts.push(`[result] ${flattenContent(b.content)}`);
    else if (typeof b.text === "string") parts.push(b.text);
  }
  return parts.join("\n");
}

function safeJson(x: unknown): string {
  try { return typeof x === "string" ? x : JSON.stringify(x).slice(0, 200); }
  catch { return ""; }
}

export function relTime(ts: number): string {
  const s = Math.max(1, Math.floor((Date.now() - ts) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  if (s < 604800) return `${Math.floor(s / 86400)}d ago`;
  return new Date(ts).toISOString().slice(0, 10);
}

/** v0.3: content fingerprint so we can tell what's already in the model's
 *  context window and avoid re-injecting it (also used for de-duplication). */
export function fingerprint(text: string): string {
  const norm = text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().slice(0, 500);
  let h = 5381;
  for (let i = 0; i < norm.length; i++) h = ((h * 33) ^ norm.charCodeAt(i)) >>> 0;
  return h.toString(36);
}

/** v0.4: stable key for an unordered entity pair (used for co-occurrence weights). */
export function pairKey(a: string, b: string): string {
  return a < b ? a + "\x00" + b : b + "\x00" + a;
}

// ── Entity–context graph ────────────────────────────────────────────────────
export class EntityGraph {
  entityUnits = new Map<string, Set<string>>();
  unitEntities = new Map<string, string[]>();
  cooc = new Map<string, number>(); // v0.4: entity-pair co-occurrence weights (relational bridges)

  rebuild(units: TraceUnit[]) {
    this.entityUnits.clear();
    this.unitEntities.clear();
    this.cooc.clear();
    for (const u of units) {
      this.unitEntities.set(u.id, u.entities);
      for (const e of u.entities) {
        let s = this.entityUnits.get(e);
        if (!s) { s = new Set(); this.entityUnits.set(e, s); }
        s.add(u.id);
      }
      // v0.4: record pairwise co-occurrence of this unit's entities (relational edges)
      for (let i = 0; i < u.entities.length; i++)
        for (let j = i + 1; j < u.entities.length; j++) {
          const k = pairKey(u.entities[i], u.entities[j]);
          this.cooc.set(k, (this.cooc.get(k) ?? 0) + 1);
        }
    }
  }
  queryEntities(qEntities: string[]): Map<string, number> {
    const scores = new Map<string, number>();
    for (const e of qEntities) {
      const s = this.entityUnits.get(e);
      if (!s) continue;
      for (const uid of s) scores.set(uid, (scores.get(uid) ?? 0) + 1);
    }
    return scores;
  }
  neighbors(unitId: string): Set<string> {
    const out = new Set<string>();
    const ents = this.unitEntities.get(unitId) ?? [];
    for (const e of ents) {
      const s = this.entityUnits.get(e);
      if (!s) continue;
      for (const uid of s) if (uid !== unitId) out.add(uid);
    }
    return out;
  }
}

// ── BM25 ────────────────────────────────────────────────────────────────────
export class BM25 {
  docs: string[][] = [];
  df = new Map<string, number>();
  avgdl = 0;
  N = 0;
  private k1 = 1.5;
  private b = 0.75;
  build(units: TraceUnit[]) {
    this.docs = units.map((u) => u.tokens);
    this.N = this.docs.length;
    this.df.clear();
    let total = 0;
    for (const d of this.docs) {
      total += d.length;
      const seen = new Set(d);
      for (const t of seen) this.df.set(t, (this.df.get(t) ?? 0) + 1);
    }
    this.avgdl = this.N ? total / this.N : 0;
  }
  score(queryTokens: string[], docIndex: number): number {
    const d = this.docs[docIndex];
    if (!d || !d.length) return 0;
    const tf = new Map<string, number>();
    for (const t of d) tf.set(t, (tf.get(t) ?? 0) + 1);
    let score = 0;
    for (const t of queryTokens) {
      const f = tf.get(t);
      if (!f) continue;
      const n = this.df.get(t) ?? 0;
      const idf = Math.log(1 + (this.N - n + 0.5) / (n + 0.5));
      const denom = f + this.k1 * (1 - this.b + this.b * (d.length / (this.avgdl || 1)));
      score += (idf * (f * (this.k1 + 1))) / denom;
    }
    return score;
  }
}

// ── Dense embedder (v0.2: meanings, not words) ─────────────────────────────
// Turns text into a vector capturing semantics. Zero-LLM: it's an encoder,
// exactly what the paper means by "encoder computation accounted separately."
// Uses transformers.js (all-MiniLM-L6-v2, ~23MB, runs in Node via onnxruntime).
// Falls back to BM25 automatically if unavailable.
export class Embedder {
  private pipe: any = null;
  ready = false;
  dim = 384;
  private loading: Promise<void> | null = null;
  model: string = "Xenova/bge-small-en-v1.5"; // v0.10 default (proven). v0.11 note: the paper uses BGE-M3, but it underperformed bge-small with mean pooling on LoCoMo (r@5 0.20 vs 0.42) and is too slow for full eval on CPU (paper used a GPU). Configurable below for opt-in.
  pooling: "mean" | "cls" = "mean"; // BGE-M3 prefers "cls" (mean crams near-matches); bge-small uses "mean"
  constructor(model?: string, pooling?: "mean" | "cls") { if (model !== undefined) this.model = model; if (pooling) this.pooling = pooling; }

  async init(): Promise<void> {
    if (this.loading) return this.loading;
    this.loading = (async () => {
      try {
        const mod: any = await import("@xenova/transformers");
        const env = mod.env;
        env.allowLocalModels = false;        // fetch from HF hub
        env.backends?.onnx?.wasm?.setThreads?.(1);
        this.pipe = await mod.pipeline("feature-extraction", this.model, { quantized: true });
        this.ready = true;
        console.log(`[zero-mem] embeddings ready (${this.model})`);
      } catch (e: any) {
        console.warn("[zero-mem] embeddings unavailable — using BM25 fallback:", e?.message ?? e);
        this.ready = false;
      }
    })();
    return this.loading;
  }

  async embed(text: string): Promise<number[] | null> {
    if (!this.ready || !this.pipe) return null;
    const out = await this.pipe(text, { pooling: this.pooling, normalize: true });
    return Array.from(out.data as Float32Array);
  }
}

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ── Embedding compression (disk only; in-memory stays float for exact cosine) ─
// v0.5: Symmetric int8 quantization. all-MiniLM embeddings are L2-normalized
// (|v| ≈ 1), so storing a per-vector absmax scale and bytes in [-127,127]
// preserves cosine ranking almost losslessly (measured drift ~0.0004).
// Cuts embedding disk footprint ~21x vs full-precision JSON text arrays.
const EMB_MAGIC = "ZMEM1";
export function quantize(vec: number[]): { scale: number; bytes: Int8Array } {
  let mx = 0;
  for (const v of vec) { const a = Math.abs(v); if (a > mx) mx = a; }
  const scale = mx || 1;
  const bytes = new Int8Array(vec.length);
  for (let i = 0; i < vec.length; i++) bytes[i] = Math.round((vec[i] / scale) * 127);
  return { scale, bytes };
}
export function dequantize(scale: number, bytes: Int8Array | number[]): number[] {
  const out = new Array<number>(bytes.length);
  for (let i = 0; i < bytes.length; i++) out[i] = (bytes[i] / 127) * scale;
  return out;
}

// ── Binary heap (for HNSW search) ───────────────────────────────────────────
class Heap<T> {
  private a: T[] = [];
  private less: (a: T, b: T) => boolean;
  constructor(less: (a: T, b: T) => boolean) { this.less = less; }
  get size() { return this.a.length; }
  peek(): T | undefined { return this.a[0]; }
  push(x: T) { this.a.push(x); this.up(this.a.length - 1); }
  pop(): T | undefined {
    const n = this.a.length; if (!n) return undefined;
    const top = this.a[0]; const last = this.a.pop()!;
    if (n > 1) { this.a[0] = last; this.down(0); }
    return top;
  }
  private up(i: number) { const a = this.a, less = this.less; while (i > 0) { const p = (i - 1) >> 1; if (less(a[i], a[p])) { const t = a[i]; a[i] = a[p]; a[p] = t; i = p; } else break; } }
  private down(i: number) { const a = this.a, n = a.length, less = this.less; for (;;) { let l = i * 2 + 1, r = l + 1, m = i; if (l < n && less(a[l], a[m])) m = l; if (r < n && less(a[r], a[m])) m = r; if (m === i) break; const t = a[i]; a[i] = a[m]; a[m] = t; i = m; } }
}

// ── HNSW (v0.6): approximate nearest-neighbour over dense embeddings ─────────
// Pure-JS (no native dep → no Windows build risk). Gated by a scale threshold in
// MemoryStore so small stores use exact brute force; HNSW activates only when the
// linear scan starts to matter (default >= 3000 units). Distance = 1 − cosine
// (embeddings are L2-normalized, so cosine = dot product).
export class HNSWIndex {
  M: number; Mmax: number; Mmax0: number; efConstruction: number; mL: number;
  unitIndex: number[] = [];            // hnsw node id -> unit array index
  private data: number[][] = [];
  private levelOf: number[] = [];
  private layers: Array<Map<number, number[]>> = [];
  private entry = -1; private entryLevel = -1;
  private seed: number;
  constructor(opts: { M?: number; efConstruction?: number; seed?: number } = {}) {
    this.M = opts.M ?? 16; this.Mmax = this.M; this.Mmax0 = this.M * 2;
    this.efConstruction = opts.efConstruction ?? 200; this.mL = 1 / Math.log(this.M);
    this.seed = opts.seed ?? 1337;
  }
  get size() { return this.data.length; }
  private rnd() { // mulberry32 PRNG (deterministic builds)
    this.seed |= 0; this.seed = (this.seed + 0x6D2B79F5) | 0;
    let t = Math.imul(this.seed ^ (this.seed >>> 15), 1 | this.seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  private dist(a: number[], b: number[]) { let dot = 0; for (let i = 0; i < a.length; i++) dot += a[i] * b[i]; return 1 - dot; }
  build(vectors: number[][]) { for (const v of vectors) this.insert(v); }
  // v0.7: chunked build that yields to the event loop every `yieldEvery` inserts,
  // so a large background build interleaves with pi's normal operation instead of
  // blocking the turn for seconds.
  async buildAsync(vectors: number[][], yieldEvery = 100) {
    for (let i = 0; i < vectors.length; i++) {
      this.insert(vectors[i]);
      if (yieldEvery > 0 && (i + 1) % yieldEvery === 0) await new Promise((r) => setImmediate(r));
    }
  }
  // v0.9: insert a SINGLE new vector into the live index and map its node to
  // `unitIdx` (the store's unit-array index). Keeps freshly-added units
  // searchable at scale instead of waiting for the next growth-gated rebuild
  // (which would otherwise leave them score-0 / invisible to semantic search
  // until embedded-count grows +50%). Incremental inserts gradually dilute graph
  // quality, but MemoryStore.ensureHnsw still triggers a periodic full rebuild
  // (which subsumes these nodes and resets quality), so this is purely additive.
  add(q: number[], unitIdx: number) { this.insert(q); this.unitIndex.push(unitIdx); }
  private neighbors(lc: number, node: number): number[] { return this.layers[lc].get(node) ?? []; }
  private insert(q: number[]) {
    const id = this.data.length; this.data.push(q);
    const l = Math.floor(-Math.log(this.rnd() || 1e-12) * this.mL);
    this.levelOf[id] = l;
    while (this.layers.length <= l) this.layers.push(new Map());
    for (let lc = 0; lc <= l; lc++) if (!this.layers[lc].has(id)) this.layers[lc].set(id, []);
    if (this.entry === -1) { this.entry = id; this.entryLevel = l; return; }
    let ep = [this.entry];
    for (let lc = this.entryLevel; lc > l; lc--) ep = this.searchLayer(q, ep, 1, lc).map((x) => x.id);
    for (let lc = Math.min(l, this.entryLevel); lc >= 0; lc--) {
      const C = this.searchLayer(q, ep, this.efConstruction, lc);
      const neigh = this.selectNeighbors(q, C, this.M);
      this.layers[lc].set(id, neigh.map((c) => c.id));
      const Mmax = lc === 0 ? this.Mmax0 : this.Mmax;
      for (const c of neigh) {
        const arr = this.layers[lc].get(c.id)!; arr.push(id);
        if (arr.length > Mmax) {
          const ci = this.data[c.id];
          const kept = arr.map((n) => ({ id: n, d: this.dist(ci, this.data[n]) })).sort((a, b) => a.d - b.d).slice(0, Mmax).map((s) => s.id);
          this.layers[lc].set(c.id, kept);
        }
      }
      ep = C.map((x) => x.id);
    }
    if (l > this.entryLevel) { this.entry = id; this.entryLevel = l; }
  }
  private searchLayer(q: number[], eps: number[], ef: number, lc: number): { id: number; d: number }[] {
    const visited = new Set<number>(eps);
    const C = new Heap<{ id: number; d: number }>((a, b) => a.d < b.d);
    const W = new Heap<{ id: number; d: number }>((a, b) => a.d > b.d); // max-heap: top = furthest
    for (const e of eps) { const d = this.dist(q, this.data[e]); C.push({ id: e, d }); W.push({ id: e, d }); }
    while (C.size) {
      const c = C.pop()!; const f = W.peek()!;
      if (c.d > f.d) break;
      for (const e of this.neighbors(lc, c.id)) {
        if (visited.has(e)) continue; visited.add(e);
        const d = this.dist(q, this.data[e]); const worst = W.peek();
        if (!worst || d < worst.d || W.size < ef) {
          C.push({ id: e, d }); W.push({ id: e, d });
          if (W.size > ef) W.pop();
        }
      }
    }
    const out: { id: number; d: number }[] = [];
    while (W.size) out.push(W.pop()!);
    out.sort((a, b) => a.d - b.d); // nearest first
    return out;
  }
  // selectNeighborsHeuristic (Malkov-Yashunin Alg.4): keep diverse neighbors,
  // backfill to M for connectivity. Improves recall vs plain nearest-M.
  private selectNeighbors(q: number[], C: { id: number; d: number }[], M: number): { id: number; d: number }[] {
    const R: { id: number; d: number }[] = [];
    for (const e of C) {
      if (R.length >= M) break;
      let good = true;
      for (const r of R) { if (this.dist(this.data[e.id], this.data[r.id]) < e.d) { good = false; break; } }
      if (good) R.push(e);
    }
    for (const e of C) { if (R.length >= M) break; if (!R.includes(e)) R.push(e); }
    return R;
  }
  searchUnitIndices(q: number[], k: number, ef?: number): number[] {
    if (this.entry === -1) return [];
    let ep = [this.entry];
    for (let lc = this.entryLevel; lc > 0; lc--) ep = this.searchLayer(q, ep, 1, lc).map((x) => x.id);
    const W = this.searchLayer(q, ep, Math.max(ef ?? 200, k), 0);
    return W.slice(0, k).map((w) => this.unitIndex[w.id] ?? -1).filter((i) => i >= 0);
  }
}

// ── Memory store ────────────────────────────────────────────────────────────
export class MemoryStore {
  embedder: Embedder | null = null; // v0.2: optional dense embedder
  units: TraceUnit[] = [];
  private path: string;
  private embPath: string;          // v0.5: int8 sidecar for embeddings (keeps store.json text-only)
  private dirty = true;
  private saveTimer: ReturnType<typeof setTimeout> | null = null;
  private counter = 0;
  scopeToProject = true;
  maxUnits = 2000;                  // v0.5: retention safety net (ZERO_MEM_MAX_UNITS)
  maxAgeMs = 90 * 24 * 3600 * 1000; // v0.5: ~90d (ZERO_MEM_MAX_AGE_DAYS)
  graph = new EntityGraph();
  bm25 = new BM25();
  hnsw: HNSWIndex | null = null;     // v0.6: built lazily above hnswThreshold
  hnswThreshold = 10000;            // v0.6: below this, exact brute force is faster AND exact (ZERO_MEM_HNSW_THRESHOLD)
  hnswEnabled = true;               // v0.6: ZERO_MEM_HNSW=0 disables
  private hnswBuiltFor = -1;        // v0.6: embedded-count the index was built for (growth-gated rebuild)
  private hnswBuilding = false;     // v0.7: background build in progress
  private hnswPromise: Promise<void> | null = null; // v0.7: awaited by tests
  extract: (text: string) => string[];

  constructor(path: string, extract: (t: string) => string[]) {
    this.path = path;
    this.embPath = path.replace(/\.json$/i, ".emb.bin");
    this.extract = extract;
  }
  load() {
    let raw: any = null;
    try {
      if (existsSync(this.path)) {
        raw = JSON.parse(readFileSync(this.path, "utf8"));
        this.units = Array.isArray(raw?.units) ? raw.units : [];
        for (const u of this.units) if (!u.tokens || !u.tokens.length) u.tokens = tokenize(u.text); // v0.7: tokens not persisted; recompute from text
        this.scopeToProject = raw?.scopeToProject ?? true;
        this.counter = raw?.counter ?? this.units.length;
      }
      // v0.5: fill embeddings from the int8 sidecar. If the sidecar is absent
      // (legacy store, pre-migration) units keep their inline arrays in memory;
      // the next persist() migrates them out into the sidecar.
      this.loadEmbeddings();
      // v0.10: re-embed if the model changed OR the store predates the stamp
      // (absent stamp ⇒ legacy vectors from an unknown model). `raw` is hoisted
      // out of the `if` above so it's in scope here (this was the v0.10 load bug).
      if (this.embedder && raw?.embedder !== this.embedder.model) {
        const stale = this.units.filter((u) => u.embedding && u.embedding.length).length;
        if (stale) console.log(`[zero-mem] embedder ${raw?.embedder ? `'${raw.embedder}'→` : ""}'${this.embedder.model}'; re-embedding ${stale} units on next use`);
        for (const u of this.units) u.embedding = undefined;
      }
      this.enforceRetention();
    } catch (e) { console.error("[zero-mem] failed to load store:", e); }
    this.dirty = true;
  }
  /** Read the int8 sidecar and dequantize into unit.embedding. */
  private loadEmbeddings() {
    let buf: Buffer | null = null;
    try { if (existsSync(this.embPath)) buf = readFileSync(this.embPath) as Buffer; } catch { /* none */ }
    if (!buf || buf.length < 14) return; // need magic(5)+ver(1)+count(4)+dim(4)
    let off = 0;
    if (buf.subarray(off, off + 5).toString("latin1") !== EMB_MAGIC) { console.warn("[zero-mem] emb sidecar: bad magic, skipping"); return; }
    off += 5;
    off += 1; // version (currently 1)
    const count = buf.readUInt32LE(off); off += 4;
    const dim = buf.readUInt32LE(off); off += 4;
    const byId = new Map<string, TraceUnit>();
    for (const u of this.units) byId.set(u.id, u);
    for (let n = 0; n < count && off + 2 <= buf.length; n++) {
      const idLen = buf.readUInt16LE(off); off += 2;
      const id = buf.subarray(off, off + idLen).toString("utf8"); off += idLen;
      if (off + 4 + dim > buf.length) break;
      const scale = buf.readFloatLE(off); off += 4;
      const q = new Int8Array(buf.subarray(off, off + dim)); off += dim;
      const unit = byId.get(id);
      if (unit) unit.embedding = dequantize(scale, q);
    }
  }
  /** Quantize every unit's embedding and write the int8 sidecar. */
  private writeEmbeddings(units: TraceUnit[]) {
    const embUnits = units.filter((u) => Array.isArray(u.embedding) && u.embedding.length);
    if (!embUnits.length) {
      try { if (existsSync(this.embPath)) unlinkSync(this.embPath); } catch { /* ignore */ }
      return;
    }
    const dim = embUnits[0].embedding!.length;
    const same = embUnits.filter((u) => u.embedding!.length === dim);
    let total = 5 + 1 + 4 + 4;
    for (const u of same) total += 2 + Buffer.byteLength(u.id, "utf8") + 4 + dim;
    const buf = Buffer.allocUnsafe(total);
    let off = 0;
    buf.write(EMB_MAGIC, off, "latin1"); off += 5;
    buf.writeUInt8(1, off); off += 1;
    buf.writeUInt32LE(same.length, off); off += 4;
    buf.writeUInt32LE(dim, off); off += 4;
    for (const u of same) {
      const idLen = Buffer.byteLength(u.id, "utf8");
      buf.writeUInt16LE(idLen, off); off += 2;
      buf.write(u.id, off, "utf8"); off += idLen;
      const { scale, bytes } = quantize(u.embedding!);
      buf.writeFloatLE(scale, off); off += 4;
      buf.set(bytes, off); off += dim; // Int8 bit pattern → Uint8 (Buffer)
    }
    writeFileSync(this.embPath, buf);
  }
  /** v0.6/v0.7: build the HNSW index over embedded units when the store crosses
   *  hnswThreshold; below it, leave null so retrieve() uses exact brute force
   *  (which is faster AND exact below ~10k units). Rebuild is growth-gated so a
   *  per-message add() never triggers a rebuild — the index refreshes only once
   *  embedded-count has grown ≥50% since the last build. The build runs in the
   *  BACKGROUND (chunked, event-loop-yielding) so a large rebuild never stalls a
   *  turn; retrieve() falls back to exact brute force until the build finishes.
   *  Returns the build promise (or null) so tests can await completion. */
  ensureHnsw(): Promise<void> | null {
    if (!this.hnswEnabled) { this.hnsw = null; return null; }
    if (this.hnswBuilding) return this.hnswPromise;
    const idx: number[] = [];
    for (let i = 0; i < this.units.length; i++) if (this.units[i].embedding && this.units[i].embedding!.length) idx.push(i);
    if (idx.length < this.hnswThreshold) { this.hnsw = null; return null; }
    if (this.hnsw && this.hnswBuiltFor > 0 && idx.length < this.hnswBuiltFor * 1.5) return null; // growth-gated: skip until +50%
    const vectors = idx.map((i) => this.units[i].embedding!);
    this.hnswBuilding = true;
    this.hnswPromise = this.buildHnswInBackground(idx, vectors);
    return this.hnswPromise;
  }
  private async buildHnswInBackground(idx: number[], vectors: number[][]) {
    try {
      const h = new HNSWIndex({ M: 16, efConstruction: 200 });
      await h.buildAsync(vectors); // chunked + yields; does not block the turn
      h.unitIndex = idx;
      this.hnsw = h;
      this.hnswBuiltFor = idx.length;
    } catch (e) { console.error("[zero-mem] HNSW build failed:", e); }
    finally { this.hnswBuilding = false; this.hnswPromise = null; }
  }
  /** v0.5: bound store growth — drop units older than maxAgeMs, then trim to maxUnits. */
  enforceRetention() {
    let changed = false;
    const cutoff = Date.now() - this.maxAgeMs;
    if (this.maxAgeMs > 0) {
      const before = this.units.length;
      this.units = this.units.filter((u) => u.timestamp >= cutoff);
      if (this.units.length !== before) changed = true;
    }
    if (this.maxUnits > 0 && this.units.length > this.maxUnits) {
      this.units.sort((a, b) => a.timestamp - b.timestamp);
      this.units = this.units.slice(this.units.length - this.maxUnits);
      changed = true;
    }
    if (changed) this.dirty = true;
  }
  ensureIndex() {
    if (!this.dirty) return;
    this.graph.rebuild(this.units);
    this.bm25.build(this.units);
    this.dirty = false;
  }
  /** v0.2: embed any units still missing a semantic vector (idempotent).
   *  v0.9: if the HNSW index is already live, fold each newly-embedded unit in
   *  incrementally so it's searchable at scale right away (no waiting for the
   *  next +50% rebuild). Skipped while a background build is in progress — that
   *  build reads all current embeddings fresh and will include the unit anyway. */
  async embedAll() {
    if (!this.embedder || !this.embedder.ready) return;
    for (let i = 0; i < this.units.length; i++) {
      const u = this.units[i];
      if (u.embedding && u.embedding.length) continue;
      u.embedding = await this.embedder.embed(u.text);
      if (this.hnsw && this.hnswEnabled && !this.hnswBuilding) {
        try { this.hnsw.add(u.embedding, i); } catch { /* best-effort; next rebuild recovers */ }
      }
    }
  }
  add(partial: Omit<TraceUnit, "id" | "entities" | "tokens" | "fp">): TraceUnit {
    const text = partial.text.slice(0, 4000);
    const id = `u${Date.now().toString(36)}_${(this.counter++).toString(36)}`;
    const unit: TraceUnit = {
      ...partial, id, text,
      fp: fingerprint(text),
      entities: this.extract(text),
      tokens: tokenize(text),
    };
    this.units.push(unit);
    this.enforceRetention();
    this.dirty = true;
    return unit;
  }
  size() { return this.units.length; }
  clear() { this.units = []; this.dirty = true; }
  persistDebounced(delay = 1500) {
    if (this.saveTimer) clearTimeout(this.saveTimer);
    this.saveTimer = setTimeout(() => { this.persist(); this.saveTimer = null; }, delay);
  }
  async persist() {
    try {
      mkdirSync(join(this.path, ".."), { recursive: true });
      // v0.5: store.json stays text-only (small, human-readable); embeddings are
      // quantized to int8 in the .bin sidecar (~21x smaller than inline JSON).
      // v0.7: tokens are NOT persisted (recomputed from text on load) to shrink the JSON further.
      const stubs = this.units.map((u) => { const { tokens, embedding, ...rest } = u; return rest; });
      writeFileSync(this.path, JSON.stringify({ units: stubs, scopeToProject: this.scopeToProject, counter: this.counter, embedder: this.embedder?.model }));
      this.writeEmbeddings(this.units);
    } catch (e) { console.error("[zero-mem] failed to persist:", e); }
  }
}

// ── Retrieval pipeline ──────────────────────────────────────────────────────
// v0.7: pairwise similarity between two units for MMR — cosine of embeddings
// (preferred) with a token-Jaccard fallback when embeddings are absent.
function unitSim(a: TraceUnit, b: TraceUnit): number {
  if (a.embedding && b.embedding && a.embedding.length === b.embedding.length) return cosine(a.embedding, b.embedding);
  const sa = new Set(a.tokens), sb = new Set(b.tokens);
  let inter = 0; for (const t of sa) if (sb.has(t)) inter++;
  const uni = sa.size + sb.size - inter;
  return uni ? inter / uni : 0;
}
// v0.7: Maximal Marginal Relevance — pick k items that are both relevant and
// mutually dissimilar. Iteratively select argmax( rel(c) − λ·max sim(c, selected) ).
function mmrSelect<T extends { unit: TraceUnit; score: number }>(cands: T[], k: number, lambda: number): T[] {
  if (cands.length <= k) return cands;
  const selected: T[] = [];
  const pool = [...cands];
  while (selected.length < k && pool.length) {
    let bestIdx = 0; let bestVal = -Infinity;
    for (let i = 0; i < pool.length; i++) {
      const c = pool[i];
      let div = 0;
      for (const s of selected) { const d = unitSim(c.unit, s.unit); if (d > div) div = d; }
      const val = c.score - lambda * div;
      if (val > bestVal) { bestVal = val; bestIdx = i; }
    }
    selected.push(pool[bestIdx]);
    pool.splice(bestIdx, 1);
  }
  return selected;
}

// v0.9: pick an MMR λ from query shape instead of a fixed 0.5. Lookups (terse,
// entity-packed: "whats my api key") want LOW λ — the single most relevant
// snippet matters and diversity just dilutes it. Exploratory intent ("summarize
// what we built") wants HIGH λ — several distinct facets beat one repeated.
// Only consulted when the caller does NOT pass an explicit mmrLambda.
export function adaptiveMmrLambda(query: string, ents: string[], toks: string[]): number {
  const q = query.toLowerCase();
  if (/\b(summari[sz]e|overview|recap|round-?up|status|catch up|list|enumerate|everything|all (the|our)|what (do|did) we|tl;?dr)\b/.test(q)) return 0.7;
  const n = toks.length;
  const density = n ? ents.length / n : 0;
  if (n > 0 && n <= 6 && density >= 0.15) return 0.25; // terse + entity-packed → lookup
  if (n >= 12) return 0.6;                              // long → likely multi-faceted
  return Math.min(0.55, Math.max(0.4, 0.5 - density));  // neutral ~0.5, leans relevance with entities
}

// v0.11: evidence answer-type compatibility (paper Eq 15 "Rank by ... answer-type
// compatibility"). Detects the expected answer form from the query and scores how well
// a candidate's text matches it (temporal → dates; quantity → numbers). Used as a
// small re-rank boost so type-compatible evidence wins near-ties.
export function evidenceCompat(query: string, text: string): number {
  const q = query.toLowerCase(), t = text.toLowerCase();
  if (/\b(when|what (time|date|day)|how long ago|how old)\b/.test(q))
    return /\b(\d{1,2}\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?|\d{4}|yesterday|today|tomorrow|last (week|month|year|night))\b/.test(t) ? 1 : 0;
  if (/\b(how many|how much|number of|count|total)\b/.test(q))
    return /\b\d+\b/.test(t) ? 1 : 0;
  return 0; // no detectable expected type → neutral
}

export async function retrieve(query: string, store: MemoryStore, opts: RetrieveOpts): Promise<Hit[]> {
  store.ensureIndex();

  // v0.2: dense semantic matching (meanings, not words). Falls back to BM25
  // automatically if no embedder is configured or it fails to load.
  let qEmb: number[] | null = null;
  let useEmb = false;
  if (store.embedder) {
    await store.embedder.init();
    if (store.embedder.ready) {
      await store.embedAll();
      qEmb = await store.embedder.embed(query);
      useEmb = !!qEmb;
    }
  }

  const scopeToProject = opts.scopeToProject ?? true;
  const topK = opts.topK ?? 5;
  const recentExcludeMs = opts.recentExcludeMs ?? 120_000;
  const rho = opts.rho ?? 0.6;

  const qTokens = tokenize(query);
  const qEnts = store.extract(query);
  if (!qTokens.length && !qEnts.length) return [];
  // v0.9: lexical coverage — fraction of query terms present in the corpus. High ⇒ BM25
  // is trustworthy (factual lookup, terms actually appear); low ⇒ synonym/paraphrase or
  // OOV terms, where dense embeddings are needed. Drives the "coverage" fusion weight.
  const cov = qTokens.length ? qTokens.filter((t) => store.bm25.df.has(t)).length / qTokens.length : 0;
  const cutoff = Date.now() - recentExcludeMs;

  // v0.9: split into in-project vs cross-project pools. Cross-project is scored
  // (and penalized) ONLY when the in-project pool yields nothing above minScore
  // — i.e. when project scoping is "too narrow". Otherwise scoping is respected
  // exactly as before, so a project that has its own answer never leaks across.
  const inIdx: number[] = [];
  const outIdx: number[] = [];
  for (let i = 0; i < store.units.length; i++) {
    const u = store.units[i];
    // v0.8: recent-exclusion is now SESSION-SCOPED. A unit from the *current*
    // session younger than recentExcludeMs is almost certainly still in the
    // model's context window (activeContext fingerprints catch exact dupes;
    // this is a backstop for near-paraphrases). Units from OTHER sessions must
    // never be time-excluded — a fact learned seconds ago in a prior session
    // has to be recallable immediately. (Previously this dropped freshly-stored
    // facts for 2 minutes across ALL sessions, so e.g. a name told to the agent
    // was unrecoverable in any new conversation opened within 2 minutes.)
    if (opts.sessionId && u.sessionId === opts.sessionId && u.timestamp >= cutoff) continue;
    if (opts.activeContext?.has(u.fp)) continue; // v0.3: skip what's already in the model's context window
    if (scopeToProject && u.cwd !== opts.cwd) outIdx.push(i); else inIdx.push(i);
  }
  if (!inIdx.length && !outIdx.length) return [];

  const gRaw = new Map<string, number>();
  if (qEnts.length) for (const [uid, cnt] of store.graph.queryEntities(qEnts)) gRaw.set(uid, cnt);

  // v0.4: relational bridges — units whose entities CO-OCCUR with query entities
  // (the paper's co-occurrence-weighted entity–context graph) earn graph score even
  // without a direct mention, so the thread is followed through strong connections.
  if (opts.useBridges !== false && qEnts.length) {
    for (const u of store.units) {
      if (!u.entities.length) continue;
      let bridge = 0;
      for (const eq of qEnts) {
        for (const eu of u.entities) {
          if (eq === eu) continue;
          bridge += store.graph.cooc.get(pairKey(eq, eu)) ?? 0;
        }
      }
      if (bridge > 0) gRaw.set(u.id, (gRaw.get(u.id) ?? 0) + bridge);
    }
  }

  // v0.6/v0.7: at scale, restrict semantic search to HNSW candidates (exact cosine
  // re-rank after); below hnswThreshold, store.hnsw is null -> brute force over the
  // whole pool. Fetch a candidate pool (not just topK) so MMR has room to diversify.
  const useMmr = opts.mmr !== false;
  const poolSize = useMmr ? Math.max(topK * 4, topK + 10) : topK;
  let semCand: Set<number> | null = null;
  if (useEmb && (opts.useHnsw ?? true)) {
    store.ensureHnsw(); // non-blocking; starts/continues a background build if needed
    if (store.hnsw) semCand = new Set(store.hnsw.searchUnitIndices(qEmb!, poolSize, opts.hnswEf ?? 200));
  }

  // v0.9: score BOTH signals per pool — lexical (BM25) always, plus dense (cosine)
  // when an embedder is loaded — so retrieve can HYBRID-fuse them. Each signal is
  // normalized by its OWN per-pool max so in-project and cross-project hits are each
  // comparable to 1.0 before the federation penalty, and lexical/dense share a scale.
  const scorePool = (idx: number[]): { bm: Map<number, number>; bmMax: number; bmMin: number; sem: Map<number, number>; semMax: number; semMin: number } => {
    const bm = new Map<number, number>(), sem = new Map<number, number>();
    let bmMax = 0, bmMin = Infinity, semMax = 0, semMin = Infinity;
    for (const i of idx) {
      const u = store.units[i];
      const b = store.bm25.score(qTokens, i); bm.set(i, b); if (b > bmMax) bmMax = b; if (b < bmMin) bmMin = b;
      let s = 0;
      if (useEmb && qEmb && u.embedding && u.embedding.length) {
        s = (semCand && !semCand.has(i)) ? 0 : (cosine(qEmb, u.embedding) + 1) / 2; // cosine [-1,1] → [0,1]
      }
      sem.set(i, s); if (s > semMax) semMax = s; if (s < semMin) semMin = s;
    }
    return { bm, bmMax, bmMin, sem, semMax, semMin };
  };
  // v0.9: hybrid fusion. v0.8 used dense-ALONE when an embedder was loaded (discarding
  // BM25), which LoCoMo10 showed UNDERPERFORMS BM25 on real factual lookups (r@5 0.27
  // vs 0.53) — and no naive fusion (max/weighted/RRF) beats BM25 there either, because
  // per-signal normalization is degenerate and MiniLM is weak out-of-domain. The fix is
  // a COVERAGE router: blend by how many query terms actually appear in the corpus, so
  // BM25 carries factual lookups (high coverage) and dense rescues synonym/paraphrase
  // queries whose terms are OOV (low coverage). hybrid:false restores v0.8 dense-only.
  const hybrid = (opts.hybrid ?? true) && useEmb;
  const fusion: "weighted" | "max" | "coverage" = opts.fusion ?? "coverage";
  const semW = Math.min(1, Math.max(0, opts.semanticWeight ?? 0.5));
  const lexW = 1 - semW;
  type Pool = { bmMax: number; bmMin: number; bm: Map<number, number>; semMax: number; semMin: number; sem: Map<number, number> };
  // v0.11: min-max normalization per signal (paper Eq 12), not max-norm. BM25's min is
  // ~0 so it's ~unchanged, but dense cosines cluster in [0.5,1]; min-max stretches that
  // to [0,1], restoring discrimination that max-norm compressed away.
  const mmN = (v: number, lo: number, hi: number) => hi > lo ? (v - lo) / (hi - lo) : (v > 0 ? 1 : 0);
  const hOf = (pool: Pool, i: number): number => {
    const bN = mmN(pool.bm.get(i) ?? 0, pool.bmMin, pool.bmMax);
    if (!hybrid) return useEmb ? mmN(pool.sem.get(i) ?? 0, pool.semMin, pool.semMax) : bN; // v0.8
    const sN = mmN(pool.sem.get(i) ?? 0, pool.semMin, pool.semMax);
    if (fusion === "max") return Math.max(bN, sN);
    if (fusion === "coverage") return cov * bN + (1 - cov) * sN; // trust lexical when query terms match the corpus
    return lexW * bN + semW * sN; // weighted
  };
  const minScore = opts.minScore ?? 0.15; // (hoisted from below; v0.9 federation needs it before the cross-project decision)
  const inPool = scorePool(inIdx);

  const entityDriven = qEnts.length > 0;
  const temporal = /\b(earlier|before|last|previous|yesterday|ago|used to|recently|back when)\b/i.test(query);
  let wGraph = rho;
  if (entityDriven) wGraph = Math.min(0.9, rho + 0.15);
  else if (temporal) wGraph = Math.max(0.3, rho - 0.2);
  const wHier = 1 - wGraph;

  let gMax = 0, gMin = Infinity;
  for (const v of gRaw.values()) { if (v > gMax) gMax = v; if (v < gMin) gMin = v; }

  type Cand = { idx: number; unit: TraceUnit; score: number; reason: string };
  const hLabel = hybrid ? `hybrid:${fusion}` : (useEmb ? "semantics" : "lexical");
  const reasonFor = (g: number, h: number, cross = false) => {
    const base = g > 0 && h > 0 ? `graph+${hLabel}` : g > 0 ? "graph" : hLabel;
    return cross ? `${base}+cross-project` : base;
  };
  const cands: Cand[] = [];
  for (const i of inIdx) {
    const u = store.units[i];
    const g = mmN(gRaw.get(u.id) ?? 0, gMin, gMax);
    const h = hOf(inPool, i);
    const score = wGraph * g + wHier * h;
    if (score <= 0) continue;
    cands.push({ idx: i, unit: u, score, reason: reasonFor(g, h) });
  }
  // v0.9: cross-project federation — ONLY when the in-project pool has nothing
  // above minScore (project scoping is too narrow). Cross-project hits are
  // h-normalized within their OWN pool, then penalized so they rank below a
  // real in-project answer of equal strength. The instant the current project
  // answers the query, this block is skipped and nothing leaks across.
  if ((opts.federate ?? true) && scopeToProject && outIdx.length > 0 &&
      !cands.some((c) => c.score > minScore)) {
    const outPool = scorePool(outIdx);
    const federatePenalty = opts.federatePenalty ?? 0.7;
    for (const i of outIdx) {
      const u = store.units[i];
      const g = mmN(gRaw.get(u.id) ?? 0, gMin, gMax);
      const h = hOf(outPool, i);
      const score = federatePenalty * (wGraph * g + wHier * h);
      if (score <= 0) continue;
      cands.push({ idx: i, unit: u, score, reason: reasonFor(g, h, true) });
    }
  }

  // Evidence closure: 1-hop graph neighbors + session-adjacent turns.
  const bySessionTurn = new Map<string, TraceUnit[]>();
  for (const u of store.units) {
    const arr = bySessionTurn.get(u.sessionId) ?? [];
    arr.push(u);
    bySessionTurn.set(u.sessionId, arr);
  }
  for (const a of bySessionTurn.values()) a.sort((x, y) => x.timestamp - y.timestamp);

  const have = new Map(cands.map((c) => [c.unit.id, c]));
  for (const c of [...cands]) {
    for (const nid of store.graph.neighbors(c.unit.id)) {
      if (have.has(nid)) continue;
      const nu = store.units.find((u) => u.id === nid);
      if (!nu) continue;
      if (scopeToProject && nu.cwd !== opts.cwd) continue;
      if (opts.activeContext?.has(nu.fp)) continue; // v0.3: don't pull in-context evidence back in via closure
      have.set(nid, { idx: -1, unit: nu, score: c.score * 0.35, reason: "closure:shared-entity" });
    }
    const arr = bySessionTurn.get(c.unit.sessionId);
    if (arr) {
      const pos = arr.findIndex((u) => u.id === c.unit.id);
      for (const delta of [-1, 1]) {
        const nb = arr[pos + delta];
        if (!nb || have.has(nb.id)) continue;
        if (opts.activeContext?.has(nb.fp)) continue; // v0.3: don't pull in-context evidence back in via closure
        have.set(nb.id, { idx: -1, unit: nb, score: c.score * 0.25, reason: "closure:adjacent-turn" });
      }
    }
  }

  const seenFp = new Set<string>();
  const ranked = [...have.values()]
    .filter((c) => c.score > minScore)
    .sort((a, b) => b.score - a.score)
    .filter((c) => { if (seenFp.has(c.unit.fp)) return false; seenFp.add(c.unit.fp); return true; }); // v0.3: de-dup near-identical evidence
  // v0.11: evidence calibration (paper Eq 15) — re-rank admissible evidence by
  // answer-type compatibility as a small boost (breaks near-ties toward type-matching
  // evidence: temporal queries favor units containing dates, quantity → numbers).
  if (opts.calibrateEvidence ?? true) {
    const cw = 0.15;
    ranked.sort((a, b) => (b.score + cw * evidenceCompat(query, b.unit.text)) - (a.score + cw * evidenceCompat(query, a.unit.text)));
  }
  // v0.7: MMR diversifies the injected set so top-K isn't several near-duplicate
  // snippets. Diversify from a pool larger than topK, then select topK.
  const pool = ranked.slice(0, poolSize);
  const picked = useMmr && pool.length > topK ? mmrSelect(pool, topK, opts.mmrLambda ?? adaptiveMmrLambda(query, qEnts, qTokens)) : pool;
  return picked.slice(0, topK).map((c) => ({ unit: c.unit, score: c.score, reason: c.reason }));
}

export function formatEvidence(hits: Hit[], snippetChars = 220): string {
  if (!hits.length) return "";
  const lines = [
    "## Prior session memory (Zero-Mem — use only if relevant; not authoritative)",
  ];
  for (const h of hits) {
    const u = h.unit;
    const snip = u.text.replace(/\s+/g, " ").trim().slice(0, snippetChars);
    const sess = u.sessionName ? ` • "${u.sessionName}"` : "";
    lines.push(`- [${relTime(u.timestamp)}${sess} • ${h.reason}] ${snip}`);
  }
  return lines.join("\n");
}

// ── Answer-level calibration (v0.6) ───────────────────────────────────────────
// Deterministic, zero-LLM checks on the reader's (model's) output. The paper's
// component #4: validate type/format, detect imbalance, and flag over-reliance on
// injected memory. NON-DESTRUCTIVE — it never rewrites the model's answer, only
// emits warnings (opt-in via ZERO_MEM_CALIBRATE=1 in index.ts).
export interface CalibrateOpts {
  hits?: Hit[];                              // evidence that was injected
  query?: string;                            // the user prompt
  expects?: "json" | "code" | "text";        // optional type expectation
  extract?: (t: string) => string[];         // entity extractor (coverage signal)
}
export interface CalibrationResult {
  ok: boolean;
  warnings: string[];
  signals: { fences: number; jsonParses?: boolean; entityCoverage: number; verbatimMemory: boolean };
}
function stripFences(s: string): string {
  return s.trim().replace(/^```[a-zA-Z0-9]*\n?/, "").replace(/\n?```$/, "");
}
export function calibrate(response: string, opts: CalibrateOpts = {}): CalibrationResult {
  const warnings: string[] = [];
  const fences = (response.match(/```/g) ?? []).length;
  if (fences % 2 !== 0) warnings.push(`unbalanced code fence (count=${fences})`);

  let jsonParses: boolean | undefined;
  if (opts.expects === "json") {
    try { JSON.parse(stripFences(response)); jsonParses = true; }
    catch (e: any) { jsonParses = false; warnings.push(`expected JSON but parse failed: ${(e?.message ?? e)}`); }
  }

  let entityCoverage = 1;
  if (opts.query && opts.extract) {
    const qEnts = opts.extract(opts.query);
    if (qEnts.length) {
      const lower = response.toLowerCase();
      const hit = qEnts.filter((e) => lower.includes(e.toLowerCase())).length;
      entityCoverage = hit / qEnts.length;
      if (entityCoverage === 0) warnings.push(`response mentions none of the ${qEnts.length} query entities`);
    }
  }

  let verbatimMemory = false;
  if (opts.hits && opts.hits.length) {
    const lower = response.toLowerCase();
    for (const h of opts.hits) {
      const snip = h.unit.text.replace(/\s+/g, " ").trim().slice(0, 80).toLowerCase();
      if (snip.length > 30 && lower.includes(snip)) {
        verbatimMemory = true;
        warnings.push(`response reproduces injected memory verbatim (unit ${h.unit.id}) — memory is not authoritative`);
        break;
      }
    }
  }

  return { ok: warnings.length === 0, warnings, signals: { fences, jsonParses, entityCoverage, verbatimMemory } };
}
