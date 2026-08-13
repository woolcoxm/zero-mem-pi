/**
 * Zero-Mem core — pure, pi-independent logic (testable in plain Node).
 * The pi wiring lives in index.ts and imports from here.
 *
 * Faithful reimplementation of Xiao et al., "Zero-Mem: Zero-Token Memory
 * Operations for LLM Agents", arXiv:2607.29377.
 */

import { join } from "node:path";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";

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
  scopeToProject?: boolean;
  topK?: number;
  recentExcludeMs?: number;
  rho?: number;
  activeContext?: Set<string>; // v0.3: fingerprints of messages already in the model's window (excluded)
  minScore?: number;           // v0.3: relevance floor (default 0.15) — drop weak/tangential hits
  useBridges?: boolean;        // v0.4: enable co-occurrence relational bridges (default true)
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

  async init(): Promise<void> {
    if (this.loading) return this.loading;
    this.loading = (async () => {
      try {
        const mod: any = await import("@xenova/transformers");
        const env = mod.env;
        env.allowLocalModels = false;        // fetch from HF hub
        env.backends?.onnx?.wasm?.setThreads?.(1);
        this.pipe = await mod.pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { quantized: true });
        this.ready = true;
        console.log("[zero-mem] embeddings ready (all-MiniLM-L6-v2)");
      } catch (e: any) {
        console.warn("[zero-mem] embeddings unavailable — using BM25 fallback:", e?.message ?? e);
        this.ready = false;
      }
    })();
    return this.loading;
  }

  async embed(text: string): Promise<number[] | null> {
    if (!this.ready || !this.pipe) return null;
    const out = await this.pipe(text, { pooling: "mean", normalize: true });
    return Array.from(out.data as Float32Array);
  }
}

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  if (na === 0 || nb === 0) return 0;
  return dot / (Math.sqrt(na) * Math.sqrt(nb));
}

// ── Memory store ────────────────────────────────────────────────────────────
export class MemoryStore {
  embedder: Embedder | null = null; // v0.2: optional dense embedder
  units: TraceUnit[] = [];
  private path: string;
  private dirty = true;
  private saveTimer: ReturnType<typeof setTimeout> | null = null;
  private counter = 0;
  scopeToProject = true;
  graph = new EntityGraph();
  bm25 = new BM25();
  extract: (text: string) => string[];

  constructor(path: string, extract: (t: string) => string[]) {
    this.path = path;
    this.extract = extract;
  }
  load() {
    try {
      if (existsSync(this.path)) {
        const raw = JSON.parse(readFileSync(this.path, "utf8"));
        this.units = Array.isArray(raw?.units) ? raw.units : [];
        this.scopeToProject = raw?.scopeToProject ?? true;
        this.counter = raw?.counter ?? this.units.length;
      }
    } catch (e) { console.error("[zero-mem] failed to load store:", e); }
    this.dirty = true;
  }
  ensureIndex() {
    if (!this.dirty) return;
    this.graph.rebuild(this.units);
    this.bm25.build(this.units);
    this.dirty = false;
  }
  /** v0.2: embed any units still missing a semantic vector (idempotent). */
  async embedAll() {
    if (!this.embedder || !this.embedder.ready) return;
    for (const u of this.units) {
      if (u.embedding && u.embedding.length) continue;
      u.embedding = await this.embedder.embed(u.text);
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
      writeFileSync(this.path, JSON.stringify({ units: this.units, scopeToProject: this.scopeToProject, counter: this.counter }));
    } catch (e) { console.error("[zero-mem] failed to persist:", e); }
  }
}

// ── Retrieval pipeline ──────────────────────────────────────────────────────
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
  const cutoff = Date.now() - recentExcludeMs;

  const poolIdx: number[] = [];
  for (let i = 0; i < store.units.length; i++) {
    const u = store.units[i];
    if (u.timestamp >= cutoff) continue;
    if (scopeToProject && u.cwd !== opts.cwd) continue;
    if (opts.activeContext?.has(u.fp)) continue; // v0.3: skip what's already in the model's context window
    poolIdx.push(i);
  }
  if (!poolIdx.length) return [];

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

  const hRaw = new Map<number, number>();
  let hMax = 0;
  for (const i of poolIdx) {
    const u = store.units[i];
    let s: number;
    if (useEmb && qEmb && u.embedding && u.embedding.length) {
      s = (cosine(qEmb, u.embedding) + 1) / 2; // cosine [-1,1] → [0,1]
    } else {
      s = store.bm25.score(qTokens, i); // word-overlap fallback
    }
    hRaw.set(i, s);
    if (s > hMax) hMax = s;
  }

  const entityDriven = qEnts.length > 0;
  const temporal = /\b(earlier|before|last|previous|yesterday|ago|used to|recently|back when)\b/i.test(query);
  let wGraph = rho;
  if (entityDriven) wGraph = Math.min(0.9, rho + 0.15);
  else if (temporal) wGraph = Math.max(0.3, rho - 0.2);
  const wHier = 1 - wGraph;

  let gMax = 0;
  for (const v of gRaw.values()) if (v > gMax) gMax = v;

  type Cand = { idx: number; unit: TraceUnit; score: number; reason: string };
  const cands: Cand[] = [];
  for (const i of poolIdx) {
    const u = store.units[i];
    const g = gMax ? (gRaw.get(u.id) ?? 0) / gMax : 0;
    const h = hMax ? (hRaw.get(i) ?? 0) / hMax : 0;
    const score = wGraph * g + wHier * h;
    if (score <= 0) continue;
    const hLabel = useEmb ? "semantics" : "lexical";
    const reason = g > 0 && h > 0 ? `graph+${hLabel}` : g > 0 ? "graph" : hLabel;
    cands.push({ idx: i, unit: u, score, reason });
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

  const minScore = opts.minScore ?? 0.15;
  const seenFp = new Set<string>();
  return [...have.values()]
    .filter((c) => c.score > minScore)
    .sort((a, b) => b.score - a.score)
    .filter((c) => { if (seenFp.has(c.unit.fp)) return false; seenFp.add(c.unit.fp); return true; }) // v0.3: de-dup near-identical evidence
    .slice(0, topK)
    .map((c) => ({ unit: c.unit, score: c.score, reason: c.reason }));
}

export function formatEvidence(hits: Hit[]): string {
  if (!hits.length) return "";
  const lines = [
    "## Retrieved memory (Zero-Mem — 0 extra LLM calls)",
    "Background context from earlier sessions in this project. Use ONLY if relevant;",
    "it is not authoritative over current instructions.",
  ];
  for (const h of hits) {
    const u = h.unit;
    const snip = u.text.replace(/\s+/g, " ").trim().slice(0, 220);
    const sess = u.sessionName ? ` • "${u.sessionName}"` : "";
    lines.push(`- [${relTime(u.timestamp)}${sess} • ${h.reason}] ${snip}`);
  }
  return lines.join("\n");
}
