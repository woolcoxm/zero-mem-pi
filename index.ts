/**
 * Zero-Mem for pi — pi agent wiring.  Core (testable) logic is in ./core.ts.
 * v0.2: dense semantic embeddings (meanings, not words) with BM25 fallback.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import { join } from "node:path";
import { homedir } from "node:os";
import { MemoryStore, Embedder, retrieve, formatEvidence, makeExtractor, flattenContent, fingerprint, calibrate, type Hit } from "./core.ts";

type Role = "user" | "assistant" | "tool";

export default async function (pi: ExtensionAPI) {
  let nlp: any = null;
  try { nlp = (await import("compromise")).default; }
  catch { console.warn("[zero-mem] compromise not installed — using regex-only NER"); }
  const extract = makeExtractor(nlp);

  const storePath = process.env.ZERO_MEM_STORE ??
    join(homedir(), ".pi", "agent", "zero-mem", "store.json");
  const store = new MemoryStore(storePath, extract);
  if (process.env.ZERO_MEM_MAX_UNITS) store.maxUnits = Number(process.env.ZERO_MEM_MAX_UNITS) || store.maxUnits; // v0.5 retention
  if (process.env.ZERO_MEM_MAX_AGE_DAYS) store.maxAgeMs = (Number(process.env.ZERO_MEM_MAX_AGE_DAYS) || 0) * 24 * 3600 * 1000; // v0.5
  // v0.6: slimmer per-request injection (the latency lever on slow local models).
  const injectTopK = Math.max(1, Number(process.env.ZERO_MEM_INJECT_TOPK ?? 3));
  const injectSnippet = Math.max(40, Number(process.env.ZERO_MEM_INJECT_SNIPPET ?? 120));
  const mmrLambda = Math.min(1, Math.max(0, Number(process.env.ZERO_MEM_MMR_LAMBDA ?? 0.5))); // v0.7: relevance vs diversity
  const calibrateOn = process.env.ZERO_MEM_CALIBRATE === "1"; // v0.6: opt-in answer calibration
  let lastInjection: { query: string; hits: Hit[] } | null = null; // v0.6: for calibrate()
  store.embedder = new Embedder(); // v0.2; loads lazily on first retrieve
  let loaded = false;
  const ensureLoaded = async () => { if (!loaded) { store.load(); loaded = true; } };

  // Warm up embeddings in the background after each capture so retrieval is instant.
  const warmEmbeddings = () => {
    store.embedder?.init()
      .then(() => store.embedAll())
      .then(() => store.persistDebounced())
      .catch(() => {});
  };

  pi.on("message_end", async (event: any, ctx: any) => {
    try {
      await ensureLoaded();
      const msg = event?.message;
      if (!msg) return;
      const role: string = msg.role;
      if (role !== "user" && role !== "assistant" && role !== "toolResult") return;
      let text = flattenContent(msg.content);
      // v0.6: opt-in deterministic calibration of the model's own answer.
      if (calibrateOn && role === "assistant" && text.trim().length > 0) {
        try {
          const r = calibrate(text, { hits: lastInjection?.hits, query: lastInjection?.query, extract });
          if (!r.ok) ctx.ui.notify?.(`[zero-mem calibrate] ${r.warnings.join("; ")}`, "warn");
          else if (ctx.hasUI) ctx.ui.setStatus?.("zero-mem-calibrate", "calibrate: ok");
        } catch { /* best-effort */ }
      }
      if (role === "toolResult") text = text.slice(0, 500);
      if (!text || text.trim().length < 3) return;
      if (/^[\s.!?,;:]+$/.test(text)) return;
      store.add({
        sessionId: ctx.sessionManager?.getSessionId?.() ?? "unknown",
        sessionName: ctx.sessionManager?.getSessionName?.(),
        cwd: ctx.cwd,
        role: (role === "toolResult" ? "tool" : role) as Role,
        text,
        timestamp: Date.now(),
      });
      store.persistDebounced();
      warmEmbeddings();
      if (ctx.hasUI) ctx.ui.setStatus?.("zero-mem", `memory: ${store.size()} units`);
    } catch (e) { console.error("[zero-mem] message_end error:", e); }
  });

  pi.on("before_agent_start", async (event: any, ctx: any) => {
    try {
      await ensureLoaded();
      const query = String(event?.prompt ?? "").trim();
      if (!query) return;
      const hits = await retrieve(query, store, { cwd: ctx.cwd, scopeToProject: store.scopeToProject, topK: injectTopK, mmrLambda, activeContext: activeContextFingerprints(ctx) });
      lastInjection = { query, hits };
      if (!hits.length) return;
      return { systemPrompt: (event.systemPrompt ?? "") + "\n\n" + formatEvidence(hits, injectSnippet) };
    } catch (e) { console.error("[zero-mem] before_agent_start error:", e); }
  });

  pi.on("session_shutdown", async () => { try { await store.persist(); } catch {} });

  pi.on("session_start", async (_e: any, ctx: any) => {
    await ensureLoaded();
    if (ctx.hasUI && store.size()) ctx.ui.setStatus?.("zero-mem", `memory: ${store.size()} units`);
  });

  pi.registerCommand("memory", {
    description: "Search Zero-Mem memory (zero LLM calls)",
    handler: async (args: string, ctx: any) => {
      await ensureLoaded();
      const q = (args || "").trim();
      if (!q) { ctx.ui.notify?.(storeStats(store, ctx.cwd), "info"); return; }
      const hits = await retrieve(q, store, { cwd: ctx.cwd, topK: 8, activeContext: activeContextFingerprints(ctx) });
      if (!hits.length) { ctx.ui.notify?.("No matching memory.", "info"); return; }
      ctx.ui.setWidget?.("zero-mem", [formatEvidence(hits), `query: ${q}`]);
    },
  });

  pi.registerCommand("memory-stats", {
    description: "Show Zero-Mem memory statistics",
    handler: async (_a: string, ctx: any) => { await ensureLoaded(); ctx.ui.notify?.(storeStats(store, ctx.cwd), "info"); },
  });

  pi.registerCommand("memory-clear", {
    description: "Wipe all Zero-Mem memory",
    handler: async (_a: string, ctx: any) => {
      const ok = await ctx.ui.confirm?.("Zero-Mem", "Delete ALL stored memory? This cannot be undone.");
      if (!ok) return;
      store.clear();
      await store.persist();
      ctx.ui.notify?.("Zero-Mem memory cleared.", "info");
    },
  });

  pi.registerTool({
    name: "recall_memory",
    label: "Recall memory",
    description:
      "Search prior session memory for the current project (deterministic, zero extra model calls). " +
      "Returns up to 8 relevant snippets with timestamps. Use when you need context from earlier work.",
    parameters: Type.Object({
      query: Type.String({ description: "What to recall (entities, file names, decisions, etc.)" }),
    }),
    async execute(_id: string, params: { query: string }, _signal: AbortSignal, _onUpdate: any, ctx: any) {
      await ensureLoaded();
      const hits = await retrieve(params.query, store, { cwd: ctx.cwd, topK: 8, activeContext: activeContextFingerprints(ctx) });
      return {
        content: [{ type: "text" as const, text: hits.length ? formatEvidence(hits) : "No matching memory." }],
        details: { hits: hits.length },
      };
    },
  });
}

function activeContextFingerprints(ctx: any): Set<string> {
  // v0.3: fingerprints of everything currently in the model's context window,
  // so retrieval skips memories the model can already see (no redundant injection).
  const fps = new Set<string>();
  try {
    const sm = ctx?.sessionManager;
    const entries = sm?.buildContextEntries?.() ?? sm?.getBranch?.() ?? sm?.getEntries?.() ?? [];
    for (const e of entries) {
      const txt = flattenContent(e?.message?.content ?? e?.content);
      if (txt && txt.trim().length > 3) fps.add(fingerprint(txt));
    }
  } catch { /* best-effort: on failure, exclude nothing */ }
  return fps;
}

function storeStats(store: MemoryStore, cwd: string): string {
  const inProj = store.units.filter((u) => u.cwd === cwd).length;
  const ents = new Set<string>();
  for (const u of store.units) for (const e of u.entities) ents.add(e);
  const emb = store.units.filter((u) => u.embedding && u.embedding.length).length;
  return `Zero-Mem: ${store.units.length} units (${inProj} this project), ${ents.size} entities, ${emb} embedded`;
}
