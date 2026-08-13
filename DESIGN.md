# Zero-Mem for pi — Design

Faithful reimplementation of **"Zero-Mem: Zero-Token Memory Operations for LLM Agents"** (Xiao et al., arXiv:2607.29377, 31 Jul 2026) as a [pi](https://pi.dev) coding-agent extension.

## The core insight (why this fits pi perfectly)

The paper's claim: *agent memory need not generate an intermediate representation of the past.* Most memory systems (MemGPT, Mem0, Zep, summaries) **burn LLM calls** to maintain memory. Zero-Mem keeps **raw interaction traces** and structures them — zero LLM calls for any memory operation; only the final answer uses the model.

pi already stores every message as a session entry. **Those entries ARE Zero-Mem's "interaction traces."** So this extension doesn't build a parallel store from scratch — it *organizes* pi's traces and *retrieves evidence per query*, injecting it before the model. That is exactly the paper's operating regime.

## Paper → pi mapping

| Paper component | This extension |
|---|---|
| **1. Provenance-preserving token-free substrate** | `message_end` event → capture unit (`MemoryStore.add`): role, text, `sessionId`, `cwd`, timestamp. Persisted to `~/.pi/agent/zero-mem/store.json`. |
| **NER (spaCy, non-generative)** | `makeExtractor`: `compromise` (if installed) **+ code-aware regex** (paths, identifiers, quoted terms, acronyms). Code regex matters more than NLP NER for a coding agent. |
| **2. Entity–context graph** (co-occurrence + adjacency) | `EntityGraph`: `entity → units` + `unit → entities`. Query = units mentioning query entities; closure = 1-hop neighbors. |
| **3. Temporal hierarchy** (session → episode → turn) | `retrieve`: BM25 over units **+ session-adjacent turn closure** (prev/next in same session). |
| **Query-conditioned routing** (ρ=0.6) | Weight `wGraph`/`wHier` shifted by query profile (entity-driven → +graph; temporal words → +hierarchy). |
| **Dual-view retrieve + fusion** | Normalize each view to [0,1], fuse `wGraph·g + wHier·h`. |
| **Evidence closure** | Add 1-hop graph neighbors + adjacent turns at discounted weight. |
| **4. Deterministic calibration** | Filter (score>0.05, provenance scope, recency guard), dedupe, rank, cap top-K. |
| **Final-QA reader = the only LLM** | The normal pi agent turn. Evidence is injected via `before_agent_start → systemPrompt`. **Zero extra LLM calls.** |

## What we kept faithful vs. simplified (v0.1)

**Faithful:** raw-trace substrate, two-view structure (graph + temporal), query-conditioned routing, fusion, evidence closure, deterministic calibration, zero-LLM-call operations, final-reader-only LLM.

**Simplified for v0.1 (documented for v0.2):**
- **Entity graph** uses entity→unit mapping + on-the-fly shared-entity closure instead of precomputed pairwise co-occurrence weights (cheaper, same relational behavior at this scale).
- **Semantic signal** is BM25 (lexical) rather than a dense embedder. Still zero-LLM; an encoder (transformers.js / llama-server `/v1/embeddings`) is the natural v0.2 upgrade.
- **Answer-level calibration** (type/format checks on the reader's output) is deferred — it needs a typed task and is lower-value for open-ended coding.
- **Context-redundancy** avoidance uses a recency guard (skip the last ~2 min) instead of exact active-context set diffing.

## Hyperparameters (paper defaults)
- Routing ρ = **0.6**, closure discounts 0.35 (graph) / 0.25 (adjacent turn), top-K = **5**, min score 0.05. (`BM25`: k1=1.5, b=0.75.)

## Version status
- **v0.1** ✅ Raw-trace substrate, entity graph, temporal hierarchy, BM25, fusion, closure, calibration, zero-LLM injection.
- **v0.2** ✅ Dense semantic embeddings (`transformers.js` + `all-MiniLM-L6-v2`) replace word-matching for the timeline view, with automatic BM25 fallback. Proven on synonym queries that BM25 misses.
- **v0.3** ✅ Context-aware retrieval: fingerprints of the model's current window are passed in (`activeContext`) so we never inject what's already visible; plus a relevance floor (default 0.15) and near-duplicate de-duplication.
- **v0.4** ✅ Co-occurrence relational bridges (`EntityGraph.cooc`, `useBridges`): units whose entities co-occur with query entities earn graph score without a direct mention.
- **v0.5** ✅ Compact int8 embedding sidecar (`store.emb.bin`) + retention policy. Embeddings quantized to int8 (~21× smaller than inline JSON, cosine drift <0.002) live outside `store.json`; `maxUnits`/`maxAgeMs` bound growth. One-shot migration in `migrate.ts`. **This fixes store bloat + I/O, not per-request tokens** (injection size is unchanged).
- **v0.6** ✅ (1) Slimmer per-request injection — header 46→19 tok, default `topK` 5→3 / snippet 220→120ch (294→112 tok/turn, ≈1.9 s prefill saved on a 95 tok/s model). (2) HNSW ANN index over embeddings (pure-JS, gated at `hnswThreshold`=10000 where it starts to beat exact brute force; growth-gated rebuild so a per-message add never rebuilds). (3) Deterministic answer-level calibration (opt-in, non-destructive): code-fence / JSON / entity-coverage / verbatim-memory checks.
- **v0.7** ✅ (1) Stripped `tokens` from `store.json` (recomputed on load; text store ~25% smaller). (2) MMR diversity pruning for injection (`mmr`/`mmrLambda`; cuts redundant snippets). (3) Async background HNSW build (`HNSWIndex.buildAsync`, event-loop-yielding; `retrieve()` falls back to exact brute force until ready, so a >10k rebuild never stalls a turn). (4) Deterministic retrieval eval harness (`eval.ts`): recall@K / MRR ablation — semantic embeddings lift recall@5 0.75→0.96 vs BM25.
- **v0.8** ✅ Session-scoped recent-exclusion: `recentExcludeMs` now applies only to the *current* session (whose recent turns are still in the model's context window and already caught by `activeContext` fingerprints). Units from *other* sessions are always recallable, no matter how recent. Fixes a real bug where a fact the user just told the agent was unrecoverable in any new conversation opened within `recentExcludeMs` (default 2 min). See `test-recall.ts` (3/3).
- **v0.9** ✅ (1) Cross-project federation (`federate`/`federatePenalty`): when a project has nothing relevant above `minScore`, reach across to other projects (penalized, tagged `cross-project`); no leak when the project can answer itself. `test-federation.ts` (5/5). (2) Adaptive MMR λ (`adaptiveMmrLambda`): lookup queries → relevance-favoring, exploratory → diversity-favoring. `test-adaptive.ts` (5/5). (3) HNSW incremental insert (`HNSWIndex.add`): freshly-embedded units fold into the live index immediately, no waiting for the +50% rebuild. `test-incremental.ts` (7/7). (4) Retrieval eval on the **real LoCoMo10 benchmark** (`eval-locomo.ts`, 1986 QA): exposed that the v0.8 dense-only default UNDERPERFORMS BM25 on real data (r@5 0.273 vs 0.529); no naive fusion (max/weighted/RRF) recovers it. Fix: a **coverage router** — `scorePool` computes both BM25 + cosine and blends by query lexical coverage (`fusion:"coverage"`, default). Validated: LoCoMo r@5 0.534 (≥ BM25) AND paraphrase eval 0.92 (vs BM25 0.75). 38/38 tests green.

## Remaining roadmap
- **v0.10** End-to-end paper metric: LoCoMo/HotpotQA final-answer F1/BLEU with an LLM reader (needs a running endpoint; `eval-locomo.ts` covers the deterministic retrieval view meanwhile).
- **v0.10** Beat BM25 outright on LoCoMo: a stronger/conversational embedder or a cross-encoder reranker over the coverage-router candidate set. (Coverage currently *ties* BM25; pure dense loses.)
- **v0.10** Incremental-HNSW graph-quality refresh tuning (insertion-count vs recall drift) now that add-time inserts are live.

## Safety / cost
- **Zero extra LLM calls** in steady state (the paper's headline property). The optional `recall_memory` tool uses a normal tool round-trip but no extra model generation.
- Storage grows with history; text is capped at 4000 chars/unit (500 for tool results). Clear with `/memory-clear`.
- All handlers are try/caught so a malformed message can never break the agent.
