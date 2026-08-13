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

## Remaining roadmap
- **v0.4** HNSW index for scale beyond a few thousand units.
- **v0.4** Pairwise co-occurrence weights + the paper's evidence-closure "relational bridges."
- **v0.4** Answer-level calibration for typed tasks (deterministic support/type/format checks on the reader's output).
- **v0.5** Eval harness on LoCoMo / HotpotQA-style splits to measure F1/BLEU as in the paper.

## Safety / cost
- **Zero extra LLM calls** in steady state (the paper's headline property). The optional `recall_memory` tool uses a normal tool round-trip but no extra model generation.
- Storage grows with history; text is capped at 4000 chars/unit (500 for tool results). Clear with `/memory-clear`.
- All handlers are try/caught so a malformed message can never break the agent.
