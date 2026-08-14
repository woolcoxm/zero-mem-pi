# Zero-Mem for pi

Zero-token long-term memory for the [pi](https://pi.dev) coding agent — a faithful
reimplementation of [Zero-Mem (arXiv:2607.29377)](https://arxiv.org/abs/2607.29377).

It remembers what happened across your pi sessions **without spending any extra
LLM calls or tokens**. It keeps your raw conversation traces, organizes them into
an entity–context graph + a temporal hierarchy, and on each new prompt
deterministically retrieves the most relevant evidence and quietly drops it into
the model's context. Your local model (e.g. Qwen3-Coder-Next) is the only thing
that generates text.

See [`DESIGN.md`](./DESIGN.md) for the paper→pi mapping, version history, and roadmap.

## Install (dev/test)

```bash
cd C:\Users\Robot\projects\zero-mem-pi
npm install            # installs compromise (NER); extension still works without it
```

Run pi with the extension loaded:

```bash
pi -e C:\Users\Robot\projects\zero-mem-pi\index.ts
```

For permanent auto-load (hot-reloadable via `/reload`), copy/symlink the folder to
`~/.pi/agent/extensions/zero-mem/` (so pi sees `index.ts`), or add to
`~/.pi/agent/settings.json`:

```json
{ "extensions": ["C:\\Users\\Robot\\projects\\zero-mem-pi\\index.ts"] }
```

> **Restart pi fully (quit the app) to pick up code changes** — reloading a session
> does not re-import the extension module, so edits to `core.ts`/`index.ts` only
> take effect after a full restart.

## How it works (in pi)

- Every finalized message is captured as a **trace unit** with provenance (session, project, time) + extracted entities, then embedded with `bge-small-en-v1.5` (v0.10; was MiniLM).
- On each prompt, a **zero-LLM pipeline** routes between two views (entity–context graph vs. semantic/temporal), fuses them, runs evidence closure, and injects up to **3** snippets (default) as a `## Prior session memory` block in the system prompt.
- Memory is **project-scoped** by default and persisted to `~/.pi/agent/zero-mem/`:
  - `store.json` — text + metadata only (small, human-readable).
  - `store.emb.bin` — embeddings, **int8-quantized** in a compact sidecar (v0.5). The full-precision JSON era is gone; a one-shot `migrate.ts` converts legacy stores automatically on first load.
- A **retention policy** bounds growth (drop units older than `maxAgeMs`; trim to `maxUnits`).
- Above ~10k embedded units, retrieval switches from exact brute-force cosine to a **pure-JS HNSW** index (growth-gated rebuild, so it never stalls a turn).

## How it compares

Zero-Mem's defining property is **zero-token memory operations**: capture is
passive (every message) and retrieval is deterministic encoder + index math —
the LLM is **never** called to decide what to remember, summarize, or forget.
The only tokens spent on memory are the injected snippet (~103/turn).

| | Plain RAG (vector DB) | MemGPT / LLM-managed | **Zero-Mem** |
|---|---|---|---|
| Decide what to remember | no | **LLM calls** | automatic, no LLM |
| Retrieval | cosine only | varies | BM25 + dense + entity graph + temporal, fused |
| Summarize / forget | no | **LLM calls** | deterministic retention policy |
| Tokens spent on memory mgmt | retrieval tokens | **many** (LLM babysits its own memory) | **zero** — encoder math only |

So vs. a bare vector store it's richer (co-occurrence graph bridges, temporal
hierarchy, hybrid retrieval — not just cosine); vs. MemGPT-style systems it's far
cheaper (they burn LLM generations managing memory). The honest trade-off:
retrieval quality is **competitive, not SOTA** — on real data it **ties BM25**
(LoCoMo r@5 0.543 vs 0.529) and beats it on paraphrase (0.98 vs 0.75). The win
is the zero-token property + the structured pipeline, not raw retrieval dominance.

## Commands

| Command | What it does |
|---|---|
| `/memory <query>` | Search memory now; shows hits in a widget (0 LLM calls). |
| `/memory-stats` | Counts (units, entities, units in this project). |
| `/memory-clear` | Wipe all stored memory (asks to confirm). |

There's also an optional `recall_memory` tool the model can call for explicit recall.

## Tuning

All via environment variables (set in your shell or `settings.json`):

**Storage / retention (v0.5)**
- `ZERO_MEM_STORE` — override the store path.
- `ZERO_MEM_MAX_UNITS` — retention cap (default `2000`).
- `ZERO_MEM_MAX_AGE_DAYS` — expire units older than N days (default `90`).

**Per-request injection (v0.6) — the latency lever on slow local models**
- `ZERO_MEM_INJECT_TOPK` — snippets injected per turn (default `3`; on-demand `/memory` + `recall_memory` still use 8).
- `ZERO_MEM_INJECT_SNIPPET` — snippet char cap for injection (default `120`).

**HNSW at scale (v0.6)**
- `ZERO_MEM_HNSW_THRESHOLD` — unit count at which HNSW activates (default `10000`; below it, exact brute force is faster *and* exact).
- `ZERO_MEM_HNSW=0` — disable HNSW entirely.

**Calibration (v0.6)**
- `ZERO_MEM_CALIBRATE=1` — opt-in deterministic checks on the model's own answers (code-fence balance, JSON validity, query-entity coverage, verbatim reproduction of injected memory). Non-destructive — emits warnings only.

**Retrieval fusion (v0.9)**
- `ZERO_MEM_FUSION` — `coverage` (default; BM25 for factual lookups, dense for synonym/paraphrase), `max`, or `weighted`.
- `ZERO_MEM_HYBRID=0` — disable fusion (restore v0.8 dense-only; not recommended — underperforms BM25 on real data).
- `ZERO_MEM_FEDERATE=0` — disable cross-project fallback (reach into other projects only when this one has nothing relevant).

**Dense embedder (v0.10)**
- `ZERO_MEM_EMBEDDER` — sentence-embedder model (default `Xenova/bge-small-en-v1.5`; was MiniLM through v0.9 — bge lifts LoCoMo pure-semantic r@5 0.27→0.42). The store re-embeds automatically if you change this.

Code-level constants (`index.ts`/`core.ts`): `rho` (routing weight), closure discounts, `recentExcludeMs`, `scopeToProject`, `minScore`, BM25 `k1`/`b`.

## Performance

Measured live on a real 236-unit store (dim 384) at a 94.77 tok/s prompt-eval
rate (`bench.ts`; **extension OFF ⇒ zero overhead** for every row):

**Store I/O** (amortized: load once/session, persist debounced per message)

| | Legacy (inline floats) | Compact (v0.5: int8 sidecar) |
|---|---|---|
| Store on disk | 2088 KB | 239 KB (json 147 + bin 93) — **−88%** |
| Load | 45.2 ms | 8.7 ms (19% of legacy) |
| Persist | 93.4 ms | 20.9 ms (22% of legacy) |

**Per-turn operations**

| op | cost |
|---|---|
| capture (`add()` on `message_end`) | 0.04 ms |
| retrieve (semantic, w/ query embed) | 5.1 ms |
| retrieve (BM25-only fallback) | 1.3 ms |

**Token injection** (v0.5 `topK5 × 220ch` → v0.6 `topK3 × 120ch`): header
46→19 tok, body avg 249→103 tok → **~276 → ~103 tok/turn (−173)**.

**Derived per-turn impact @ 94.77 tok/s**: injection prefill 2912 → 1087 ms
(**−1825 ms/turn**); total per-turn overhead 2917 → 1092 ms, against ~20 s of
base system-prompt + tools prefill.

HNSW recall@10 vs exact brute force: 90–96% at `ef=200` (dim 384); below 10k
units brute force is faster, so HNSW stays dormant until it pays off.

## Eval — retrieval quality

`eval.ts` is a deterministic, **zero-LLM** retrieval harness: 24 facts seeded
across 3 sessions (+40 distractors), each with 1–2 paraphrase queries whose
gold answer is that fact. Zero-Mem owns the *retrieval* step (the paper's
headline is zero extra LLM calls); the reader is your model, so we measure
recall@K / MRR + token cost rather than end-to-end F1/BLEU.

| config | recall@3 | recall@5 | MRR | tok/turn |
|---|---:|---:|---:|---:|
| BM25 only (no embeddings) | 0.75 | 0.75 | 0.73 | 32 |
| coverage fusion (v0.11: min-max + calibration) | 0.96 | **0.98** | 0.90 | 63 |

Headline: the **coverage router** lifts recall@5 **0.75 → 0.98** over BM25 on
synonym/paraphrase queries (v0.11's min-max normalization lifted it 0.92 → 0.98).
(This dataset stresses semantic matching; the v0.8 dense-only path scored 0.96
here, but see LoCoMo below for why that default was wrong for real data.)

### LoCoMo10 — the paper's real benchmark

`eval-locomo.ts` runs the retriever on the actual
[LoCoMo10](https://github.com/snap-research/locomo) dataset (10 long
conversations, 1986 QA). For each QA we check whether the gold-evidence
utterance (by `dia_id`) lands in the top-K.

| config | recall@5 | MRR |
|---|---:|---:|
| BM25 only | 0.529 | 0.393 |
| pure semantic (MiniLM, v0.8 default) | 0.273 | 0.183 |
| RRF hybrid (k=60) | 0.503 | 0.323 |
| **coverage fusion (v0.11: min-max + calibration)** | **0.543** | **0.404** |

On real conversational factual lookups, **BM25 beats MiniLM** (0.529 vs 0.273),
and no naive fusion (max/weighted/RRF) recovers it — the dense model is weak
out-of-domain. The **coverage router** fixes it: blend BM25 + dense by the
query's lexical coverage, so BM25 carries factual lookups (high coverage) while
dense rescues synonym/paraphrase queries whose terms are OOV (low coverage).
v0.11 added the paper's **min-max normalization** (Eq 12 — stretches dense
cosines from [0.5,1] to [0,1]) and **evidence calibration** (Eq 15 — answer-type
compatibility re-rank). Result: **0.543 (≥ BM25) on LoCoMo** *and* **0.98 on the
paraphrase eval** (vs BM25's 0.75) — best-of-both, no regressions. For the
paper's actual metric — end-to-end answer F1/EM/BLEU with an LLM reader —
`eval-reader.ts` runs it: on a 50-QA LoCoMo sample (top-10 context + session
dates, Qwen3-Coder reader, temp 0), **answer F1 0.200 / EM 0.040 / BLEU-1
0.224** (retrieval hit@10 0.40; was F1 0.155 at top-5). It's a strict RAG setup
so it's a lower bound — the LoCoMo paper itself reports models "lag behind human
performance." Embeddings are bit-exact deterministic across runs.

## Tests

```bash
node --experimental-strip-types test-storage.ts    # int8 round-trip, size, retention, migration (7/7)
node --experimental-strip-types test-hnsw.ts       # HNSW recall vs brute, async build, threshold guard (5/5)
node --experimental-strip-types test-mmr.ts        # MMR reduces pairwise redundancy (3/3)
node --experimental-strip-types test-calibrate.ts  # fence/json/coverage/verbatim checks (8/8)
node --experimental-strip-types test.ts            # v0.4 co-occurrence relational bridges
node --experimental-strip-types test-recall.ts     # v0.8 session-scoped recent-exclusion (3/3)
node --experimental-strip-types test-federation.ts # v0.9 cross-project federation (5/5)
node --experimental-strip-types test-adaptive.ts   # v0.9 adaptive MMR lambda (5/5)
node --experimental-strip-types test-incremental.ts # v0.9 HNSW incremental insert (7/7)
node --experimental-strip-types eval.ts            # retrieval eval: recall@K / MRR ablation
node --experimental-strip-types eval-locomo.ts     # retrieval eval on real LoCoMo10 (caches ~2.8MB dataset)
node --experimental-strip-types eval-reader.ts      # END-TO-END: LLM-reader F1/EM/BLEU on LoCoMo (falls back to hit-rate w/o endpoint)
node --experimental-strip-types bench.ts           # A/B benchmark (store I/O + token overhead)
```

Requires Node ≥ 22 (for `--experimental-strip-types`). The live MiniLM embedder is
loaded on demand; tests that need it will fetch `all-MiniLM-L6-v2` once (~23 MB).

## Status

**v0.11** — everything in v0.10 plus three **paper-fidelity fixes** (vs arXiv:2607.29377):
**min-max per-view normalization** (Eq 12), **evidence calibration** (Eq 15 —
answer-type compatibility re-rank), and **top-10 retrieval budget** (their ablation's
best). All validated on the real LoCoMo benchmark with no regressions (38/38 tests):
retrieval r@5 **0.534 → 0.543** (≥ BM25), MRR 0.393 → 0.404; paraphrase **0.92 →
0.98**; end-to-end answer F1 **0.155 → 0.200** (BLEU-1 0.177 → 0.224). Also audited
the paper's **BGE-M3** embedder — tested it; bge-small empirically wins on CPU (made
`Embedder(model,pooling)` configurable for opt-in BGE-M3). Paper LoCoMo target
(GPT-4o-mini reader): F1 59.15; our gap is mostly the reader (chat vs coder), not the
pipeline. Remaining work in [`DESIGN.md`](./DESIGN.md).
