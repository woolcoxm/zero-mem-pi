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

- Every finalized message is captured as a **trace unit** with provenance (session, project, time) + extracted entities, then embedded with `all-MiniLM-L6-v2`.
- On each prompt, a **zero-LLM pipeline** routes between two views (entity–context graph vs. semantic/temporal), fuses them, runs evidence closure, and injects up to **3** snippets (default) as a `## Prior session memory` block in the system prompt.
- Memory is **project-scoped** by default and persisted to `~/.pi/agent/zero-mem/`:
  - `store.json` — text + metadata only (small, human-readable).
  - `store.emb.bin` — embeddings, **int8-quantized** in a compact sidecar (v0.5). The full-precision JSON era is gone; a one-shot `migrate.ts` converts legacy stores automatically on first load.
- A **retention policy** bounds growth (drop units older than `maxAgeMs`; trim to `maxUnits`).
- Above ~10k embedded units, retrieval switches from exact brute-force cosine to a **pure-JS HNSW** index (growth-gated rebuild, so it never stalls a turn).

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
| coverage fusion (v0.9 default) | 0.90 | **0.92** | 0.88 | 81 |

Headline: the **coverage router** lifts recall@5 **0.75 → 0.92** over BM25 on
synonym/paraphrase queries. (This dataset is designed to stress semantic
matching; the v0.8 dense-only path scored 0.96 here, but see LoCoMo below for
why that default was wrong for real data.)

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
| **coverage fusion (v0.9 default)** | **0.534** | **0.393** |

On real conversational factual lookups, **BM25 beats MiniLM** (0.529 vs 0.273),
and no naive fusion (max/weighted/RRF) recovers it — the dense model is weak
out-of-domain. The v0.9 **coverage router** fixes it: blend BM25 + dense by
the query's lexical coverage, so BM25 carries factual lookups (high coverage)
while dense rescues synonym/paraphrase queries whose terms are OOV (low
coverage). Result: **0.534 (≥ BM25) on LoCoMo** *and* **0.92 on the paraphrase
eval** (vs BM25's 0.75) — best-of-both, no regressions. For the paper's actual
metric — end-to-end answer F1/EM/BLEU with an LLM reader — see `eval-reader.ts`
(runs the moment an OpenAI-compatible endpoint is up; falls back to retrieval
hit-rate otherwise). Embeddings are bit-exact deterministic across runs.

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

**v0.9** — cross-project federation (reach across projects when this one has nothing
relevant), adaptive MMR λ (relevance-favoring for lookups, diversity-favoring for
exploratory queries), HNSW incremental insert (fresh facts searchable at scale
immediately, no waiting for the +50% rebuild), and a **retrieval eval on the real
LoCoMo10 benchmark** that exposed and fixed a real deficiency: the v0.8 dense-only
default underperformed BM25 on real data (r@5 0.273 vs 0.529). The fix is a
**coverage router** that blends BM25 + dense by query lexical coverage — 0.534
(≥ BM25) on LoCoMo, 0.92 on the paraphrase eval. 38/38 tests green. Remaining
work (end-to-end F1/BLEU with an LLM reader, a stronger/conversational embedder
or cross-encoder reranker to beat BM25 outright, incremental-HNSW quality) is in
[`DESIGN.md`](./DESIGN.md).
