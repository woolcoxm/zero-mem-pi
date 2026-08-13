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

Code-level constants (`index.ts`/`core.ts`): `rho` (routing weight), closure discounts, `recentExcludeMs`, `scopeToProject`, `minScore`, BM25 `k1`/`b`.

## Performance

Measured on a real ~240-unit store at a 94.77 tok/s prompt-eval rate (`bench.ts`):

| | Before (v0.4) | After (v0.6) |
|---|---|---|
| Store on disk | ~2.1 MB | ~0.3 MB (**−84%**) |
| Load (per session) | ~28 ms | ~3.7 ms (8.6× faster) |
| Persist (debounced/msg) | ~65 ms | ~17 ms (3.8× faster) |
| Tokens injected/turn | ~294 | ~112 (−182) |
| Prefill added by injection | ~3.1 s | ~1.2 s (−1.9 s/turn) |

HNSW recall@10 vs exact brute force: 90–96% at `ef=200` (dim 384); below 10k units brute force is faster, so HNSW stays dormant until it pays off.

## Tests

```bash
node --experimental-strip-types test-storage.ts   # int8 round-trip, size, retention, migration (6/6)
node --experimental-strip-types test-hnsw.ts      # HNSW recall vs brute, retrieve overlap, threshold guard (3/3)
node --experimental-strip-types test-calibrate.ts # fence/json/coverage/verbatim checks (8/8)
node --experimental-strip-types test.ts           # v0.4 co-occurrence relational bridges
node --experimental-strip-types bench.ts          # A/B benchmark (store I/O + token overhead)
```

Requires Node ≥ 22 (for `--experimental-strip-types`). The live MiniLM embedder is
loaded on demand; tests that need it will fetch `all-MiniLM-L6-v2` once (~23 MB).

## Status

**v0.6** — raw-trace memory, dense semantic embeddings, context-aware + co-occurrence-bridge
retrieval, compact int8 storage with retention, slim per-request injection, HNSW at scale,
and opt-in answer calibration. Remaining work (eval harness, async HNSW build, MMR injection
pruning) is documented in [`DESIGN.md`](./DESIGN.md).
