# Zero-Mem for pi

Zero-token long-term memory for the [pi](https://pi.dev) coding agent — a faithful
reimplementation of [Zero-Mem (arXiv:2607.29377)](https://arxiv.org/abs/2607.29377).

It remembers what happened across your pi sessions **without spending any extra
LLM calls or tokens**. It keeps your raw conversation traces, organizes them into
an entity–context graph + a temporal hierarchy, and on each new prompt
deterministically retrieves the most relevant evidence and quietly drops it into
the model's context. Your local model (e.g. Qwen3-Coder-Next) is the only thing
that generates text.

See [`DESIGN.md`](./DESIGN.md) for the paper→pi mapping and roadmap.

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

## How it works (in pi)

- Every finalized message is captured as a **trace unit** with provenance (session, project, time) + extracted entities.
- On each prompt, a **zero-LLM pipeline** routes between two views, fuses, runs evidence closure, calibrates, and injects up to 5 snippets as a `## Retrieved memory` block in the system prompt.
- Memory is **project-scoped** by default and persisted to `~/.pi/agent/zero-mem/store.json`.

## Commands

| Command | What it does |
|---|---|
| `/memory <query>` | Search memory now; shows hits in a widget (0 LLM calls). |
| `/memory-stats` | Counts (units, entities, units in this project). |
| `/memory-clear` | Wipe all stored memory (asks to confirm). |

There's also an optional `recall_memory` tool the model can call for explicit recall.

## Tuning

Environment variables / code constants:
- `ZERO_MEM_STORE` — override the store path.
- In `index.ts`: `rho` (routing), `topK`, closure discounts, `recentExcludeMs`, `scopeToProject`.

## Status

v0.3 — raw-trace memory + dense semantic embeddings + context-aware injection
(all proven by `test.ts`). Remaining work (HNSW scale, answer-level calibration,
eval harness) is documented in `DESIGN.md`.
