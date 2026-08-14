/**
 * Zero-Mem v0.14b regression test — the "what is your name?" confusion.
 *
 * Reported live failure: tiny name-related chat lines ("what is my name?",
 * "your name is Henry.", "my name is mark.") got injected for the query
 * "what is your name?", and the reader hallucinated a "conflict" between two
 * unrelated statements. Root cause: min-max normalization stretched the weak
 * pool (only matching term: "name") to [0,1], so garbage scored ~0.5 and beat
 * minScore. v0.14b adds pool-confidence gating — weak pools stay weak.
 */
import { MemoryStore, makeExtractor, retrieve, Embedder } from "./core.ts";

let pass = 0, fail = 0;
const check = (name: string, cond: boolean, extra = "") => { if (cond) { pass++; console.log(`  ✓ ${name} ${extra}`); } else { fail++; console.log(`  ✗ ${name} ${extra}`); } };

const extract = makeExtractor(null);
const store = new MemoryStore("C:/Users/Robot/projects/zero-mem-pi/.relevance-test.json", extract);
store.embedder = new Embedder(); // production path: dense embeddings active
await store.embedder.init();
const cwd = "C:/proj", now = Date.now();
const add = (sid: string, role: "user" | "assistant", text: string, agoMin: number) =>
  store.add({ sessionId: sid, cwd, role, text, timestamp: now - agoMin * 60_000 });

// The exact transcript from the live failure (plus a bit of same-flavor noise).
add("s1", "user", "what is my name?", 3);
add("s1", "assistant", "your name is Henry.", 2);
add("s1", "user", "my name is mark.", 4);
add("s1", "assistant", "Hi Mark! Nice to meet you.", 2);
add("s1", "user", "what's your name?", 1);
// And one real fact that SHOULD be retrievable.
add("s2", "user", "The deploy key for the staging cluster is dk_8842 — it expires Friday.", 30);
add("s2", "assistant", "Noted: staging deploy key dk_8842, expiry Friday.", 30);
store.ensureIndex();

console.log("ZERO-MEM v0.14b — weak-pool confidence gating");

// 1. The confusion case: nothing in memory answers "what is your name?" —
// the name chatter is tangential (only "name" matches). Expect NO injection.
const weak = await retrieve("what is your name?", store, { cwd, topK: 3 });
check("weak pool injects NOTHING for 'what is your name?'", weak.length === 0,
  `(got ${weak.length}: ${weak.map((h) => JSON.stringify(h.unit.text.slice(0, 24))).join(", ")})`);

// 2. A question about the USER's name IS answerable — should surface the fact.
const user = await retrieve("what did i say my name was", store, { cwd, topK: 3 });
check("real signal still retrieved: 'my name is mark.' surfaces", user.some((h) => h.unit.text === "my name is mark."),
  `(top: ${user[0]?.unit.text.slice(0, 30) ?? "(none)"})`);

// 3. Strong facts unaffected: deploy key lookup must work.
const key = await retrieve("staging deploy key expiry", store, { cwd, topK: 3 });
check("strong pool unaffected: deploy-key fact surfaces", key.some((h) => h.unit.text.includes("dk_8842")),
  `(top: ${key[0]?.unit.text.slice(0, 30) ?? "(none)"})`);

// 4. Greetings may legitimately surface the prior greeting exchange (cos
// 0.73 — semantically it IS the closest memory, and it carries the user's
// name), but must never leak unrelated facts.
const hi = await retrieve("hello there", store, { cwd, topK: 3 });
check("greeting query never surfaces unrelated facts", !hi.some((h) => h.unit.text.includes("dk_8842")),
  `(got ${hi.length}: ${hi.map((h) => h.unit.text.slice(0, 20)).join(" / ")})`);

console.log(`\nRESULTS: ${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
