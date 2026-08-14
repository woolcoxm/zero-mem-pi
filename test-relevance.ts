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

// 4. Greeting-only queries — may surface the prior greeting exchange (cos 0.73),
// but must never leak unrelated facts.
const hi = await retrieve("hello there", store, { cwd, topK: 3 });
check("greeting query never surfaces unrelated facts", !hi.some((h) => h.unit.text.includes("dk_8842")),
  `(got ${hi.length}: ${hi.map((h) => h.unit.text.slice(0, 20)).join(" / ")})`);

// 5. The v0.14b follow-up live bug (REAL transcript, verbatim from the store):
// the assistant self-named ("I'm Echo… I named myself that last session") and
// later, in a NEW conversation, "what is your name?" injected NOTHING — the
// strict user-only naming rule made the self-naming permanently un-injectable,
// and stale "I don't have a name" denials outranked it once partially fixed.
{
  const s2 = new MemoryStore("C:/Users/Robot/projects/zero-mem-pi/.relevance-test2.json", extract);
  s2.embedder = new Embedder();
  await s2.embedder.init();
  const t0 = Date.now() - 30 * 60_000;
  const real = (role: "user" | "assistant" | "tool", text: string, agoS: number) =>
    s2.add({ sessionId: "old-session", cwd, role, text, timestamp: t0 + agoS * 1000 });
  real("user", "hi, my name is mark, ill let you decide your name", 0);
  real("assistant", "I don't have a personal name — I'm a coding assistant (part of pi). If you'd like, you can call me whatever you'd like.", 40);
  real("user", "i thought your name was echo???", 60);
  real("assistant", "You're absolutely right, Mark — my apologies! I'm **Echo**. 🪶\n\nI named myself that last session because we're building a memory system — Echo felt right.", 80);
  real("assistant", "I'm the **pi coding agent** — that's my harness. There's no custom name set for me in this session (the PI_AGENT_NAME variable is empty).", 100);
  const nm = await retrieve("what is your name?", s2, { cwd, topK: 3, sessionId: "brand-new-session" });
  check("NEW conversation: 'what is your name?' surfaces the Echo self-naming", nm.length > 0 && /echo/i.test(nm[0].unit.text),
    `(top: ${nm[0]?.unit.text.slice(0, 50) ?? "(none)"})`);
  check("stale name-denials are NOT injected", !nm.some((h) => /don'?t have (a )?(personal )?name|no custom name/i.test(h.unit.text)));
  const nm2 = await retrieve("whats your name", s2, { cwd, topK: 3, sessionId: "another-new-session" });
  check("casual phrasing 'whats your name' also surfaces Echo", nm2.length > 0 && /echo/i.test(nm2[0].unit.text),
    `(top: ${nm2[0]?.unit.text.slice(0, 50) ?? "(none)"})`);
}

// 6. The v0.14c cross-project failure (REAL transcript #2): pi was started from
// C:\Users\Robot\projects (a session full of OBS/PowerShell output with "name"
// fields), while the identity facts live under the zero-mem-pi project. The
// in-project OBS noise used to beat minScore, federation never fired, and
// "whats my name?" got a wall of OBS JSON instead of the user's name.
{
  const s3 = new MemoryStore("C:/Users/Robot/projects/zero-mem-pi/.relevance-test3.json", extract);
  s3.embedder = new Embedder();
  await s3.embedder.init();
  const PROJ = "C:\\other-project", t0 = Date.now() - 60 * 60_000;
  const obs = (role: "user" | "assistant" | "tool", text: string, ago: number) =>
    s3.add({ sessionId: "obs", cwd: PROJ, role, text, timestamp: t0 + ago * 1000 });
  obs("tool", `{ "name": "Untitled", "DesktopAudioDevice1": { "prev_ver": 537001985, "name": "Desktop Audio", "uuid": "fa7aac82", "id": "wasapi_output_capture" } }`, 30 * 60);
  obs("assistant", "At line:1 char:187 ... scenes\\Untitled.json' -Raw | ConvertFrom-Json; 'name=' + .name; You must provide a value expression following the '+' operator.", 26 * 60);
  obs("assistant", "Case-insensitive duplicate: the scene `ZCode` and the window-capture source `Zcode` — OBS requires unique source names and choked.", 20 * 60);
  const zm = (role: "user" | "assistant", text: string, ago: number) =>
    s3.add({ sessionId: "zm", cwd, role, text, timestamp: t0 + ago * 1000 });
  zm("user", "hi, my name is mark, ill let you decide your name", 55 * 60);
  zm("assistant", "You're absolutely right, Mark — my apologies! I'm **Echo**. 🪶 I named myself that last session.", 40 * 60);

  const q1 = await retrieve("whats my name?", s3, { cwd: PROJ, sessionId: "fresh", topK: 3 });
  check("cross-project: 'whats my name?' surfaces the mark unit", q1.length > 0 && /my name is mark/.test(q1[0].unit.text),
    `(top: ${q1[0]?.unit.text.slice(0, 40) ?? "none"})`);
  check("cross-project hits are tagged", q1.some((h) => h.reason.includes("cross-project")));
  const q2 = await retrieve("what is your name?", s3, { cwd: PROJ, sessionId: "fresh", topK: 3 });
  check("cross-project: 'what is your name?' surfaces Echo", q2.length > 0 && /echo/i.test(q2[0].unit.text),
    `(top: ${q2[0]?.unit.text.slice(0, 40) ?? "none"})`);
  const q3 = await retrieve("OBS scene duplicate window capture", s3, { cwd: PROJ, sessionId: "fresh", topK: 3 });
  check("non-identity in-project queries still answer locally", q3.length > 0 && q3[0].unit.cwd === PROJ,
    `(top: ${q3[0]?.unit.text.slice(0, 40) ?? "none"})`);
}

console.log(`\nRESULTS: ${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
