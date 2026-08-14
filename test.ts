/**
 * Zero-Mem v0.4 test — co-occurrence relational bridges.
 * Two things co-occur often → a strong "thread" between them → asking about one
 * now recalls the other's context, even with no direct mention.
 * v0.13: real assertions (this used to be a print-only demo that exited 0 even
 * if bridges broke).
 */
import { MemoryStore, Embedder, makeExtractor, retrieve } from "./core.ts";
import nlp from "compromise";

let pass = 0, fail = 0;
const check = (name: string, cond: boolean, extra = "") => { if (cond) { pass++; console.log(`  \u2713 ${name} ${extra}`); } else { fail++; console.log(`  \u2717 ${name} ${extra}`); } };

const extract = makeExtractor(nlp);
const store = new MemoryStore("/tmp/zeromem-test.json", extract);
store.embedder = new Embedder();
const cwd = "C:/Users/Robot/projects/myapp";
const now = Date.now();
const add = (sessionId: string, sessionName: string, role: "user" | "assistant", text: string, agoMin: number) =>
  store.add({ sessionId, sessionName, cwd, role, text, timestamp: now - agoMin * 60_000 });

// ReactApp + TailwindUI appear together (strong co-occurrence).
add("s1", "frontend", "assistant", "The frontend is built with ReactApp and styled using `TailwindUI`.", 60);
// TailwindUI + ComponentLib appear together (so ComponentLib is tied to ReactApp via TailwindUI).
add("s2", "component-lib", "assistant", "Our shared `ComponentLib` is written entirely on top of `TailwindUI`.", 50);
// Unrelated distractor (no thread to ReactApp).
add("s3", "auth", "assistant", "Authentication uses JWT tokens stored in http-only cookies.", 40);

console.log("ZERO-MEM v0.4 — co-occurrence relational bridges");
await store.embedder.init();
store.ensureIndex();

const q = "tell me about the ReactApp frontend";
// v0.13: pure retrieval (no graph thread): bridges OFF *and* closure OFF —
// s2 has zero lexical/semantic overlap with the query, so only the graph
// (co-occurrence bridge + shared-entity closure) can surface it.
const off = await retrieve(q, store, { cwd, topK: 5, useBridges: false, useClosure: false });
const on = await retrieve(q, store, { cwd, topK: 5, useBridges: true });

console.log("bridges OFF:", off.map((h) => `[${h.unit.sessionName}] ${h.reason}:${h.score.toFixed(2)}`).join(", ") || "(no hits)");
console.log("bridges ON :", on.map((h) => `[${h.unit.sessionName}] ${h.reason}:${h.score.toFixed(2)}`).join(", ") || "(no hits)");

check("pure retrieval (bridges+closure OFF) does not surface the component-lib note (s2)", !off.some((h) => h.unit.sessionId === "s2"));
check("graph thread (bridges+closures ON) surfaces the component-lib note (s2)", on.some((h) => h.unit.sessionId === "s2"));
check("auth distractor (s3) never surfaces", !on.some((h) => h.unit.sessionId === "s3") && !off.some((h) => h.unit.sessionId === "s3"));

console.log(`\nRESULTS: ${pass} passed, ${fail} failed`);
if (fail) process.exit(1);
