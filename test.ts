/**
 * Zero-Mem v0.4 test — co-occurrence relational bridges.
 * Two things co-occur often → a strong "thread" between them → asking about one
 * now recalls the other's context, even with no direct mention.
 */
import { MemoryStore, Embedder, makeExtractor, retrieve } from "./core.ts";
import nlp from "compromise";

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

console.log("=".repeat(70));
console.log("ZERO-MEM v0.4 — co-occurrence relational bridges");
console.log("=".repeat(70));
await store.embedder.init();
store.ensureIndex();
console.log("co-occurrence edges recorded:",
  [...store.graph.cooc.entries()].map(([k, v]) => `${k.replace("\x00", "↔")}=${v}`).join(",  ") || "(none)");
console.log();

const q = "tell me about the ReactApp frontend";
console.log("QUERY:", q, "  [query entities:", extract(q).join(", ") + "]\n");

console.log("v0.3 — bridges OFF (only direct entity matches):");
const off = await retrieve(q, store, { cwd, topK: 5, useBridges: false });
console.log(off.map((h) => `  • [${h.unit.sessionName}] ${h.reason}:${h.score.toFixed(2)} — ${h.unit.text.slice(0, 52)}`).join("\n") || "  (no hits)");
console.log();

console.log("v0.4 — bridges ON (follows co-occurrence threads):");
const on = await retrieve(q, store, { cwd, topK: 5, useBridges: true });
console.log(on.map((h) => `  • [${h.unit.sessionName}] ${h.reason}:${h.score.toFixed(2)} — ${h.unit.text.slice(0, 52)}`).join("\n") || "  (no hits)");

console.log("\n" + "=".repeat(70));
console.log("Result: v0.4 surfaces the component-library note (s2) because TailwindUI");
console.log("co-occurs with ReactApp — a relational bridge — without ReactApp being");
console.log("mentioned there at all. v0.3 missed it entirely.");
console.log("=".repeat(70));
