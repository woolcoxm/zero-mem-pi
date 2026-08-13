/**
 * One-shot: migrate a legacy inline-embedding store.json to the v0.5
 * int8 sidecar format. Backs up the original first. Safe to re-run.
 */
import { MemoryStore, makeExtractor } from "./core.ts";
import { homedir } from "node:os";
import { join } from "node:path";
import { copyFileSync, existsSync, statSync } from "node:fs";

const p = join(homedir(), ".pi", "agent", "zero-mem", "store.json");
const bak = p.replace(/\.json$/i, ".json.bak");
const emb = p.replace(/\.json$/i, ".emb.bin");

const before = existsSync(p) ? statSync(p).size : 0;
console.log("store.json before:", (before / 1024).toFixed(1), "KB");

// Back up the original (only if no backup yet, so re-runs don't clobber it).
if (existsSync(p) && !existsSync(bak)) {
  copyFileSync(p, bak);
  console.log("backed up original →", bak.replace(/.*\.pi/, "~/.pi"));
}

const store = new MemoryStore(p, makeExtractor(null));
store.load();
const inline = store.units.filter((u) => Array.isArray(u.embedding) && u.embedding!.length).length;
const inBin = existsSync(emb);
console.log(`loaded ${store.units.length} units; inline embeddings: ${inline}; sidecar present: ${inBin}`);

await store.persist();

const afterJson = statSync(p).size;
const afterBin = existsSync(emb) ? statSync(emb).size : 0;
console.log("store.json after :", (afterJson / 1024).toFixed(1), "KB");
console.log("store.emb.bin    :", (afterBin / 1024).toFixed(1), "KB");
console.log("total after      :", ((afterJson + afterBin) / 1024).toFixed(1), "KB");
console.log("reduction        :", ((1 - (afterJson + afterBin) / before) * 100).toFixed(1), "%");
