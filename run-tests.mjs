/**
 * Zero-Mem test runner — every test-*.ts file, sequentially, one command.
 * Usage: npm test   (or: node --experimental-strip-types run-tests.mjs [name-filter])
 */
import { spawnSync } from "node:child_process";
import { readdirSync } from "node:fs";

const filter = process.argv[2] ?? "";
const files = readdirSync(new URL(".", import.meta.url))
  .filter((f) => /^test-.*\.ts$/.test(f) || f === "test.ts")
  .sort();

let pass = 0, fail = 0;
for (const f of files) {
  if (filter && !f.includes(filter)) continue;
  const r = spawnSync(process.execPath, ["--experimental-strip-types", f], { stdio: "inherit" });
  if (r.status === 0) { pass++; console.log(`── ${f} OK`); }
  else { fail++; console.log(`── ${f} FAILED (exit ${r.status})`); }
}
console.log(`\n${pass} test file(s) passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
