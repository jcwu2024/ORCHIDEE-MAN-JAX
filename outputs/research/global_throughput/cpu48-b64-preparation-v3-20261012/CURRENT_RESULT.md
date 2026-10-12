# CPU48 / FULL64 B64 v3 preparation result

Status: preparation complete; no production deployment or scheduler submission.

The immutable execution ref `codex/global-cpu48-b64-v3` resolves to
`03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8`, based on
`b1b5cf0835da298bddb388e61d754feeb6c14cf2`. The pin manifest and verified
bundle are in this directory. Bundle SHA-256 is
`940a6f7e2aa6bf545661ef1c815cbd0156fa92b62b18452cbad7398d7c49d47c`.
The execution tree changes only the six declared harness/test paths; all 104
`jax_orchidee` files match the base tree. Certified science identity remains
`11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`, and
the actual source identity is `b33450cae9cd8c9b2d5f7c6075fe270f461887edc09c77417c5e820a5c18fab8`.

Local source admission passed in the clean detached verification checkout.
The focused preparation suite passed 15/15. The exact ten previously failing
capacity protocol/launcher cases passed 10/10 in a separate pinned rerun. A
different additional selector of four AOT interface and six source-certificate
cases also passed 10/10; it is supplemental coverage and does not substitute
for the original ten. Ruff and Bash syntax checks passed. The remote canonical
environment does not provide pytest, and no package installation was attempted.

On `test01`, copies of the pinned launcher, coordinator, CPU runner and its
resource helper were staged under the preparation output namespace. Their
hashes were checked against the local pinned copies. Launcher expansion and
coordinator scalar/B64 runner dry-runs passed. The resulting CLI contract names
`cnall`, one node/task, 48 CPUs, zero GPUs, 5400 seconds, no memory reservation,
and cost ceilings of CNY 3.60 primary / CNY 5.04 conservative. Both runner
results reported `model_execution=false` and `JAX_imported=false`. The target
v3 production checkout path is still absent. Raw target stdout, commands and
read-only inspections are retained in `target-launcher.stdout.txt`,
`target-coordinator.stdout.json` and `TARGET_SMOKE.json`. The exact unique
staging directory was inventoried, removed and checked absent. Focused pytest
runs have now been repeated on the immutable pin with raw stdout, stderr, exit
codes, runtime metadata, and file hashes recorded in
`regression-recheck-20261013/REGRESSION_RECEIPT.md` and its adjacent log files.
These remain local regression checks, not target-runtime or scientific
acceptance evidence.

This evidence supports proposal review only. It does not establish CPU
throughput, numerical acceptance, CPU/GPU speed ratio, billed cost, or
production runtime acceptance. The GPU B64 record is the same-workload
comparator; B1024 is a normalized reference only. The execution thread's
bounded artifact review passed; see [PREPARATION_ACCEPTANCE.md](PREPARATION_ACCEPTANCE.md).
Supervisor review and fresh single-job approval are the next decision.
