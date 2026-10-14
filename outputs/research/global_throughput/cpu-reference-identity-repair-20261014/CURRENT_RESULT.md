# CPU reference identity repair, 2026-10-14

The bounded repair passes the real saved-reference preparation chain on the
accepted target CPU runtime. The historical failure is now reproduced and
localized: `stage_identity` omitted `metadata.git_head`, so the scalar reuse
guard compared the actual execution commit against `None`. The other three
identity comparisons already matched. No physical model was lowered,
compiled or executed, and scientific acceptance remains false.

## Before and after

Both diagnostics ran on `cntest`, `ibtest02`, one task and **one actual CPU**,
affinity `[0]`, with a 300-second limit. Canonical CPU Python initialized
JAX/jaxlib **0.4.38**, one CPU device and x64. The original FULL64 input reader,
screening, actual metadata producer, stage identity producer and real GPU/CPU
`load_reference` paths were used; no producer or reference loader was mocked.

| Evidence | Before | After |
| --- | --- | --- |
| Free job | 15855041 | 15855062 |
| Diagnostic exit/status | 1 / reference_preparation_failed | 0 / reference_preparation_passed |
| Scheduler parent | COMPLETED 0:0, 65s | COMPLETED 0:0, 23s |
| GPU reference | Loaded | Loaded, 760 leaves |
| CPU scalar reference | Original provenance ValueError | Loaded, 760 leaves |
| Physical runner/lower/compile/run | None | None |

The before shell deliberately accepts the expected diagnostic exit1; its
COMPLETED scheduler parent does not mean reference preparation passed.
These parent durations describe restricted allocations and are not model
throughput samples, cold compile times or a speedup comparison.

All values are in the preserved reports and `summary.json`:

| Guard comparison | Candidate | Before stage | After stage |
| --- | --- | --- | --- |
| baseline_ref | 133e35aec7b83b9d79888c46092f64d518e97299 | Same | Same |
| certified_science_sha256 / science_sha256 | 11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5 | Same | Same |
| gpu_reference_sha256 / numerical_reference_sha256 | a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6 | Same | Same |
| execution_commit / runtime.git_head | 60dccbddde10be4295aab7e7fe55fa7864d3c3f3 | null | 60dccbddde10be4295aab7e7fe55fa7864d3c3f3 |

## Minimal shared repair and isolation

`benchmark_global_batch.stage_identity` gains an explicit `include_git_head`
option, default false so existing generic stage contracts retain their schema.
Production server preparation always opts in, copying the actual metadata
producer's value. No field is supplied by a test or consumer to conceal the
producer omission, and every scalar/reference guard remains unchanged.

The existing production screening/identity/reference-load block was extracted
into `benchmark_global_server_cpu.prepare_references`. The normal benchmark
calls it before its existing StageStore/runner construction. The diagnostic
entry calls that same function after the same `read_prepared` and `metadata`,
then returns immediately. It has no physical runner creation or model execution
endpoint. Normal CPU48 allocation guards remain in the production entry;
the diagnostic separately requires actual cntest one-CPU receipt and does not
claim those 48-CPU gates. The full science arguments are in each `actual-argv.txt`;
budget1 is disclosed, and original FULL64 inputs, config, reference roles,
five-repeat declaration, native precision and tolerances are retained.

Before uses the exact historical `benchmark_global_batch.py` bytes SHA
`bc8ce1d62f6947228c8c8438344987345bf18095f307c90a2f45dc4655c6891e`
plus only the shared extraction/compact observation and diagnostic entry.
After changes the producer option and its production call. Exact before/after
overlay hashes are in `overlay-provenance.json` and each run's `overlay-hashes.txt`.

Remote staging is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/diagnostics/global-cpu-reference-identity-20261014`,
a historical detached 60dcc… checkout sharing the preserved bare Git objects.
It truthfully declares uncommitted harness overlays; it is not a new production
pin. The original production checkout remained clean at 60dcc… with physical
tree `75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba`; all 104 physical files and
historical scientific bytes are unchanged. Research carbon changes stay separate.

The original admission bytes, SHA
`f71eb5db4e638615ef5e06be5783d193bbfc1bd7563dd30c2424e04b40c14a8d`,
are retained as `original-admission.json`. The explicitly diagnostic derivative,
SHA `41f59ab72cd620cd892398ae3b069d1867b6b7d72cd98dede40dbf93c0cdde51`,
changes only `candidate_identity.execution_code_dir` to the actual staging path.
Original source audit, old job identity, scalar data/manifest/report/checkpoint
and every scientific field remain identical; no new audit was generated and
validators were not relaxed. Candidate job IDs are the real free allocations.

## Checks and remaining limits

Focused producer, guard-rejection and affected shared-interface tests passed:
**55 passed**, with one existing pytest configuration warning, in 10.01s.
The exact three-file command is recorded in `test-results.txt`. Producer checks
use actual metadata/git HEAD. The negative loader unit uses synthetic arrays,
a synthetic science digest, declared accepted runtime fields and a stub of the
immutable-admission validator solely to isolate the real loader commit guard;
it first accepts the correct produced identity, then rejects a wrong candidate
commit. These local unit fixtures do not establish target-runtime acceptance;
the separate real target before/after chain provides that evidence.
The after target loads the exact old scalar checkpoint
`1cddeaf80c2fc36f173835a5ff1bd4d8d520b144d4c96289ef6fe86dde898dfa`
and accepted GPU reference SHA shown above, preserving 64 points and the two
seven-day blocks. The input SHA remains
`be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7`.

This closes the preparation identity fault only. Actual CPU48 admission of a
future revised executable, physical lower/compile/run, numerical candidate
gates, five warm samples and a same-workload CPU/GPU throughput comparison are
unverified. Prior scalar 61/64 GPU passes, three finite failing points/four
comparisons, sealed-reset-proof-only limitation and scientific false persist.
No scalar generation, GPU run, environment change, paid submission/proposal,
production update or automatic retry occurred. The paid v4 attempt and all old
paid grants remain closed; further model execution requires a new dispatch.

Evidence is under this directory, mirrored remotely at
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu-reference-identity-repair-20261014`:
`before/` and `after/` contain full reports, original stdout/stderr, actual argv,
source/overlay identities and actual free allocation records; `replay-terminal.sacct.raw`
preserves parent/step terminal rows. `summary.json`, `overlay-provenance.json`
and operation receipts provide compact review entry points without saved array
copies. Before report SHA is
`8f18494d4c731a527937e800d0b1eb4f68562ee32de9bd7dde0370ba3bddb0e7`;
after report SHA is
`2d28e1e3c83fd9728816254d9e45a32f3776a57e878abb084bf8ff4809384d20`.
