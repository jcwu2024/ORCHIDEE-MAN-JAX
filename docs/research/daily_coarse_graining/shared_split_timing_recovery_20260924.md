# Full-batch timing timeout and cache-observed recovery

Free launcher11143 exited124 at180min, completed_updates0. It did not produce
a full-update time, sample-throughput measurement or actual full-batch peak.
Original report still says running; the shell exit log establishes termination.
Report/log: runtime/outputs/diagnostics/shared-split-caed647-full-timing.*.

Completed compile stages:

| Component | Compile path seconds | Compiled live bytes |
| --- | ---: | ---: |
| daily micro4 |2849.259|11256930436|
| T7 |5442.468|1971235093|
| T30 |unfinished when timeout interrupted it|unknown|

T30 began8569.9s. Fresh initialization/checkpoint survives, but no optimizer
update occurred. The predicted180min envelope was insufficient. Do not describe
this as training failure, OOM, failed AD, or an observed throughput result.

## Why existing admission evidence did not imply cache reuse

Read-only Luna/high comparison identifies different bootstrap artifact sources:
recurrent admission819/982 versus learning742/275. Both use the same recurrent
objective/native AD, architecture, scales and loss. `load_shared_runtime` builds
the retained template from the first loaded artifact, and `make_canonical_step`
captures it. This is a concrete possible captured-constant difference, not a
demonstrated compiler-key cause. Runtime sample values alone do not require new
compilation. The learning runner holds its bootstrap/head fixed across updates.

The server cache contains new daily and T7 entries created at13:38 and15:07.
Presence is not proof they load for the next run. No cache files were deleted.

## Recovery execution

Code1c70fa4 adds a standalone compiler observer around the unchanged
`compile_gradient_evaluators` and allows its module in the existing shell.
Model, inputs, initialization, loss, update order and mathematical graph are
unchanged. The observer records actual compiler-boundary cache hits/backend
calls, without bulk data hashing. Daily/T7 backend recompilation is blocked if
their completed entries cannot be reused; T30 may compile. The hook is restored
before optimizer updates and on exceptions. Reports use new names, preserving
the failed run and its checkpoint.

Three local tests pass, checking hit/miss behavior and hook restoration.
Actual container JAX0.4.38 test on test02 passes: tiny cold compile followed by
clear_caches and persistent hit, identical outputs; no Teacher read. Target-shell
dry-run passes. Source checkout was fast-forwarded while idle; unchanged model
evidence reused rather than rerunning physical/AD admission.

Free GPU1 launcher21040 started20:29:31CST,240min ceiling; GPU0/1 empty and host
load0.00/0.01/0.05 at launch. No paid allocation. Report/log:
runtime/outputs/diagnostics/shared-split-1c70fa4-full-timing.{json,log}; separate
compiler evidence:shared-split-1c70fa4-full-timing.cache.json.
Command: runtime/transfer/launch_shared_split_timing_recovery.sh, unchanged
shared_split_learning_v1.json with --technical-updates2 and passed recurrent
resume1 admission. First two planned artifacts remain742/275.

Next collect actual hit evidence, T30 completion, first/second full update
component timings and device memory. Then calculate exposures/hour and500-update
budget. On a daily/T7 miss inspect the observed graph difference before another
cold retry. Do not assert speed or persistent cache success before the report.
