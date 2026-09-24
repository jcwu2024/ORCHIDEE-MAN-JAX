# Full-batch timing timeout and cache-observed recovery

## Completed recovery: 22:19 CST September 24

Free launcher21040 exited0; report status passed, completed_updates2,
elapsed6553.07s (109.22min). Both changing-data full-network updates completed
and update2 checkpoint was saved. This supersedes the RUNNING snapshot below.
Raw JSON/log/cache evidence is also copied to local outputs/ with the unchanged
shared-split-1c70fa4-full-timing stem. No new paid job was submitted.

| Measurement | First update | Second update, different TRAIN artifact |
| --- | ---: | ---: |
| daily32, micro4 |45.942s|4.538s|
| T7 |2.257s|2.014s|
| T30 |8.424s|8.140s|
| optimizer |2.851s|0.025s|
| full update wall |66.423s|16.473s|

Second reported internal total14.742s versus wall16.473s leaves1.731s outside
that timer; do not label the entire difference IO without profiling. One warm
changing-data observation is not a long-run average. Device peak actual usage
is12323709184bytes (11.477GiB); the25.391GiB pool is reservation, not usage.

Daily/T7 both hit persistent cache with no backend compilation. T30 performed
one backend compilation; whole T30 trace/lower/compile stage5304.27s (88.40min),
observer compiler boundary4817.56s. The XLA55m6s alarm covers a narrower stage,
not the full T30 startup. T30 cross-process reuse is still unmeasured.

At the single warm wall measurement, arithmetic throughput is218.5updates/hour,
6993daily draws/hour plus218.5windows/hour at each horizon. The15079combined
day exposures/hour count repeated dates and recurrent computation, not unique
Teacher coverage. Five hundred updates extrapolate to2.29hours of update work
only; startup, fixed science panels, checkpoint IO and variability are extra.
This is not a paid wall-time estimate or a final training budget.

The result establishes bounded full-batch trainability and measured speed,
not scientific accuracy or sustained training acceptance. Fixed science monitors
were deliberately skipped; differing batch losses do not measure learning.
Earlier daily/recurrent physical and native-AD admission evidence remains the
authority for those checks. Next prepare the declared fresh500 learning budget,
including monitor/startup costs, using this unchanged architecture. Do not
continue the technical checkpoint as formal training or repeat cold compilation
solely for another timing sample. Any subsequent throughput revision should
address compilation and execution together.

## Previous attempt and recovery chronology

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

## Observed cache reuse at 21:50 CST

Launcher21040 remains active after1h21m; no optimizer update yet. Daily and
T7 explicitly report cache_hit with zero backend_compile calls. Whole
trace/lower/load stages took347.6s and572.0s, respectively, versus the prior
cold stages2849.3s and5442.5s. The narrower observed compiler boundaries took
46.43s and156.84s; these must not be presented as whole startup times.

T30 has only cache_lookup followed by an XLA slow-compilation message. Its
lookup-time zero-call snapshot does not establish a hit. T30 completion and
full-update throughput remain pending. This establishes reuse for the unchanged
daily/T7 timing graphs only, not compatibility across different runners or
future graph changes. The earlier budget assumed reuse before verifying it;
future budgets must separate observed cold work from demonstrated warm startup.

Next collect T30 completion, first/second full update
component timings and device memory. Then calculate exposures/hour and500-update
budget. On a daily/T7 miss inspect the observed graph difference before another
cold retry. Do not assert speed or persistent cache success before the report.
