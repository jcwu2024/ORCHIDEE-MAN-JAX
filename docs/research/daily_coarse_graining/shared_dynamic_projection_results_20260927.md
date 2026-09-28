# FP32 union dynamic-projection diagnostic

Date: 2026-09-27. This is an opt-in exact-formula execution experiment, not
production adoption, a scientific gradient, training or full-update throughput.
Default expert128 execution remains `dynamic_projection_execution="field"`.
The union option is implemented: its synthetic-component strict gradient gate
failed, while the actual row0 daily-loss/parameter-gradient comparison below
passes. No GPU job remains active. Component/lower-only runs were free; the
subsequently approved integrated diagnostic used a paid allocation.

## Source and implementation

`shared_decoder_expert_execution.py` now supports `"union"`: original
`_decoder_dynamic_tokens` materializes globally unique axis/local identities;
original `_field_dynamic_projection_cache` projects once through the same two
shared coordinate blocks; field-specific gathers restore the existing bank
order and retain each expert's family/process/carbon constants. Owners are
`shared_transformer.py::_decoder_dynamic_plan:810-900`,
`_decoder_dynamic_tokens:904-977`, `_field_dynamic_projection_cache:424-448`.

No attention/FFN, precision, weights/capacity, loss, physical process, data or
sampling change. GPP keeps its original decoder/carbon route; masks, bias and
undefined-local sentinel behavior and full output restoration are unchanged.
Native AD remains native; no custom backward or stop-gradient is introduced.

## Actual layout and input admission

Reuse immutable checkout `c599236bcf57b745e9bafed678298010f4c3b58f`, retained
update-0 checkpoint and prepared artifact742 micro4 **row0**, without Teacher
regeneration. CPU-only layout inspection completed in 30.501938s. There are
161 fields, 339 unique axis rows and 1,798 local rows including one sentinel,
versus 1,380 / 3,249 rows across repeated field banks. Width384/two-block union
tokens occupy 3,282,432 nominal bytes; union K/V 13,129,728 bytes (12.52MiB),
versus 28,440,576 bytes of all returned field banks. These are interfaces, not
live GPU peaks. Source-level projection dots are 1,288 field / 8 union.

The original eager source-reuse context assembler and token producer stopped
before decoder/physics/loss: capture 85.676307s, process 100.350529s. The finite
5,688,681-byte payload contains FP32 `(339,384)` and `(1798,384)` token banks,
original K/V parameters and field maps. Saved masks have 3,834/3,854 defined
states, 801/980 axis entries and 2/2 PFT entries. Those masks were not modified.

## Two measurement boundaries

Both routes use identical actual tokens, maps, FP32 K/V weights and cotangent
values: seed20260927, `fold_in` by output leaf, normal divided by square root
of leaf size. This is a **synthetic linear component cotangent**, not the
canonical scientific-loss adjoint. All 161 field banks are retained, including
empty banks. Neither boundary executes integrated decoder attention/physics.

The initial boundary returns all banks plus native parameter/token VJPs:
1,298 input/output leaves, including 1,288 bank/cotangent leaves. The scalar
correction consumes the same cotangent in `sum(vdot(bank,cotangent))`, packs
cotangent values into one input vector and returns scalar plus native parameter/
token gradients: 11 input/output leaves. It does not shorten gradients.

| Measurement | Field | Union |
| --- | ---: | ---: |
| Banks-return: lower / compile (s) | 3.699213 / 25.410116 | 3.638074 / 6.716895 |
| Banks-return: first execution / warm median (ms) | 69.3345 / 21.6110 | 132.1037 / 21.9766 |
| Banks-return: IR operations / native-AD dots | 25,826 / 3,684 | 18,806 / 24 |
| Banks-return: estimated live / temporary (decimal MB) | 349.252 / 281.064 | 108.910 / 40.722 |
| Scalar: lower / compile (s) | 5.675983 / 48.475710 | 5.514280 / 32.322661 |
| Scalar: first execution / warm median (ms) | 750.3176 / 9.48289 | 24.8703 / 6.73777 |
| Scalar: IR operations / dots, including scalar consumers | 42,749 / 6,260 | 34,649 / 2,420 |
| Scalar: estimated live / temporary (decimal MB) | 369.243 / 329.507 | 69.969 / 30.233 |

Warm repeats are seven, synchronized, fixed field-then-union order, not a
randomized performance study. Banks-return shows no warm gain; scalar medians
give **1.407422x component throughput**, with ranges 9.458-10.412ms field and
6.270-7.752ms union. Already-ready blocking traversal of the original 1,298
leaves measured 0.352539ms median, so blocking alone does not explain 22ms;
combined argument/result marshaling, dispatch and device work are not separately
attributed. Do not mix the two compile budgets/counts or extrapolate either
timing/memory boundary to integrated decoder, recurrent or complete updates.

## Strict numerical outcome

Unchanged `rtol=1e-5`, `atol=1e-6`; all values/gradients finite. Banks-return
outputs (7,110,144 coordinates) and token VJP (820,608) pass; projection-parameter
VJP fails **26/591,360**, maximum delta 5.245209e-6, relative L2 4.507057e-7.
Scalar value and token VJP pass; parameter gradient fails **27/591,360**,
maximum delta 5.722046e-6, relative L2 4.507584e-7. These failures are kernel
coordinates; original bias gradients pass. No tolerance was changed.

A read-only independent analytic FP64 reference checked only the scalar's
27 failed coordinates: `math.fsum(float(token[row,i])*float(cotangent[row,j]))`
over 4,629 field-axis/local products per coordinate, using saved actual tokens,
maps and packed cotangent. Field max/median absolute error is
1.302393e-6 / 2.282545e-7; union 1.905233e-6 / 1.106052e-6. Union is closer
at only 1/27. At these checked coordinates, the discrepancies are consistent
with FP32 accumulation differences rather than missing large derivative terms;
this is not all-gradient admission or a whole-graph/scientific reference.
The strict gate still fails. Small numerical differences are not a training
tool bug, and the analytic check does not authorize adoption.

## Validation, resources and artifacts

Five targeted local tests passed (115.76s): existing expert128/default checks,
union mixed masks/nonzero biases/sentinel full selected-field forward and
native parameter/context/state AD, plus a two-block component VJP test.
Assertions include nonzero key/local-token/context/state gradient norms and
zero undefined-state gradients. The scalar packed-boundary regression passed
separately (4.55s), with explicit FP32 test cotangents under X64 runtime.
These small selected-field fixtures are not all-field/full-loss acceptance.

Free gln01 `ibc13b02n04`, V100-SXM2-32GB GPU0; occupancy empty before each
GPU process, load0.06/0.09, affinity0-111, OMP4/OpenBLAS1. Capture producer cap
180s; each component lower/compile cap180s; process590s / outer600s. Capture,
banks and scalar process times 100.350529 / 55.234338 / 106.375227s imply about
**0.0728 one-GPU card-hours** of free wall exposure, excluding CPU inspection.
Canonical runtime capability evidence was reused; no environment rebuild.
Cache configured before helper/JAX/model compilation. Actual target-shell
dry-runs exercised CPU and scalar argument expansion; an old-Bash empty-array
launcher issue was repaired before CPU inspection, with no GPU computation.

Runner: `scripts/hpc/probe_shared_dynamic_projection.py` and
`run_shared_dynamic_projection.sh`. Small overlay lives at
`runtime/transfer/shared-dynamic-projection-20260927`; immutable checkout and
previous five neural-stage executable caches are untouched. Local and remote
diagnostics retain prefix `shared-dynamic-projection-20260927`: inspection/
capture JSON and logs, payload, both `compare` and `compare-scalar` JSON/logs,
field/union StableHLO and result pickles, packed cotangent, local error-analysis
and `compare-scalar.fp64-oracle.json`. No data or prior result was removed.

## Full neural-transpose lower/IR-only comparison

The authorized bounded comparison completed with `status="lowered_only"` in
176.794668s; no process remains active. It reuses the same immutable checkout,
update-0 checkpoint and artifact742 micro4 **row0**. All 64,884,481 parameters
(300 leaves) and the 3,854-coordinate continuous state are abstract inputs;
there is no parameter device upload or full-model execution. The paired wrapper
is `jit(native_vjp(checkpointed_jitted_neural_stage))`, with original raw masks
and native `float0` cotangents for integer/bool outputs. No physical updater or
scientific loss is run. Outputs, cotangents and all parameter/state gradient
leaf shapes/dtypes match exactly across the two routes: 322 neural/cotangent
leaves and 301 gradient leaves, 259,568,756 nominal gradient bytes. Both IR
main signatures have 462 arguments / 301 results, with 259,880,804 /
259,568,756 nominal bytes. Shape equality is not numerical equivalence.

| Paired lowered IR / timing | Field | Union |
| --- | ---: | ---: |
| Module operations | 218,689 | 140,541 |
| Module `dot_general` / `while` | 5,704 / 16 | 1,564 / 16 |
| Function definitions / main-reachable unique definitions | 855 / 855 | 834 / 834 |
| Reachable-function operations / dots / while | 218,688 / 5,704 / 16 | 140,540 / 1,564 / 16 |
| Main operations / dots / while | 4 / 0 / 0 | 4 / 0 / 0 |
| Largest outlined `neural_stage_0` operations / dots / while | 197,480 / 5,164 / 16 | 120,185 / 1,024 / 16 |
| Other main callee `neural_stage` operations | 2,284 | 1,551 |
| StableHLO text bytes | 27,832,051 | 19,715,362 |
| Output shape / lower (s) | 13.591814 / 71.028421 | 10.537754 / 46.511516 |
| Shape + lower / including save and inspection (s) | 84.620235 / 96.581176 | 57.049270 / 65.081206 |

The four-op main is only an outer wrapper calling `neural_stage` and
`neural_stage_0`; main-only counting would hide the reduction. Unique reachable
function definitions are counted once, not weighted by calls, loop iterations
or tensor sizes. Module operation count falls **35.7348%**, dots **72.5806%**,
and text **29.1631%** relative to the **new paired field route**. These are not
FLOPs, optimized backend instruction counts, peak memory or GPU speedups.
Field was lowered first, so timings are diagnostic rather than randomized
compiler-speed evidence. The graph reduction is material but not admission.

All 16 recorded while summaries match: same outlined owner, sampled operand
types and per-region operation/dot counts. Each condition has 3 operations;
body counts range 873-937, with no direct body dots. Helpers called by bodies
may contain dots; direct-region counts do not imply dot-free loops or unchanged
dynamic work. This is summary equality, not a complete textual body comparison.

Historical stage05 remains a separate comparator: 216,415 operations,
5,704 dots, 853 functions; its separately compiled pjit transpose has 2,671
arguments / 301 results, unlike the new outer VJP wrapper. Do not derive paired
percentages or backend-time predictions by mixing these boundaries. The saved
historical unit and five completed executable caches remain untouched.

The guard records **zero full-graph executable lookups, backend compiles,
forward executions and transpose executions**, including cache-hit prevention
during lowering. Initialization alone made 23 small executable lookups/backend
calls (largest result360 bytes), before the abstract comparison; these are not
neural-stage compilation. Persistent cache was configured before helper/JAX
initialization. Actual node `ibc13b02n04`, JAX0.4.38, visible GPU0 V100;
occupancy empty before launch, load0.02, affinity0-111, OMP4/OpenBLAS1.
Each shape/lower cap180s, process480s, outer500s; one visible GPU over the
176.795s process gives **0.04911 free wall-exposure card-hours**, not measured
GPU compute usage. No paid job, training update or environment rebuild.

Runner: `scripts/hpc/probe_shared_neural_transpose_lower.py` and
`run_shared_neural_transpose_lower.sh`; actual target-shell `bash -n` and
`--dry-run` passed. Two lightweight guard/masked-native-VJP lowering tests
passed in0.75s; they do not establish real-model numerical parity. Local and
remote artifacts under `outputs/diagnostics/` use prefix
`shared-neural-transpose-lower-20260927`: JSON, log and field/union HLO pickles.
The overlay is `runtime/transfer/shared-neural-transpose-lower-20260927`.

Default field stays unchanged; union stays opt-in and unadopted. The component
strict failure27 remains unchanged. Next, define comparison objects, budgets
and acceptance gates for bounded integrated native-gradient/canonical-loss
validation and compile/run timing, first reusing available references rather
than blindly repeating the timed-out baseline. This is an engineering decision,
not an added user-approval requirement within the already authorized free scope;
paid work retains existing approval rules. This lower-only round did not execute
or automatically start backend compilation, recurrent runs or training.
No tolerance change or production adoption follows from these results.

## Integrated actual daily check completed

The bounded check in `probe_shared_union_integrated.py` completed:
same artifact742 **row0**, update0, FP32 union route, checkpointed native
staged AD and public joint daily loss, including all parameter/input-state
gradients. New neural forward/transpose caps300/600s, whole process1000s,
outer1020s,24GiB and three warm repeats. The four unchanged physical/loss
units must hit the existing cache; a miss stops before backend compilation.
No unchanged timed-out field transpose is compiled.

The original serialized canonical field micro4 binary accepts dynamic
parameters and the complete sample pytree. Exact `c599236` source binds
`make_shared_joint_daily_objective` with the same scales/weights, then vmaps
the scalar objective and averages four rows. Repeating **every** prepared
row0 leaf four times (inputs/day inputs/targets/masks/weights/discrete state)
therefore supplies a scalar/full-parameter-gradient reference without a new
model graph. The harness verifies shapes/dtypes, entire treedef and serialized
input tree, forbids new graph lookups during deserialization/replay, and saves
the reference. Strict1e-5/1e-6 loss/all-parameter comparison and six meaningful
sample-mean auxiliary terms are recorded; summed four-row counts are not
blindly compared to one-row counts. This gives no state-gradient truth or
matched-workload timing baseline. All candidate results/timings are preserved
even if strict comparisons fail; the component27 failure remains unchanged.

Four local tests pass (1.28s), including actual tiny serialized dynamic-row
replay and full-sample mean-gradient equivalence. Undefined-name lint and
`git diff --check` pass. Both target-shell scripts pass `bash -n` and actual
free/Slurm `--dry-run`; GPU1 free and GPU3 Slurm bindings were preserved.
The fresh deployed tiny preflight passed in66.645216s on JAX0.4.38 before the
changed real graph. The in-process checkpointed staged synthetic gate passed
too; these gates are runtime plumbing evidence, not actual-model acceptance.

At11:35:53CST free gln01 GPU0/GPU1 remained occupied (81%/82% utilization),
so no free run launched. The user subsequently approved the paid fallback;
job **15418739** was submitted on2026-09-27T15:15:21 (scheduler time):
gnall, one V100,8 allocated CPUs,64GiB,25min, non-exclusive, same immutable
checkout and overlay. Worst-case additive charge
`(8*0.07+2.20)*25/60 = CNY1.15`. At11:39CST the only CPU-idle gnall node,
`ibc13b03n07`, had CPUAlloc0/CPULoad0.01 but AllocTRES GPU8; other nodes
were mixed/drained. No consistent wholly idle GPU-node candidate was proven,
so the proposal does **not** pin a node; Slurm selects from available resources.
The query-visible queue was empty despite allocated counters, another reason
not to claim verified GPU availability. No whole-node reservation is requested.
Executed submission after explicit resource/time/directory/command/cost approval:

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1
/rmprog/slurm/v22.05.7/bin/sbatch /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-union-integrated-20260927/slurm_shared_union_integrated.sh
```

The thin wrapper uses `srun` CPU binding and preserves the scheduler's single
`CUDA_VISIBLE_DEVICES`; the free launcher still restricts direct execution to
gln01. Log path is `runtime/outputs/diagnostics/shared-union-integrated-paid-%j.log`.
Job15418739 completed **0:0**,13min12s, on `ibc13b03n06`; no task remains
active. Actual launch15:31:51CST, load4.13/4.30/4.51, affinity
`0-2,11,56-58,67`, eight allocated CPUs and visible GPU0. No other GPU process
line was recorded at launch; node was not exclusively reserved. Report elapsed
708.350054s is the integrated runner, not the entire allocation/preflight.
One V100 for792s is **0.22card-hours**; additive elapsed-time estimate
`(8*0.07+2.20)*792/3600 = CNY0.6072`, not a billing statement. Approved
worst-case ceiling was CNY1.15.

### Numerical and physical outcome

`status="bounded_validity_passed"`. Repeated-row0 field binary loss is
0.5339762271443766; union loss0.53397628857042. Unchanged1e-5/1e-6 strict
comparison passes the scalar and **all64,884,481 parameter-gradient
coordinates**, with zero failures, maximum delta3.8743019104e-7 and relative
L2 difference8.7937337078e-7. Scalar delta6.1426043363e-8. All16 semantic
parameter groups also pass strict comparison. This is one actual scientific
daily-loss adjoint at the declared checkpoint/sample, not all-input admission.

All six sample-mean auxiliary comparisons pass with zero failures:

| Scalar | Maximum absolute difference |
| --- | ---: |
| process_day | 1.4518476699e-7 |
| process_labels | 2.8240351302e-7 |
| state_change_anchor | 7.9660210650e-9 |
| flux | 2.5064935683e-8 |
| stock | 9.2215154401e-8 |
| joint_daily_contribution | 6.1426043363e-8 |

All11 canonical hard counts are zero, including budgets, negative water/carbon,
nonfinite predictions/state/science, physical/science masks and discrete state.
The 300 parameter-gradient leaves and all161 neural-output leaves are finite.
Continuous-state gradient has3,854 coordinates,3,834 nonzero, norm
2,049,901.976637; structurally undefined input-state gradients are exactly zero.
Large finite state-gradient magnitude is descriptive, not accuracy evidence.
There is **no independent state-gradient reference**. Raw masks are Boolean,
8,538/39,006 defined; predicted/expected state masks have5,738/5,737 defined
of5,757 coordinates, with canonical mask-mismatch count zero. This does not
prove raw-output/full-mask parity against the saved binary, which does not
return them. The prior synthetic-component27-coordinate failure is unchanged;
the actual scalar's strict pass does not erase that separate failure.

### Compilation, persistence and timing limits

| Native unit | Lookup/cache seconds | Backend seconds | Cache hit |
| --- | ---: | ---: | --- |
| Union neural forward | 76.044101 | 74.816676 | No |
| Physical forward | 15.169597 | 0 | Yes |
| Loss forward | 0.992227 | 0 | Yes |
| Loss transpose | 0.181323 | 0 | Yes |
| Physical transpose | 2.971440 | 0 | Yes |
| Union neural transpose | 336.564711 | 331.985349 | No |

The final unit now completes within600s. The earlier field unit timed out at
600s and has no finite completed compile duration; this is not a measured
paired compile-speed ratio. Four unchanged physical/loss units were reused,
not recompiled. Read-only cache-file metadata confirms both new union neural
executables persisted (2,811,088 /9,525,707 bytes), alongside the four reused
units (12,884,710 bytes total): **six route executables,25,221,505 bytes**.
The old field neural-forward executable is retained as a seventh stage cache
file. No bulk hashes, cache replay, full baseline recompilation or new job
was performed during this result inspection.

Field binary deserialization44.451985s and one reference execution16.533138s
used **zero new graph executable lookups**. It is four repeated rows and
different AD/output scheduling, not a timing baseline for the single-row union.
Union cold first call579.769127s includes tracing, compilation/cache retrieval,
artifact persistence, dispatch and first execution. The synchronized subsequent
calls are **33.938836 /2.579293 /2.572464s**. Do not silently discard the first
or call the three repeats uniformly steady state. Source shows each call uses
the outer **non-jitted** native value-and-grad dispatcher, retaining all
parameters/state gradients plus many-leaf neural outputs/masks/auxiliary data.
The compiler observer covers only the cold call and named stage units; the
three repeats are outside it. Existing logs contain no repeat-specific compile
or profiling records, so the33.94s cause is **unknown**; additional helper
compilation, dispatch or device work cannot be separately attributed. No claim
of zero warm compilation or production execution overhead is supported.

Whole-process device statistics peak at11,275,374,080 live bytes (10.501GiB),
with peak allocator pool18,251,513,856 bytes (16.998GiB). They span reference
replay and all candidate stages, not isolated union-transpose peak memory;
the allocator pool is not actual live memory. Warm timings are not optimizer
updates, changing-sample throughput, recurrent performance or10x evidence.

JSON/runtime/log are retained locally under `outputs/diagnostics/`; full field
reference259,562,418bytes, union result260,063,579bytes and six stage HLOs stay
on Explore1000 under the same diagnostic prefix. No large result download was
needed. The log is `shared-union-integrated-paid-15418739.log`.

### Next bounded questions

Default field remains; production adoption, training and tolerance changes
were not made. First reuse the six persisted units in a cache-hit-only replay
with all compiler/cache events and per-call stages observed across every
repeat. Separate diagnostic host/many-leaf return costs from device stages;
retain full parameter/state derivatives and physical checks. Report cold replay,
first repeat and stable later calls separately before defining a production-
interface, unchanged-work complete-update benchmark; do not compare micro4
reference speed to row0. Removing diagnostics from the measured interface is
an execution-boundary change requiring bounded checks, not permission to drop
scientifically active outputs.

Before production adoption, obtain small independent state-gradient evidence
by reusing the saved **field** binary's
dynamic full-sample inputs for a small, predeclared set of central directional
finite differences in defined physical-state groups (row0 repeated four times),
with physical-scale-aware step convergence and all hard/mask checks. Compare
those slopes to the saved union state gradient's dot products. This requires
no default neural-transpose compile and provides bounded directional evidence,
**not** all3,854-coordinate exact gradient parity. Complete coordinate-wise
parity remains unestablished, not a newly imposed prerequisite; this is not a
renewed long-term gradient-infrastructure workstream. After the short replay clarifies this diagnostic's
execution costs, return to production microbatch/unchanged-work complete-update
comparison to answer training throughput. The saved union FP32 result can also
be a reference asset for a later explicitly scoped precision comparison; no
precision change or renewed broader-FP16 run was made here. These are proposed
next checks, not a new paid submission or automatic GPU execution.

## Bounded observed cache replay: cold cap reached

Authorized free gln01 launcher69488 started16:42:38CST on `ibc13b02n04`,
GPU0 empty, load0.06/0.03/0.05, affinity0-111; it ended124 with no active
process/GPU use remaining. `shared-union-replay-20260927.{json,log}` is
preserved remotely and locally. New small overlay leaves the paid evidence
unchanged. Two observer tests pass (0.95s), lint/diff checks and target-shell
syntax/argument dry-run pass. No field-binary replay or unchanged runtime
preflight was repeated. All compiler/cache lookups and backend entries were
observed; model-stage misses were forbidden before backend compilation.

The **100s cold-call cap**, not the480s whole-process or120s unit-cache cap,
stopped the process at149.720641s. All290 initialization lookups hit cache
(aggregate lookup3.767335s; intercepted dispatch0.158201s). Neural forward
was the only reached model unit: cache hit3.080414s,1798 result leaves /
7,952,651 nominal bytes; intercepted execution submission17.523577s.
**Zero backend calls** occurred, including tiny helpers. Physical/loss/reverse
lookups and every completed full result, warm call, cProfile and JAX trace
were not reached. The preserved event sums account for20.603991s of the100s
call; approximately79.396s remain unattributed to these two intervals. No
per-event absolute timestamps separate neural frontend tracing from work
before the next transition lookup. Initialization is roughly49.72s before the
call cap, inferred from total elapsed, not a independently timed phase.

The17.52s interception spans host argument/result handling and asynchronous
submission, potentially blocking device work; it is **not GPU execution time**.
JSON flushes on lookup entry/exit add observer overhead. Python dispatch hooks
miss ready C++ jit fast paths, as the unit test demonstrates; planned profiler
tracing was never reached. Neither pure Python/dispatch cost nor the original
33.94s/2.57s causes can be attributed from this partial run. No actual-model
instrumentation parity result exists. Earlier paid strict parity remains valid;
this timeout does not invalidate cache persistence or numerical evidence.
Runner149.721s corresponds to about0.04159 free one-GPU wall-exposure
card-hours, not kernel usage. No retry, cap extension or paid submission.

The cold budget was underestimated: the original579.769s first call minus
406.802s measured neural backend compilation already leaves about172.967s
for frontend tracing, cache loads, persistence, dispatch and first execution.
The2.57s late repeat was not a basis for a100s cold-retrace cap. The remaining
questions should not trigger another generic instrumented replay loop.

### Concrete next production-interface comparison

Reuse artifact742's **original four rows**, update0 checkpoint, prepared
context and existing admitted field/combined serialized micro4 executables
and saved results. Reuse exact source owners:
`shared_recurrent_training.py::make_shared_joint_daily_objective:243-277`,
`shared_daily_microbatch.py::make_microbatch_objective:66-89`, and
`compile_microbatch_evaluator:92-103`; existing integration pattern is
`probe_shared_expert_real_daily.py::gpu_run:439-478`. Bind the opt-in union
adapter with source-reuse backbone, but do not change loss, sample count,
weights, precision or architecture. The existing production evaluator is
`jit(value_and_grad(mean(vmap(daily_objective)), has_aux=True))`: scalar loss,
original auxiliary/hard counts and all300 parameter-gradient leaves. Omit
only additional diagnostic raw outputs/masks and state-reporting returns, not
active neural computations or physical checks. This matches the saved field
binary's work/output contract; do not time row0 full-input-state AD against
micro4 parameter-only AD.

This **does require a new union micro4 native graph/lower/backend compile**;
the six single-row staged cache units are not a compatible substitute. No new
field gradient compile is needed. Previous combined micro4 lower115.308s and
backend707.667s inform a proposed lower180s/backend900s/whole1500s ceiling,
including reference deserialization, first execution100s, bounded strict
comparisons and three interleaved matched warm repeats after explicit warmup.
Save the executable/full result before comparison and stop on resource or
numerical gates. These are proposed finite time-to-measurement budgets, not
predicted union timings or permission for a new free/paid launch. Check24GiB
device admission; retain cold/startup cost separately from same-work warm
timing. Reuse the accepted deployed runtime evidence where unchanged.

Keep the model fully differentiable in input state and all parameters: no
stop-gradient or shortened recurrent path. Production training's existing
parameter-only returned interface is the timing contract, not a removal of
state-derivative capability. Retain the saved full native row0 state-gradient
result as a separate admission boundary. Before production adoption, a small
field-binary central directional check can perturb every required row0 sample
leaf consistently, preserve masks/physical validity and verify step convergence
against saved union state-gradient dot products. It does not provide all-
coordinate state parity and is not the main throughput workstream. The daily
micro4 benchmark precedes any unchanged daily32+T7+T30 full-update measurement;
no new recurrent compilation, training, precision change or paid job is part
of this closeout.
