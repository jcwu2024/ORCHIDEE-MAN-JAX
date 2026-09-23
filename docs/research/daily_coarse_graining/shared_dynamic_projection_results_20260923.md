# Dynamic projection reuse: partial results and memory regression

## Latest: field-local daily result collected

Free wrapper103428 completed successfully. Immutable f220013, daily32/micro4:
warm6.80851/6.62601/6.60759s (median6.62601s); lowering+compile2958.44s,
total3188.76s. All75,578,497 gradient coordinates pass against saved reference,
maximum absolute1.490116e-7, relativeL2 5.50415e-7; all hard counters zero.
Loss0.298775508397 vs reference0.298775521382.

Compiled-live21.972GiB; actual cumulative allocator peak23.405GiB. The25.391GiB
allocator pool is reserved capacity, not actual live use. Field-local reuse
removes the previous27.214GiB rejection but does not beat the original
gather_reuse21.015GiB compiled footprint. Daily time is about15% below the
preceding new-model7.77s context; nodes differ, so not a matched speedup claim.
Revised T7/T30 and full joint-update parity remain unmeasured. No training
updates were retained and no scientific acceptance follows from this check.

The user's historical comparison is correct: old109,464,375-parameter model
used64 daily samples/micro8 in median6.53646s, not32. Its measured full dual
update was7.22209s. Evidence:`outputs/mixed-learning-dual-paid-15230086.json`.
The6.53646s includes daily-only optimizer/checks (underlying daily branch
median6.42775s); the new6.62601s is daily gradient accumulation only.
New daily throughput32/6.62601=4.83 samples/s vs old64/6.53646=9.79 samples/s,
about49.3% of the historical throughput. Different architectures/losses/hardware
allocations prevent interpreting this as a matched science or efficiency
ablation; they do not justify ignoring the gap. New-model training stays on
hold while execution grouping and actual device costs are investigated.

Raw completed evidence:`outputs/shared-dynamic-local-v2-daily.json` and remote
`runtime/outputs/diagnostics/shared-dynamic-local-v2-daily.json/.log`.
Earlier running/pending statements below are historical launch snapshots.

Source cfc518d, free gln01 wrappers25756/25757 have exited. Neither is still
running. No paid charge and no retained training updates. Do not adopt this
execution candidate or proceed to the joint replay on these results.

| Component | Result | Compiled live GiB | Warm seconds |
| --- | --- | ---: | ---: |
| daily micro4 | resource_limited, exit2; not executed |27.214| unavailable |
| T7 | full gradient/value/hard-count parity passed |3.419|2.254|
| T30 | timeout during compilation, exit124 | unavailable | unavailable |

The preceding gather_reuse daily micro4 was21.015GiB. The candidate's increase
is in temporary storage (28,614,351,648bytes), not weights/arguments. The24GiB
operating envelope is unchanged. This is a compiler estimate, not an observed
OOM or actual27GiB allocation. T7 previous compiled-live was3.283GiB.

T7 checked all75,578,497 gradient coordinates: zero failures, maximum absolute
difference3.7253e-8, relativeL2 3.1352e-7; all physical hard counters zero.
Warm replays2.459/2.248/2.254s; allocator peak4.278GiB. Prior paid T7~2.610s
is different-node context, not a matched-allocation speedup proof.

Daily lowering+compilation2417.43s; total2606.57s. T7 lowering+compilation
4247.48s; its gradient and exact executable were saved before T30 started at
elapsed4473.53s. The120min process limit interrupted T30 compilation. Its JSON
still says running/compile_started because SIGTERM did not execute a final
Python report write; log exit124 and absent wrapper process establish timeout.
Retain`T7.pkl` and`T7.executable.pkl`; no need to rebuild accepted T7 evidence.

Raw evidence: runtime diagnostics`shared-dynamic-reuse-daily.json/.log` and
`shared-dynamic-reuse-recurrent.json/.log`, local JSON copies under`outputs/`.

## Next bounded work

Read the existing compiled daily candidate and previous micro executables to
inspect memory shapes without retracing, model execution or backend compilation.
In parallel, inspect native AD residuals around coordinate-chunk decoding on a
small representative model. Establish a specific mechanism before patching.
Preserve network capacity, all potentially active outputs and gradient semantics.
Hold T30 and the joint replay until daily memory is resolved. No new paid job.

## Targeted revision in progress

Native reverse-JAXPR inspection finds a concrete amplification mechanism:
each field's coordinate-map transpose returns dense cotangents for the entire
global339-axis/1798-local projected banks, despite consuming a small subset.
The summed field/block/KV cotangent surface is
161*2*2*(339+1798)*768*4bytes=3.94GiB per sample. This is a static surface count,
not measured simultaneous live memory; scheduling/fusion can change allocation.

The revision gathers each field's unique tokens before K/V projection, then
reuses the compact field-local projected bank inside its chunk map. Across
fields these banks contain1380axis+3249local identities, compared with41147
coordinate-token occurrences. Projection reuse still eliminates about88.7%
of these repeated token projections while avoiding whole-global-bank
cotangents inside each field map. Capacity, masks, weights, precision, losses
and normalization semantics are unchanged; no custom reverse or remat added.

Before another real-model compile: verify focused numerical parity and inspect
bounded compiled reverse memory with production width/microbatch shapes. Then
use one free GPU for daily micro4 only, keeping the24GiB gate and90min ceiling.
Do not repeat T7/T30 before daily numerical/memory/time evidence warrants it.
Rejected expensive compiles will also preserve a diagnostic executable; such
files have no passed component record and cannot enter joint replay.

SSH authentication temporarily stalled, then recovered. The failed candidate's
actual optimized GPU HLO was recovered directly from existing persistent cache
`jit_objective-be77c9c96552514b98b02d997dd6e74c5f527468abce827bd40459f89902c272-cache`
without model tracing/execution or backend compilation. Native JAX0.4.38 lacks
the newer optional buffer-assignment export; the compressed HLO survived this
inspection API exception and was analyzed on free test02 without reloading GPU.

Actual HLO contains432 scatter instructions for each of[1798,4,768] and
[339,4,768] projected-bank shapes, with583/460 adds and405 copies per bank shape.
This corroborates dense global-bank cotangent work surviving GPU compilation.
Counts include fused computation bodies; they are NOT simultaneous allocation
counts or proof that this accounts for all6.20GiB of regression. Largest logical
arrays also include a pre-existing source-family broadcast/reduction; do not
change that separate path in this candidate or equate fusion shapes with buffers.
Local compact summary:`outputs/shared-dynamic-hlo-summary.json`; remote HLO:
`runtime/outputs/diagnostics/shared-cache-memory-20260923/`. Actual revised GPU
memory/runtime remain pending.

Revised field-local token construction and projection passed the focused local
forward/static pair (2tests,56.92s) and parameter/state gradient pair
(2tests,167.42s), including nonzero K/V biases and all-PFT-inactive state.
The executable persistence/update plumbing3tests also passed2.63s. No tolerances
were changed. `audit_shared_projection_ad.py` records native reverse shapes;
optional bounded compilation separates outer microbatch vmap from queries
within a day and must not be reported as full-model GPU memory acceptance.

## Deployed field-local fragment and single-day launch

Model candidate checkout`runtime/checkouts/shared-dynamic-local-v2` is immutable
f220013852d1ad7777a154090c360b9ef9cdc920. It shares Git objects with
`shared-dynamic-reuse-v1`, which in turn depends on`shared-staged-large-v1` and
its pre-existing alternates; preserve that dependency chain while active.

The deployed JAX0.4.38 CUDA runtime check passed56.62s. The first fragment
invocation stopped before compilation because canonical metadata fixtures were
local-only. The standalone audit now accepts`--bounded-only`, reusing already
recorded local layout evidence and requiring no dataset metadata. This helper
was transferred separately; the candidate model checkout was not mutated.
Actual shell daily argument expansion and launch shell syntax passed.

Free V100 bounded compilation, width768, outer micro4,64queries,4fields,
2chunks,1represented block:

| Executable | Compile seconds | Compiler temporary bytes |
| --- | ---: | ---: |
| global projected banks |18.251|327804888|
| field-local banks |2.688|101676920|

Thus this fragment's compiler temporary estimate falls312.62->96.97MiB,
about69.0%. It is neither full-model peak memory nor a runtime speed benchmark.
The two shapes also have different token-input/output storage; temporary bytes
are reported separately rather than conflating that with weight memory.
Raw report:`outputs/shared-dynamic-local-v2-fragment.json`; remote equivalent
under runtime diagnostics. Runtime report:`shared-dynamic-local-v2-runtime.json`.

With local parity and this deployed mechanism check passed, started only the
full daily micro4 probe on free gln01GPU0, wrapperPID103428 at11:19:56CST
2026-09-23(server clock),90min ceiling. Both GPUs were0MiB/0% immediately
beforehand; host load0.14/0.16/0.81. Source checkpoint100/artifact742, reference
gradients and24GiB limit are unchanged. Logs/JSON:
`runtime/outputs/diagnostics/shared-dynamic-local-v2-daily.log/.json`;
component artifacts/PID:`runtime/outputs/diagnostics/shared-dynamic-local-v2`.
No paid allocation and no retained updates. Next collect numerical parity,
compiled/actual memory and warm daily32 time. T7/T30 and joint replay remain
on hold until this result supports them.
