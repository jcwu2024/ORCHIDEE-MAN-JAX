# Union neural precision preparation

Status: both attempts ended. First15469456 failed at a harness/schema boundary;
repaired15469965 passed saved-reference validation and compiled neural-forward
unit0, then required physical-forward unit1 cache hit failed and stopped before
backend compilation. No full candidate, numerical acceptance or timing exists.
No further retry, physical recompilation or separate mixed micro4 is authorized.

## Completed repaired outcome

Job15469965 ran once after daily15469919 ended. Before submission n04 was
IDLE/CPUAlloc0/CPULoad0.11/AllocTRES empty. Launch2026-09-29T10:59:37+0800
recorded load0.08/0.10/0.12, affinity0-3,56-59,8CPUs/GPU0 and no GPU compute
PID. It endedFAILED1:0 after211s allocation /201.287s process. Exposure
0.0586111card-hours; estimate `(8*0.07+2.20)*211/3600 = CNY0.1617667`, not a bill.
Both neural attempts total296s/0.0822222card-hours/estimated CNY0.2269333.
The approved sequential daily+repaired-neural jobs total416s/0.1155556card-hours/
estimated CNY0.3189333, below the CNY1.61 new-allocation maximum.

New neural-forward unit0 completed one backend entry in77.482903s, lookup/
unit78.701456s;301 arguments/259,568,756 nominal bytes and1,800 results/
7,952,939 nominal bytes, estimated live362,983,967bytes. Prior FP32 forward
had1,798 results/7,952,651bytes; this changed neural residual interface is
recorded, not numerical acceptance. Unit1 `jit_transition_stage` still has
125 arguments/158,808bytes and5,189 results/52,078,668bytes, but missed the
mandatory completed-unit cache. The guard stopped before physical backend
entry: zero unit1 backend calls. No loss/reverse units, complete candidate
gradients, engineering screen or warm result were produced.

Aggregate261 compiler lookups/259 backend entries/completions/88.041976s cover
initialization, reference validation and dispatch together; only unit0 is a
named new model unit. Do not attribute the entire total to initialization.
Status `completed_unit_cache_miss` means cache compatibility is inconclusive,
not a numerical rejection. Preserve the copied private cache, including unit0,
and both old/new StableHLO records. Small evidence is
[repaired JSON](../../../outputs/diagnostics/shared-union-neural-precision-paid-20260929-15469965.json)
and [log](../../../outputs/diagnostics/shared-union-neural-precision-paid-15469965.log).

## CPU-only cache diagnosis and next preparation

The generated [comparison](../../../outputs/diagnostics/compare_union_transition_hlo_cpu.py)
parsed preserved old/new unit1 text with standard MLIR APIs on test01 CPU,
under600s timeout and compiler-entry guards. Final run9.158s, JAX/JAXlib0.4.38,
zero backend calls; no model lowering, executable lookup, GPU or allocation.
The [JSON](../../../outputs/diagnostics/shared-union-transition-cache-diagnosis-cpu-20260929.json)
records exact input/normalized fingerprints, typed operations/constants,
operand identities, attributes, module metadata and available key evidence.

Old/new operation totals27,915/27,917 and constant counts1,194/1,196 differ only
in the operation/type/attribute multisets by an extra copy each of67/i32 and
11016/i64. Literal placement/order also differs. JAX's standard strip-debug
cache computation bytecode differs, so source debug locations alone cannot
explain the miss. Distinct typed literals and nonconstant typed operand/attribute
graphs match after explicit literal aliasing. This supports representation
differences, but standard strip-debug+CSE whole-module text/bytecode equality
fails; no prior-IR substitution hook is prepared or admitted.

Saved reports agree JAX0.4.38, `cuda:0`, c599236, remat and multiplier1.
Original serialized GPU compile options, actual XLA flags, JAXlib version,
accelerator topology/platform/runtime versions and per-component key digests
were not recorded; the assumed old launcher path is unavailable. CPU version
and new launcher declarations cannot reconstruct those original GPU inputs.
Additional cache/environment differences therefore remain unresolved.

Next preparation is to align captured sample construction with the prior
staged owner: `jnp.asarray(value)[0]` in `probe_shared_neural_precision.py:515`
versus `np.asarray(value)[0]` in the candidate probe:268. Keep host-only
reference validation separately. This is a source candidate, not a proved cause;
require exact normalized per-unit1-4 IR/ordered argument+result interfaces before
any later replay. Record scoped native `jax._src.cache_key` per-component DEBUG
digests, serialized compile-options hash, actual flags/JAXlib/platform/runtime/
topology/compression/custom-hook inputs in a future authorized execution.
Retain the mandatory-hit guard. Do not rename/fabricate cache keys, replace
physical IR, recompile physical units or infer numerical acceptance. Preparation
is not authorization for GPU/retry/paid execution; no implementation expansion
is part of this diagnosis.

## Actual attempt and repair

Job15469456 used the approved30min envelope on `ibc13b03n04`, which was IDLE,
CPUAlloc0, CPULoad0.01 and AllocTRES empty immediately before submission.
Launch load0.08/0.03/0.08, affinity0-3,56-59,8CPUs, GPU0 and empty GPU process
query are recorded. It endedFAILED1:0 after85s allocation /77.319s process,
before candidate compilation. Exposure0.0236111 card-hours; additive estimate
`(8*0.07+2.20)*85/3600 = CNY0.0651667`, not a bill. MaxRSS0 is not usable
host-memory evidence. Initialization made23 backend calls /13.192s. Cache copy
took23.866s,1,653 files /30,134,528 nominal bytes; the reference was preserved.
No candidate result, numerical screen or warm timing was produced.

Failure was `np.asarray` on a ragged seven-leaf mask tree. Repair traverses
boolean leaves and pairs predicted/expected values with declared masks. It
restores the existing owner's finite scope: loss, outputs and all parameter/
state gradients, explicitly attached to `all_measured_values_finite`. Nonfinite
defined diagnostics fail; undefined values remain unchanged. Actual reference
has one NaN in each `['state_diagnostics']['expected'].other` and
`['state_diagnostics']['predicted'].other`, each length3,854. Both positions
are declared undefined; defined-nonfinite count is zero.

Final test01 CPU check: JAX0.4.38,11.463s, exact saved FP32 result/context/
update0 identities, full validator and strict/engineering self-comparison pass.
It covers300 leaves /64,884,481 parameter coordinates,3,854 state gradients,
161 output/raw-mask fields /39,006 mask coordinates, seven leaves /5,757
coordinates per state mask,16 parameter groups and257 state fields. Exact
masks and undefined-gradient zeros pass. No neural forward, retained transition,
model AD or GPU execution occurred; import/mask-assembly CPU primitives remain.
Nine focused local tests include ragged masks, inactive NaNs and active-NaN
rejection; Ruff passes. Thresholds are unchanged. Preserve original source,
failed cache and small artifacts:
[failed JSON](../../../outputs/diagnostics/shared-union-neural-precision-paid-20260929-15469456.json),
[failed log](../../../outputs/diagnostics/shared-union-neural-precision-paid-15469456.log),
[actual-reference CPU check](../../../outputs/diagnostics/shared-union-neural-precision-saved-reference-cpu-20260929.json).

## First stage

The candidate uses native `F16_F16_F32` neural dot algorithms through the
existing scoped `neural_policy`. Norms, GELU, softmax, attention scores/value
sums, controls and source physical calculations retain their existing
precision. Architecture, capacity, sampling, objective and gradients are
unchanged. Loss multiplier is exactly1; there is no multiplier sweep.

The numerical reference is the saved full result
`runtime/outputs/diagnostics/shared-union-integrated-20260927.pkl`, artifact742
prepared row0, update0 checkpoint, union execution, all300 parameter leaves /
64,884,481 coordinates and all3,854 continuous-state gradients. Reference and
candidate use the same union execution. This reference does not establish
independent field-route state-gradient parity. The repeated-row0 field result
is not used as original-four truth or a performance comparator.

The probe makes one candidate call and saves the full loss, physical outputs,
raw masks, auxiliary/state masks and native gradient pair before engineering
screening. It reuses the existing contract unchanged: loss relative difference
at most1e-3, semantic parameter-group and state-field gradient relative L2 at
most1e-2 and cosine at least0.9999; near-zero norm at most1e-12 uses absolute
max at most1e-6. Strict1e-5/1e-6 results are reported separately. All measured
values must be finite, physical hard counts zero, raw/state masks exact and
undefined state gradients exactly zero. Output-normalized discrepancies are
descriptive, with no new threshold. A failed engineering screen stops
progression; its saved result is retained.

The inherited contract's `inactive_reference_gradient_coordinates_must_remain_zero`
declaration refers to structural inactivity, not every zero in a reference
gradient. `reference_zero_became_nonzero` is descriptive because reference
zeros can arise from cancellation. The implemented exact-zero gate covers
undefined continuous-state inputs; raw/state masks are checked for exact
equality. A separate complete structural-inactive parameter/output coordinate
audit is absent. Reports explicitly retain this coverage gap, so an engineering
pass does not claim every inactive-coordinate declaration has been verified.

The completed FP32 union report has six units. Ordinals1/2/3/4 are physical
forward/loss forward/loss reverse/physical reverse and must hit their completed
cache before any backend entry. Only neural forward/reverse units0/5 may be
new. The existing roughly29MiB nominal cache is copied to the new output-specific
directory; reference cache files and the immutable checkout are preserved.
The probe records aggregate compiler lookups/backend entries/completions and
named-unit evidence, startup, copy time, cold first call, active-buffer memory
and GPU wall exposure. No FP32 field transpose, reference replay, generic
profiler, completed FFN/micro4 experiment or row0 warm timing is repeated.

Caps: whole process1200s; new neural forward300s; new neural reverse600s;
outer process1240s with10s kill grace; estimated executable/live-buffer memory
24GiB. A mandatory cache miss stops before backend compilation. Reference
validation, runtime/schema failures, numerical failures and resources are
reported separately. A timeout is not numerical rejection or acceptance.

## Separate second stage

`run_shared_union_neural_precision.sh --micro4 --screen-report PATH` is ready
for a passing full parameter/state screen. The micro4 owner enforces the exact
same reference identity, contract, multiplier1 and complete gradient count in
that report before compiling the candidate. The first-stage paid wrapper
rejects `--micro4`; this stage needs separate execution authorization/resources.

It uses four distinct original artifact742 rows, existing admitted combined
FP32 field executable/result and canonical reference, the public mean daily
objective, original auxiliary and all parameter gradients. Only the opt-in
candidate union neural dots change. Both numerical results are saved before
gates. Finite meaningful gradients and physical checks are retained, strict
discrepancies are independently recorded, and the unchanged loss/semantic
parameter-group engineering gate must pass before three alternating matched
warm pairs. Cache/output stems are separate from prior diagnostics. No new
field graph compilation, optimizer or recurrent work occurs.

Its default preparation budgets remain180s lower,900s backend,1500s whole
process and100s per first/warm execution, with24GiB memory admission. Actual
timing can only support original-four production-interface parameter AD.
Neither stage certifies T7/T30, a complete update, production precision or the
10x target. Subsequent route-specific measurements must include T1/daily,
T7 and T30 followed by the unchanged original loss/data mix complete update.

## Checks and authorization

Local `ORCJAX`:17 focused tests pass. Tests cover complete reference units,
cache separation, required-cache-miss stop before backend, scoped restoration,
full-state prerequisite, unchanged engineering gates and save-before-screen /
no row0 warm replay. Ruff passes for new/extended probes and focused tests;
the older shared neural helper has pre-existing lint findings outside these
changes. `git diff --check` passes.

Target Bash4.2 syntax and actual argument expansion pass freeGPU0/1,
schedulerGPU3 and UUID bindings, empty/forwarded argument arrays, output stem,
separate micro4 overlay/cache and failed multiple-GPU/paid-micro4 guards.
The deployed JAX0.4.38 CPU-only hook/schema check passes, confirms six saved
union units, and forces a mandatory physical cache miss with zero backend
calls and no model/GPU execution. It does not establish candidate numerical
validity or real cache compatibility; those remain first-stage gates.

Repaired staging path:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-union-neural-precision-20260929-v2`.
Original path without `-v2` remains the failed-job source record.
Immutable code root:
`runtime/checkouts/shared-split-width-v1` at
`c599236bcf57b745e9bafed678298010f4c3b58f`.
Preparation JSON is local under `outputs/diagnostics/` and copied to the overlay.

The earlier free gln01 inspection found both V100s occupied; zero utilization
did not qualify a resident external process as free. The repaired paid job
then ran once under the approved25min/8CPU/one V100/64GiB/nonexclusive,
maximum CNY1.15 envelope, sequentially after daily15469919. Its1200/300/600s
caps remained unchanged. Both allocations ended. The completed result grants
no further retry, separate micro4, larger budget, physical compile or training.

Under remote `runtime/outputs/diagnostics/`, preserve
`shared-union-integrated-20260927.stage-01.hlo.pkl`,
`shared-union-neural-precision-paid-20260929-15469965.stage-00.hlo.pkl`,
its `.stage-01.hlo.pkl` and `.stages-cache/`, alongside the original reference
`shared-neural-reference-20260927-staged.stages-cache/` and failed15469456
cache/source. The comparison script/JSON also remain there; their compact
copies are local under `outputs/diagnostics/`. No executable/cache is modified
by CPU IR inspection.
