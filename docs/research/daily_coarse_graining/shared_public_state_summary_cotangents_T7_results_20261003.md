# Original FP32 public T7 summary-input cotangent capture

Completed 2026-10-03: **the instrumented original FP32 graph fails the complete
initial-state bridge at five of3,854 coordinates.** Loss, all19 auxiliaries,
all64,884,481 parameter gradients and hard/memory gates pass. The first raw
result is preserved, but its cotangents are not admitted as original-binary
evidence. This is a numerical bridge failure after successful compilation,
not a precision-candidate result or a compilation/resource failure.

## Scope and mandatory bridge

The source-cloned original assembler adds eight independent FP32 zero deltas
only at normalized-state summary inputs: field at
`strong_baseline_architecture.py::assemble_strong_baseline_context`1123–1127,
seven expert-axis consumers1167–1171. Dynamic scan days carry distinct
`[8,3854]` slices of `[7,8,3854]` deltas. The complete shared public T7 loss
`(flux + stock)/9`, native checkpointed seven-day recurrence, expert128/backbone,
original update0 weights and artifact742 case remain. Native AD returns all
parameters, all initial states and the delta derivatives; no custom VJP,
stop-gradient, shortened recurrence, new precision or optimizer update is used.

The exact same-window original saved binary is replayed before the new capture.
Its comparison to the historical original result is independent context, not
a waived gate. Capture comparison uses unchanged `atol=1e-6, rtol=1e-5` against
that fresh original replay. All full parameter/state/loss arrays are saved
before gates. Separate zero-delta arguments may change compiler optimization;
even a passing public-output bridge could not prove identical hidden
original-binary cotangents.

Source AST, CLI and tiny CPU native recurrent interface checks pass, including
distinct day factors127/63/31/15/7/3/1. Actual test01 Bash syntax and both
full/resume argument expansions pass. The deployed canonical JAX0.4.38 GPU
runtime preflight also passes. These interface checks do not certify the real
model derivative.

## Actual numerical result

| Scope | Coordinates | Failed | Maximum absolute error |
| --- | ---: | ---: | ---: |
| Loss |1 |0 |1.392025e-11 |
| Auxiliary/physical values |19 |0 |5.271533e-8 |
| Complete parameter gradients |64,884,481 |0 |7.450581e-8 |
| Complete initial-state gradients |3,854 |5 |15.935692 globally; this maximum passes its allowance |

The corrected saved-array NumPy verifier agrees for the entire original output
scope:300 numerical parameter leaves/64,884,481 coordinates pass,20 loss/auxiliary
coordinates pass, and the same five state coordinates fail. Artifact acceptance
found that the first CPU walker incorrectly converted two static `None` nodes
to NaN and counted them as two extra parameters. This was an analyzer/reporting
error; the GPU parameter result did not change. The repair validates container
and static-node ABI explicitly, records both matching `None` paths, and enforces
the declared300-leaf/64,884,481-coordinate scope. Matching-`None`, changed/missing
static-node, changed-container and incomplete-numeric-scope controls pass;
analysis and result summaries were recomputed. The actual failed coordinates are:

| Coordinate | Source state field/index | Original replay | Capture | Absolute error / strict allowance |
| --- | --- | ---: | ---: | ---: |
|32 |`diffuco_previous_step_state.snowdz[0,2]` |−1.87067508698 |−1.87065458298 |2.050400e-5 /1.970675e-5 |
|33 |`diffuco_previous_step_state.snowgrain[0,0]` |0.368530541658 |0.368672460318 |1.419187e-4 /4.685305e-6 |
|34 |`diffuco_previous_step_state.snowgrain[0,1]` |−1.54780745506 |−1.54778456688 |2.288818e-5 /1.647807e-5 |
|199 |`hydrol_previous_step_state.canopy2ground[0,0]` |−0.0491824704411 |−0.0491846862367 |2.215796e-6 /1.491825e-6 |
|385 |`hydrol_previous_step_state.nroot[0,0,10]` |−0.724096357822 |−0.724086344242 |1.001358e-5 /8.240964e-6 |

The maximum error **within the failed set** is1.419187e-4 at33. The global
maximum15.935692 is at1785 and passes its139.960752 allowance; it is not the
size of a failed comparison. Coordinate428 is not among the five failures.
Source field mapping follows the existing serialized `StateLeafSpec` row-major
contract, with source references explicit snow, hydrol canopy and hydrol soil.

All14 physical hard counts are zero, `declared_mask_change=14` is unchanged,
and all21 undefined-state gradients remain exactly zero. The112 source input
masks are preserved. Captured cotangents are finite FP32 `[7,8,3854]` with
215,824 values. Estimated live memory3.375275GiB and whole-process observed
active peak3.393435GiB pass the24GiB gate. Process pool peak is a separate
allocation statistic, not an arm-specific or full-training memory peak.

The first state failure is terminal under this protocol. No repeat call,
paired timing, candidate binary or full HLO text is generated. No tolerance,
reference, architecture, loss, sampling or production policy changes.

## Bounded cancellation screen

The predeclared screen requires a valid complete first bridge and two stable
repeat captures before admitting interpretation. Neither condition is met.
The saved values remain available as **conditional descriptions of this
instrumented graph only**:

| Day0 coordinate | `sum(abs(partials))/abs(sum(partials))` | `gamma7*sum(abs(partials))/source_F32_scale` | Original public strict allowance |
| --- | ---: | ---: | ---: |
|33 |4.335834 cancellation ratio |6.669475e-7 |4.685305e-6 |
|199 |1.001215 cancellation ratio |3.745899e-11 |1.491825e-6 |
|428 |38.666552 cancellation ratio |7.432208e-7 |1.326489e-5 |

All three conditional day-zero bounds are below their public raw-gradient
allowances. **This does not exclude cancellation in the original graph or
identify the T7 discrepancy's cause**, because the instrument failed its
bridge and has no stability evidence. The seven-add FP32 `gamma7` is a
conditional possible-error bound, not measured GPU error. It excludes
upstream partial-generation error, normalization-division rounding, other
direct state paths and recurrent Jacobians. Later days are normalized-state
cotangents only and cannot directly reconstruct initial-state gradients.
The analyzer's `screen_interpretation_admitted` is explicitly false.

An existing-array five-coordinate FP64 supplement verifies exact source/profile/
checkpoint/case identity, all300 weight bridges,112 input masks and byte-identical
initial raw state. Both original and capture fail the unchanged comparison to
the saved full-neural-FP64 state vector at all five coordinates. At33,
original/capture are+0.368531/+0.368672 versus FP64−0.198487; this cross-precision
sign difference cannot select a true derivative or validate the capture.
Current original-versus-historical state self-comparison independently fails
one coordinate; its global maximum8.240276 is not the failed-coordinate error.
The prior original native-JVP report's33 self drift2.712011e-5 remains historical
context. `five-state-fp64-offline.json` retains every row, identity gate and
unchanged allowance. No new model/JAX/GPU call is used for this supplement.

## Runtime, closure and artifacts

Free gln01 actual host `ibc13b02n04`, one V100 GPU1, canonical JAX0.4.38
container, OMP4/OpenBLAS1 and immutable checkout `c599236`. GPU1 was idle with
0MiB/0%, no contexts and host load0.02/0.05/0.05 before dispatch. Bash origin
1791011854.624285 precedes Python imports; the total3600s cap includes10s
kill grace. All stages share that first origin.

Preflight82.582s, lowering193.519042s and backend2508.716800s complete within
their caps. The sole synchronized full capture call takes2.063887s; it is
not a stable throughput measurement. Whole dispatch2935.555290s is
0.815432025 free card-hours, CNY0, one capture compilation/one capture call and exit1 for the
bridge failure. There is no resource exception, repair or resume. Cleanup at
2026-10-03T08:08:25Z confirms GPU1 0MiB/0% and no compute contexts. The window
is closed; unused time authorizes no continuation.

Local family: `outputs/diagnostics/public-state-summary-cotangents-T7-20261003/`.
Remote counterpart is under the project `runtime/outputs/diagnostics/`.
The first capture260,453,461B and original replay259,589,887B are transferred
once each. Key records are `result-summary.json`, `cancellation-analysis.json`,
`state-bridge-compact.pkl`, `five-state-fp64-offline.json`, `actual-exit.json`, `terminal-gpu-cleanup.txt`,
`model/model.json`, source/input/mask snapshots and preparation checks.
CPU scripts `analyze_public_state_summary_cotangents_T7_cpu.py` and
`finalize_capture_result_cpu.py` reproduce the new saved-array evidence;
synthetic and read-only analyzer-review records are under `preparation/`.

The first outcome does not test the earlier approved local component's
F64 reverse fan-in mechanism on the real public T7 model. The next preparation
is a separate source-backed original-forward candidate applying that exact
mechanism at the eight normalized-state summary consumers, retaining complete
native public AD and the original ABI. GPU execution of that new precision
boundary remains unapproved. This preparation does not reopen batching or the
closed projected-bank adjoint route and does not make exhaustive internal
gradient decomposition a prerequisite for complete-update optimization.
