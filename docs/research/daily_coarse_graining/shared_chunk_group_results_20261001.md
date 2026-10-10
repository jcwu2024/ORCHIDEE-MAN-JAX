# Native grouped chunk execution G2

Status: G2 is closed. Both the bounded component and actual public daily
parameter/state-AD candidate finished with strict differences. Persistent
disjoint daily state intervals prevent attributing the failures to self noise.
No recurrent/full-update acceptance or training is claimed. Prior R4/V4
negative evidence remains preserved.

## Exact execution scope

G2 uses native `jax.lax.map(..., batch_size=2)` for expert5 only: 7,542
actual rows, 59 original full128 chunks, ten padded rows, 29 groups of two
and one native remainder chunk. The original chunk function retains its
coordinate blocks, projections, dynamic/source masks, native AD, readout
and complete output restoration. All other routes call the original helper.
No architecture, precision, loss, sampling, tolerance or gradient length
changes are introduced.

Source provenance is `shared_decoder_macro_packed._decode_mapped_rows`
lines25–76; coordinate execution is
`shared_transformer._run_coordinate_blocks_reusing_projections` lines537–610.
`scripts/hpc/shared_chunk_group_execution.py` reproduces the canonical map
body with this one map argument change. The component reuses
`probe_shared_command_buffer_loops.original_component`, adding only the
optional native chunk batch-size argument. Graph transport remains closed;
these are bounded exact-source claims, not graph completeness claims.

This route reduces serial chunk invocations while limiting the chunk-axis
batching compared with the rejected full59-chunk vmap. That reasoning does
not imply preserved compiled arithmetic or successful numerical parity.
Actual JAX0.4.38 `lax.map` signature with `batch_size` was verified on GPU0.
No2/4/8 sweep is performed.

## Component result

The captured expert5 sample0 raw-bank payload and original unscaled
`jax.sin(arange(...)*0.11)` cotangent are unchanged. This cotangent is a
component diagnostic, not the public scientific loss. Original forward and
native fullVJP saved binaries are reused with zero baseline compilations.
Scope: 2,896,128 outputs, all3,548,928 decoder parameters, all21,579,648
captured raw-bank/input coordinates. Native parameter and raw-input VJPs
are both retained; readout/public physical loss belong to the next gate.

| Scope | Initial failed coordinates | Maximum absolute difference |
|---|---:|---:|
| Output | 217 /2,896,128 | 2.384186e-6 |
| Complete decoder parameter VJP | 968,728 /3,548,928 | 1.399517e-4 |
| Complete captured raw-bank/input VJP | 2,003 /21,579,648 | 1.838803e-5 |

All seven cross pairs fail: parameter counts range968,461–969,017; input
counts1,981–2,016; output remains217 in every pair. Both arms' output self
replays are bitwise. Parameter self noise affects1,555–2,228 coordinates,
with maxima1.287460e-5–2.038479e-5; every raw-input self comparison passes.
The cross differences are retained and not waived as self noise.

Original→G2 warm forward median is0.024673713→0.017036278 seconds
(1.448304x); full nativeVJP median is0.100129455→0.070344590 seconds
(1.423414x). Candidate backend compilation took3.968174 seconds forward
and11.928809 seconds nativeVJP. All values and gradients are finite and
meaningful: both arms have3,548,928 nonzero parameter and4,186,752 nonzero
raw-input coordinates. Candidate estimated live memory is2,232,640,800
bytes (baseline2,260,028,960); observed process peak2,633,146,880 bytes.
Launcher42.496991 seconds /0.011804720 free card-hours, one GPU, paid0.
Numerical status is`parity_failed`; launcher exit0 records successful
completion of the diagnostic, not numerical acceptance.

All initial complete trees, fourteen full repeated output/parameter/input
VJP trees and candidate forward/nativeVJP executables are retained locally
and remotely. The measured1.42x component speed makes an independent
actual-loss screen useful; it does not grant public-model admission.

## Actual public daily dual gate

The candidate starts on node`ibc13b02n04`, GPU0, at
2026-10-01T08:55:32+0800; both GPUs are empty at launch, host load
0.33/0.39/0.83, affinity0–111. JAX/JAXlib0.4.38, original CUDA libraries,
OMP4, LLVM parallelism false, canonical existing environment and immutable
`shared-split-width-v1` checkout c599236 are unchanged. The accepted runtime
preflight is reused. Budgets240s lower /1500s backend /4200s Python /4210s
launcher match the measured public daily route. No paid job is submitted.

Dedicated `probe_shared_chunk_group_daily_dual.py` reuses the unchanged
Luna-owned public dual gate and shared objective/helper. It compares G2
against the saved original field expert128+backbone reference without
recompiling that reference. Explicit identity/schema/variant and adapter
hash distinguish G2. The V4 parameter-only report records only the origin
of the original reference's completed parameter bridge; it grants no G2
admission. The failed V4 daily state-AD route remains closed.

ABI:`(parameters, initial_states[4,3854], daily_samples)` with all daily
sample leaves dynamic and `inputs.continuous_state` replaced by the
explicit states. The result includes full64,884,481 parameter coordinates
and15,416 state coordinates, public loss/auxiliary and11 physical hard
counts. Initial plus seven complete paired loss/auxiliary/parameter/state
comparisons use unchanged`rtol=1e-5, atol=1e-6`; self replay is diagnostic.
Finite/meaningful and24GiB peak/live gates remain mandatory. Diagnostic
states never enter AdamW.

The actual daily gate finished `parity_failed`: loss(1), auxiliary(160),
all64,884,481 parameters and11 zero physical hard counts pass. Parameter
maximum difference1.788139e-7; gradient norm10.394025 with64,862,209 nonzero
parameter coordinates. All15,416 state coordinates are finite with15,334
nonzero. Initial state fails25; seven paired failures are
`[22,24,25,20,23,24,22]`. Field self counts are`[6,9,5,6,8,9,8]`, G2
self counts`[2,5,4,4,5,4,6]`; self comparisons remain diagnostic.

Raw evidence has33 unique failed sample/state coordinates,17 failing every
pair,15 with disjoint ranges beyond the largest unchanged tolerance:

| Sample/state | Original seven-repeat range | G2 range | Gap / largest allowance |
|---|---|---|---|
| 0/509 | [-12.191044807,-12.191041946] | [-12.191296577,-12.191296577] | 2.517700e-4 /1.229104e-4 |
| 1/385 | [112.205627441,112.205726624] | [112.207588196,112.207687378] | 1.861572e-3 /1.123057e-3 |
| 2/34 | [-0.670689225,-0.670688808] | [-0.670653343,-0.670653045] | 3.546476e-5 /7.706892e-6 |
| 2/1777 | [-5.897720247,-5.897711982] | [-5.897860699,-5.897846235] | 1.259877e-4 /5.997720e-5 |

Both arms pass all self comparisons at these examples. Initial maximum
among failed coordinates is0.535737702, from1/1799; global1.842353046
belongs to passing large coordinates and is not the failed maximum.
Full raw intervals are retained in
`shared-chunk-group-daily-dual-state-intervals-20261001.json`.

Warm daily dual median original0.380717173→G2 0.331364447 seconds is
1.148937903x, inadmissible because cross-state gates fail. Lowering took
108.236332 seconds, backend878.829029 seconds and Python1038.141902
seconds. Observed peak11,694,431,744 bytes and candidate estimated live
11,416,285,332 bytes pass24GiB. Completed launcher1045.421921 seconds /
0.290394978 free card-hours, one GPU, paid0. Its executed unchanged public
gate returns exit0 after a recorded numerical failure; exit0 is not
admission. No timeout occurred. All initial result trees, candidate binary,
failed paired complete trees and raw repeated states are preserved.

No G2 T7 or complete update is launched. The independent originalT30
reference work is a baseline asset, not progression of G2. Prepared
daily-only-group2 integration stays blocked by this actual public gate.

## Subsequent bounded direction

The original optimized component HLO preserves canonical128-row GEMMs and
has independent attention/FFN/projection operations with operation queue0
and empty wait queues. Queue metadata alone does not establish missing
overlap. A separate original-map latency-hiding scheduler screen will check
actual default/option acceptance, HLO scheduling and unchanged contraction
dimensions before any model progression. It changes no chunk grouping,
formulas or nativeAD boundary and is not a deterministic fix for the failed
G2/V4 state bias. It retains seven full component comparisons and raw trees.

A cheap numerical feasibility check found7542 unique query rows and7542
active rows in each of four captured expert5 samples. There is no measured
row-deduplication or further inactive-row omission benefit here. This check
would not by itself establish symbolic equivalence even if duplicates existed.

## Artifacts

Dedicated overlay:`runtime/transfer/shared-chunk-group-20261001`.
Component stem:`runtime/outputs/diagnostics/shared-chunk-group-component-20261001-g2`.
Public daily stem:`runtime/outputs/diagnostics/shared-chunk-group-daily-dual-g2-20261001`.
Local evidence:`outputs/diagnostics/shared-chunk-group-20261001/`.
Original daily reference:
`shared-chunk-vmap-daily-dual-v4-20261001-reference.json`,
`.reference.executable.pkl`,`.reference.pkl` under remote diagnostics.
Actual target-shell dry-run expansions are retained; existing running
snapshots and Luna-owned files are untouched.
