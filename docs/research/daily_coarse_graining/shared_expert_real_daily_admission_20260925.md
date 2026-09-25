# Real daily admission for expert-grouped decoder execution

Latest18:26CST: CPU1517 completed0,256.18s. Context is14,465,399bytes and
passed pickle round-trip plus canonical objective construction. FreeGPU0
launcher104632 started18:25:25CST on ibc13b02n04 (host load0.00; both GPUs
empty before dispatch). The deployed JAX0.4.38 runtime gate passed in22.25s;
real probe is loading the original saved daily executable. GPU report/log:
`runtime/outputs/diagnostics/shared-expert-real-daily-20260925.{json,log}`;
runtime check uses the same stem with `.runtime.json`. No measured real
candidate speed or gradient result yet. Earlier preparation status follows.

Status: all six local adapter/probe tests pass. Free test02 preparation1517
started2026-09-25 18:19:16CST (server clock),15minute ceiling. Diagnostic only; no
scientific training, retained weight update or paid allocation.

The adapter tests compare canonical raw outputs, restored shapes and masks,
including explicitly inactive PFTs and two fields sharing an expert. Native
AD parity covers full parameter, context and assembled-input gradient trees
on the tiny fixture at unchanged1e-5/1e-6 tolerance. Ruff and compilation
checks pass. This is not deployed real-objective numerical acceptance.

The three probe tests cover prepared-input reuse and pickle round-trip with
daily objective reconstruction, complete-tree comparison including nonfinite
rejection, and explicit sibling adapter loading. Actual CPU/GPU target-shell
dry-runs pass. Source checkout stays c599236; staged diagnostic code is under
`runtime/transfer/shared-expert-real-daily-20260925`. Preparation log/report:
`runtime/outputs/diagnostics/shared-expert-real-daily-20260925-prepare.{log,json}`.
No reference gradient is recompiled. A candidate daily binary and both full
results will be retained if the GPU phase reaches those stages.

## Question

Does expert128 execution reduce the cost of the actual daily training
contribution while preserving that contribution and its complete parameter
gradient? The preceding synthetic helper test took about276ms for micro4,
passed24.11M gradient-coordinate comparisons, and localized two old-FP32
output discrepancies with an independent FP64 oracle. It does not answer this
question or establish a complete mixed-update speedup.

## Fixed method and data

Use the original c599236 shared split architecture and saved fresh update0
checkpoint. The existing compact input bundle identifies TRAIN artifacts742
and275 and contains their exact prepared daily32/T7/T30 pytrees. This probe
uses four rows of the existing daily input; it does not sample new Teacher
data, initialize another model or alter the training/validation split.

Reconstruct the actual joint daily contribution using the existing public
objective: process/day loss divided by3, plus flux and stock loss divided by9.
Keep canonical input scaling, output transforms, physical constraints,
retained tail, loss statistics and scale validation. Preserve backbone and
the original GPP decoder, including its raw-carbon route. Only non-GPP
coordinate execution is grouped within expert; all eligible coordinates,
full output shapes, masks and shared parameters remain.

## Preparation and execution

1. Prepare one trusted pickled context on free test02: head, retained template,
   required frozen scales/statistics and compact real daily arguments. Do not
   serialize the entire runtime or closures. Round-trip it before GPU use.
2. Run the small deployed training-runtime check before the changed gradient
   graph. Load the original saved daily micro4 gradient executable. Execute
   and preserve its result before candidate compilation.
3. Scope the candidate decoder binding to a newly constructed objective and
   compile native JAX value-and-gradient for that same daily contribution.
   Record trace/lower, backend compilation, memory, synchronized warm timings,
   loss, auxiliary values, physical hard counts and all parameter gradients.

The free launcher bounds preparation to15minutes and GPU execution to30minutes,
plus at most3minutes for the deployed runtime check. Both shell branches must
pass actual target-shell argument expansion. No Slurm submission is involved.

## Decision

Use existing rtol1e-5/atol1e-6 numerical comparisons and exact physical hard
count checks. Do not relax thresholds to obtain acceptance. On failure retain
the completed result and distinguish physical/numerical disagreement from
compile, memory or infrastructure failure. Tiny adapter tests additionally
cover input/context gradients and applicability masks; these are not real
T7/T30 state-gradient acceptance.

On daily admission, evaluate whether the measured gain warrants candidate
T7/T30 compilation and a complete mixed-update comparison. Do not add candidate
daily gradients to old recurrent gradients and label the result a complete
candidate update. Subsequent throughput reports must distinguish same-work
speed, changed batch composition, extra GPUs/card-hours and startup overhead.
The user's approximately10x throughput objective remains a target, not a
result inferred from chunk counts or helper timings.
