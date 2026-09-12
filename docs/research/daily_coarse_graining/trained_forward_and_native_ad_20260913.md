# Trained 7/30-day forward screen and native T7 parameter AD

User approved both tracks. Reuse the final update1998 checkpoint from job15048786,
which starts from one declared fresh initialization and has no exploratory
weight-chain dependency. No new training or paid submission is authorized here.
Architecture, physical state, input conditioning, scales and model parameters
remain unchanged.

## Fixed forward panel

Use NumPy PCG64 seed20260913 to select8 seven-day and8 thirty-day windows from
the existing prepared extension1001--3000. Selection uses metadata only,
excludes every one-day Teacher day in the training catalogue (including its
monitor dates), and prevents overlap among selected windows. Preserve the
actual flattened component window index and storage-row identity; actual
calendar day indices are read from payloads, not inferred from row numbers.
At most one window per artifact per horizon, favor distinct point-years.
The selection report is written before any trained-model evaluation.

Only the initial state is Teacher supplied. Every later state/discrete carry
and state-derived PFT view comes from the canonical bound model. The existing
per-window annual retained bundle is reused with each actual daily input.
No48-step forcing reconstruction, hidden-state scan or Teacher-state injection.

Report all8 accepted flux keys (including GPP, maintenance/growth/autotrophic
respiration, NPP and heterotrophic respiration) and all5 accepted carbon-stock
keys. Use the existing target-oriented13-key acceptance scales and thresholds:
T7 normalized RMSE<=0.35, absolute normalized bias<=0.10; T30<=0.25/0.07.
Also preserve physical-unit daily curves, complete-window cumulative errors,
and terminal stock changes from the same initial stock. A nearly correct stock
level is not evidence of a correct biomass increment. Mark uninformative zero
Teacher fluxes and nonfinite/missing observations explicitly.

Keep existing finite/state-mask/discrete/budget/stock hard checks; report
failures as observed model divergence, not automatically as trainer defects.
Do not silently select replacement windows after a poor result. This panel is
disjoint by Teacher day within train/train, not a sealed spatial/temporal/joint
validation. Even a complete pass is only a preliminary screen.

## Native T7 AD at the intended trained state

Take the first preselected T7 window, before looking at its outputs/gradients.
Use the unchanged canonical diagnostic science scalar (four cumulative fluxes
and final AGB/BGB), with initial state fixed. Differentiate the full neural
parameter tree using native `jax.value_and_grad(checkpointed_loss)`.
JAX owns the full recurrent backward calculation; no hand-assembled cotangent.

Two parameter-only Rademacher directions are frozen at seeds20260913 and
20260914, generated from canonical PyTree order with PCG64. Each coordinate is
exactly +/-1 in the parameter dtype. Their norm is sqrt(parameter_count),
recorded explicitly. Keeping per-coordinate magnitude1 avoids unresolvable
projections from dividing a109-million-coordinate direction by its norm; it
does not perturb primal parameters or change the loss. Directions are chosen
independently of gradients.

Save the complete native parameter gradient before compiling the JVP stage.
Compare host-float64 dot products against the JVP of the identical checkpointed
loss, at existing derivative tolerances atol1e-6/rtol1%, and primal agreement
at atol1e-6/rtol1e-5. Require finite, nonzero gradients, physical gates and both
projections resolved above1e-5. An unresolved direction is not a correctness
pass. Compile/memory-check both executables before numerical evaluation,
with18GiB per-executable buffer ceiling.

A pass establishes only this trained checkpoint/window/scalar and the two
parameter directions. It does not close the old fresh-T2 finite-difference
plateau, prove every initial-state coordinate, admit T30 training or Gate D2,
or authorize a multi-day training job. This is a measured intended-use check
of the simpler native-checkpointed route.

## Execution order and bounds

Local selector/window-layout/metrics tests, real metadata preparation, and
actual-data CPU abstract tracing precede the GPU work. Each new execution path
runs the exact-runtime tiny gate and real-shell argument dry run. No bulk data
rehashing. Forward screen: one free GPU,75-minute cap, per-window progress.
Native reverse and JVP stages: separate90/60-minute caps and durable gradient,
so a later timeout cannot discard the reverse result. These are maximum bounds,
not runtime predictions. Use idle free GPUs; no new paid job is submitted.

## Pre-execution evidence and resource adjustment

The metadata selector completed in148.18s on free test02. It excludes63,771
distinct one-day/monitor dates and selects16 windows,14 locations and7 years,
with296 mutually disjoint Teacher days. Its first T7 window is artifact1887,
window5, point277.0-091.0/year1995/start-row73. No model output selected it.

The real T7/T30 forward graph and actual metric payload preflight passed at
1b3aaa8 in141.17s. Eleven local selector/metrics/layout tests pass after review
added exact panel/population checks and nonfinite-result rejection. These
bookkeeping changes do not alter the traced forward graph.

The combined native-gradient/JVP CPU preflight did not complete within its
15-minute limit on test03; the last observation showed about7.5minutes CPU
time on the shared node. No shape/type exception was captured, and the report
remains an incomplete bootstrap-stage record, not a passed AD trace. Repeating
that large abstract trace before GPU tracing would duplicate substantial work.
Proceed with the already validated real forward boundary and the exact-runtime
tiny AD gate, then let each bounded GPU lowering supply the actual gradient/JVP
trace and memory result before evaluation. This resource adjustment does not
promote numerical correctness or change the AD acceptance tolerances.

## Launch record

At c76c553, the forward screen runs on free GPU1 (launcher PID4617),75-minute
limit after runtime preflight. The native AD chain runs independently on free
GPU0 (PID6852),90minutes for reverse and60minutes for JVP after reverse success.
Logs: `runtime/outputs/diagnostics/trained-forward-ad-v1-forward-pipeline.log`
and `trained-forward-ad-v1-ad-pipeline.log`. Reports use the same prefix with
`forward.json`, `ad-reverse.json` and `ad-verify.json`. Neither is a training job.
The source checkout is fixed during execution; the final checkpoint remains
the immutable result of job15048786. Results are pending.
