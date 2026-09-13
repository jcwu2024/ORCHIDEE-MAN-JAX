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
the immutable result of job15048786. See completed results below.

## Completed results (2026-09-13)

Both chains completed at c76c553 with exit0, with launcher PIDs absent.
No training update or paid allocation was performed. Reports are retained at
the declared remote paths and copied to local outputs/ with unchanged names.

Forward completed all16 windows in708.62s. Metric populations and physical
checks pass; both horizon science gates fail. These are train-domain disjoint
days, not spatial/temporal holdouts. All heterotrophic Teacher totals are zero,
so their exact matches provide no fitting evidence.

Mean cumulative fluxes per window, Teacher -> model, in gC/m2:

| Horizon | GPP | NPP | Autotrophic respiration |
| --- | --- | --- | --- |
| 7 days | 14.2708 -> 9.9132 | 4.9814 -> 1.7077 | 9.2894 -> 8.2055 |
| 30 days | 52.3565 -> 35.8220 | 17.2429 -> 5.3923 | 35.1136 -> 30.4297 |

The30-day aggregate relative biases (sum error / sum Teacher, not mean
per-window percentages or training-normalized bias) are-31.58% GPP,-68.73%
NPP and-13.34% autotrophic respiration. Mean30-day AGB change is+1.1731 ->
-7.4233, BGB+0.1528 ->-2.9903 gC/m2. Small normalized final biomass errors
do not establish accurate increments. Nonliving-carbon change is reported as
+125.2697 ->-11925.4042 gC/m2; attribute constituent pools and verify extractor
semantics before concluding which process caused this large discrepancy.
Existing budget hard checks passing do not certify this full carbon trajectory.

Forward active GPU peak is1.280GiB (pool20.631GiB). After warmup, measured
window calls take about0.272s/T7 and1.142-1.244s/T30. Initial T7 call takes
17.97s beyond explicit compilation. These are single-window diagnostic calls,
not a batched production benchmark or a Teacher speedup measurement.

Native T7 reverse passes in2180.85s (1821.90s compilation); the full parameter
gradient is finite with norm7.55134 and109,464,375 entries. Estimated compiled
buffers2.266GiB, active peak2.683GiB. JVP verification passes in666.97s:

| Direction seed | Reverse projection | JVP | Absolute error |
| --- | --- | --- | --- |
| 20260913 | -1.4729198971 | -1.4729273688 | 7.47168e-6 |
| 20260914 | -1.0286162576 | -1.0286147868 | 1.47087e-6 |

Both projections are resolved and meet the predeclared gate; relative errors
are5.073e-6 and1.430e-6. Primal difference is3.493e-9 and all hard counts
are zero. This admits only the stated checkpoint/window/scalar/two parameter
directions. T30, initial-state gradients and the historical fresh-T2 finite
difference discrepancy remain open. Do not repeat the completed reverse stage.

Next work is forward carbon-pool/GPP attribution and bounded native parameter
AD extension toward T30. The forward failures do not identify an architectural
cause and do not authorize a new paid training run.
