# First bounded learning run of the consolidated shared Transformer

Status: declared before training;83621 failed at0 updates (monitor API),
same-method free recovery87042 COMPLETED100 updates at4408daf in97.07min.
See `shared_learning_results_20260922.md` for outcomes and limitations;
`shared_learning_runtime_fix_20260922.md` for the compatibility fix.
The original launch below is history.
The user's2026-09-22 continuation authorizes this bounded free-node work;
no paid computation is authorized. Numerical daily/T7/T30 admission7359 has
passed. This is a learning pilot, not the final paper training run.

## Fixed method

Use `shared_learning_v1.json`, daily-v3 architecture and seed20260921,
fresh75,578,497 parameters and fresh AdamW. No exploratory/daily/joint-admission
checkpoint inheritance. Keep LR1e-5, clip1.0, existing AdamW constants,
TRAIN calibration and shared trajectory losses unchanged. No backbone,
precision, output coordinate, curriculum or loss-weight search in this run.

100 updates. Each uses all32 single-day rows from one existing prepared
TRAIN artifact, plus its first7-day and first30-day window. Select100 distinct
artifact IDs by seed20260922 permutation of sorted available IDs. Record the
exact ordered selection before inspecting predictions. These artifacts can
share source days:3200 draws must not be presented as3200 unique days without
the data preparation count. Recurrent day exposures also overlap daily data.
This is not an epoch of either the ready daily pool or the full Teacher release.

At identical parameters, average32 already weighted single-day gradients,
then add the weighted T7 and T30 gradients, clip once and update once.
Sequential daily accumulation preserves the admitted single-row AD graph
and memory use; it is an initial execution choice, not a claim of optimal
throughput. Reuse persistent compile caches and native checkpointed recurrent
AD. Do not redo directional AD checks for unchanged graphs and ordinary data.

## Fixed evaluation and preservation

Record science at updates0/20/100. Fixed fit monitor: first8 selected training
artifacts, window0 at both horizons (16 windows). Reuse all48 already prepared
`fresh-method-validation-20260921` windows, separately reporting temporal,
spatial and joint held-out splits. They do not participate in parameter
updates or calibration. They are development validation, not a sealed test
or an assertion that they have never been consulted in earlier research.

Report physical-unit daily/cumulative GPP, source NPP, maintenance/growth
respiration, RA and RH; AGB/BGB and the seven carbon stock levels/changes.
Preserve per-window curves and teacher magnitudes, signed bias, RMSE, sample
coverage, nonzero-RH coverage and hard failures. Do not substitute a lower
aggregate training loss for this scientific comparison. Use the fixed final
update100 as endpoint; retain update20 without silently selecting the better
looking checkpoint. There is no newly invented final accuracy threshold.

Save parameters, AdamW and schedule cursor at0, every5 updates and before
milestone evaluation. Resource interruption may resume only the same method
and next declared data ID. Keep last valid recovery on numerical/physical
failure. Stop on nonfinite values or physical hard failure; do not stop or
retune solely because an ordinary minibatch loss rises. Small sample/short
pilot failure does not prove the architecture can never learn.

## Execution envelope

Free gln01, one idle V10032GiB, main240min upper bound. First confirm metadata,
actual selected batch populations, shape classes and held-out point-year
separation on free CPU. Reuse7359 deployed runtime/AD evidence; this changes
host accumulation and sampling, not the underlying AD/state boundary.
Check compiled memory against24GiB and record actual peak and warm update
timings. Do not repeat JVP or full gradient serialization each update. If
runtime hits its resource ceiling, preserve progress and reassess actual
remaining cost; a later paid run requires a separate concrete approval.

After this pilot, decide the next data/learning budget from fixed TRAIN-fit
and held-out physical curves, including tradeoffs and uncertainty.365-day,
long-site, GateD2 and product acceptance remain outstanding.

## Completed data preparation

Free test02 preparation at3af6fe8 completed in58.50seconds. It inspected the
100 selected existing artifacts without executing a neural model or generating
Teacher data. The declared draws contain3200 distinct single-day samples,
700 distinct days within T7 windows and3000 within T30 windows. Their union
contains6899 point/year/day keys, covering523 landpoints,1720 point-years and
39 years (1961-2004, not every year). This remains a bounded pilot, not an epoch.
There are zero point-year overlaps with the48-window held-out development panel.
Each training component has one shape/dtype class, identical to admission819.
Evidence: `outputs/shared-learning-prep-3af6fe8.json` locally and the same name
under remote `runtime/outputs/diagnostics/`.

## Local implementation verification

The runner, plan and monitor test suite passes13 tests, including direct
comparison of32-gradient averaging plus T7/T30 against the complete toy loss,
one AdamW update, exact same-method optimizer resumption, physical/nonfinite
rejection, fixed panels, stock curves and input-signature cache reuse.
The monitor plus adjacent recurrent loss/validation tests passes19 tests
(30 distinct tests across both suites). Ruff passes. These are local contract
tests, not actual-model GPU execution or scientific accuracy evidence.

## Free-node launch

Immutable codebd50f51 deployed to
`runtime/checkouts/shared-learning-bd50f51`. Actual target-shell CPU/GPU
argument expansion and shell syntax pass. Free test02 configuration execution
completed0 in22.7s; both gln01 GPUs were idle immediately before launch.
GPU1 wrapper83621 runs the declared pilot with240min ceiling. No paid job.
The runtime/AD boundary is unchanged, so accepted7359 evidence is reused,
without repeated synthetic runtime tests or directional full-model gradients.

Report/log prefix:
`runtime/outputs/diagnostics/shared-learning-bd50f51-gpu`.
Query from the workstation with `ssh gln01 'ps -p 83621 -o pid,etime,pcpu,comm'`;
no process row means read the report/log to distinguish completion from failure.
The first observed stage is configuration/runtime loading, not a completed
optimizer update. Wait for fixed0/20/100 physical outcomes before judging
learning. Recovery checkpoints retain the declared fresh-method cursor and
optimizer plus completed milestone metrics; do not substitute admission7359
or an exploratory checkpoint as initialization.
