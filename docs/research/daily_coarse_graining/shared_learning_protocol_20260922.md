# First bounded learning run of the consolidated shared Transformer

Status: declared before training, implementation/preparation in progress.
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
