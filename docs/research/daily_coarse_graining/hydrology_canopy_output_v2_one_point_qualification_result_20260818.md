# Hydrology And Canopy Output v2 One-Point Qualification Result

Date: 2026-08-18.

Status: execution completed; scientific qualification rejected. The candidate
passes all five primary-science gates and all hard state/budget gates, but
misses the total-loss gate narrowly and misses four independent expert-loss
gates. No checkpoint is retained.

## Execution And Identity

Approved Explore1000 job `14608555` ran at commit
`806cffd1293f65ecf19931de5c7d98261987fad1` on one V100 with four allocated
CPU cores. It completed all 100 epochs and 4,600 batch-8 updates. The report
runtime is `2167.67 s`; all gradients are finite and gradient clipping is
disabled.

The run uses the frozen v6 Teacher release and all 364 valid train/train days
from `001.0-071.0`, 1961, once per epoch. It does not regenerate Teacher data,
read historical v5 rows, reconstruct 48 forcing records, execute a 48-step
state scan, load a checkpoint, or write a checkpoint.

The candidate contains 49,743,001 parameters. It learns 161 fields/39,006
scalar coordinates, reconstructs 20 fields exactly, and materializes the
complete 181-field/39,371-value process boundary before the canonical retained
exact tail.

Formal report:

`outputs/research/daily-coarse-graining/mechanistic_output_v2_one_point_qualification_20260818.json`

SHA256:

`c1d1a027e23f149cf59a9d6530255a07d839a359ad7e3dfce8dac2c2264c6661`

## Gate Result

Complete process loss falls from `4.5707906` to `0.0498893`, a `98.9085%`
reduction. This misses the preregistered `99%` gate by `0.0915` percentage
points.

All five primary normalized-RMSE gates pass their `0.05` limit:

- GPP: `0.02416`;
- maintenance respiration: `0.00729`;
- NPP: `0.02903`;
- aboveground-biomass tendency: `1.29e-5`;
- belowground-biomass tendency: `2.60e-5`.

All prediction, defined next-state, water/carbon budget, and nonnegative-
inventory hard counts are zero.

Three expert reductions pass the independent `95%` gate:

- maintenance respiration: `99.9847%`;
- soil thermal: `99.5681%`;
- daily interface/finalize: `99.1563%`.

Four expert reductions fail:

- hydrology/snow: `93.7780%`;
- canopy/GPP: `93.4923%`;
- surface energy: `89.3930%`;
- litter/POC/DOC: `80.2522%`.

## Interpretation

The hydrology/canopy v2 revision is positive mechanism evidence, not a failed
rewrite. Hydrology improves from the v1 formal result's `83.0539%` reduction
to `93.7780%`. Its support-aware final `nroot`, exact post-update hydrology
fields, and signed canopy endpoint should therefore be retained unless a later
source audit contradicts them.

The candidate is no longer in the earlier regime where it cannot fit the
training point-year at all. It now fits the five science outputs that matter
most for later biomass and productivity rollouts. That does not yet establish
rollout validity: the independent process gates deliberately prevent those
aggregate outputs from hiding residual internal-process errors.

The residual is concentrated rather than uniform. The largest final
normalized field MSE values include `fluxsens=1.0201`, `vevapp=0.9899`,
`temp_hydro=0.6416`, `ae_ns=0.6381`, `wspeed_daily=0.5935`,
`evapot=0.5413`, `ok_leak_aux.litterpart=0.5000`, `evapot_corr=0.3833`,
`litter_above.incoming_amount=0.2689`, `qsurf=0.2595`, DOC diffusion
`0.1784`, and `nroot=0.1245`. Several litter and surface coordinates become
worse than their fresh-initialization loss.

Hydrology, canopy, and surface energy still improve slowly through epoch 100.
Litter/POC/DOC remains near an `80%` reduction and is effectively plateaued.
The learning rate has already decayed to `1e-5`, so extending the unchanged
schedule is not a justified next experiment.

The Slurm job was recorded as `FAILED 1:0` only because the old runner returned
exit code 1 for a completed scientific rejection. The report itself records
`execution_status=completed` and `failure_reason=null`. Runner semantics are
corrected after this result so `completed_failed_gate` exits zero and exit code
1 is reserved for execution failure.

## Bounded Next Diagnostic

Do not open 64-point-year training, rollout, Gate D2, sealed-test, or product
stages, and do not repeat or extend this training schedule.

The immediate task is a zero-training output-coordinate attainability audit on
the same existing 364 v6 rows. It bypasses the neural body but retains the v2
learned-coordinate transforms, exact reconstruction, conservative update,
retained tail, masks, and complete mechanistic loss. For every learned
continuous coordinate it constructs or solves the per-day coordinate that
best reproduces the Teacher target; canopy categories use deterministic
correct-class logits. Results must be reported by expert, label, and field.

The audit passes representation attainability only if continuous learned
coordinates reach normalized MSE at or below `1e-8`, canopy categories and
defined masks are exact, every declared-defined materialized output is finite,
all hard counts remain zero, and any exact-derived diagnostic agrees at its
existing numerical tolerance. A nonzero floor is not averaged away across
fields.

This produces a direct decision:

- if litter/POC/DOC or coupled surface-energy coordinates retain a floor, fix
  that output/target representation before another network fit;
- if they are attainable, the floor is in neural mapping or optimization, so
  the next experiment is an isolated one-expert memorization diagnostic for
  litter and surface energy, not another complete paid run;
- hydrology v2 remains in place while these two dominant families are tested.

Any resulting network-architecture change must be discussed with the user
before implementation.
