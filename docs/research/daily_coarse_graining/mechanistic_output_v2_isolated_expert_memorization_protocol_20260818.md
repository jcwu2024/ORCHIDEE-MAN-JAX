# Mechanistic Output v2 Isolated-Expert Memorization Protocol

Date: 2026-08-18.

Status: frozen bounded diagnostic protocol. This is not a new architecture,
an Attention/control comparison, or a checkpoint-promotion experiment.

## Question

The full mechanistic-output v2 model can attain every Teacher output
coordinate, but joint seven-expert training leaves the largest residuals in
`litter_poc_doc` and `surface_energy`. Test each family independently to
separate two explanations:

1. the existing neural mapping cannot memorize the family even when trained
   alone;
2. the mapping is adequate, but the joint equal-expert optimization causes
   interference or an ineffective update allocation.

## Fixed Boundary And Data

Use the accepted `MechanisticStrongBaselineProcessHead` without changing its
49,743,001-parameter architecture, source routing, Attention blocks, output
coordinates, conservative updater, or retained tail.

Use the same 364 valid train/train v6 days from `001.0-071.0`, 1961. Every day
appears once per epoch through the existing deterministic padded batch-8
schedule. Do not regenerate Teacher data, read historical v5, read validation
or sealed test, reconstruct 48 forcing records, or execute a 48-step state
scan.

## Two Independent Diagnostics

Run from the same fresh initialization seed:

1. `litter_poc_doc` objective only;
2. `surface_energy` objective only.

For the selected expert, the training scalar is the existing
`loss.expert_mse[selected_expert]`. Its equal-label weighting and every field
transform remain unchanged. No label is duplicated or manually reweighted.
The complete 181-field prediction and conservative state transition are still
materialized on every forward pass. Unselected expert losses and the five
primary science metrics are reported but do not enter the isolated training
scalar.

## Budget

- fresh AdamW, no historical or failed-run checkpoint;
- batch size 8;
- at most 100 epochs and 4,600 updates per expert;
- existing 2% warmup plus cosine schedule from `1e-4` to `1e-5`;
- weight decay `1e-4` with the existing exclusions;
- no gradient clipping;
- evaluation at initialization, epoch 1, every 5 epochs, and the final epoch;
- one-epoch zero-retention smoke for each expert before either full run;
- no checkpoint or optimizer state written by smoke or full diagnostic.

The two independent full diagnostics may use the two free `gln01` GPUs in
parallel only after both one-epoch smokes pass. No paid Slurm job is opened.

## Required Reporting

For each run report:

- selected-expert and complete seven-expert loss histories;
- every selected label and field at initialization and final evaluation;
- unchanged active-day and defined-value counts;
- first and final observed gradient and AdamW update coverage for shared
  parameters, the selected expert blocks, other expert blocks, selected label
  heads, and other label heads;
- all five primary science metrics as diagnostics;
- prediction/state nonfinite counts, water/carbon budgets, and negative
  inventory counts.

## Gates

The one-epoch smoke opens its full diagnostic only if all 46 updates complete,
all gradients and parameters remain finite, the selected expert loss falls,
all reporting counts are unchanged, and every hard count is zero.

The full isolated-expert diagnostic passes neural memorization only if:

- all scheduled updates are finite;
- selected-expert loss reduction is at least `99%`;
- every selected field whose initial mean daily loss is at least `1e-4`
  either falls by at least `95%` or finishes at or below `1e-4`;
- selected expert-block and selected label-head gradient coverage are nonzero,
  with nonzero realized AdamW updates;
- all active/count masks are unchanged;
- every state/budget/nonfinite/nonnegative hard count is zero.

Unselected expert or primary-science regression is reported and prevents any
checkpoint promotion, but does not invalidate the narrow memorization answer.
No checkpoint is promoted or retained under any outcome.

## Decision

- Both pass: retain the architecture and address joint-objective interference
  or training allocation before another complete run.
- One fails aggregate memorization: the existing mapping is inadequate for
  that process family; discuss an architecture change with the user before
  implementation.
- Aggregate passes but the dominant-field gate fails: diagnose intra-expert
  label/field weighting before changing the backbone.

This protocol does not open 64-point-year training, rollout, Gate D2, sealed
test, or product acceptance.
