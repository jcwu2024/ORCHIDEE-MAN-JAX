# Target-Oriented Snow Thermal Recurrence Intervention v1

Date: 2026-08-27

Status: frozen before zero-training execution

Machine contract:
`manifests/coarse_graining/target_oriented_snow_thermal_recurrence_intervention_v1.json`

Contract SHA256:
`959413d639a6bd853b15ca2fc896beea4ea5e690463f4a78e8c3cac1ccaf96ca`

## Question

Does the coupled snowheat/snowtemp cross-day state explain the maintenance-
gradient residual left by the accepted snowheat-only intervention in the
30-day extreme window?

## Frozen Intervention

Use only the accepted 30-day extreme and matched-control windows. The parent
is the accepted snowheat Teacher-carry mode. The treatment additionally
replaces `hydrol_previous_step_state.snowtemp`, so both snowheat and snowtemp
come from the current Teacher day-start state before each daily transition.

All other continuous coordinates and the complete discrete state remain
predicted. Parameters, forcing, Attention operator, conservative updater,
retained exact tail, and maintenance target are unchanged. Both state slices
must be resolved by key from `DailyMarkovContract.state_leaves`; flat indices
are forbidden.

This is a diagnostic intervention, not a training method or legal final
inference path.

## Measurements And Gates

Compute exactly two snowheat-plus-snowtemp maintenance gradients, one for each
30-day window. Each covers all `83,755,173` parameters and reports complete
parameter-family and output-head attribution. Compare the results with the
hash-bound snowheat-only parent report.

Engineering acceptance requires both gradients to be finite, complete, and
nonzero and every physical hard failure to be zero. Coupled snow thermal
recurrence is causally confirmed if the extreme full gradient and its
snowtemp-head gradient both fall by at least a factor of ten relative to the
snowheat-only parent. The control is reported but not thresholded. A full-
gradient reduction below two rejects the coupled slice as the remaining
cause; intermediate evidence remains mixed.

These thresholds diagnose a path. They cannot promote the model or authorize
training.

## Execution Boundary

The test takes no optimizer or parameter step, retains no checkpoint or
recovery, does not regenerate Teacher v6, and does not read historical v5,
validation values, or sealed test. It is restricted to free `gln01`. Paid
training, architecture edits, 365-day rollout, Gate D2, and product acceptance
remain closed.
