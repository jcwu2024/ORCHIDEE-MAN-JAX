# Target-Oriented Snowheat Recurrence Intervention v1

Date: 2026-08-26

Status: frozen before zero-training execution

Machine contract:
`manifests/coarse_graining/target_oriented_snowheat_recurrence_intervention_v1.json`

Contract SHA256:
`006e9d3d8ff63253ea2e70c995735856e9c3e93285934934fbbe3f7b16cab122`

## Question

Is the predicted `hydrol_previous_step_state.snowheat` state carried between
days the causal path that amplifies the two extreme maintenance-respiration
gradients?

The preceding attribution strongly implicates that path but compares complete
free rollout with complete Teacher-state daily evaluation. This intervention
changes exactly one cross-day state slice so that the causal claim does not
depend on all other Teacher state coordinates.

## Frozen Intervention

Use the same update-1000 Attention parameters, prepared train/train windows,
native forcing, daily operator, conservative update, retained exact tail, and
maintenance-respiration score in two modes:

1. **free reference**: carry the complete predicted next state into the next
   day;
2. **snowheat Teacher carry**: before each daily transition, replace only the
   `hydrol_previous_step_state.snowheat` coordinates of the predicted carry
   with that day's Teacher day-start snowheat coordinates.

The first day is an identity operation because its free initial state already
is the Teacher day-start state. All other continuous coordinates and the full
discrete state remain predicted. The exact slice must be resolved from
`DailyMarkovContract.state_leaves`; a hard-coded flat index is forbidden.

The intervention is diagnostic Teacher information. It is not training,
scheduled sampling, a proposed architecture, or legal final inference.

## Measurements And Gates

Use the same four 7/30-day extreme and matched-control windows as the accepted
gradient-path report. Compute one maintenance score and complete parameter
gradient per mode and window, for exactly eight gradients over all
`83,755,173` parameters. Report complete coarse/fine family norms, state-head
norms by learned field, and the cosine between the two modes.

Engineering acceptance requires all eight gradients to be finite, complete,
and nonzero; free-reference scores to reproduce the accepted report within
`atol=1e-8` or `rtol=1e-3`; and all physical hard failures to be zero.

Snowheat recurrence is causally confirmed only if both extreme windows reduce
the full maintenance gradient and the snowheat-head gradient by at least a
factor of ten. Matched controls are reported but not thresholded because the
accepted path audit already shows that Teacher-state behavior is strongly
scenario dependent. A reduction below two in both extremes rejects snowheat
as the isolated cause; a split result is partial evidence and opens only the
smallest coupled snow-thermal diagnostic.

These are diagnostic thresholds, not promotion gates.

## Execution Boundary

The test takes no optimizer or parameter step, retains no checkpoint or
recovery, does not regenerate Teacher v6, does not read historical v5,
validation values, or sealed test, and does not modify the Attention
architecture. It is restricted to free `gln01`. Paid training, 365-day
rollout, Gate D2, and product acceptance remain closed.
