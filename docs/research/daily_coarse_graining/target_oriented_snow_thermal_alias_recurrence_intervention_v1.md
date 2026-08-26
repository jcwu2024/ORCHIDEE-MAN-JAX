# Target-Oriented Snow Thermal Alias Recurrence Intervention v1

Date: 2026-08-27

Status: frozen before zero-training execution

Machine contract:
`manifests/coarse_graining/target_oriented_snow_thermal_alias_recurrence_intervention_v1.json`

Contract SHA256:
`5ab495baf5c140d71e841defaa6e799d8c6419717944e6aa38ab37b5ecd77f63`

## Question

Did the prior coupled intervention leave a cross-day path open because the
Markov state stores both canonical `hydrol` snowheat/snowtemp coordinates and
their `diffuco` mirrors?

## Mechanism And Intervention

The accepted v6 Markov contract contains four prognostic slices:

- `diffuco_previous_step_state.snowheat` at `[36:39]`;
- `diffuco_previous_step_state.snowtemp` at `[42:45]`;
- `hydrol_previous_step_state.snowheat` at `[510:513]`;
- `hydrol_previous_step_state.snowtemp` at `[520:523]`.

The fast-owner contract declares each `diffuco` leaf a mirror of the matching
`hydrol` canonical owner. The previous intervention replaced only the two
`hydrol` slices. Its residual therefore cannot distinguish a same-day path
from a cross-day path retained through the two mirrors.

The new treatment replaces all four slices from the current Teacher day-start
state before each transition. It must resolve the slices by Markov key and
verify both mirror relations from the fast-owner contract. Every other
continuous coordinate and the full discrete state remain predicted. The
Attention operator, forcing, parameters, conservative updater, retained exact
tail, and maintenance target remain unchanged.

This is a causal diagnostic only. It is forbidden in training and final
inference.

## Matched Control And Gates

Reuse exactly the prior 30-day extreme window and maintenance-loss-matched
control. Compute one complete maintenance gradient for each window, covering
all `83,755,173` parameters. Report the full gradient, all parameter families,
all primary/state output heads, and the parent-to-treatment reductions.

Engineering acceptance requires both gradients to be finite, complete, and
nonzero; all four slices and both mirror relations must resolve; and every
physical hard failure must be zero.

The alias path is confirmed when the extreme snowheat-head norm falls by at
least tenfold and the full gradient falls by at least twofold relative to the
accepted coupled parent. It is classified as non-explanatory only when the
head falls by less than twofold and the full gradient by less than 1.25-fold.
Intermediate evidence is mixed. The matched control is reported but is not a
threshold gate.

These are bounded diagnostic thresholds, not model-promotion gates.

## Execution Boundary

The audit takes no optimizer or parameter step, retains no checkpoint or
recovery, does not regenerate Teacher v6, and does not read historical v5,
validation values, or sealed test. It is restricted to free `gln01`. Paid
training, architecture edits, 365-day rollout, Gate D2, and product acceptance
remain closed.
