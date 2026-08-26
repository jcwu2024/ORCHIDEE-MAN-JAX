# Target-Oriented Maintenance State Recurrence Intervention v1

Date: 2026-08-27

Status: frozen before zero-training execution

Machine contract:
`manifests/coarse_graining/target_oriented_maintenance_state_recurrence_intervention_v1.json`

Contract SHA256:
`cf4f11670df3fe3dc279802d24c23a5f8ebf6e6744ba47dac32de419cfa4e839`

## Question

After the complete snow-thermal alias cut, is the dominant direct-maintenance
gradient amplified by a nonphysical cross-day edge from the previous day's
persisted `resp_maint` state into the next day's maintenance Attention expert?

## Source Boundary

`stomate_resp.f90::maint_respiration` lines 122-151 does not consume prior
maintenance output. `stomate_npp.f90::npp_calc` lines 279-320 overwrites
`resp_maint` from the current day's accumulated `resp_maint_part`, and line
530 consumes that result in current-day NPP. The JAX day-end packet retains
`npp_update.resp_maint` as restart/Markov state, while the current neural
maintenance expert admits every continuous `state_carbon` field token.

`resp_maint_part` is explicitly excluded from this intervention. Fortran
`stomate.f90::stomate_main` lines 4943-4958 and the canonical JAX daily reset
set that accumulator to zero after its current-day use.

## Intervention And Matched Control

Reuse exactly the prior 30-day extreme and maintenance-loss-matched control.
The parent is the accepted four-slice snow-thermal alias Teacher-carry mode.
The treatment preserves those four carries and additionally replaces only
`slowproc_stomate_previous_step_state.resp_maint` from the current Teacher
day-start state before every transition. Every other continuous coordinate,
all discrete state, the Attention operator, parameters, forcing, conservative
updater, retained exact tail, and maintenance objective remain unchanged.

Compute exactly two complete gradients over all `83,755,173` parameters.
Report the full gradient, every parameter family, all primary/state heads,
and parent-to-treatment reductions for both the full gradient and direct
maintenance head.

Engineering acceptance requires both gradients to be finite, complete, and
nonzero; the single state slice and the four retained snow-thermal slices must
resolve by Markov key; and every physical hard failure must be zero.

The recurrence path is confirmed when both the extreme full gradient and
direct-maintenance-head norm fall by at least twofold. It is non-explanatory
when both fall by less than `1.25`-fold. Intermediate evidence is mixed. The
matched control is reported but not thresholded. These are causal-diagnostic
thresholds, not model-promotion gates.

## Execution Boundary

The audit takes no optimizer or parameter step, retains no checkpoint or
recovery, does not regenerate Teacher v6, and does not read historical v5,
validation values, or sealed test. It is restricted to free `gln01`. Input-
mask, source-routing, and architecture changes remain forbidden pending the
result and user discussion.
