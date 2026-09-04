# Coupled GPP Day-5 Crossed Attribution V1

Date: 2026-09-04

Status: frozen before zero-training execution

## Question

The accepted artifact-550 trace contains a one-day GPP jump from `2.76733630`
to `4.86989972`. Matching both snowheat mirrors leaves `100%` of that jump,
while the existing potential factor carries virtually its entire log ratio.

This audit asks whether the potential jump is caused primarily by the rejected
parameter displacement acting directly on the Day-5 map, by the different
state carried into Day 5, or by their interaction. It then resolves any state
effect across the six GPP-admitted state owners and any parameter effect
across the two trainable parameter families. It changes no model mechanism.

## Frozen Inputs

Reuse only artifact `550`, runtime year `1976`, seven-day window index `3`,
and Day-5 index `4`. Reconstruct exactly the accepted update-2 control
parameters and historical update-3 `1/64` treatment from the same 26 nonzero
gradient sources used by the localized and snowheat audits.

Run the ordinary control and treatment seven-day trajectories only to obtain
their matched Day-5 start states. No Teacher future state is used. The Day-5
forcing, static inputs, PFT inputs, retained inputs, and discrete state are
identical in every crossed evaluation.

## Parameter-State Cross

Let `G(P, S)` be routed Day-5 PFT14 GPP with parameter set `P` and same-day
start state `S`. Evaluate all four cells:

| Cell | Parameters | Day-5 start state |
| --- | --- | --- |
| `G_CC` | control | control |
| `G_TC` | treatment | control |
| `G_CT` | control | treatment |
| `G_TT` | treatment | treatment |

`G_CC` and `G_TT` must reproduce the accepted free trajectories. For each
cell, evaluate the complete bound daily operator once and report physical
budget/finite-state checks. Also expose the existing normalization, potential,
limitation, correction, and final `vcmax25 / 100` factors.

Define `J = G_TT - G_CC` and

```text
R_matched_state = abs(G_TC - G_CC) / abs(J).
```

- `state_path_dominant_for_day5_jump` when `R_matched_state <= 0.10`;
- `state_path_not_explanatory_for_day5_jump` when
  `R_matched_state >= 0.75`;
- `mixed_parameter_state_path_for_day5_jump` otherwise.

Also report the symmetric two-factor Shapley decomposition:

```text
parameter = 0.5 * ((G_TC - G_CC) + (G_TT - G_CT))
state     = 0.5 * ((G_CT - G_CC) + (G_TT - G_TC)).
```

These terms must sum to `J` at floating-point tolerance. They are explanatory
measurements; the bounded matched-state rule controls the classification.

## State-Owner Controls

At fixed treatment parameters, evaluate both necessity and sufficiency for
the six continuous state owners admitted by `canopy_gpp`:

| Source family | Markov component | Coordinates |
| --- | --- | ---: |
| `state_canopy` | `diffuco_previous_step_state` | `52` |
| `state_energy` | `enerbil_previous_step_state` | `141` |
| `state_water` | `hydrol_previous_step_state` | `622` |
| `state_thermal` | `thermosoil_previous_step_state` | `512` |
| `state_carbon` | `slowproc_stomate_previous_step_state` | `2,386` |
| `state_driver` | `driver_previous_step_state` | `11` |

For necessity, start from treatment state and replace exactly one complete
owner with its control value. For sufficiency, start from control state and
insert exactly one complete treatment owner. No defined mask or discrete
state is changed. `state_finalize` is excluded because the frozen router does
not admit it to `canopy_gpp`.

Owner capture and recreation are normalized to the full state effect at fixed
treatment parameters. An owner is reported as individually material only when
the primary cross does not reject state mediation and either its necessity
capture or sufficiency fraction is at least `10%`. This flag does not authorize
an architecture edit; interacting owners remain possible.

## Parameter-Family Controls

The historical displacement changes exactly two accepted trainable families:

- `target_oriented.structured_gpp`: `12,879,363` parameters;
- `target_oriented.coupled_gpp_carbon_feedback`: `3,080,832` parameters.

At both control and treatment Day-5 states, evaluate a hybrid with only one
family taken from the treatment and the other retained from the control.
Report each family's sufficiency, necessity, and the two-family interaction.
The hybrids are diagnostic values only and are never retained.

## Bounded Path Check

At interpolation fractions `0, 1/8, 1/4, 1/2, 3/4, 1`, report four finite
potential/GPP paths:

1. control-to-treatment parameters at control state;
2. control-to-treatment parameters at treatment state;
3. control-to-treatment state at control parameters;
4. control-to-treatment state at treatment parameters.

This distinguishes a smooth large sensitivity from a narrow nonlinear
transition without searching a result-dependent grid. The endpoint values
must reproduce the four crossed cells.

## Engineering Acceptance

Acceptance requires all of the following:

- the accepted snowheat report has SHA256
  `a19c6230e14d50b204d3d5ebf0eb70c721c3d0e7683d06d03083313da15a2e23`;
- free control and treatment GPP traces reproduce that report;
- Day-5 control/treatment discrete start states are identical;
- all six source masks have the frozen coordinate counts, are disjoint, and
  resolve to GPP-admitted source families;
- both trainable parameter family counts are exact;
- every crossed, owner, family, and interpolation factor is finite, and routed
  GPP is nonnegative;
- factorized and complete-operator GPP agree at relative tolerance `2e-7` for
  all four crossed cells;
- each crossed complete-day result has finite defined state and water/carbon
  budget residual no larger than `2e-12`;
- both Shapley additivity and all interpolation endpoints pass their declared
  numerical checks.

An engineering failure must still write the complete report before returning
a nonzero status.

## Execution Boundary

This is one free-`gln01` zero-training diagnostic. It applies one historical
parameter displacement in memory and writes one report. It creates no
optimizer, retains no parameters, writes no checkpoint or recovery, regenerates
no Teacher data, reads neither v5 nor fresh validation nor sealed test,
reconstructs no half-hour forcing records, performs no 48-step state scan, and
opens no paid training, 365-day rollout, Gate D2, or product acceptance.

If the result implicates a network parameter family, source route, or
architecture mechanism, implementation stops for discussion with the user.
