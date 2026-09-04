# Coupled GPP Snowheat Causal Intervention V1

Date: 2026-09-04

Status: frozen before zero-training execution

## Question

Does the small rejected parameter displacement create the Day-5 GPP jump in
artifact 550's fourth seven-day window through the two mirrored snowheat
state coordinates, or is snowheat merely the earliest visible passenger of a
different recurrent path?

## Matched Intervention

Reconstruct the exact update-2 control parameters and historical update-3
`1/64` treatment exactly as in the accepted localized-jump trace. Reuse only
artifact `550`, runtime year `1976`, seven-day window index `3`.

Evaluate exactly three trajectories:

1. **control free**: update-2 parameters and ordinary predicted state carry;
2. **treatment free**: the rejected `1/64` treatment and ordinary predicted
   state carry;
3. **treatment matched-snowheat carry**: the same treatment parameters, but
   before every daily transition replace only
   `diffuco_previous_step_state.snowheat` and
   `hydrol_previous_step_state.snowheat` with the control-free trajectory's
   same-day start values.

Day 1 is necessarily unchanged because both free trajectories start from the
same state. All other continuous state, all discrete state, forcing,
parameters, the conservative updater, and the retained exact tail remain
those of the treatment. Both aliases are replaced together so the diagnostic
does not create a physically inconsistent mirror pair. This uses no future
Teacher state.

## Measurements

For all three trajectories, report the seven daily GPP values, the selected
window science objective, physical hard counts, and the two snowheat slices
at every day start.

At each day, also expose the existing structured-GPP factorization without
changing it:

```text
normalization scale * softplus(potential)
                    * sigmoid(limitation)
                    * exp(bounded correction)
                    * vcmax25 / 100
```

Report each positive factor, pre-VCMAX GPP, routed GPP, and the factor whose
absolute log-ratio contributes most to the free treatment-control Day-5 GPP
ratio. The reported routed product must reproduce the process-head GPP.

## Bounded Classification

Let `D_free` be the absolute Day-5 GPP difference between treatment free and
control free, and `D_cut` the absolute difference between matched-snowheat
treatment and control free.

- `snowheat_path_causal_for_day5_jump`: `D_cut <= 0.1 * D_free`;
- `snowheat_path_not_explanatory_for_day5_jump`: `D_cut >= 0.75 * D_free`;
- otherwise: `snowheat_path_partial_for_day5_jump`.

The selected-window objective displacement is corroborating evidence and is
reported but does not replace the direct Day-5 criterion. Engineering
acceptance additionally requires the free control and treatment trajectories
to reproduce the accepted parent trace, finite factors, exact mirror
resolution, factorized routed-GPP agreement, and zero physical hard failures.

Only a causal result can justify discussing a narrower GPP input route or
snow-state treatment. A partial or negative result continues attribution to
the other Day-5 GPP inputs. Any architecture or routing change must be
discussed with the user before implementation.

## Execution Boundary

This is one free-`gln01` zero-training diagnostic. It applies one historical
parameter displacement in memory only and writes a report only. It does not
retain parameters, invoke an optimizer, regenerate v6, read v5, inspect fresh
validation or sealed test, reconstruct 48 half-hour forcing records, run a
48-step candidate scan, change the architecture, or open paid training,
365-day rollout, Gate D2, or product acceptance.
