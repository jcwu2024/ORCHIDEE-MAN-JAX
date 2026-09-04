# Coupled GPP Localized Jump Result (2026-09-04)

Status: accepted zero-training localization; a matched snowheat intervention
is required before assigning causality

Report SHA256:
`cdca61cdcdf88ff0fd1fc8a07cfe07ead7180438844b7618c3f81f94281f618e`

Git commit: `e436b96e6a63b49bf9cc541b5231a569fe72a78f`

## Execution And Integrity

The free-`gln01` GPU execution completed in `1218.624 s` with exit code zero.
Peak active JAX memory was `11,016,873,728` bytes. All control and treatment
physical hard-failure counts are zero.

The rejected logical-update-3 direction was reconstructed from the 26
nonzero rows in its original 43-source order: 24 direct-GPP artifact rows,
the reachable-state row, and the 30-day row. Its reconstructed norm is
`0.0253903907913`, matching the historical norm. The historical nominal
`1/64` displacement is `0.000121233958610`; the realized float32 parameter
displacement is `0.000121277465951`.

Exact-zero control and the reconstructed treatment reproduce the historical
artifact-550 aggregate exactly:

| Mode | Seven-day objective |
| --- | ---: |
| control | `0.0832977795` |
| treatment | `0.0883991438` |

## Localization

Only the fourth of artifact 550's eight seven-day windows has a material
regression. Its objective changes from `0.2560620074` to `0.3111115816`, a
`21.4985%` worsening. The other seven windows are unchanged at practical
scale or improve.

The selected window has no discrete-state difference and no defined-mask
difference on any day. Its first material continuous-state difference occurs
on Day 2 in snow layer 2 of both mirror copies of `snowheat`:

- `diffuco_previous_step_state.snowheat`, global coordinate `38`;
- `hydrol_previous_step_state.snowheat`, global coordinate `512`.

The normalized difference grows from `0.00185488` on Day 2 to `0.00324600`
on Day 3 and `0.00417342` on Day 4. GPP remains nearly identical through Day
4. On Day 5 it jumps from `2.76733630` in the control to `4.86989972` in the
treatment, a difference of `+2.10256342`. Growth respiration and NPP then
increase by `+0.58871836` and `+1.51384721`; AGB and BGB increase by
`+0.79463513` and `+0.71857271`. Days 6 and 7 retain the biomass offset while
the largest normalized state differences move to leaf-age fractions.

## Source-Path Audit

The current source router classifies the `diffuco` mirror as `state_canopy`
and the `hydrol` mirror as `state_water`. The `canopy_gpp` expert admits both
families. Consequently both snowheat copies can reach GPP through the parent
state-token/axis-token bank and through the inherited GPP hidden state. The
structured specialist's potential, limitation, and correction branches all
retain those non-forcing source tokens. The newer coordinate-preserving
carbon-feedback adapter reads only
`slowproc_stomate_previous_step_state`, so it is not the direct snowheat
route.

This establishes reachability, not causality. The first material state and
the later GPP jump are separated by three daily transitions. A matched
intervention must remove only the treatment-control snowheat difference and
show whether the Day-5 GPP jump disappears.

## Decision

The localized trace gate passes and is closed. No architecture defect is yet
declared. The next gate is the frozen matched snowheat-carry intervention in
[`target_oriented_coupled_gpp_snowheat_causal_intervention_v1.md`](target_oriented_coupled_gpp_snowheat_causal_intervention_v1.md).

No optimizer update, checkpoint, recovery, Teacher regeneration, historical
v5 read, fresh-validation read, sealed-test read, 48-step scan, architecture
change, or paid execution occurred.
