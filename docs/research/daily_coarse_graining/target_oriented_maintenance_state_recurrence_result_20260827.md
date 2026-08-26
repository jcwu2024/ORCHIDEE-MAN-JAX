# Target-Oriented Maintenance State Recurrence Result

Date: 2026-08-27

Status: accepted zero-training diagnostic; persisted maintenance recurrence is
not explanatory

Report SHA256:
`a95ae1291a1d4cc77bb026448b4c31e8244604dee9485886c5f7079272f809b9`

Contract SHA256:
`cf4f11670df3fe3dc279802d24c23a5f8ebf6e6744ba47dac32de419cfa4e839`

Git commit: `d08d6ebb20ecb8ea724a181d493ef6572e8e9778`

## Result

Both complete 30-day gradients are finite and nonzero over all `83,755,173`
parameters. The four accepted snow-thermal slices and the added
`slowproc_stomate_previous_step_state.resp_maint [3516:3518]` slice resolve
exactly. Every physical hard-failure count is zero. The 30 declared dynamic
state-mask differences in each rollout are the preregistered Teacher-carry
intervention, not unexpected mask failures.

| Window | Full gradient before | Full gradient after | Reduction | Maintenance head before | Maintenance head after | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 30-day extreme | `33.69151431` | `33.69151394` | `1.000000011x` | `33.46682558` | `33.46682500` | `1.000000017x` |
| matched control | `15.67715819` | `15.67715838` | `0.999999988x` | `15.52365488` | `15.52365499` | `0.999999993x` |

The extreme result is far below the frozen `1.25x` no-effect boundary for
both measurements. The preregistered classification is therefore
`maintenance_state_recurrence_not_explanatory`.

No optimizer or parameter step was taken. No checkpoint or recovery was
written. Teacher v6 was not regenerated, historical v5 was not read, and no
validation values or sealed-test data were accessed. The Attention model,
input mask, source routing, conservative updater, and retained exact tail were
unchanged.

Peak active JAX memory was `2,203,474,432` bytes, about `2.05 GiB`. The audit
took `1973.44 s`; total process time was `2093.07 s`. It ran on free `gln01`
without Slurm or a paid allocation.

## Interpretation

Persisted `resp_maint` is visible to the broad neural `state_carbon` route,
but that nonphysical cross-day edge has negligible causal effect on these
maintenance gradients. This result supplies no evidence for deleting the
field from the maintenance input route. `resp_maint_part` remains outside the
hypothesis because its day-end reset is source exact.

After the accepted four-slice snow-thermal carry, the remaining extreme norm
is almost entirely the direct gradient of the 30-day maintenance objective
with respect to `daily.fast_maintenance_respiration`. That is a loss/output-
head optimization question, not another demonstrated Teacher-carry state
failure. The absolute norm `33.47` is not itself a failure threshold; its
meaning must be tested through a realized finite step and the required
7/30-day science protections.

## Next Gate

Return to non-recurrence objective and optimization design. Freeze a
zero-training maintenance-head conditioning audit at the update-1000
checkpoint. It must keep the current Attention architecture and all routing
unchanged, isolate the existing independent maintenance final head, measure
7/30-day group geometry and realized finite response, and protect GPP, growth
respiration, total autotrophic respiration, NPP, one-day state, and carbon
stocks.

Do not continue arbitrary state-by-state Teacher carry. Do not edit the
maintenance input mask, source routing, output architecture, or training loss
before that audit is frozen and locally qualified. Paid training, 365-day
rollout, Gate D2, sealed test, and product work remain closed.
