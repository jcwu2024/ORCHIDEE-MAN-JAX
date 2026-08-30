# Structured-GPP Batch-Four Joint Training Result

Date: 2026-08-30

Status: training-stage capability accepted; scientific promotion deferred to
the frozen independent short-horizon screen

Slurm job: `14878155`

Executable commit:
`8115588bb9c0dbff7ced2e8d2db702453169d601`

Joint-training contract SHA256:
`168d44bdf8e1094ab4d683edaf8ebf576ff7dd27053d754419956cce30798832`

Report SHA256:
`cdfa8ac246350d516c7f9ac77d8ffe92b241a307c168c2bdb6af7f54b947dbe9`

Selected checkpoint SHA256:
`77cb405fc7cbcea3a394041a141c7d5180bd7068eaa309ccdc33a3995d295f2b`

## Execution

The paid job completed with exit code zero in `08:11:36`. All 248 effective
batch-four updates and 992 artifact presentations executed. The optimizer
continued from direct-GPP warm-start step 250 and finished at absolute step
498. All four train-only epoch panels strictly improved direct GPP, 7-day
science, and 30-day science, so epoch 4 supplied the selected diagnostic
checkpoint.

Only the `12,879,363` structured-GPP specialist coordinates were trainable.
All `93,504,164` retained coordinates remained exactly unchanged. Fourteen of
248 updates were clipped at the frozen `46.802861793665286` threshold. Every
gradient, optimizer, finite-value, state, budget, inventory, and checkpoint
gate passed.

The cumulative train-only panel reductions from the pre-joint endpoint were:

| Source | Reduction |
| --- | ---: |
| Direct GPP | `49.6886%` |
| 7-day science | `50.7018%` |
| 30-day science | `42.8952%` |

Teacher v6 was not regenerated, historical v5 was not used, validation and
sealed-test values were not read, and the model reconstructed neither 48
forcing records nor a 48-step state scan.

## Decision

This is training-stage success only. The checkpoint remained
`promotion_eligible=false` and could not open 365-day rollout, Gate D2, or
product acceptance. Its only next gate was the pre-existing independent
temporal/spatial/joint 96-window 7/30-day screen.

The screen result is recorded in
[`target_oriented_structured_gpp_short_horizon_screen_result_20260830.md`](target_oriented_structured_gpp_short_horizon_screen_result_20260830.md).
