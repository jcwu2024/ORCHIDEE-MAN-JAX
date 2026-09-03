# Persistent Scale-Selected Coupled-GPP Training Result

Date: 2026-09-03

Status: stopped at the update-10 scientific gate; no checkpoint selected

Slurm job: `14911228`

Executable commit: `c0273d99b61e9b02c41e22a340823af27184e07f`

Result SHA256:
`764b14e169345ba119407b9e3dcf3a7e9141900c2710c8afa2775dd0fe881168`

Progress SHA256:
`ab185bd22a0fbd2da1e7891f07ab950e17e4b8ebec7a69f30170a38f5be28197`

Recovery SHA256:
`1f63d6c5326200253762c48fe004cf18977e4fdd34aed4f6af3db73f4eca8b52`

## Execution

The corrected paid run completed with Slurm state `COMPLETED`, exit code
`0:0`, and elapsed time `02:18:50`. The runner executed ten persistent updates
and then returned status `stopped_scientific_gate`, classification
`milestone_scientific_gate`, and next action
`stop_and_attribute_direct_gpp_generalization_failure`.

The ten selected scales were:

```text
1, 1/2, 1/2, 1, 1/2, 1/2, 1, 1, 1, 1
```

All ten updates found a finite common direction and decreased all four source
losses on their own eight-artifact training batch. Update time was
`356.464-778.356` seconds, with median `406.215` seconds and total
`4,318.087` seconds. The first update included the new production gradient
compilation.

The run presented 80 distinct train/train artifacts spanning 39 runtime years
from 1961 to 2004. This corresponds to 16,640 direct-GPP teacher days, 640
seven-day windows, and 320 thirty-day windows. The result is therefore not a
one-point or repeated-tiny-batch experiment.

At the published additive rates, the `02:18:50` allocation corresponds to
approximately CNY 1.30 for eight CPUs plus CNY 5.09 for one GPU, or CNY 6.39
total before any center-side billing granularity.

## Milestone-10 Result

Lower values are better. The fixed independent monitor changed as follows:

| Source | Initial | Update 10 | Change |
| --- | ---: | ---: | ---: |
| direct GPP on teacher states | 0.06676850 | 0.08289720 | +24.1562% |
| reachable one-day state | 0.02383366 | 0.02021957 | -15.1638% |
| 7-day component-resolved science | 0.07522360 | 0.05298558 | -29.5626% |
| 30-day component-resolved science | 0.08703911 | 0.05934533 | -31.8176% |

Direct GPP improved on only one of the eight monitor artifacts. The seven
regressions range from `+5.75%` to `+134.02%`. This violates the frozen
update-10 condition that direct GPP may not be worse than the selected
update-7 state, so `continuation_passed=false` and the runner stopped at the
first production milestone.

The longer-horizon behavior moved in the intended direction. Cumulative GPP
loss fell `30.2653%` at seven days and `33.9401%` at thirty days. The science
macro normalized RMSE fell from `0.292588` to `0.250504` at seven days and
from `0.321744` to `0.278150` at thirty days.

| Science key | 7-day change | 30-day change | Update-10 absolute gate |
| --- | ---: | ---: | --- |
| GPP | -30.27% | -33.94% | fail / fail |
| maintenance respiration | -2.72% | -8.56% | pass / pass |
| growth respiration | -31.24% | -35.84% | fail / fail |
| heterotrophic respiration | unchanged at zero | unchanged at zero | pass / pass |
| autotrophic respiration | -28.38% | -32.46% | fail / fail |
| NPP | -30.61% | -33.99% | fail / fail |
| NEP | -30.61% | -33.99% | fail / fail |
| NEE | -30.61% | -33.99% | fail / fail |
| aboveground biomass carbon | -29.08% | -30.97% | pass / pass |
| belowground biomass carbon | -31.55% | -35.78% | pass / pass |
| total live biomass carbon | -30.62% | -34.00% | pass / pass |
| nonliving ecosystem carbon | +18.38% | -0.66% | pass / pass |
| ecosystem carbon | +17.03% | -2.27% | pass / pass |

The stock errors remain small and pass, but the absolute GPP, NPP,
growth/autotrophic respiration, NEP, and NEE gates still fail. The long-horizon
improvement is material evidence, not product acceptance.

## Physical And Persistence Result

All physical, finite, budget, stock, and exact-discrete gates passed. Fixed
parameters remained exactly unchanged. Peak JAX bytes in use remained
`4,114,680,576`; the allocator limit was approximately 25.6 GB.

No Teacher data were regenerated, no historical v5 rows were used, no fresh
or sealed validation was read, and no 48-record forcing reconstruction or
subdaily state scan occurred.

No checkpoint was written or selected. The final parameter-only
`recovery.pkl` is retained remotely for deterministic attribution, but it is
not promotion eligible and must not be used to resume unchanged training.

## Interpretation And Next Gate

This run rejects the current per-update admission rule as sufficient for
persistent training. Every scheduled batch accepted its own direct-GPP step,
yet the held-aside direct-GPP monitor regressed on seven of eight artifacts.
The dominant failure is therefore batch-to-batch direction transfer for the
day-local GPP mapping, not an inability to decrease the four objectives on a
single batch.

The result does not reject the Attention architecture. It also does not erase
the approximately 30-34% improvements in cumulative GPP, NPP, flux balance,
and live-biomass errors. Those gains may nevertheless depend on compensating
rollout states, so relaxing the direct-GPP gate without attribution would be
unsafe.

Before any further training, reconstruct the accepted ten-update trajectory
without retaining new parameters and evaluate the fixed monitor after every
step. At the first direct-GPP regression, compare the scheduled-batch and
monitor-group directional derivatives and per-artifact gradient alignment.
This bounded zero-training audit must distinguish:

1. a sampling-variance failure repairable by a larger or multi-group effective
   batch;
2. incompatible direct-GPP directions requiring robust group constraints; or
3. a systematic teacher-state versus free-rollout compensation that requires
   changing the loss or acceptance logic.

Do not resume the recovery, read fresh validation, run 365 days, open Gate D2,
or change the architecture until this attribution is complete.
