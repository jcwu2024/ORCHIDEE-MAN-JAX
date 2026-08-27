# Target-Oriented Source-Shaped Branch Training Result

Date: 2026-08-27

Status: completed engineering run; scientific gate failed; checkpoint is
diagnostic only

Slurm job: `14701638`

Executable commit: `7fdce6538869289a638cab4763bd506d17dcb8c1`

Report SHA256:
`a5143c066f611ef3678216cb76c1c8eebf679e5726050ed5cc6265516b2d4b22`

Branch-only checkpoint SHA256:
`fd6b866ec7a7c74f13dd7485a11d14126a7c39a689e1b9e2836bfd11079bd153`

Screen contract SHA256:
`a06beb815834142f44939022eea6ebd044e0a096b5d3c443754a9d87fb0f8cee`

## Execution Result

The paid `gnall` job completed with exit code zero in `00:58:51`. All 100
planned optimizer updates executed against prepared train/train artifacts
`252..351`. The complete forward retained the 84,148,646-parameter Attention
candidate, while only the 394,242 source-shaped maintenance-branch parameters
were differentiated and updated. All 83,754,404 retained coordinates remained
exactly unchanged.

Four of 100 updates were clipped at the frozen p90 norm `0.1973866669`. The
raw branch-gradient norm ranged from `0.0028030` to `0.3266585`, with median
`0.0475639`. Every update had finite, nonzero gradients and parameter steps.
The final branch-only checkpoint was written as required and remains
`promotion_eligible=false`.

Teacher v6 was not regenerated, historical v5 was not used, and validation or
sealed-test values were not read. The run reconstructed no 48-record forcing
sequence and executed no 48-step state scan.

## Frozen Screen

The source mechanism's maintenance improvement became larger after training:

| Horizon | Update 0 reduction | Update 100 reduction | Change |
| --- | ---: | ---: | ---: |
| 7 days | `73.6076%` | `90.7604%` | `+17.1528` percentage points |
| 30 days | `87.3457%` | `96.5044%` | `+9.1587` percentage points |

The same update moved most protected carbon quantities in the wrong direction.
The entries below are median score changes relative to the fixed update-1000
parent; positive means larger error than the parent and negative means smaller
error.

| Protected score | 7-day update 0 | 7-day update 100 | 30-day update 0 | 30-day update 100 |
| --- | ---: | ---: | ---: | ---: |
| GPP | `+0.5403%` | `+0.5403%` | approximately `0%` | approximately `0%` |
| Growth respiration | `+69.9249%` | `+85.4734%` | `-8.6031%` | `+0.5013%` |
| Autotrophic respiration | `-24.2984%` | `-45.7827%` | `-63.8734%` | `-71.0083%` |
| NPP | `+27.9226%` | `+33.2066%` | `-20.7417%` | `-5.8534%` |
| AGB | `+18.3846%` | `+39.2159%` | `-22.4605%` | `-10.4991%` |
| BGB | `+19.0799%` | `+25.1768%` | `-4.9776%` | `+2.2964%` |
| Total live biomass | `+29.1662%` | `+39.2284%` | `-19.1439%` | `-3.6291%` |
| Ecosystem carbon | `-2.8360%` | `-2.4990%` | `-0.9523%` | `-0.3368%` |

The relative maintenance gate passes, but the protected-score gate and the
complete relative source gate fail. No 7-day group (`0/8`) and only one
30-day group (`1/8`) passes every absolute target-oriented key. Scientific
promotion therefore fails with classification
`source_mechanism_not_yet_supported`.

All prediction/state nonfinite, unexpected-mask, discrete, budget, negative-
stock, and final-screen hard-failure counts are zero. Declared-dynamic mask
changes remain diagnostic and are not hard failures.

## Interpretation

This result does not reject the coordinate-aligned Attention backbone, and it
does not show that the source-shaped maintenance formula lacks signal. The
formula now removes most maintenance error on both horizons. It shows that
optimizing only the two maintenance branches cannot restore the coupled GPP,
growth-respiration, NPP, and biomass balance while every other output head is
frozen. The parent contains compensating flux errors; improving one term in
isolation can expose rather than repair the remaining imbalance.

More updates with the same branch-only objective are not admitted. The
update-100 checkpoint is retained only to attribute the observed coupling,
not as a continuation or validation checkpoint.

## Next Gate

Freeze and execute a zero-training source-coupling attribution on the existing
fixed parent, update-0 source candidate, and update-100 branch checkpoint. It
must use only the same train/train panel and must separate direct daily carbon
identity effects from cross-day state recurrence. The output is a signed
residual and intervention account for GPP, maintenance respiration, growth
respiration, autotrophic respiration, NPP, and live-biomass tendency. Its
decision is whether a bounded successor must jointly expose/train additional
flux heads or whether the failure is dominated by rollout-state coupling.

No optimizer continuation, validation read, broader training, 365-day
rollout, Gate D2, sealed test, or product work is authorized by this result.
