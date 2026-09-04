# Coupled GPP Snowheat Causal Result (2026-09-04)

Status: accepted zero-training negative causal result; snowheat is not the
carrier of the localized Day-5 GPP jump

Report SHA256:
`a19c6230e14d50b204d3d5ebf0eb70c721c3d0e7683d06d03083313da15a2e23`

Executable commit:
`67fae75d5da7080336cdec320f2182233ce45802`

## Execution And Integrity

The free-`gln01` GPU execution completed in `890.625 s` with status `passed`.
Peak active JAX memory was `11,016,873,728` bytes. It reconstructed the
historical logical-update-3 `1/64` parameter displacement in memory and wrote
only this report. All three trajectories have zero physical hard failures.
The free control and treatment GPP traces and selected-window scores reproduce
the accepted parent trace.

The exposed structured-GPP factor product differs from the fused process-head
output by at most `8.04796e-7`, or `1.65260e-7` relatively. The first complete
report at commit `8648a59` preserved these values but failed an overly strict
`1e-7` relative check. The accepted runner uses the bounded `2e-7` relative
tolerance, still rejects a `2.2e-7` regression case, records the tolerance in
the report, and passes all three factorization checks. This changes no model
value or scientific classification.

## Matched Intervention

Both snowheat mirrors in the treatment trajectory were replaced together by
their control-free same-day start values before every daily transition. All
other state, forcing, parameters, the constrained updater, and the retained
exact tail remained those of the treatment.

| Day | Control free GPP | Treatment free GPP | Treatment matched-snowheat GPP |
| ---: | ---: | ---: | ---: |
| 1 | `0.87114341` | `0.87099466` | `0.87099466` |
| 2 | `5.32855706` | `5.32856674` | `5.32856674` |
| 3 | `5.32856178` | `5.32857117` | `5.32857117` |
| 4 | `4.86990465` | `4.86989667` | `4.86989667` |
| 5 | `2.76733630` | `4.86989972` | `4.86989972` |
| 6 | `2.76733476` | `2.76711556` | `2.76711556` |
| 7 | `3.78151127` | `3.78155177` | `3.78155177` |

The Day-5 free jump is `2.10256342`. Matching snowheat removes exactly none of
it at report precision: retained jump fraction is `1.0`. The selected-window
objective is likewise unchanged from `0.311111589937405` to
`0.311111589937248`; its displacement from the `0.256062005456361` control is
retained at `0.999999999997151`.

The frozen classification is therefore
`snowheat_path_not_explanatory_for_day5_jump`.

## GPP Factor Attribution

At Day 5, routed GPP's treatment-control log ratio is `0.56518781`. Almost all
of it comes from the existing potential branch:

| Factor | Control | Treatment | Log ratio |
| --- | ---: | ---: | ---: |
| potential | `0.99127793` | `1.74440932` | `0.56517633` |
| limitation | `0.72920895` | `0.72921473` | `7.92863e-6` |
| correction | `3.27080553` | `3.27081716` | `3.55307e-6` |
| normalization scale | `1.69713719` | `1.69713719` | `0` |
| `vcmax25 / 100` | `0.689672502` | `0.689672502` | `0` |

The matched-snowheat factors are identical to the free-treatment factors.
Thus the jump is inside the potential path, but is not transmitted by either
snowheat mirror.

## Decision And Next Gate

Snowheat reachability and temporal precedence are real, but the causal
hypothesis is rejected. This result does not justify deleting or narrowing a
snowheat input route.

Before any architecture or training change, the next gate is a zero-update
Day-5 crossed attribution. It evaluates control and treatment parameters
against both control and treatment same-day states, then uses matched
source-owner interventions only if the crossed evaluations locate a state
effect. This separates direct parameter-map sensitivity from cross-day state
feedback before choosing a repair.

No optimizer, checkpoint, recovery, Teacher regeneration, historical v5 read,
fresh-validation read, sealed-test read, half-hour forcing reconstruction,
48-step scan, architecture change, or paid execution occurred.
