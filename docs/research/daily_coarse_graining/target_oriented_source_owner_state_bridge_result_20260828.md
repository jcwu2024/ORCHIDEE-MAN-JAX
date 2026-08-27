# Target-Oriented Source-Owner State Bridge Result

Date: 2026-08-28

Status: accepted zero-training causal result; four exact source owners
contribute but are not sufficient

Executable commit: `1c862a43bd24853a1a76460655981abecbc37f5c`

Report SHA256:
`7defdf47707e6729c7012f4140dd312ed36658274ae7f984a1516a12c6642d5c`

Contract SHA256:
`c8bcadc2ed94a38afaa8788ba5775070725ed1dc00d58db29b2753099ae86a2a`

## Execution

The free `gln01` V100 run completed in `834.30 s` (`13.91 min`). It loaded the
same source-shaped update 0 and branch-only update 100 states and evaluated
only the new four-owner bridge. The accepted free and full-Teacher modes were
reused by report hash.

No optimizer was created, no parameter step was applied, and no checkpoint was
written. All prediction/state finite, unexpected-mask, discrete, budget, and
negative-stock hard-failure counts are zero. The raw autotrophic-respiration
and NPP bias identities close with maximum absolute residual `3.0198e-14`,
below the frozen `1e-10` tolerance.

## Matched Result

The table reports median update-100 versus update-0 acceptance-score change.
Negative values improve and positive values worsen.

| Horizon/key | Free rollout | Four-owner bridge | Full Teacher day start |
| --- | ---: | ---: | ---: |
| 7-day growth respiration | `+4.15%` | `+3.63%` | `-8.89%` |
| 7-day NPP | `+6.58%` | `+5.63%` | `-8.67%` |
| 7-day AGB | `+8.66%` | `-16.32%` | `-8.80%` |
| 7-day BGB | `+5.03%` | `-7.62%` | `-8.54%` |
| 7-day total live biomass | `+6.60%` | `-12.27%` | `-8.64%` |
| 30-day growth respiration | `+10.90%` | `+10.28%` | `-6.88%` |
| 30-day NPP | `+11.94%` | `+11.56%` | `-7.24%` |
| 30-day AGB | `+4.20%` | `-2.96%` | `-7.63%` |
| 30-day BGB | `+14.99%` | `+10.26%` | `-7.26%` |
| 30-day total live biomass | `+11.14%` | `+3.04%` | `-7.43%` |

Exactly `4/10` frozen cells reverse: all three 7-day biomass measures and
30-day AGB. The remaining six cells are still materially regressed. The formal
classification is
`source_owner_state_drift_contributes_but_not_sufficient`.

GPP remains numerically invariant. Maintenance retains a large improvement in
the bridge (`61.82%` at 7 days and `77.57%` at 30 days), rather than showing
the full Teacher-start deterioration. This confirms that carrying only the
four direct physical branch inputs does not reproduce the complete state-
distribution change.

## Interpretation

The four exact source owners are causal for a substantial part of live-
biomass recurrence, especially at seven days. They are not the missing link
for flux balance: the bridge covers only about `4.0%/3.5%` of the movement
from free to full Teacher state for 7/30-day growth respiration and
`6.3%/2.0%` for NPP.

Therefore a revised model cannot be limited to stabilizing only `biomass`,
`lai`, `veget_max`, and `t2m_longterm`. Broader recurrent state paths or the
Attention hidden context used by the maintenance branches remain necessary to
explain why the trained correction works on its own rollout distribution but
fails to transfer to full Teacher states.

This result does not reject Attention, and it does not support another
unchanged branch-only continuation.

## Next Gate

Stop for architecture discussion before implementation. That discussion must
preserve the successful coordinate-aligned Attention backbone as an open
option while deciding how the maintenance mechanism should condition on the
broader recurrent state distribution. No training, validation read, 365-day
rollout, Gate D2, sealed test, or product work is authorized by this result.
