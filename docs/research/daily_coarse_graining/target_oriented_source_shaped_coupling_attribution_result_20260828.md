# Target-Oriented Source-Shaped Coupling Attribution Result

Date: 2026-08-28

Status: accepted zero-training causal result; joint flux expansion is not yet
supported

Executable commit: `4c22a4c46a49bd74ab462d4206e5abeca7822f4c`

Report SHA256:
`e2a0b90f3972887da3e494da943f0a9d9fb8389eb8c551587ff71563426a0d6b`

Contract SHA256:
`12b327c81851551dfd7ef5e4878b03845290d78b5d51d26168046fe25dce93b2`

## Execution

The free `gln01` V100 run completed in `809.48 s` (`13.49 min`). It evaluated
the complete frozen eight-group panel for both source-shaped update 0 and
branch-only update 100. No optimizer was created, no update was applied, and
no checkpoint was written. All prediction/state finite, unexpected-mask,
discrete, budget, and negative-stock hard-failure counts are zero.

The raw cumulative autotrophic-respiration and NPP bias identities close with
maximum absolute residual `5.6843e-14`, below the frozen `1e-10` tolerance.

## Matched Result

The table reports median acceptance-score change from update 0 to update 100.
Negative values improve and positive values worsen.

| Key | 7-day free rollout | 7-day Teacher day start | 30-day free rollout | 30-day Teacher day start |
| --- | ---: | ---: | ---: | ---: |
| GPP | approximately `0%` | exactly `0%` | approximately `0%` | exactly `0%` |
| Maintenance respiration | `-62.9347%` | `+680.3003%` | `-71.7403%` | `+792.2957%` |
| Growth respiration | `+4.1536%` | `-8.8892%` | `+10.8980%` | `-6.8778%` |
| Autotrophic respiration | `-29.9664%` | `+22.9273%` | `-31.6947%` | `+18.9994%` |
| NPP | `+6.5809%` | `-8.6701%` | `+11.9426%` | `-7.2449%` |
| AGB | `+8.6595%` | `-8.8047%` | `+4.1960%` | `-7.6310%` |
| BGB | `+5.0256%` | `-8.5422%` | `+14.9886%` | `-7.2633%` |
| Total live biomass | `+6.5979%` | `-8.6382%` | `+11.1408%` | `-7.4275%` |
| Ecosystem carbon | `+0.3467%` | `+0.1789%` | `+0.9481%` | `+0.7400%` |

Every material free-rollout regression in growth respiration, NPP, AGB, BGB,
and total live biomass disappears under Teacher day starts and becomes an
improvement. Those regressions therefore require predicted cross-day state.

The Teacher-day-start intervention exposes a different same-day effect:
update 100 strongly worsens maintenance itself and therefore worsens total
autotrophic respiration, while GPP is exactly invariant and growth respiration
improves. The frozen classification is `direct_daily_coupling_confirmed`, but
`joint_flux_scope_required=false` because Teacher-day-start NPP and biomass do
not regress.

## Interpretation

The trained branches are not a uniformly improved mapping from physical day
state to maintenance respiration. They compensate effectively on the model's
own recursively predicted state distribution, but move in the wrong direction
on Teacher day-start states. The apparent maintenance breakthrough and the
protected-carbon regression are therefore coupled to state distribution and
feedback, not just to an underpowered direct maintenance head.

This closes immediate joint GPP/growth/maintenance expansion. It also closes
unchanged branch-only continuation. Neither conclusion rejects the retained
coordinate-aligned Attention backbone.

## Next Gate

Freeze a zero-training source-owner state bridge on the same train/train panel.
Between ordinary free rollout and the already accepted full Teacher-day-start
endpoint, replace only the four exact cross-day source owners with their
Teacher values:

- `slowproc_stomate_previous_step_state.biomass`;
- `slowproc_stomate_previous_step_state.lai`;
- `slowproc_stomate_previous_step_state.veget_max`;
- `slowproc_stomate_previous_step_state.t2m_longterm`.

If that bounded carry reproduces the full Teacher-day-start reversal, the
physical source-input state drift is sufficient. If it remains free-rollout-
like, the remaining cause lies in broader Attention context or other state
paths. That decision must precede discussion of a revised branch architecture.

No training, validation read, broader panel, 365-day rollout, Gate D2, sealed
test, or product work is authorized by this result.
