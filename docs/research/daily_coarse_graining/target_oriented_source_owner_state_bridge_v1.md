# Target-Oriented Source-Owner State Bridge V1

Date: 2026-08-28

Status: completed on free `gln01`; accepted result is recorded in
`target_oriented_source_owner_state_bridge_result_20260828.md`

Contract SHA256:
`c8bcadc2ed94a38afaa8788ba5775070725ed1dc00d58db29b2753099ae86a2a`

## Question

The accepted source-shaped branch checkpoint improves maintenance on its own
recursive state distribution, but worsens NPP and live biomass. Those
protected regressions disappear when every day starts from the full Teacher
state. This audit asks whether drift in the exact physical inputs of the new
maintenance branches is sufficient to cause that reversal.

It does not train, rank, or implement another architecture.

## Intervention And Matched Controls

The intervention runs an otherwise ordinary free rollout and Teacher-carries
only these four exact day-start state owners before each daily transition:

- `slowproc_stomate_previous_step_state.biomass`;
- `slowproc_stomate_previous_step_state.lai`;
- `slowproc_stomate_previous_step_state.veget_max`;
- `slowproc_stomate_previous_step_state.t2m_longterm`.

The first day is unchanged because its initial state is already the prepared
Teacher state. Every other continuous coordinate and all discrete state remain
predicted. Forcing, parameters, neural operator, constrained updater, and
retained exact tail remain unchanged.

The audit compares three matched modes for both deterministic source-shaped
update 0 and branch-only update 100:

| Mode | Four source owners | Other cross-day state |
| --- | --- | --- |
| Free rollout | predicted | predicted |
| Four-owner bridge | Teacher-carried | predicted |
| Full Teacher day start | Teacher-carried | Teacher-carried/reset |

The accepted free and full-Teacher results are reused by report hash. Only the
four-owner bridge is executed.

## Frozen Data And Metrics

The panel remains train/train groups `251, 358, 465, 572, 679, 786, 893,
1000`, with eight 7-day and four 30-day windows per group. It reads no
validation values or sealed test and does not regenerate Teacher v6.

For GPP, maintenance, growth respiration, autotrophic respiration, NPP, AGB,
BGB, total live biomass, and ecosystem carbon, the report uses the accepted
score:

```text
0.5 * (normalized RMSE^2 + normalized signed bias^2)
```

The required reversal set is growth respiration, NPP, AGB, BGB, and total live
biomass at both 7 and 30 days, for 10 fixed cells. The raw autotrophic and NPP
carbon-bias identities must still close within `1e-10`; all state, mask,
budget, and nonnegative-stock hard counts must remain zero.

## Bounded Decision

- `source_owner_state_drift_sufficient`: all 10 bridge cells have update-100
  versus update-0 median score change less than or equal to zero.
- `source_owner_state_drift_contributes_but_not_sufficient`: 1 to 9 cells
  reverse.
- `source_owner_state_drift_not_explanatory`: no cell reverses.

The report also gives each bridge cell's progress from the accepted free value
toward the full Teacher-start value. This is descriptive and does not weaken
the all-10 sufficiency gate.

Every outcome returns to architecture discussion. A sufficient result focuses
that discussion on correcting source-owner recurrence; a partial or absent
result requires broader Attention context or other recurrent state paths to
remain in scope. No architecture implementation is authorized by this audit.

## Execution Boundary

The run creates no optimizer, applies no parameter step, writes no checkpoint,
reconstructs no 48-record forcing series, and executes no 48-step scan. A free
`gln01` V100 may run the bounded panel. Paid work, broader training, 365-day
rollout, Gate D2, sealed test, and product work remain closed.

## Qualification And Execution

The contract, classifier, multi-leaf intervention boundary, forward-only
launcher, and related source-shaped regressions pass locally. The real-v6 run
completed at commit `1c862a4`; report SHA256 is
`7defdf47...642d5c`. See the result document for the accepted partial and
insufficient classification.
