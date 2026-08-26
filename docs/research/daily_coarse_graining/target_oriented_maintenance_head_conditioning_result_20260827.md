# Target-Oriented Maintenance Head Conditioning Result

Date: 2026-08-27

Status: accepted zero-training diagnostic; common maintenance-head direction
exists, but no evaluated finite step satisfies every frozen protection gate

Report SHA256:
`85ab83a120f1f82f09b8783391e564e5632e84c8582fa74a6b250089bad27757`

Contract SHA256:
`b354c4e8309e878899ce61c83f851b08279977a9984eccf0b9a60bc7089fc369`

Git commit: `79b2e36283c315170858ce392f2c90b074ed68b2`

## Result

All 16 head-only gradients are finite and nonzero over the complete
769-parameter maintenance final head. Their L2 norms range from `0.7703` to
`15.7215`. Some individual group/horizon gradients oppose each other strongly
(minimum pairwise cosine `-0.9795`), but the minimum-norm simplex solution
still realizes positive descent for every objective. Its minimum constraint
cosine is `0.056523`, above the frozen `0.01` gate.

The common direction is therefore supported. The existing final head is not
incapable of reducing maintenance error.

Four in-bound virtual steps were evaluated exactly. Negative values below are
improvements in the named normalized score; positive values are regressions.

| Predicted fraction | Step norm | Maintenance 7 d | Maintenance 30 d | NPP 7 d | NPP 30 d | Growth respiration 7 d | Ecosystem carbon 7 d | Formal pass |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0.1%` | `0.0001846` | `-0.1453%` | `-0.0820%` | `-0.0055%` | `-0.0081%` | `+0.0075%` | `+0.00017%` | no |
| `0.3%` | `0.0005539` | `-0.4350%` | `-0.2459%` | `-0.0163%` | `-0.0243%` | `+0.0227%` | `+0.00051%` | no |
| `1%` | `0.0018462` | `-1.4415%` | `-0.8188%` | `-0.0533%` | `-0.0803%` | `+0.0754%` | `+0.00217%` | no |
| `3%` | `0.0055387` | `-4.2508%` | `-2.4508%` | `-0.1597%` | `-0.2371%` | `+0.2249%` | `+0.00569%` | no |

The `0.1%` proposal misses the required `0.1%` median 30-day maintenance
improvement. The `0.3%` and `1%` proposals improve maintenance at both
horizons and also improve median NPP, AGB, BGB, total live biomass, and all
30-day stock scores. They nevertheless fail the frozen exact protection rule:
the 7-day growth-respiration score and 7-day ecosystem-carbon score have small
positive median changes. The 7-day GPP median also increases by only about
`8e-9` to `2e-8` in relative-score units, but the preregistered threshold is
nonpositive rather than tolerance based.

The `3%` proposal additionally exceeds the `1%` any-group limits for growth
respiration, a derived flux, and a stock. The declared `10%` proposal has norm
`0.01846`, above the frozen `0.01` head-step bound, and was correctly not
evaluated. All evaluated proposals retain the parent-passing acceptance keys,
leave the parent-failing stock RMSE unchanged, and have zero hard and
nonfinite failures.

The formal classification is therefore
`finite_head_step_not_protected`. No virtual parameters were retained.

## Interpretation

The result separates two claims that must not be conflated:

1. The existing head has a useful common maintenance direction. This audit
   does not reject the Attention body or show that the network cannot learn
   maintenance.
2. Along the frozen common direction, a single shared `768 -> 1` linear final
   projection did not realize the tested maintenance correction while
   satisfying the frozen requirement of nonpositive median change in every
   protected score. The smallest useful steps are close to protected, but
   they are not formal passes. This audit does not exhaust every direction in
   the 769-dimensional head space.

The current head shares the same 769 parameters across all active maintenance
coordinates. Coordinate-aligned Attention supplies distinct hidden vectors,
but one linear projection materializes every PFT/plant-part base-maintenance
value before the existing nonnegative transform and exact temperature-factor
route. GPP has a separate final head; its tiny later-day changes arise through
the physical state recurrence, not through changed GPP-head parameters.

This evidence points to output parameterization or conditional correction
capacity, not to another Teacher-carry intervention and not to automatic
abandonment of Attention. It also does not justify weakening the frozen gate
after seeing the result.

## Next Boundary

The current architecture, input mask, source routing, and loss remain frozen.
Before any implementation, discuss whether the next candidate should preserve
the coordinate-aligned Attention body and replace only the maintenance output
materialization with a structured conditional head. That discussion must
state the mechanism, parameter budget, initialization from or separation from
the current checkpoint, matched diagnostic, and bounded 7/30-day gate before
training.

No unchanged AdamW continuation, paid training, validation-value read,
365-day rollout, sealed test, Gate D2, or product work is authorized by this
result.

## Execution

The audit ran on free `gln01` without Slurm. Total runtime was `4252.70 s`
(`70.88 min`): baseline forward `719.09 s`, head gradients and geometry
`2913.63 s`, and virtual steps `444.74 s`. Peak active JAX memory was
`2,403,405,824` bytes, about `2.24 GiB`; the allocator limit was about
`23.80 GiB`.

Teacher v6 was not regenerated. Historical v5, validation values, and sealed
test were not read. No optimizer was invoked, and no checkpoint or recovery
was written.
