# Coupled GPP Direct Recalibration Smoke Result

## Outcome

The corrected free `gln01` smoke completed at commit `a32db38` in
`2,552.884` seconds. Direct common-direction updates 3 and 4 passed, but
update 5 failed the frozen finite-step gate. The classification is
`common_direction_finite_step_rejection`.

The JSON report is
`outputs/research/daily_coarse_graining/target-oriented-coupled-gpp-direct-recalibration-a32db38.json`,
SHA256 `0c2d3e219f4dce16be40d15ce6a1ccbdb7c5f4f239e88e83a874e42b23242fd2`.

This is not a rejection of the coupled-GPP Attention architecture or of
per-update common-direction recalibration. It is positive two-update transfer
evidence followed by a sharply localized finite-step curvature failure.

## Consecutive Updates

All values below are relative loss changes; negative values are improvements.

| Update | Minimum source cosine | GPP | State | 7-day | 30-day | Weighted | Accepted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 3 | `0.226096` | `-0.9890%` | `-6.5908%` | `-3.7805%` | `-6.5116%` | `-3.6710%` | yes |
| 4 | `0.207390` | `-4.9910%` | `-4.1031%` | `-2.8312%` | `-4.3690%` | `-3.8419%` | yes |
| 5 | `0.495394` | `+58.1509%` | `-28.6952%` | `-9.9005%` | `-10.6083%` | `+9.1686%` | no |

The actual float32-preserving parameter displacement norms are
`0.00775897292`, `0.00775897308`, and `0.00775897285`. Thus update 5 is not a
direction-feasibility failure: its common direction is stronger than those of
the two accepted updates, and all four realized first-order cosines are about
`0.495394`. The same fixed full displacement is too large for its GPP
curvature.

## Bounded-Scale Rationale

For update 5 GPP, the measured gradient norm, direction cosine, current loss,
and actual displacement predict a first-order relative change of `-19.7633%`
at full scale. The realized change is instead `+58.1509%`. A local quadratic
fit through the zero-step derivative and full-step response gives coefficient
`0.779141`, an estimated second zero crossing at scale `0.253654`, and an
estimated local minimum at `0.126827`.

This estimate is diagnostic, not an acceptance result. It motivates a frozen
bounded scale grid `1,1/2,1/4,1/8,1/16,1/32`, with `1/4` bracketing the
predicted crossing and `1/8` bracketing the predicted minimum. The full-scale
record is reused; smaller scales require real eight-artifact evaluation.

## Engineering Correction

The first free launch at commit `40b66a2` timed out after 90 minutes without a
report. It accepted update 3, then promoted float32 parameters to float64
because a float64 scalar step norm entered the direct update. That triggered a
second compilation family: the next direct-gradient compile took
`16m14.9s`, and the 7-day gradient compile took `25m17.8s` before timeout.

Commit `a32db38` casts the common direction and scalar norm to each parameter
leaf's dtype. The corrected update-4 first artifact took `35.5s`, followed by
approximately `32.5s` artifacts, rather than another 40-minute compilation.
The correction changes neither data, direction solver, nominal displacement,
nor scientific gate.

## Integrity

All physical hard gates pass. The `93,504,164` fixed parameters, parent
recovery, source checkpoint, and common-direction predecessor report remain
unchanged. No proposal, optimizer state, recovery, or checkpoint is retained.
No validation or sealed-test value is read; no Teacher data is regenerated;
no v5 data, 48-record forcing reconstruction, or subdaily state scan is used.

The JAX memory profile peaks at `4,043,065,600` bytes after the dtype fix.

## Next Gate

The next bounded gate is a zero-retention update-5 finite-scale audit at the
exact in-memory state after accepted direct updates 3 and 4. It must reproduce
the full-scale rejection, evaluate the frozen smaller scales, and select no
state for retention. Architecture changes, paid continuation, fresh
validation, 365-day rollout, Gate D2, sealed test, and product acceptance
remain closed.
