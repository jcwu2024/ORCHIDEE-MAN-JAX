# Coupled GPP Fresh-Moment Admission Result

## Outcome

The free `gln01` admission completed at commit `eb4eb19` in `1,519.041`
seconds. Scheduled update 2 passed, but scheduled update 3 failed the frozen
every-source finite-step gate. The classification is
`stop_and_review_fresh_moment_direction_or_finite_step_without_paid_training`.

The JSON report is
`outputs/research/daily_coarse_graining/target-oriented-coupled-gpp-fresh-moment-admission-eb4eb19.json`,
SHA256 `3e07fb8a313c9885e069d44b5266212d4fe12e0b9295876db8a701ecdc15f834`.

## Consecutive Proposals

| Source | Update 2 | Update 3 |
| --- | ---: | ---: |
| Direct GPP | `-4.04281%` | `+0.758192%` |
| Reachable one-day state | `-2.91495%` | `-4.32042%` |
| 7-day science | `-0.609017%` | `+0.0372167%` |
| 30-day science | `-0.345835%` | `-1.01398%` |
| Weighted objective | `-1.86632%` | `-0.766949%` |

Negative values are improvements. Update 2 reproduces the accepted reset
full-step geometry result on the production proposal path. Update 3 improves
the aggregate objective, state, and 30-day loss, but its GPP and 7-day losses
increase beyond the frozen `1e-6` relative tolerance.

Both proposals used optimizer-internal step one and clip scale `1.0`. Their
raw gradient norms were `2.40601` and `1.78844`; parameter-displacement norms
were `0.00778863` and `0.00775897`. All `15,960,195` combined-gradient values
were finite, all 190 gradient leaves were nonzero, and both trainable families
changed on each proposal.

## Attribution

The update-3 weighted raw gradient has source cosines:

| Source | Cosine with weighted raw gradient |
| --- | ---: |
| Direct GPP | `+0.444107` |
| Reachable one-day state | `+0.486760` |
| 7-day science | `-0.274594` |
| 30-day science | `-0.0310940` |

Thus update 3 is not the same inherited-moment failure as the rejected paid
update 2. Fresh moments remove the known historical rotation, but the fixed
weighted objective itself is no longer a common first-order descent direction
for the two horizon sources on the next scheduled group. The realized GPP
regression despite its positive raw cosine also leaves an AdamW-transform or
finite-curvature contribution to quantify. The result rejects fresh moments
as a sufficient training rule; it does not reject the coupled-GPP architecture
or prove that no four-source common direction exists.

## Integrity

Every pre/post budget, unexpected-status, discrete, nonfinite, and
negative-stock hard count is zero. Fixed parameters and the structured-GPP
source checkpoint are exactly unchanged. The parent update-1 recovery is
unchanged. No optimizer update or neural parameter was persisted, no
checkpoint was written, and peak live JAX allocation was `3,987,660,288`
bytes (about `3.71 GiB`). Teacher v6 was not regenerated; historical v5,
fresh validation values, and sealed test were not read.

## Next Gate

Paid continuation remains closed. The next bounded gate is a zero-retention
update-3 direction audit at the in-memory state after the admitted fresh-moment
update 2. It must use the same four source gradients and scheduled group,
determine whether a common first-order direction exists, and compare direct
and fresh-AdamW finite realizations on a frozen scale grid. It may not change
the architecture, objective labels, data schedule, or read validation values.
