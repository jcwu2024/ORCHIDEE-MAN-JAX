# Group-Aware Backtracking Smoke, 2026-08-15

## Decision

Close repeated use of parent-frozen scalarization weights. Backtracking shows
that the second-step failure is directional, not merely an oversized finite
step: the worst regression decreases almost exactly linearly from norms
`0.002` to `0.0005` but never changes sign.

Do not test still smaller fixed-weight steps and do not continue training.
The common direction must be recalibrated after the first accepted update.

## Protocol

The frozen contract SHA256 is
`8c9a97141f0eab05fe54e545ad8056e399eb12008e786eff4f0d35ddb347c11d`.
It repeats the efficient eight-VJP parent scalarization from update 1000. At
each of at most five updates, it tries global norms
`0.002, 0.001, 0.0005` in descending order and applies the largest norm for
which every one of the 40 group-component losses is nonincreasing. A norm
below `0.0005` is not admitted as scientifically useful.

The update-1000 parent remains the matched no-training control. Final
in-memory parameters are evaluated on the established six-artifact non-test
7/30-day subset, then discarded. No checkpoint or recovery is written.

## Result

The free-`gln01` run at commit `4118c72` completed with valid evidence:

- runtime report:
  `outputs/training/successor-group-aware-backtracking-smoke-4118c72/report.json`;
- report SHA256:
  `c2691824b6548be05337620ffb91cab1994b58bd83293a56968d89574a074261`;
- elapsed Python time: `1113.28 s`;
- peak JAX device bytes: `13,215,706,112`;
- hard failures and nonfinite values: zero;
- accepted updates: one;
- checkpoint, recovery, and final parameter retention: none.

Update 1 again selects `0.002` and improves all 40 constraints. At update 2,
no candidate is admitted:

| Step norm | Worst group-component relative change |
|---:|---:|
| `0.002` | `+8.610857e-4` |
| `0.001` | `+4.308843e-4` |
| `0.0005` | `+2.155203e-4` |

The near 2:1 scaling as the step halves is first-order behavior. Curvature at
`0.002` is not the primary cause; the parent-frozen scalarized gradient at the
new parameters has a non-descent projection for at least one constraint.

Because only the same first update is accepted, independent rollout results
match the preceding smoke: all six cells and all four primary aggregates
improve, while the combined primary improvement is `0.00922%`, below the
`0.5%` gate. This positive one-step signal remains valid but does not support
repeated fixed-weight updates.

## Next Gate

Design one second-state recalibration audit before implementation. It should:

1. reconstruct the admitted first update from the frozen parent coefficients;
2. at the resulting parameters, recompute the exact 40 group-component
   gradients and solve a new common-direction simplex problem;
3. test a bounded finite-step grid on all ten groups without retaining a
   second update;
4. report support size and overlap with the parent's 14-constraint support.

If no common direction exists at the second state, close the current
repeated-update mechanism and redesign the objective or sampling groups. If a
common direction exists and its support remains sparse with useful overlap,
the next implementation should use an active-set recalibration algorithm:
start from the prior support, test all 40 directional derivatives, and add
violated constraints. That is the plausible path to fewer than 40 reverse
passes per update; fixed weights and further fixed-step searches are closed.
