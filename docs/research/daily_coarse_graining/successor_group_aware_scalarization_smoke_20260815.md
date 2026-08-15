# Frozen Group-Aware Scalarization Smoke, 2026-08-15

## Decision

Reject a fixed `0.002` step with parent-frozen scalarization coefficients for
repeated updates. The efficient scalarization exactly reproduces the admitted
parent direction and its first step improves every local constraint. After
that step, the same coefficients drift enough that a second `0.002` proposal
regresses several group-component losses and is stopped before application.

The result does not reject the group-aware direction or the Attention
architecture. It shows that a single parent calibration cannot be reused at a
fixed step without a trust-region response.

## Protocol

The frozen contract SHA256 is
`42423d0b62bd27d632809cfbbd494412213d4fcc777f087e54aef43ae861642a`.
The 14 positive simplex constraints from the accepted parent audit are
converted to raw-loss coefficients by dividing each weight by its parent
gradient norm. Same-group components share one VJP, reducing each proposed
update from 40 reverse passes to eight. All ten groups are evaluated before
and after every proposal.

At most five in-memory updates use global norm `0.002`. A proposal is not
applied unless every group-component is nonincreasing and all hard/nonfinite
checks pass. The final accepted in-memory parameters are compared with the
unchanged update-1000 parent on the established six-artifact non-test 7/30-day
subset. No checkpoint is retained.

## Result

The free-`gln01` run at commit `1bbb163` completed with valid evidence:

- runtime report:
  `outputs/training/successor-group-aware-scalarization-smoke-1bbb163/report.json`;
- report SHA256:
  `1aeb93ae49e29496c43dfd9589f7e87f6e6be20c214a23a6d266aaca57360ca6`;
- elapsed Python time: `1052.59 s`;
- peak JAX device bytes: `13,215,706,112`;
- hard failures and nonfinite values: zero;
- checkpoint, recovery, and final parameter retention: none.

Update 1 is an exact realization of the accepted parent direction:

- direction-norm relative error: `4.82e-15`;
- maximum virtual relative-change error: `6.55e-15`;
- worst of all 40 relative changes: `-1.8028e-5`;
- update applied: yes.

At update 2, fixed norm `0.002` is rejected before application. The largest
regression is group 5 NPP at `+8.6109e-4` (`+0.0861%`). Group 5 AGB tendency,
group 10 NPP/AGB tendency, and several other constraints also regress. This
is coefficient/step drift across multiple constraints rather than one noisy
outlier.

## Independent Rollout Signal

The validation compares the one accepted step with the update-1000 parent.
All six temporal/spatial/joint by 7/30-day cells improve, and all four primary
aggregates improve:

| Aggregate | Relative change |
|---|---:|
| NPP | `-9.2206e-5` |
| AGB | `-9.5624e-5` |
| BGB | `-1.2252e-4` |
| Total live biomass | `-9.8222e-5` |

The combined research-primary improvement is `9.2210e-5` (`0.00922%`), below
the frozen `0.5%` smoke threshold. Maintenance aggregate RMSE regresses only
`0.0497%`, within its guard. This is small but directionally consistent
independent evidence, unlike the earlier direct-direction smoke in which all
six cells and every primary key worsened.

## Next Gate

Freeze one matched backtracking smoke using the same eight-VJP direction. At
each update, try norms `0.002`, `0.001`, and `0.0005` in descending order and
apply the largest norm that keeps all 40 losses nonincreasing. Stop if no norm
of at least `0.0005` is admitted; do not shrink toward a numerically harmless
but scientifically ineffective step.

The backtracking contract SHA256 is
`8c9a97141f0eab05fe54e545ad8056e399eb12008e786eff4f0d35ddb347c11d`.
Passing would open rotating-group weight recalibration. Failure would close
parent-frozen weights and require recalibrating the robust direction itself.
