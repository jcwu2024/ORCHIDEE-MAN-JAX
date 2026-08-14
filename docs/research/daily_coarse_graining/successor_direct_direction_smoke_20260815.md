# Direct-Direction 10-Update Smoke, 2026-08-15

## Decision

Reject fixed-norm, per-batch direct minimum-norm training without more
updates. The direct realization removes AdamW distortion, but a direction
estimated from one four-point-year mixed batch does not transfer reliably to
other batches or to the independent 7/30-day rollout subset.

This result does not reject the Attention architecture. It localizes the next
question to batch coverage and batch-to-rollout generalization.

## Protocol

The frozen contract SHA256 is
`faff94bf1b651e9a805da59125f63ca6baaf2835888035c4eaa6b8fd3580f1e1`.
Starting from the accepted update-1000 Attention parameters, each of ten
scheduled updates uses four different point-years, 32 one-day anchors, and
four 30-day windows. It recomputes the minimum-norm combination of the
one-day-anchor, NPP, AGB-tendency, and BGB-tendency gradients and applies an
exact global parameter delta norm of `0.001`.

No AdamW, momentum, coordinate-wise scaling, weight decay, or gradient
clipping is used. The frozen update-1000 parameters are the no-training
control for the accepted six-artifact non-test 7/30-day subset. No checkpoint
or recovery is written and no sealed-test value is used.

## Result

The free-`gln01` run at commit `621cf57` completed all ten updates and all six
validation artifacts:

- runtime report:
  `outputs/training/successor-direct-direction-smoke-621cf57/report.json`;
- report SHA256:
  `48e8e7da98a5b2c52e045af5653be5aea0bb85f970cb34b6f7ac40a65b50abf7`;
- elapsed Python time: `1002.89 s`;
- peak JAX device bytes: `12,934,680,576`;
- hard failures and nonfinite values: zero;
- checkpoint, recovery, and retained final parameters: none.

All four component median relative changes over the ten training batches are
negative: anchor `-0.0222%`, NPP `-0.1372%`, AGB tendency `-0.1589%`, and BGB
tendency `-0.0752%`. This aggregate hides unstable transfer. Only `6/10`
updates decrease every component. Updates 1, 2, 5, and 6 regress at least one
component; the maximum single-update regression is `19.11%`. Thus the step
norm admitted on four diagnostic groups is not uniformly safe on the next ten
scheduled mixed batches.

Independent rollout evidence is also negative. Relative to the frozen
update-1000 parent, the research-primary composite worsens `1.4348%`, and
zero of six slice/horizon cells improve. Every primary key worsens:

| Research-primary key | Aggregate RMSE relative change |
|---|---:|
| NPP | `+1.4354%` |
| Aboveground biomass carbon | `+0.1761%` |
| Belowground biomass carbon | `+1.2562%` |
| Total live biomass carbon | `+0.5887%` |

Maintenance-respiration RMSE improves `5.7566%`. The same maintenance versus
productivity tradeoff therefore persists after removing AdamW.

## Next Gate

Do not continue fixed-norm per-batch direct training. Before any further
updates, run a zero-update multi-batch direction-transfer audit at the frozen
update-1000 parameters. Use ten distinct scheduled mixed batches and all 40
batch-component gradients to test whether a single robust common-descent
direction exists across materially more data than one update consumes.

The audit must compare a preregistered global step-norm grid on all ten
batches, retain no parameters, and use no validation or sealed-test values.
If no robust direction exists, close gradient-combination optimizer work and
redesign the training objective or sampling unit. If it exists, only then
consider a bounded robust multi-batch update smoke. No network architecture
change is authorized by this result.
