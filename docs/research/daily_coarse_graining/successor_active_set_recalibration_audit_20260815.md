# Active-Set Recalibration Audit, 2026-08-15

## Decision Before Execution

Freeze one zero-retention implementation audit of efficient second-state
recalibration. This follows the exact 40-gradient admission and does not
change `source_routed_process_attention_v2`, the data, or the objective.

The exact reference found a 13-constraint support with 12 constraints shared
with the parent's 14-constraint support. The implementation therefore starts
from the prior support instead of recomputing all 40 component gradients.

## Mechanism

1. Reconstruct the accepted first norm-`0.002` update with the existing eight
   group VJPs.
2. At the second state, compute the 14 parent-support component gradients
   individually and solve their minimum-norm simplex.
3. Normalize that parameter direction and evaluate one JVP per group. Each of
   the ten JVPs returns all four loss directional derivatives, so all 40
   constraints are checked without 40 reverse passes.
4. Add every nonpositive constraint whose gradient is not available, then
   re-solve and repeat. Stop after at most three rounds or six additions.
5. Evaluate the same finite norm grid `0.0005, 0.001, 0.0015, 0.002` on all ten
   groups with the exact reference's all-component and median-effect gates.

The frozen contract is
`manifests/coarse_graining/successor_active_set_recalibration_audit_v1.json`,
SHA256 `284b328fa4c635499a2562d059e58b35e03918d3759f91e659d6595f3546670f`.

## Gates

- all 40 final relative directional derivatives are strictly positive;
- all ten groups pass the finite component nonincrease gate;
- each component median improves by at least `1e-4` at the selected step;
- the selected step equals the exact reference norm `0.002`;
- at most 20 individual component reverse passes, versus 40 for the exact
  reference;
- at most three ten-group JVP rounds and six added constraints;
- zero hard failures and nonfinite values.

## Boundaries

- free `gln01` only and no paid job;
- one reconstructed in-memory first update and no new applied update;
- no checkpoint, recovery, or parameter retention;
- no validation-value or sealed-test evaluation;
- no Teacher regeneration and no v5 data;
- no repeated training until this audit passes.

Passing opens design of a bounded repeated active-set training smoke. Failure
requires revising the recalibration mechanism, not changing the network
architecture.
