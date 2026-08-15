# Second-State Exact Recalibration Audit, 2026-08-15

## Decision Before Execution

Freeze one zero-retention audit at the parameter state after the accepted
first direct update. This is not training and does not change the selected
`source_routed_process_attention_v2` architecture.

The audit answers whether the 40 local group-component objectives still have
a useful common direction after one step, and whether the new minimum-norm
simplex support overlaps the parent's 14 active constraints enough to make an
efficient active-set recalibration implementation plausible.

## Frozen Mechanism

1. Load the accepted update-1000 Attention recovery.
2. Reconstruct the accepted norm-`0.002` first update from the parent's 14
   positive simplex constraints, reduced to eight group VJPs by
   `simplex_weight / parent_component_gradient_norm`.
3. At the reconstructed second state, compute each of the 10 groups x 4
   component gradients separately and solve the exact minimum-norm simplex
   over their 40 unit gradients.
4. Evaluate global step norms `0.0005, 0.001, 0.0015, 0.002` on all ten
   groups. Every group must satisfy the component nonincrease tolerance, and
   every component median must improve by at least `1e-4` for a step to pass.
5. Report minimum cosine, selected finite step, new support size, parent/new
   support overlap, hard/nonfinite counts, device memory, and runtime.

The contract is
`manifests/coarse_graining/successor_second_state_recalibration_audit_v1.json`,
SHA256 `5379e26771d1e37b90983b8bc7242c5feb3c26a091de212153b1bc58b3023014`.

## Boundaries

- free `gln01` only; no paid job;
- 8 parent-direction VJPs plus 40 second-state component VJPs;
- one reconstructed in-memory first update and zero new applied updates;
- no checkpoint, recovery, or proposed parameter retention;
- no validation-value or sealed-test evaluation;
- no Teacher regeneration and no historical v5 data;
- no 7/30-day rollout, because this audit is strictly a local mechanism test.

## Decision Rule

- If no second-state common direction exists, redesign the objective or
  sampling groups.
- If a common direction exists but no bounded finite step passes, redesign
  its finite-step realization before training.
- If the support is dense or disjoint from the parent support, redesign the
  efficient recalibration mechanism before training.
- If a finite common direction has sparse overlapping support, implement an
  active-set recalibration: initialize from the prior support, check all 40
  directional derivatives, and add violated constraints.

No fixed-parent-weight or smaller fixed-step experiment may replace this
audit.

## Result

The free-`gln01` run completed at commit `1902936` with valid evidence:

- runtime report:
  `outputs/training/successor-second-state-recalibration-1902936/report.json`;
- report SHA256:
  `c4ddb39b551da4ea2d26f030ca2a202482043de080a684c8e60cfd6873710e97`;
- elapsed Python time: `1051.42 s`;
- peak JAX device bytes: `15,667,732,736`;
- hard failures and nonfinite values: zero;
- checkpoint, recovery, validation-value evaluation, and parameter retention:
  none.

The first update is reproduced to `4.96e-15` relative direction-norm error
and `6.55e-15` maximum virtual relative-change error. At the reconstructed
second state, an exact common direction still exists across all 40
constraints. Its minimum cosine is `0.0486966`.

Every finite step tested decreases all 40 losses. Norms `0.0005, 0.001,
0.0015` fail only because their anchor median improvement is below the frozen
`1e-4` effect threshold. Norm `0.002` passes all gates with median relative
changes:

- one-day anchor: `-1.17110e-4`;
- NPP: `-9.93916e-4`;
- AGB tendency: `-1.29845e-3`;
- BGB tendency: `-3.33780e-4`.

The recalibrated simplex support has 13 constraints. Twelve overlap the
parent's 14-constraint support. It drops `group_04.npp` and
`group_06.belowground_biomass_tendency`, and adds only
`group_06.one_day_anchor`.

## Accepted Next Gate

Implement a bounded active-set recalibration audit. Start from the parent
support, solve using its individual gradients, check all 40 directional
derivatives, add violated constraints, and compare the efficient result with
this exact second-state reference. Do not start repeated training until that
mechanism reproduces a common finite direction without requiring 40 reverse
passes.
