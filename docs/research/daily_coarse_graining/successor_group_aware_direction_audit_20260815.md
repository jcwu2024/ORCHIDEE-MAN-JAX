# Group-Aware 40-Objective Direction Audit, 2026-08-15

## Decision

The selected Attention checkpoint has a strong common first-order descent
direction across all 10 train/train groups and all four scientific loss
components. This resolves the sign-conflict failure of ordinary large-batch
averaging. It does not yet admit a finite training step because the frozen
effect-size gate was not reached within the original grid.

Do not classify this result as absence of a usable direction, do not train it
yet, and do not change the network architecture. Run one bounded zero-update
finite-step extension to determine whether the same direction has a shared
step with useful effect size.

## Protocol

The frozen contract SHA256 is
`c6db0816b33f2333a8b94f7daaf0fb167be906c66051b036afda69f158fc120d`.
At the accepted update-1000 `source_routed_process_attention_v2` parameters,
the audit independently normalizes the 40 gradients formed by 10 scheduled
mixed groups times one-day anchor, NPP, AGB tendency, and BGB tendency. It
solves the minimum-norm simplex problem over those 40 unit gradients and
evaluates virtual global step norms `0.00001, 0.00003, 0.0001, 0.0003` on
every contributing group.

The effective audit covers 40 point-years, 320 one-day anchors, and 40
30-day windows. It performs no optimizer update, retains no proposed
parameters, and reads no validation or sealed-test values.

## Result

The free-`gln01` run at commit `b0ebda9` completed with valid evidence:

- runtime report:
  `outputs/training/successor-group-aware-direction-b0ebda9/report.json`;
- report SHA256:
  `f9c55f15dbd4744097dc5bc5089a35447d2cc63fcad41a00650622e3d809a4e3`;
- geometry sidecar SHA256:
  `1acb9e3029a0359abcee218383bbed3c79344c468f83d299f15ff867d20dc8b9`;
- elapsed Python time: `929.87 s`;
- peak JAX device bytes: `15,530,472,704`;
- hard failures and nonfinite values: zero;
- optimizer updates and retained proposals: zero.

The minimum cosine over all 40 constraints is `0.050356`, well above the
frozen `0.001` direction gate. The simplex support contains 14 constraints;
its largest individual weight is `0.16696`. The support includes constraints
from every component family, so the solution is not an unconstrained mean
direction mislabeled as group-aware.

Every tested step improves every component in all `10/10` groups. At the
largest original step, `0.0003`, the component median relative changes are:

| Component | Median relative change | Frozen magnitude gate |
|---|---:|---:|
| One-day anchor | `-1.621744e-5` | fail |
| NPP | `-1.532049e-4` | pass |
| AGB tendency | `-2.008961e-4` | pass |
| BGB tendency | `-5.177755e-5` | fail |

The least negative group-component change is `-3.182271e-6`; there is no
hidden regression. Formal classification is nevertheless
`group_aware_finite_step_not_supported` because the preregistered contract
requires every component's median decrease to reach `1e-4`.

## Interpretation

The earlier `4/10` ordinary-averaging result was an aggregation failure, not
evidence that the Attention architecture lacks a joint productivity/biomass
direction. Group-aware combination solves the observed direction-sign
problem. The remaining question is step size: the original grid ends before
the anchor reaches the effect threshold.

Changes through `0.0003` are close to linear. Linear extrapolation places the
anchor threshold near global norm `0.00185` and the BGB threshold near
`0.00058`. This supports one explicit extension through `0.002`; it does not
justify an open-ended search or training continuation.

## Next Gate

The frozen extension contract is
`successor_group_aware_step_extension_v1.json`, SHA256
`e78db1bac583d6adf853eb38207c7c410b5befb1cbb087e01da527d28fd3bda1`.
It re-evaluates the same 40-objective construction at global norms
`0.0005, 0.001, 0.0015, 0.002`. It remains zero-update and report-only.

If no step passes, close the current finite-step effect-size gate and redesign
the training mechanism rather than extending the grid again. If a shared step
passes, review a computationally bounded update mechanism before any repeated
training, because recomputing 40 full gradients per update is not itself a
practical production training strategy.
