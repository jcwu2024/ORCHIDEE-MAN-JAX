# Distribution-Aware Teacher-GPP Bridge Result

Date: 2026-08-29

Status: accepted zero-training causal audit; GPP boundary supported as the
next architecture scope

Executable commit:
`5480f51852146210fa391ea5c9c1601a043323b8`

Bridge contract SHA256:
`ce2eefb1360bb5b6f49f727ab470d1faef3952a7649d411539d81ba3b28c5647`

Report SHA256:
`fa9bc44556ba4376ebb03e55490b20f4385dd9929f8a83cf01a4ce7998531c52`

## Execution

The forward-only audit completed on free `gln01` GPU 0 in `934.80 s`. It
loaded the exact update-250 adapter checkpoint and all `93,504,933` candidate
parameters while executing zero optimizer updates and zero parameter steps.
The accepted ordinary free-rollout groups were reused; only the Teacher-GPP
bridge and Teacher-day-start modes were reexecuted.

Teacher v6 was not regenerated, historical v5 was not used, and no validation
or sealed-test value was evaluated. The run reconstructed no 48-record forcing
sequence and executed no subdaily state scan. Both executed modes have zero
prediction/state/discrete/budget/negative-stock hard failures.

## Teacher-GPP Result

The evaluated-value-weighted aggregate normalized RMSE values are:

| Key | 7-day free | 7-day bridge | 30-day free | 30-day bridge |
| --- | ---: | ---: | ---: | ---: |
| GPP | `0.413106` | `0.000000` | `0.496008` | `0.000000` |
| Maintenance respiration | `0.093415` | `0.083801` | `0.082147` | `0.079441` |
| Growth respiration | `0.245293` | `0.018578` | `0.262907` | `0.020382` |
| Autotrophic respiration | `0.195314` | `0.057495` | `0.188159` | `0.052377` |
| NPP | `0.535110` | `0.068528` | `0.610814` | `0.062428` |
| AGB | `0.002228` | `0.000361` | `0.011841` | `0.001412` |
| BGB | `0.002733` | `0.000295` | `0.011645` | `0.002082` |
| Total live biomass | `0.003275` | `0.000419` | `0.015935` | `0.001625` |
| Ecosystem carbon | `0.007038` | `0.005406` | `0.021273` | `0.021140` |

Every frozen absolute required key passes at both horizons. Across the eight
groups, the median acceptance-score reductions are:

| Downstream key | 7 days | 30 days |
| --- | ---: | ---: |
| Growth respiration | `99.6757%` | `99.5925%` |
| NPP | `98.6264%` | `98.8312%` |

These values comfortably exceed the preregistered `50%` causal threshold.
The frozen classification is therefore
`gpp_boundary_supported_as_next_scope`.

## Matched Control

Teacher-day-start is not a remedy. Its aggregate normalized RMSE is
`3.3236/3.6892` for GPP, `1.9402/2.1660` for growth respiration, and
`3.7670/4.2048` for NPP at 7/30 days. Replacing the entire recurrent state
with Teacher state moves the learned flux head far outside the state
distribution on which its current mapping works. This control rules out a
generic recommendation to Teacher-force all state coordinates.

## Interpretation

The remaining long-horizon carbon failure of this checkpoint is causally
dominated by the learned GPP boundary. Exact GPP repairs retained growth
respiration and NPP, total autotrophic respiration, and the protected biomass
and ecosystem-carbon outputs while preserving the model's own cross-day state,
maintenance branch, forcing, and every other learned output.

This result does not prove that the shared Attention representation is
finished or product-ready. It does show that replacing or deleting Attention
before repairing GPP is not supported by current evidence. The next candidate
should preserve the accepted coordinate-aligned Attention backbone initially
and redesign the GPP output boundary, its physical conditioning, and its
targeted loss/rollout supervision.

## Decision

The zero-training audit is complete. Validation, 365-day rollout, long held
points, Gate D2, sealed test, and product acceptance remain closed. Per the
user-required boundary, implementation stops for an architecture discussion
before any structured GPP head is built or trained.
