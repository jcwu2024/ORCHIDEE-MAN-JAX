# Distribution-Aware Adapter Training Result

Date: 2026-08-29

Status: engineering run passed; maintenance mechanism passed; complete
scientific promotion failed; checkpoint is diagnostic only

Slurm job: `14856147`

Executable commit:
`90c6636b5ed479d76c2ccab4753b267cb628f2bc`

Bounded training contract SHA256:
`0c60924ee12d0ea555b082f76990ca55e34f86f67effb052f5eec44c01210051`

Report SHA256:
`7f6d5686a560fac9533da505a829f0eb2a392a20d5aef14e470421f670ba9257`

Adapter checkpoint SHA256:
`baae0c13aa7f22f04e56bec43b915e7e2dab8c2ca775e91d260f26136d55c7da`

## Execution

The paid `gnall` job completed with exit code zero in `03:38:12`. All 250
preregistered optimizer updates executed and persisted. The complete forward
used the `93,504,933`-parameter candidate, while direct autodiff and fresh
AdamW updated only the `9,750,529` distribution-aware maintenance-adapter
parameters. All `83,754,404` retained parent coordinates remained exactly
unchanged.

No update was clipped at the frozen threshold `2.692620468992339`. Raw
combined gradient norms ranged from `0.0072233` to `2.2248812`, with median
`0.1494472`; all 36 adapter leaves had finite nonzero gradients throughout.
The first compiled update took `809.91 s`; the median hot update took
`44.05 s`.

Teacher v6 was not regenerated, historical v5 was not used, and validation
and sealed-test values were not read. The candidate reconstructed no
48-record forcing sequence and executed no subdaily state scan. Every final
screen prediction/state finite, discrete, unexpected-mask, budget, and
negative-stock hard count is zero.

## Frozen Screen

The source-backed maintenance formula already supplied most of the mechanism
gain at initialization. Adapter training produced a further material gain:

| Horizon | Update 0 median reduction | Update 250 median reduction |
| --- | ---: | ---: |
| 7 days | `73.6621%` | `90.7403%` |
| 30 days | `87.3779%` | `95.7084%` |

The evaluated-value-weighted aggregate normalized RMSE values are:

| Key | 7-day parent | 7-day update 250 | 30-day parent | 30-day update 250 |
| --- | ---: | ---: | ---: | ---: |
| GPP | `0.412859` | `0.413106` | `0.536770` | `0.496008` |
| Maintenance respiration | `0.312606` | `0.093415` | `0.363984` | `0.082147` |
| Growth respiration | `0.207491` | `0.245293` | `0.273346` | `0.262907` |
| Autotrophic respiration | `0.282469` | `0.195314` | `0.356376` | `0.188159` |
| NPP | `0.563722` | `0.535110` | `0.621040` | `0.610814` |
| AGB | `0.002243` | `0.002228` | `0.012193` | `0.011841` |
| BGB | `0.002718` | `0.002733` | `0.011360` | `0.011645` |
| Total live biomass | `0.003294` | `0.003275` | `0.016132` | `0.015935` |
| Ecosystem carbon | `0.006348` | `0.007038` | `0.021293` | `0.021273` |

Maintenance and all four reported carbon-stock aggregates pass their
absolute RMSE/bias gates. GPP, growth respiration, and NPP still fail at both
horizons. Seven-day autotrophic respiration also fails; its 30-day value
passes.

Across the eight independent groups, median protected-score changes relative
to the fixed parent are mixed. At 7 days, growth respiration and NPP regress
by `95.04%` and `27.02%`; at 30 days they change by `+2.27%` and `-1.67%`.
The 7-day relative biomass increases occur around very small absolute errors;
the aggregate AGB, BGB, total-live, and ecosystem-carbon gates still pass.
They nevertheless correctly prevent promotion under the frozen no-regression
contract.

## Decision

The maintenance-specific mechanism is supported, but the complete candidate
is not promoted. The checkpoint remains `promotion_eligible=false`, and
unchanged adapter-only continuation is closed. Validation, 365-day rollout,
Gate D2, sealed test, and product acceptance remain closed.

This result does not reject the coordinate-aligned Attention backbone. It
shows that the newly trainable maintenance boundary can repair maintenance
without directly changing the frozen GPP head. GPP remains nearly unchanged
at 7 days and materially above its absolute gate at both horizons. Source
identity in `stomate_npp.f90::npp_calc:314-530` then propagates GPP and
maintenance into exact growth respiration and NPP.

## Next Gate

The zero-training Teacher-GPP causal bridge is frozen at contract SHA256
`ce2eefb1360bb5b6f49f727ab470d1faef3952a7649d411539d81ba3b28c5647`.
It uses the same eight train/train groups and the update-250 diagnostic
checkpoint. It replaces only predicted daily fast GPP with matched Teacher
fast GPP before the conservative update and retained exact tail, while
retaining the model's own cross-day state, maintenance adapter, forcing, and
all other outputs.

Compare ordinary free rollout, Teacher-GPP bridge, and Teacher-day-start
controls at 7 and 30 days. If Teacher GPP repairs growth respiration, NPP, and
protected carbon without new hard failures, the next architecture discussion
should be scoped to a structured GPP boundary. If it does not, the discussion
must include shared Attention/state representation. No network change or
optimizer update is authorized before that result.

The local implementation and bounded launcher are qualified. The next
operation is the two-mode free-`gln01` execution described in
[`target_oriented_distribution_aware_gpp_bridge_v1.md`](target_oriented_distribution_aware_gpp_bridge_v1.md).
