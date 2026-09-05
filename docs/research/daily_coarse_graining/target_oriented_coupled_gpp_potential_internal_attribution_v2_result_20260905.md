# Coupled GPP Potential Internal Attribution v2 Result (2026-09-05)

Status: accepted zero-training localization; the sampled GPP jump is a
full-precision block-0 Q/K attention-selection collapse

Report SHA256:
`f781e4a92eeac1dd9bc0f986a55fa83478d18d8c20025eb12a2d74067207e0a3`

Executable commit: `56ec8f2cca80a78ef37585110c1e3169214ac6a8`

## Execution And Integrity

The corrected free-`gln01` V100 run completed in `896.484` seconds with
status `passed` and classification
`potential_block_conditioning_persists_in_full_float64`. All engineering
gates pass: historical direction and displacement, control trajectory,
isolated endpoints and maximum interval, exact 32-leaf group coverage,
factorization, finite values, float32 shadow reproduction, and shadow dtypes.

The target remains artifact update 550, runtime year 1976, seven-day window
index 3, Day-5 index 4, and vegetation PFT index 1. No state or parameter was
retained.

## Precision Result

The isolated production path changes from GPP `2.7673221604` at `11/16` to
`4.8699012434` at `12/16`, an adjacent effect of `+2.1025790830`. That single
interval carries `1.0000076766` of the endpoint effect.

The full-float64 potential branch remains sharp on the same interval. Its
endpoint jump is `2.1025628812`, versus `2.1025629425` for float32, with the
same sign and a magnitude ratio of `0.9999999709`. Intermediate float32
precision is therefore not the sampled cause.

## Block And Component Attribution

Only the first potential residual block is material:

| Parameter group | Sufficiency | Necessity capture |
| --- | ---: | ---: |
| `block_0` | `1.0000001583` | `0.9999992086` |
| `block_1` | `-0.0000007914` | `-0.0000001583` |

Replacing block 0 alone moves the lower tree to GPP `4.8699015762`; replacing
it back in the upper tree reduces GPP to `2.7673204964`. Block 1 parameter
changes are immaterial. Its large activation changes are downstream responses
to the block-0 output, not evidence that its parameters caused the jump.

Within block 0, only `attention.query` and `attention.key` cross the `0.10`
materiality threshold. Each treatment projection inserted alone is
`1.0000001583` sufficient, but removing either one alone from the upper tree
captures zero of the jump. The result is redundant threshold crossing: either
Q or K displacement can push the lower state across the boundary, while the
other upper displacement keeps the upper state across it. It does not support
choosing query or key as a uniquely necessary leaf family.

## Activation Mechanism

The block-0 input is identical between the two parameter trees. Its projected
query changes by only `1.2416e-6` RMS. The projected key bank has RMS about
`7.1674e7`; its small parameter displacement changes it by `89.24` RMS and
changes the attention logits by `455.82` RMS. The attention context then
changes by `6.1962e8` RMS before LayerNorm.

Head 7 is the first clear discrete selection change:

| Quantity | `11/16` lower | `12/16` upper |
| --- | ---: | ---: |
| Winning bank index | `205` | `153` |
| Maximum weight | `0.459139` | `1.0` |
| Attention entropy | `1.418627` | approximately `0` |
| Maximum float32 logit | `8.1263` | `7796.5381` |

The full-float64 trace makes the same index switch, with upper maximum logit
`8771.1590` and weight `1.0`. Canonical bank-layout reconstruction maps index
`205` to the day-start
`slowproc_stomate_previous_step_state.litterpart` field-summary token and
index `153` to `slowproc_stomate_previous_step_state.DOC`. Both are fixed
`state_carbon` bank inputs in this parameter-only intervention.

Several block-1 heads change after block 0, as expected from their changed
query input. That downstream response does not alter the causal parameter
localization to block 0.

## Interpretation And Next Gate

For this sample, the failure is no longer an undifferentiated architecture or
training problem. Tiny Q/K parameter movement acts on extremely large key
features, flips one attention head from a mixed litterpart-centered selection
to a saturated DOC selection, and changes the final potential. More stored
weight precision, a larger parameter count, or changes confined to block 1
are not supported remedies.

This is one sampled jump, not proof that the same head and tokens explain all
GPP error. Before discussing an architecture edit, the active gate is a
zero-training head/token causal audit at the same lower/upper trees: paired
per-head Q/K slice controls, explicit head-7 logit records for DOC and
litterpart, and matched token-mask interventions. That audit must distinguish
head-7 mediation from correlated downstream changes and report whether the
DOC token is necessary at the upper state.

No optimizer, parameter update, checkpoint, recovery, architecture change,
Teacher regeneration, historical v5 read, validation read, sealed-test read,
half-hour forcing reconstruction, 48-step scan, paid execution, 365-day
rollout, Gate D2, or product acceptance occurred.
