# Coupled GPP Attention Head And Token Attribution Result (2026-09-05)

Status: accepted zero-training causal localization; block-0 head 7 and its
DOC selection mediate the sampled Day-5 GPP jump

Report SHA256:
`2c8a28d82eec1824163a3796a2777716cca8455a42241591bfa9d21478d94836`

Executable commit: `7788d6b18a0405876d425166fa8f193fb10b73d6`

## Execution And Integrity

The bounded free-`gln01` V100 run completed in `904.470` seconds with
status `passed` and classification
`head7_qk_pair_and_doc_token_mediate_sampled_jump`. Every engineering gate
passed: parent and direction identity, control trajectory, endpoint
factorizations, float32 shadow equivalence, non-potential multiplier,
384/8/48 head geometry, exact Q/K slice coverage, canonical token mapping,
baseline head-7 reproduction, exact mask changes, and finite nonnegative
intervention values.

JAX exposed a device-memory limit of about `25.56` GB. Peak bytes actually in
use were about `11.02` GB and bytes in use at report time were about `1.27`
GB. No optimizer or parameter update ran.

## Per-Head Causal Result

Only block-0 head 7 is material:

| Q+K head slice | Sufficiency | Necessity capture |
| --- | ---: | ---: |
| heads 0 through 6, each | `0` | approximately `0` |
| head 7 | `1.0000001583` | `0.9999992086` |

Putting only head 7's upper Q/K kernel columns and bias slices into the lower
tree moves GPP from `2.7673221604` to `4.8699015762`. Replacing those same
slices with their lower values in the otherwise upper tree moves GPP from
`4.8699012434` to `2.7673204964`. Thus head 7 is both sufficient and
necessary for this sampled jump; the other seven heads do not contribute a
measurable parameter-intervention effect.

All eight 48-coordinate slices exactly reconstruct the full 384-coordinate
Q and K parameter deltas. Head 7's Q and K combined delta norms are
`1.58005e-6` and `1.57708e-6`, respectively. Other heads have comparable
orders of magnitude, so the result is not explained by a uniquely large
head-7 parameter update.

## Q/K Crossed Result

Inside head 7, inserting either upper Q alone or upper K alone into the lower
tree is fully sufficient (`1.0000001583`). Conversely, removing only Q or
only K from the upper tree captures approximately zero of the jump. Removing
Q and K together captures `0.9999992086`.

This reproduces the earlier component-level redundancy at the correct head:
either side of the dot product can cross the attention threshold when the
other side remains lower, but restoring only one side cannot bring the upper
tree back across it. Q and K are therefore a coupled conditioning mechanism,
not competing unique causes.

## Token-Mediation Result

The unmodified head-7 endpoints reproduce exactly:

| Quantity | `11/16` lower | `12/16` upper |
| --- | ---: | ---: |
| winner | litterpart, index `205` | DOC, index `153` |
| winner logit | `8.126287` | `7796.538086` |
| winner weight | `0.459139` | `1.0` |
| DOC logit/rank | `-6281.570801`, rank `423/423` | `7796.538086`, rank `1/423` |
| litterpart logit/rank | `8.126287`, rank `1/423` | `8.126174`, rank `2/423` |

The decisive intervention removes DOC only from block 0, head 7, PFT 1:

- lower GPP remains exactly `2.7673221604`;
- upper GPP becomes `2.7673204964`;
- the residual jump is approximately zero and the upper removal captures
  `1.0000007914` of the original effect;
- head 7 returns to litterpart with weight `0.459123`.

Removing litterpart only from head 7 leaves the complete lower-to-upper jump
unchanged. Removing DOC and litterpart together gives the same GPP result as
removing DOC alone, with bank index `200` becoming the next head-7 winner.

The global DOC mask also eliminates the jump, but shifts both absolute
endpoints: it raises the lower state by `0.714516` of the original jump and
lowers the upper state by `0.285484`. This shows that DOC participates in
other attention rows too. The head-7-only mask is the cleaner mediation test
because it removes the full jump without changing the lower endpoint.

## Interpretation

For this sampled discontinuity, the causal chain is now resolved:

```text
tiny coupled Q/K parameter displacement
  -> block-0 head-7 DOC logit changes sign and grows to 7796.5
  -> softmax collapses from a mixed litterpart-centered distribution to DOC
  -> potential and routed GPP jump by about 2.10
```

The parent audit already showed a projected-key bank RMS around `7.17e7` and
that float64 retains the same collapse. Combined with the per-head result,
the supported defect is unnormalized Q/K dot-product conditioning against a
heterogeneous, very large-magnitude source bank. Merely increasing parameter
count, using float64, modifying block 1, or reducing only head 7's learning
rate is not supported by this evidence.

This remains one deliberately selected pathological sample. It proves the
mechanism of that discontinuity, not that the same head or DOC token explains
all GPP error, and not that a conditioning change alone will satisfy 7/30-day
or long-term scientific gates.

## Next Gate

The active gate is user discussion of the first conditioning-controlled
architecture before implementation. The evidence favors preserving the
current process-head boundary and attention/value pathway while bounding Q/K
selection geometry, for example through per-head Q/K normalization with a
bounded temperature. Pre-Q/K bank normalization and bounded-logit attention
remain alternatives to compare at the design level.

No architecture change, training, Teacher regeneration, historical v5 read,
validation read, sealed-test read, half-hour forcing reconstruction, 48-step
scan, paid execution, 365-day rollout, Gate D2, or product acceptance
occurred.
