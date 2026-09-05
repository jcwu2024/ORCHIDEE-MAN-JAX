# Coupled GPP Potential Internal Attribution v2

Date: 2026-09-05

Status: completed; the accepted result is recorded in
`target_oriented_coupled_gpp_potential_internal_attribution_v2_result_20260905.md`

## Question

Within the isolated `structured_gpp.potential_blocks` path, which residual
block and internal parameter components cause the observed
`11/16 -> 12/16` Day-5 GPP jump, and does that jump survive when every
operation inside the potential branch is evaluated in float64?

This is a corrected causal localization inside the existing model. It is not
an architecture, optimizer, training, or candidate/control experiment.

## Frozen Evidence And Sample

- accepted structured-precision parent:
  `outputs/diagnostics/target-oriented-coupled-gpp-structured-precision-587c3fd.json`,
  SHA256 `40c617ba096f5b9b50482316473d416b17dafb90b7162cbf0d59c9fffaf5b693`;
- engineering-inconclusive v1 report:
  `outputs/diagnostics/target-oriented-coupled-gpp-potential-internal-aae98e3.json`,
  SHA256 `faf1e4c9a1220a4139372fe59765080cd3979e1fcbfff8395777e3d78d4e5f64`;
- artifact update `550`, runtime year `1976`, seven-day window index `3`,
  Day-5 index `4`, vegetation PFT index `1`;
- the accepted control and rejected finite-step treatment are reconstructed
  exactly as before;
- all top-level `structured_gpp` parameters remain at control except the
  interpolated `potential_blocks`.

The isolated production path retains all 17 fractions `0/16, ..., 16/16`.
It must reproduce the parent control and potential-block-only endpoint, and
its maximum adjacent change must be `11/16 -> 12/16`. That interval carried
more than the complete endpoint effect in v1 because the remaining changes
are small and opposite-signed. Its adjacent magnitude must therefore also
remain at least `0.75` of the endpoint effect.

## Matched Causal Controls

The lower and upper parameter trees are now fixed at `11/16` and `12/16`.
Evaluate the same two whole residual blocks and the same 16 non-overlapping
component groups used by v1:

```text
block_0, block_1

attention.query, attention.key, attention.value, attention.output
attention_norm, ffn.first, ffn.second, ffn_norm
```

For each group, insert its upper value into the lower tree and replace its
upper value with the lower value in the otherwise upper tree. A group is
material when absolute sufficiency or necessity capture is at least `0.10`.
All 32 array leaves must be covered exactly once. A whole-block effect without
a material child is reported as distributed within-block interaction; no
arbitrary child winner is selected.

## Activation-Precision Control

Over the same 17 fractions, compare:

- the production-equivalent float32 shadow, which must reproduce production;
- a full-float64 potential branch including inputs, Q/K/V, logits, softmax,
  residuals, LayerNorms, FFNs, output projection, and softplus.

The sharpness, smoothing, sign, and endpoint-comparability thresholds remain
identical to v1. Adjacent traces are taken at `11/16` and `12/16` and report
per-stage RMS changes, LayerNorm gain, and attention-head entropy, maximum
weight, and selected bank index.

## Acceptance And Decision Boundary

All parent, direction, endpoint, interval, coverage, factorization, dtype,
finite-value, and float32 reproduction gates must pass before component or
precision attribution is accepted.

- A smoothed comparable float64 path opens discussion of numerical rescaling
  or internal precision control.
- A sharp comparable float64 path establishes intrinsic conditioning for this
  sample. Material block/component controls identify the smallest supported
  parameter family.
- A material block with no material child is a distributed interaction.
- Any engineering failure remains inconclusive and does not authorize a model
  change.

No architecture edit, optimizer, parameter update, checkpoint, recovery,
Teacher regeneration, historical v5 read, validation read, sealed-test read,
half-hour forcing reconstruction, 48-step scan, paid execution, 365-day
rollout, Gate D2, or product acceptance is allowed. One corrected
zero-training free-`gln01` execution is permitted after local acceptance.
