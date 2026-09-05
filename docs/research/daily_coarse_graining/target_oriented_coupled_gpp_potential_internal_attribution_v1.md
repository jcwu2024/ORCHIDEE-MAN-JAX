# Coupled GPP Potential Internal Attribution v1

Date: 2026-09-05

Status: frozen after user approval, before implementation and zero-training
execution

## Question

The accepted structured-precision attribution assigns essentially the entire
sampled Day-5 GPP jump to `structured_gpp.potential_blocks`, while float64
endpoint-weight interpolation leaves the jump unchanged. Which of the two
residual blocks and which internal parameter components cause the adjacent
`10/16 -> 11/16` jump, and does the jump survive when every operation inside
the potential branch is evaluated in float64?

This is causal localization inside the existing model. It is not a new
architecture, an optimizer experiment, or a candidate/control comparison.

## Frozen Parent And Sample

- accepted parent result:
  `outputs/diagnostics/target-oriented-coupled-gpp-structured-precision-587c3fd.json`;
- parent SHA256:
  `40c617ba096f5b9b50482316473d416b17dafb90b7162cbf0d59c9fffaf5b693`;
- artifact update `550`, runtime year `1976`, seven-day window index `3`,
  Day-5 index `4`, vegetation PFT index `1`;
- the accepted in-memory control and rejected finite-step treatment are
  reconstructed exactly as in the parent diagnostic;
- all top-level `structured_gpp` parameters remain at control except the
  interpolated `potential_blocks`.

The isolated production path uses the already frozen 17 fractions
`0/16, ..., 16/16`. It must reproduce the parent control GPP and the accepted
potential-block-only endpoint, and its largest adjacent change must remain
`10/16 -> 11/16`. Those two adjacent parameter trees are the lower and upper
states for every causal replacement below.

## Frozen Parameter Attribution

First evaluate the two whole-block groups independently with matched
sufficiency and leave-one-out necessity controls:

```text
block_0
block_1
```

Then evaluate all 16 non-overlapping component groups, eight per block:

```text
attention.query
attention.key
attention.value
attention.output
attention_norm       = scale + bias
ffn.first
ffn.second
ffn_norm             = scale + bias
```

For each group, insert its upper value into the lower tree and replace its
upper value with the lower value in the otherwise upper tree. A group is
individually material when the absolute sufficiency or necessity-capture
fraction is at least `0.10`, matching the accepted parent attribution. The 16
component groups must cover all 32 potential-block parameter leaves exactly
once; dictionary insertion order is immaterial.

## Frozen Activation-Precision Attribution

A diagnostic shadow forward records the existing operation order for the
selected PFT:

```text
query/key/value projections
attention logits and softmax weights
attention output and residual
attention LayerNorm
FFN first projection and GELU
FFN second projection and residual
FFN LayerNorm
final potential logit and softplus
```

It runs in two modes over the same 17 potential-block fractions:

- production-equivalent float32 mode, which must reproduce the production
  potential and routed GPP within the existing factorization tolerance;
- full float64 potential mode, which casts the fixed branch input and bank,
  interpolated block parameters, attention/softmax, residuals, both
  LayerNorms, FFNs, final projection, and softplus to float64. Upstream parent
  context and the other GPP factors remain fixed at their production-float32
  control values.

The accepted sharpness thresholds remain unchanged: a path is sharp when one
adjacent interval carries at least `0.75` of its endpoint effect and smoothed
when no interval carries more than `0.25`. Endpoint effects must retain their
sign and have a magnitude ratio in `[0.75, 1.25]` before a precision
classification is made.

The lower and upper traces report finite-value summaries, per-stage RMS
changes, LayerNorm input variance and change gain, and per-head attention
entropy, maximum weight, and selected bank index. These are explanatory
measurements, not post-hoc acceptance thresholds.

## Decision Boundary

- A smoothed full-float64 path classifies intermediate float32 activation
  precision as dominant and opens discussion of numerical rescaling or
  precision control without changing the learned mechanism.
- A sharp comparable full-float64 path classifies intrinsic high conditioning
  inside the current potential blocks. Material block/component controls then
  identify the smallest causal parameter family available from this audit.
- A material whole block with no material child component is classified as a
  distributed within-block interaction; no arbitrary leaf winner is chosen.
- A non-comparable endpoint, failed float32 reproduction, nonfinite trace, or
  failed parent reproduction is an engineering/inconclusive result and does
  not authorize a model change.

No architecture edit, optimizer, parameter update, checkpoint, recovery,
Teacher regeneration, historical v5 read, validation read, sealed-test read,
half-hour forcing reconstruction, 48-step scan, 365-day rollout, Gate D2, or
product acceptance is allowed. The complete diagnostic is restricted to one
free-`gln01` execution after local tests pass.
