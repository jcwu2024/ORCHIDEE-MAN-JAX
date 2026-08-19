# Hierarchical Coordinate-Query Attention v3 Local Acceptance

Date: 2026-08-19.

Status: local admission accepted. The candidate is eligible for a fresh
one-epoch litter-only smoke on a free `gln01` GPU after its source is
versioned. No GPU execution, checkpoint, rollout, paid job, sealed-test read,
Gate D2, or product promotion is accepted by this record.

## Accepted Candidate

The implemented candidate is
`hierarchical_coordinate_query_attention_v3`. It preserves the canonical
assembler, mechanistic-output v2 learned/exact partition, complete 181-field
boundary, one conservative update, and retained exact daily tail.

Only the failed `litter_poc_doc` mapping changes:

- the rejected three-block global litter expert and 13 shared litter label
  heads are absent from the candidate parameter tree;
- 42 fields are routed through four fixed stages: 12 litter-composition, five
  DOC-reaction, 15 DOC-transport, and ten inventory-transition fields;
- every stage has two width-384, eight-head coordinate cross-Attention/FFN
  blocks and one shared width-512 decoder;
- every field has its own final scalar projection;
- each coordinate attends admitted source-family tokens, its own physical
  axis tokens, its local day-start owner token, and upstream stage contexts;
- stage context is field-balanced, not scalar-count-weighted.

The instantiated tree has 58,736,698 float32 parameters, below the frozen
60,000,000 hard cap. Attention-bank size is bounded by admitted source-family
count, field rank, and at most three upstream stage tokens. There is no dense
37,518-coordinate self-Attention.

## Trainability Corrections

Local official-loss gradients exposed two output-parameterization defects in
the rejected mapping.

First, zero-mean nonnegative fields used a softplus origin at
`1e-12 * scale`. Its derivative was about `1e-12`, and the resulting
`litter_above.incoming_amount` field-head gradient was only about `1e-28` in
the local official-loss probe. This is consistent with that field's 5.6326%
reduction in the rejected isolated run.

Second, zero-centered non-direct unit-interval fields used a `1e-6` center
and multiplied raw displacement by inverse local sigmoid slope. Ordinary
random raw values could therefore saturate the sigmoid before the first
update; `lignin_struc_below` had an exact-zero field-head gradient in the
first candidate probe. This is consistent with its 26.4348% rejected-run
reduction.

Candidate litter fields now use a `0.05 * scale` nonnegative origin and a
`0.05` bounded center with direct logit displacement. Both transformations
remain physically constrained and have initial derivatives above 0.04 in the
local synthetic normalization panel. All non-litter fields retain the old
transform defaults exactly.

## Local Evidence

The candidate-specific suite passes 7/7 tests:

- exact capacity and complete 42-field route closure;
- usable nonnegative and bounded-transform initialization gradients;
- eager/JIT 161-field, 39,006-scalar finite forward;
- one complete `bind_daily_operator` transition;
- finite official litter-expert loss and gradients through all eight new
  Attention blocks, four decoders, and every supervised field head;
- explicit physical-parameter ownership remains outside opaque PFT tokens;
- inactive-PFT isolation and PFT permutation equivariance at
  `atol=1e-10, rtol=1e-12`.

The bound transition passes the existing state, nonnegative-inventory, and
water/carbon budget validation. It uses neither reconstructed half-hour
forcing nor a 48-step state scan.

Twenty-five existing output-transform, strong-baseline, and mechanistic-
output regressions pass. Nineteen qualification/launcher tests also pass.
Ruff, Python compilation, Bash syntax, `git diff --check`, and runner `--help`
pass. Total focused and related tests: 51.

## Runner Admission

The existing one-point runner and free-node launcher now accept the explicit
`--hierarchical-coordinate-attention` /
`HIERARCHICAL_COORDINATE_ATTENTION=1` selection. Candidate coverage records
the new coordinate blocks and field heads rather than removed litter expert
and label-head parameters. The same 46-update one-epoch schedule, complete
mechanistic loss, mask/count checks, science diagnostics, and zero-hard-
failure gate remain active. Smoke and full memorization runs write no
checkpoint.

## Next Gate

After versioning the current source, run exactly one fresh litter-only,
one-epoch smoke on one free `gln01` GPU:

```text
MECHANISTIC_OUTPUT=1
HIERARCHICAL_COORDINATE_ATTENTION=1
ISOLATED_EXPERT=litter_poc_doc
TRAINING_SMOKE=1
```

The smoke opens the 100-epoch isolated diagnostic only if all 46 updates are
finite, litter loss decreases, counts are unchanged, both selected coordinate
coverage groups receive gradients and realized updates, and every hard count
is zero. The smoke is execution admission, not scientific promotion.
