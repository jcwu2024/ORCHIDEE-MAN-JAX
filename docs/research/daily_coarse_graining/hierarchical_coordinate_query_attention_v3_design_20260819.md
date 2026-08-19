# Hierarchical Coordinate-Query Attention v3 Design

Date: 2026-08-19.

Status: approved by the user for design freeze and local implementation only.
No GPU training, rollout, sealed-test access, Gate D2, or product promotion is
authorized by this document.

## Decision

The next formal neural process-head candidate is
`hierarchical_coordinate_query_attention_v3`. It preserves the accepted
mechanistic-output v2 boundary and replaces only the failed
`litter_poc_doc` coordinate mapping. The previous 49,743,001-parameter model
is retained as rejected evidence and is not treated as a promotable baseline.
No separately trained weak control is required for this absolute-capability
gate.

## Evidence Addressed

The isolated 364-day diagnostic reduced the `litter_poc_doc` expert loss by
only 91.8014%. Twelve of 28 dominant fields failed even though selected
expert-block and label-head gradients and updates were nonzero. The current
mapping compresses all admitted litter/POC/DOC sources into one expert token,
then gives each output coordinate only additive axis/local-state features and
a label-local MLP. One token is therefore the only shared learned process
state for 42 fields and 37,518 scalar coordinates.

The new mechanism removes that bottleneck without using dense all-coordinate
self-Attention. Dense Attention over 37,518 coordinates is neither necessary
nor an acceptable memory cost.

## Frozen Input And Parameter Ownership

The candidate consumes the unchanged canonical daily input:

- day-start continuous and discrete state with exact defined masks;
- native six-hour forcing records and their interval metadata;
- explicit PFT state, traits, fraction, and active mask;
- named static, annual, process-static, and calendar conditions.

Physical PFT parameters retain their existing explicit ownership. `vcmax25`
and the three `maint_resp_slope` coefficients remain source-backed factors
around learned GPP and maintenance-respiration environmental responses.
`alloc_min` and `residence_time` remain inputs to retained exact daily
processes. None of these six channels is absorbed into an opaque learned PFT
embedding.

## Candidate Mechanism

The shared forcing encoder, source tokens, factorized axis tokens, seven
expert registry, bounded expert graph, output transforms, and all non-litter
expert mappings remain unchanged. The rejected litter global expert blocks
are removed from the candidate parameter tree; route-local coordinate stages
consume admitted source-family and axis tokens directly.

For every learned `litter_poc_doc` field, the candidate first constructs one
query per output coordinate from:

- the field identity;
- the relevant PFT, deep-layer, litter-pool, carbon-pool, DOC-form, DOC-pool,
  element, soil-tile, and transfer-direction axis tokens;
- the corresponding day-start owner value when one exists;
- the original litter expert context.

Each coordinate query then cross-attends a bounded route-local source bank:

- one token per admitted source family;
- the coordinate's own axis tokens;
- its local day-start owner token;
- already-computed upstream process-stage context.

The bank size is bounded by source-family count plus output rank and four
stage tokens. It does not grow with the 37,518 output scalar count.

The shared coordinate blocks execute in a fixed feed-forward order:

1. `litter_composition`: litter respiration/input, lignin, litter partition,
   dead leaves, fuel, and litter-to-DOC coordinates;
2. `doc_reaction`: POC/DOC decomposition and free/adsorbed equilibration;
3. `doc_transport`: external DOC input/export, diffusion, advection,
   cryoturbation, and permafrost/peat transfer;
4. `inventory_transition`: all litter, carbon-pool, DOC, and interception
   outgoing-fraction/incoming-amount coordinates, conditioned on the three
   upstream stage contexts.

Each stage uses shared Transformer-style multi-head cross-Attention and a
shared feed-forward decoder. The final scalar projection is field-specific,
so heterogeneous fields formerly sharing one label head no longer share the
same last mapping. A masked stage pool provides process context to downstream
stages and to all decoders in the same stage. The pool first averages active
coordinates within each field and then averages fields equally, so large DOC
tensors cannot erase small lignin, litter-partition, or inventory fields.

Zero-centered nonnegative litter fields use a candidate-local well-conditioned
softplus origin of `0.05 * scale`. The rejected mapping used `1e-12 * scale`,
which placed zero-mean incoming amounts, dead leaves, and fuel fields in a
near-saturated softplus region. The new origin remains strictly nonnegative
while preserving a usable initialization gradient. Non-litter fields retain
their existing transform exactly.

Zero-centered unit-interval litter endpoints use a candidate-local center of
`0.05` and a direct logit displacement. The rejected mapping used a center of
`1e-6` and multiplied the raw displacement by the inverse local sigmoid
slope, so ordinary random initialization could saturate the transform before
the first update. The candidate remains bounded in `[0, 1]` without that
initial gradient collapse.

This is one true-daily feed-forward graph. It has no learned cross-day hidden
state, reconstructed half-hour forcing, 48-step state scan, or recurrence over
the Teacher trajectory.

## Output And State Use

The candidate emits exactly the learned mechanistic-output v2 fields. It does
not predict next-day water or carbon stocks directly. Existing transforms
produce bounded fractions, nonnegative amounts, finite residual coordinates,
and exact masks. The existing conservative updater applies inventory
transfers, the existing mechanistic reconstructor materializes exact fields,
and the canonical retained-tail adapter completes the daily transition.

The required production binding remains:

```python
bind_daily_operator(
    input_assembler=canonical_assembler,
    process_head=candidate_head,
    retained_tail=canonical_retained_tail_adapter,
)
```

## Capacity Bound

The working width remains 384 with eight Attention heads. Each of the four
new stages has two coordinate cross-Attention/feed-forward blocks and a shared
512-wide decoder. Rejected litter expert blocks and label heads are removed
from the new parameter tree. The instantiated count must remain at or below
the existing 60,000,000 hard cap. Capacity is spent on coordinate routing
rather than a larger global token.

## Bounded Acceptance

Local admission requires:

- eager and JIT forward output identity in names, shapes, masks, and finite
  values;
- nonzero finite loss gradients through every new stage and every selected
  field projection;
- exact inactive-PFT isolation and PFT permutation equivariance;
- one bound daily transition with finite state, nonnegative inventories, and
  accepted water/carbon budgets;
- parameter count at or below 60,000,000;
- no forbidden inference source path.

Only after local admission may a fresh one-epoch free-GPU smoke be proposed.
The subsequent isolated 364-day litter diagnostic retains the frozen absolute
gate: at least 99% aggregate reduction, and every initially dominant field
must fall at least 95% or finish at or below `1e-4`. Passing that gate opens a
joint one-point diagnostic, then multi-point training and 7/30-day rollout.
It does not directly open 365-day, held-out, Gate D2, or product acceptance.

## Data And Cost Policy

- use only the frozen v6 Teacher release;
- do not regenerate v6 or combine v5 data;
- do not read sealed test data for architecture or training decisions;
- do not retain a checkpoint from local or isolated memorization diagnostics;
- use free nodes for smoke work; any paid job still requires a separate
  resource, command, wall-time, and worst-case-cost approval.
