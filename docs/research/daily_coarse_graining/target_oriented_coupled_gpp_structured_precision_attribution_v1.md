# Coupled GPP Structured Precision Attribution V1

Date: 2026-09-05

Status: frozen before zero-training implementation and execution

## Question

The accepted Day-5 cross assigns the entire `+2.1025634193` GPP jump to the
existing `structured_gpp` parameter family. Its float32 endpoint interpolation
stays near `2.7673` through fraction `1/2` and reaches `4.8699` at `3/4`, even
though the implemented factor path has no parameter-dependent hard branch.

This audit asks two bounded questions:

1. Is the sampled jump primarily caused by float32 endpoint quantization, or
   does an equally sharp response remain when the same endpoint weights are
   interpolated in float64?
2. Which existing top-level `structured_gpp` parameter subtree is necessary or
   sufficient for the endpoint jump?

It does not redesign the specialist or test a new architecture.

## Frozen Parent And Inputs

The sole scientific parent is the accepted crossed-attribution report with
SHA256
`c014ac09558f30671bc511cfa9a63690f46f42434da428a414664581987780c5`.
Reuse only artifact update `550`, runtime year `1976`, seven-day window index
`3`, and Day-5 index `4`. Reconstruct the accepted update-2 control parameters
and the historical rejected update-3 `1/64` treatment exactly as in the parent.

Run the control seven-day trajectory only to obtain its Day-5 start state. The
accepted cross already proves that substituting the treatment Day-5 state has
zero effect on GPP. Every diagnostic evaluation therefore uses the same
control Day-5 state, forcing, static/PFT inputs, retained inputs, defined masks,
and discrete state. No Teacher future state is used.

## Float32 Delta And ULP Inventory

Audit every float32 leaf under `target_oriented.structured_gpp`. For each leaf
and each top-level subtree, report parameter count, exact changed-coordinate
count, L2 and maximum absolute endpoint displacement, sign-crossing count, and
ULP-distance counts for exactly one ULP, exactly two ULPs, and more than two
ULPs. ULP distance uses monotonic IEEE-754 binary32 bit ordering over finite
values. All leaf totals must add exactly to the frozen specialist count
`12,879,363`.

The frozen top-level registry is:

1. `forcing_input`
2. `forcing_field_embedding`
3. `forcing_position_embedding`
4. `branch_embedding`
5. `parent_hidden_projection`
6. `local_input`
7. `pft_query_projection`
8. `forcing_blocks`
9. `potential_blocks`
10. `limitation_blocks`
11. `correction_blocks`
12. `potential_output`
13. `limitation_output`
14. `correction_output`

At every fixed interpolation fraction, also report among endpoint-changed
coordinates how many realized float32 values equal the control endpoint, equal
the treatment endpoint, or are a distinct intermediate value. These counts are
descriptive evidence; no minimum one-ULP fraction is assumed in advance.

## Two Precision Paths

Use the fixed fractions

```text
0, 1/16, 2/16, ..., 15/16, 1.
```

Evaluate two paths with all non-`structured_gpp` parameters held at control:

1. `production_float32`: interpolate the stored float32 endpoint arrays using
   the production dtype and execute the unchanged factor graph.
2. `float64_weight_interpolation`: cast only both endpoint
   `structured_gpp` trees to float64, interpolate those weights in float64, and
   execute the same graph. Existing explicit activation casts remain unchanged,
   so this is a weight-storage diagnostic rather than a proposed float64
   production model.

Report potential, limitation, correction, and routed GPP at every point. For
each path define

```text
endpoint jump = GPP(1) - GPP(0)
maximum adjacent fraction = max(abs(GPP[i+1] - GPP[i])) / abs(endpoint jump)
```

The float64 endpoint jump is comparable only when it has the same sign as the
production endpoint jump and its absolute magnitude ratio lies in
`[0.75, 1.25]`.

The frozen classification is:

- `float32_weight_quantization_dominant_for_sampled_jump` when the endpoint
  jumps are comparable, production maximum adjacent fraction is at least
  `0.75`, and float64 maximum adjacent fraction is at most `0.25`;
- `structured_gpp_conditioning_persists_with_float64_weights` when the endpoint
  jumps are comparable and both maximum adjacent fractions are at least
  `0.75`;
- `mixed_or_inconclusive_precision_effect` otherwise.

This classification concerns the sharpness of this one accepted Day-5 path.
It does not accept the model or prove global optimizer behavior.

## Top-Level Subtree Controls

At the fixed control Day-5 state, evaluate for every frozen top-level subtree:

- sufficiency: insert only that treatment subtree into the complete control
  `structured_gpp` tree;
- necessity: replace only that subtree in the complete treatment
  `structured_gpp` tree with its control value.

All other trainable families remain at control. Report the full factorization
for both evaluations. Relative to the full endpoint GPP effect, report signed
sufficiency and necessity capture

```text
sufficiency = (GPP_only_group - GPP_control) / full_effect
necessity = 1 - abs(GPP_without_group - GPP_control) / abs(full_effect).
```

A subtree is individually material when absolute signed sufficiency or absolute
necessity capture is at least `0.10`. This flag is descriptive and does not
authorize an architecture edit; interactions between subtrees remain possible.

## Engineering Acceptance

Acceptance requires all of the following:

- the parent report identity, status, classification, and engineering gate are
  exact;
- the reconstructed direction norm and endpoint parameter displacement pass
  the existing historical tolerances;
- the control and treatment factorized GPP reproduce parent cells `CC` and
  `TC` at the accepted factorization tolerance;
- the frozen subtree registry and total parameter count are exact, every
  parameter leaf is finite float32, and all ULP/membership counts conserve;
- every subtree and precision-path factor is finite and routed GPP is
  nonnegative;
- both precision paths reproduce their own endpoints exactly and contain all
  17 fixed fractions;
- the report is written before a nonzero engineering-gate exit.

## Execution Boundary And Decision Branches

This is one free-`gln01` zero-training diagnostic with a 60-minute hard timeout.
It reconstructs one historical displacement in memory and writes one report.
It creates no optimizer, changes no network, retains no parameters, writes no
checkpoint/recovery, regenerates no Teacher data, reads neither v5 nor fresh
validation nor sealed test, reconstructs no half-hour forcing, and performs no
48-step state scan.

After the report:

- a quantization-dominant result opens discussion of update realization or
  master-weight precision, not automatic architecture replacement;
- persistent float64 sharpness opens discussion of conditioning controls or a
  different GPP output parameterization;
- a mixed result uses the subtree controls to define at most one narrower
  zero-training follow-up.

Every branch pauses for user discussion before architecture or training work.
