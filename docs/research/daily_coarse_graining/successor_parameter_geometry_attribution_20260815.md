# Parameter-Geometry Attribution, 2026-08-15

## Question

At the cumulative-microstep stopped state, the recalibrated raw Euclidean
direction decreases all 40 objectives infinitesimally but a norm-`0.0005`
finite step regresses BGB tendency by up to `28.7222%`. This audit asks whether
one existing architecture block dominates that first-order/finite-step gap.

It does not change the network. It diagnoses the parameter coordinates in
which the accepted Attention architecture is being updated.

## Owned Blocks

The complete 23-key parameter tree is partitioned once into seven disjoint
blocks:

1. input projections;
2. input and source embeddings;
3. forcing interaction blocks;
4. process queries;
5. process interaction/attention blocks;
6. output embeddings;
7. the two-stage output decoder.

The audit reports parameter count, parameter norm, direction norm, squared
direction fraction, norm-`0.0005` delta, and delta-to-parameter norm for every
block. Coverage must be exact and all direction fractions must sum to one.

## Projection Test

Reconstruct the three accepted displacements from the update-1000 parent:
`0.002`, `0.002`, and the one safe `0.0005` microstep. Reproduce the following
failed full direction against the accepted cumulative-microstep report.

Then form exactly seven leave-one-block-out projections. Each projection is
renormalized to global norm `0.0005`. First run all 40 directional derivatives;
if any is nonpositive, record the projection and skip its finite step. Only a
projection that preserves all-40 first-order descent is evaluated on the 40
finite-step losses.

A block is attributed only when excluding it preserves first-order descent
and makes the finite step nonincreasing across all 40 objectives within
`1e-6`. This gate identifies a block suitable for a later preconditioner
audit; it does not authorize freezing that block in training.

## Boundaries

- one full reference plus seven leave-one-block-out candidates maximum;
- one frozen norm, with no smaller-step search;
- same ten train/train groups and four objectives;
- free `gln01` only;
- no optimizer update, validation read, checkpoint, recovery, or retained
  candidate parameters;
- no v6 regeneration, v5 use, sealed test, architecture change, or promotion.

The machine contract is
`manifests/coarse_graining/successor_parameter_geometry_attribution_v1.json`,
SHA256 `fa04601177a9a51b847d27a9832996f2751f2ba6a3a9c6b529caafc4e7c87153`.

## Local Admission

The parameter-block registry covers all 23 top-level keys exactly. Geometry
reporting and exclusion preserve the pytree shape, and the attribution
classifier distinguishes a safe owned projection from an unattributed
failure. The independent runner binds the accepted stopped-state report and
skips finite-step evaluation for projections that fail the all-40 first-order
gate. The complete related unit file passes `77` tests; Ruff, Python
compilation, Bash syntax, canonical contract hash, and `git diff --check`
pass.

## Result

The free-`gln01` audit completed at commit `c51dd87`. The report is
`runtime/outputs/training/successor-parameter-geometry-c51dd87/report.json`,
SHA256 `a7a170c79ef3e9cecb062adee1f0643c3d94b075216f5b4a7bd02ad47ac1a003`.
The three accepted displacements, stopped-state support, directional
derivative, and failed finite step reproduce to errors below `3.4e-15`.

The full direction is highly concentrated:

- output decoder: `96.1526%` of squared direction norm and `0.000490287` of
  the norm-`0.0005` step;
- input projection: `2.65136%`;
- process interaction: `0.704095%`;
- output embedding: `0.455063%`;
- input embedding: `0.0369012%`;
- forcing interaction and process query: effectively zero.

Excluding the output decoder makes the direction cease to be common; the
minimum relative directional derivative becomes `-0.285257`. Excluding any
of the other six blocks preserves all-40 first-order descent, but none makes
the finite step safe. Their maximum regressions remain between `28.7220%` and
`28.7225%`, essentially the full direction's `28.7222%`.

The formal classification is `parameter_geometry_not_attributed`: no
top-level leave-one-block-out projection passes the finite-step gate. Hard and
nonfinite counts are zero. Runtime is `2265.52` seconds, peak JAX device bytes
are `13,826,391,552`, and no validation values, optimizer updates,
checkpoints, recovery, or candidate parameters are produced.

## Decision After Execution

The finite-step failure is located inside the decoder-dominated part of the
direction, but the decoder as a whole is also essential for common descent.
Do not freeze the full decoder, alter Attention, or continue top-level block
searches. The next bounded attribution should split only the decoder into
`decoder_first.kernel`, `decoder_first.bias`, `decoder_second.kernel`, and
`decoder_second.bias`, report their relative update geometry, and apply the
same first-order-before-finite-step rule. If no decoder leaf exclusion is
safe, parameter-space direct steps should close in favor of an output-space
or function-space trust region.
