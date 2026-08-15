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
