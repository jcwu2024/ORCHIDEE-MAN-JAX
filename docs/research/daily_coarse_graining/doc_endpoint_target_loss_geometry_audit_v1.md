# DOC Endpoint Target/Loss Geometry Audit v1

Date: 2026-08-20.

Status: frozen zero-training protocol. Implementation and one free CPU
execution are authorized; no neural training or architecture promotion is
authorized by this document.

## Question

V4 proved that coordinate-aligned DOC transfer tokens improve DOC incoming,
but its joint signed transition decoder severely regressed DOC outgoing. This
audit asks whether the physical one-scalar identity is poorly conditioned
under the two separately normalized training targets.

## Inputs

Read the accepted v6 train/train point `001.0-071.0`, year 1961, with all 364
valid days. Reuse the accepted output normalization, loss scales, mechanistic
schema, and DOC day-start owner mapping. Do not regenerate Teacher data, join
v5, read sealed test data, initialize a neural network, or write a checkpoint.

## Representations

The audit compares three target coordinates without fitting parameters:

1. the v4 signed change represented through a positive softplus day-end DOC;
2. the v3 separate nonnegative incoming and unit-interval outgoing transforms;
3. an explicit incoming/unchanged/outgoing status plus a branch-native
   loss-scaled magnitude.

For each representation, report finite inverse-target coverage, exact finite
reconstruction, raw-target magnitude, and the local physical-output Jacobian
after division by the frozen field loss scale. Also report status imbalance,
zero-end and unit-outgoing boundaries, and each endpoint's zero-predictor loss
contribution.

## Frozen Diagnostics

The meaningful physical magnitude is `1e-12`, the normalized-Jacobian floor
is `1e-6`, and finite inverse reconstruction must be within `1e-10` absolute
error. The current joint coordinate is classified as poorly conditioned if
any admitted active target lacks a finite inverse, any admitted active target
has a normalized local Jacobian at or below the floor, or the median
incoming/outgoing normalized-Jacobian ratio lies outside `[0.1, 10]`.

This audit may reject the joint coordinate and shortlist an alternative. It
cannot promote a new network. The user must review and freeze any revised DOC
endpoint mechanism before implementation or GPU training.

The machine contract is
`manifests/coarse_graining/doc_endpoint_target_loss_geometry_audit_v1.json`,
SHA256 `4f5003f3750928b3aa9119b56045d22337bd253ac758c85c690523a935b7d6e7`.
