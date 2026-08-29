# Structured GPP Batch-Four Joint Preflight V1

Date: 2026-08-29

Status: completed and accepted; bounded batch-four training design is next

Machine contract:
`manifests/coarse_graining/target_oriented_structured_gpp_joint_batch4_preflight_v1.json`

Contract SHA256:
`06d8c3f43beeac9c0c64dc8fd679ff57d09dc7b34df64a233df9f9d0dcc19169`

## Question

The completed heterogeneity audit found batch size 4 to be the minimum size
whose calibrated raw gradient is a common descent direction in every tested
group. This preflight asks the remaining finite-step question: does the exact
inherited AdamW transformation at step 251 preserve that aggregate direction
and actually reduce all three aggregate losses?

## Frozen Proposals

The two groups are `512, 548, 584, 620` and `656, 692, 728, 763`. Each group
aggregates four train/train artifacts, totaling 832 direct Teacher-GPP days,
32 seven-day windows, and 16 thirty-day windows per proposed update.

Both proposals independently load the same accepted warm-start specialist and
AdamW state at step 250. Each computes the arithmetic mean of the four direct,
7-day, and 30-day gradients, combines them with the frozen coefficients
`1.0`, `5.065197169020285`, and `4.594433600967992`, applies the frozen
`46.802861793665286` clip and learning rate `4e-6`, and proposes step 251.
The second proposal cannot inherit the first proposal's parameters or
optimizer state.

## Acceptance

For each group, the negative realized parameter step must have a strictly
positive cosine with all three aggregate source gradients. The group's mean
direct-GPP, 7-day, and 30-day losses must each strictly decrease. Individual
artifact changes are reported for diagnosis but are not individual gates;
the effective update unit established by the preceding audit is the group of
four artifacts.

Every aggregate gradient must be finite, nonzero, cover all `12,879,363`
specialist values, and reach all specialist leaves. Every pre/post artifact
must have zero one-day, 7-day, and 30-day physical hard failures. Both
proposals must remain finite and change every specialist component, while the
retained parent and warm-start checkpoint remain unchanged.

## Boundary

The two proposals and optimizer states are discarded. No checkpoint or
recovery is written. Teacher data is not regenerated; historical v5,
validation, 365-day rollout, Gate D2, sealed test, and product acceptance
remain outside this gate. Final inference remains a true daily transition
without reconstructing 48 forcing records or performing a subdaily scan.

A complete pass opens only a separately frozen batch-four bounded-training
contract and explicit paid-resource request. Failure closes paid continuation
under this accumulated AdamW realization and requires an optimizer or
objective review.

The contract parser, arithmetic aggregation, two-proposal negative design,
runner import, and free-node shell boundary pass 30 focused and related tests,
Ruff, Python compilation, Bash syntax, and diff checks locally.
