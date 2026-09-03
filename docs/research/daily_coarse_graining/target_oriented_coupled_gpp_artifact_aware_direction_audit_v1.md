# Coupled GPP Artifact-Aware Direction Audit V1

## Question

The completed five-group audit finds a strong common direction for five
train-only direct-GPP group means plus scheduled state and 7/30-day science.
Every train-only group mean improves at every tested scale, yet the independent
monitor still splits `4/8` and worsens by `0.4731%`. Group means therefore hide
the within-group heterogeneity that matters for transfer.

This zero-training audit asks whether exposing each train-only point-year batch
to direction construction resolves that failure. It does not compare neural
architectures and does not treat the predecessor as a model-quality control.
The Attention architecture and output boundary remain unchanged.

The frozen machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_artifact_aware_direction_audit_v1.json`,
canonical SHA256
`0606e7e22822a7d01d704929d72056f7a20e95b67299a86d22ea49053f37fc52`.

## Forty-Three Constraints

The runner reconstructs the accepted update-7 state and persistent update 1,
then evaluates the same logical pre-update-2 state as the preceding audits.
Its direction constraints are:

1. 40 separate direct-GPP gradients, one for each artifact in scheduled update
   2 and train-only guard updates 11 through 14; and
2. the scheduled-group mean gradients for reachable one-day state, 7-day
   component-resolved science, and 30-day component-resolved science.

Each direct-GPP artifact is already a point-year batch containing 208 accepted
Teacher days. This is not a 40-day fit. All 43 gradients are independently
unit-normalized, and the existing large active-set minimum-norm simplex solver
uses their `43 x 43` Gram matrix. A first-order direction exists only if every
constraint cosine is at least `1e-4`.

The implementation flattens and stacks unit gradients only for the Gram and
one matrix-vector combination. It then releases those buffers before rollout
evaluation. No `43 x 43` enumeration is used; the existing iterative
active-set solver handles more than 20 constraints.

## Finite-Step Selection

If a common direction exists, it is realized without AdamW or momentum at
full displacement norm `0.007758973351050004`. The descending scale ladder is
`1/2,1/4,1/8,1/16,1/32,1/64`. A scale passes only if:

1. all four scheduled sources pass the existing finite-step rule;
2. scheduled physical hard gates pass;
3. every one of the 40 train-only artifact direct-GPP losses is nonincreasing
   within relative tolerance `1e-6`; and
4. their combined mean strictly decreases.

The first passing scale is necessarily the largest passing scale, so the
runner stops the ladder immediately when one passes. This removes the five
smaller-scale evaluations that were unnecessary in the preceding audit. The
fixed monitor is excluded from direction and scale selection and receives
only the selected proposal. If no common direction or train-only step exists,
the monitor is not read.

## Decision Boundary

A monitor-transferring direction opens a separately frozen three-update
artifact-aware backtracking smoke. A monitor failure closes fixed-panel
direction-only correction and moves to broader randomized sampling or a CVaR
training objective. A missing common direction rejects strict samplewise
common descent and moves directly to a quantile-risk objective. A missing
finite step requires changing direction conditioning or realization.

The audit retains no optimizer update, proposal, recovery, checkpoint, or
trainable parameter. It uses no fresh validation, sealed test, historical v5,
or regenerated Teacher data and performs no 48-record forcing reconstruction
or 48-step state scan. One free `gln01` execution is allowed with a 120-minute
timeout. Paid execution remains closed.

## Implementation

- Contract:
  `manifests/coarse_graining/target_oriented_coupled_gpp_artifact_aware_direction_audit_v1.json`
- Contract logic:
  `research/daily_coarse_graining/daily_operator/target_oriented_coupled_gpp_artifact_aware_direction.py`
- Runner:
  `scripts/hpc/audit_target_oriented_coupled_gpp_artifact_aware_direction.py`
- Free-node launcher:
  `scripts/hpc/run_target_oriented_coupled_gpp_artifact_aware_direction.sh`
