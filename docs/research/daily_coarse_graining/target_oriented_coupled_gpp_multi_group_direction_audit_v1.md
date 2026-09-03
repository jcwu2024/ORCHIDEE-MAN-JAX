# Coupled GPP Multi-Group Direction Audit V1

## Question

At persistent update 2, shrinking the existing four-source direction lets the
scheduled group and four disjoint train-only GPP guard groups improve together
at scale `1/4`, but the fixed monitor still worsens by `0.5107%`. The old
direction gives direct GPP only one scheduled-group mean constraint.

This zero-training audit asks whether direction construction itself improves
when direct GPP from five distinct train-only groups enters as five separate
constraints. It does not compare two neural architectures and does not treat
the predecessor as a model-quality control.

The frozen machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_multi_group_direction_audit_v1.json`,
canonical SHA256
`1c103e79663d6da22e7fcc1275432d4caec08bbe4390c695dd17ee1596b82d19`.

## Eight Direction Constraints

The runner reconstructs the accepted update-7 state and persistent update 1,
then evaluates one direction at the same pre-update-2 parameters as the
completed transfer audit. Its eight gradients are:

1. scheduled-group direct GPP;
2. guard-group 1 direct GPP;
3. guard-group 2 direct GPP;
4. guard-group 3 direct GPP;
5. guard-group 4 direct GPP;
6. scheduled-group reachable one-day state;
7. scheduled-group 7-day component-resolved science; and
8. scheduled-group 30-day component-resolved science.

Each entry is a mean gradient over eight accepted v6 train/train artifacts.
The five direct-GPP gradients therefore cover 40 distinct artifacts. Each
mean gradient is unit-normalized, and the existing minimum-norm simplex solver
constructs one common direction from the resulting `8 x 8` Gram matrix. A
first-order direction exists only if every constraint cosine is at least
`1e-4`.

No new matched control is run. The completed single-group transfer report at
SHA256 `df2593b...966c94` is the same-state reference evidence.

## Finite-Step Gate

If a common first-order direction exists, it is realized without AdamW or
momentum at the same full parameter-displacement norm
`0.007758973351050004`. Scales
`1/2,1/4,1/8,1/16,1/32,1/64` are evaluated in descending order. A scale passes
only if:

1. all four scheduled-group sources pass the existing finite-step rule;
2. scheduled physical hard gates pass;
3. every guard-group direct-GPP mean is nonincreasing within `1e-6`; and
4. the combined 32-artifact guard mean strictly decreases.

The largest passing scale is selected using train-only values. The fixed
monitor does not enter the direction or scale selection and receives only the
one selected proposal. If no common direction or train-only step exists, the
monitor is not read.

## Boundary And Decision

The audit retains no optimizer update, proposal, recovery, checkpoint, or
trainable parameter. It uses no fresh validation, sealed test, historical v5,
or regenerated Teacher data and performs no 48-record forcing reconstruction
or 48-step state scan. One free `gln01` execution is allowed with a 120-minute
timeout; paid execution remains closed.

A monitor-transferring direction opens a separately frozen three-update
multi-group GPP-protected backtracking smoke. A monitor failure requires an
artifact-aware or distributionally robust GPP direction before training. A
missing common direction closes group-mean common descent and returns to the
objective sampling unit. A missing finite step requires changing direction
conditioning or realization.

## Implementation

- Contract:
  `manifests/coarse_graining/target_oriented_coupled_gpp_multi_group_direction_audit_v1.json`
- Contract logic:
  `research/daily_coarse_graining/daily_operator/target_oriented_coupled_gpp_multi_group_direction.py`
- Runner:
  `scripts/hpc/audit_target_oriented_coupled_gpp_multi_group_direction.py`
- Free-node launcher:
  `scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh`
