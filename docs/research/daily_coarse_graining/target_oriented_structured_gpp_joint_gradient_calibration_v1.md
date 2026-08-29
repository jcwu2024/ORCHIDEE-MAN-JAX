# Structured GPP Joint-Gradient Calibration V1

Date: 2026-08-29

Status: frozen train-only zero-update contract; local implementation accepted;
free-`gln01` execution is next

Machine contract:
`manifests/coarse_graining/target_oriented_structured_gpp_joint_gradient_calibration_v1.json`

Contract SHA256:
`c00ebf0242744c1583c12d1475b20021e325c4816c0391fb78a928ac874868fe`

## Purpose

The direct warm start established that the structured specialist can learn
daily GPP. This calibration now measures whether recursive 7- and 30-day
science losses remain finite and connected to all `12,879,363` specialist
parameters at the retained update-250 endpoint. It resolves the two horizon
coefficients and the combined-gradient clip before any joint optimizer exists.

The coordinate-aligned Attention parent, distribution-aware maintenance
adapter, and warm-start specialist are all immutable during this task. No
optimizer is created, no parameter update is executed, and no checkpoint is
written.

## Data And Gradient Sources

The calibration reuses train/train artifacts `352-357`, `359`, and `360`.
They were excluded from the 250 warm-start updates and the eight-artifact
endpoint panel. Every artifact supplies three exact effective-batch gradients:

- direct GPP over all 208 existing Teacher-v6 day starts;
- component-resolved 7-day science over eight recursive windows;
- component-resolved 30-day science over four recursive windows.

The short-horizon flux basis is GPP, maintenance respiration, growth
respiration, and heterotrophic respiration. The independent stock basis is
AGB, BGB, and nonliving ecosystem carbon. Derived autotrophic respiration,
NPP, total live biomass, and ecosystem carbon are not duplicated in the
optimization weight.

## Coefficient Policy

Direct GPP is the fixed reference with coefficient `1.0`. The target gradient
norm ratios are `0.25` for the 7-day source and `0.25` for the 30-day source,
so the declared long-horizon gradient budget totals `0.5` relative to the
direct anchor. For each source and artifact, the coefficient candidate is

```text
target ratio * direct-GPP gradient norm / horizon gradient norm.
```

The frozen coefficient is the per-source median across all eight artifacts.
It must lie in the broad numerical interval `[1e-4, 100]`. The same values
will be used for every later joint update and cannot be recalibrated from
validation. The p90 of the eight resulting combined-gradient norms freezes
the proposed joint clip threshold.

Pairwise source cosines and source-to-combined cosines are reported but not
thresholded. Strong disagreement is a mechanism result that may require a
multi-objective update rule; it must not be hidden by an arbitrary post-hoc
angle gate.

## Acceptance

All three source gradients and the calibrated combined gradient must be
finite, nonzero, cover all specialist elements, and reach every specialist
leaf on every artifact. All objectives must be finite and positive. One-day,
7-day, and 30-day physical hard failures must be zero. The coefficients and
combined p90 clip must be finite and inside their frozen bounds, every parent
parameter object must remain unchanged, and the warm-start checkpoint hash
must remain identical.

A pass opens only a separately frozen discarded joint-update preflight using
the calibrated coefficients. Paid joint training, validation, 365-day
rollout, Gate D2, sealed test, and product acceptance remain closed.

## Implementation

Contract loading and coefficient aggregation are in
`daily_operator/target_oriented_structured_gpp_joint_calibration.py`.
`scripts/hpc/calibrate_target_oriented_structured_gpp_joint_gradients.py`
loads the accepted endpoint, differentiates only the specialist, and writes a
single JSON report. The free-node boundary is
`scripts/hpc/run_target_oriented_structured_gpp_joint_gradient_calibration.sh`.
Focused and related local tests pass `17` tests; Ruff, Python compilation, CLI
import, Bash syntax, and diff checks pass.
