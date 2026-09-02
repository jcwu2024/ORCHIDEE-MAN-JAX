# Coupled GPP Update-5 Scale Audit V1

## Question

The direct-recalibration smoke accepts updates 3 and 4, then rejects update 5
at the same full displacement. Update 5 nevertheless has positive cosine about
`0.495394` with every source gradient. This audit asks whether a bounded
finite realization of that already accepted first-order direction exists.

The frozen machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_update5_scale_audit_v1.json`,
canonical SHA256
`835a8ee86feb6691596552ac4047bc4f02905702d6534282c1f6be190ee69cbd`.

## Exact State Reconstruction

The runner reproduces the corrected direct-recalibration smoke in one process
without writing its intermediate report. It must reconstruct accepted updates
3 and 4 and the rejected full-scale update 5 on artifacts
`965,351,272,363,660,670,548,740`. Numeric differences from the accepted
predecessor report are recorded but are not fatal tolerances.

The post-update-4 parameters, update-5 common direction, and all evaluation
kernels remain only in memory. The predecessor result SHA256 is
`0c2d3e219f4dce16be40d15ce6a1ccbdb7c5f4f239e88e83a874e42b23242fd2`.

## Frozen Scale Grid

The full parameter displacement norm is `0.007758973351050004`, with parameter
dtype preserved. The audit evaluates scales
`1,1/2,1/4,1/8,1/16,1/32`. The reconstructed full-scale evaluation is reused;
every smaller scale receives a real eight-artifact 1/7/30-day evaluation.

The grid is frozen from the update-5 GPP derivative/full-response diagnostic:
the local quadratic estimate crosses zero at scale `0.253654` and reaches its
minimum at `0.126827`. Thus `1/4` and `1/8` directly bracket the predicted
transition and minimum, while `1/16` and `1/32` verify the small-step regime.
The diagnostic does not count as acceptance evidence.

## Gate And Retention

A scale passes only if direct GPP, reachable one-day state, 7-day science, and
30-day science are all non-worsening at relative tolerance `1e-6`, the
weighted objective decreases strictly, and every physical/nonfinite hard gate
passes. All scales are reported; no accepted scale is applied or retained.

If any bounded scale passes, the next task is a separate scale-selecting
direct-recalibration smoke. If none passes down to `1/32`, review the
parameterization and finite curvature without changing the architecture.

Paid training, fresh validation, 365-day rollout, Gate D2, sealed test, and
product acceptance remain closed.

## Local Qualification

The frozen loader, predecessor boundary, in-process reconstruction path,
bounded evaluator, CLI, and free-node launcher pass 39 focused and related
coupled-GPP/reachability tests. Ruff, formatting, Python compilation, Bash
syntax, and whitespace checks pass. One free `gln01` execution is the active
gate.
