# Structured GPP Joint-Gradient Heterogeneity Audit V1

Date: 2026-08-29

Status: frozen before free-GPU zero-update audit; local implementation
accepted

Machine contract:
`manifests/coarse_graining/target_oriented_structured_gpp_joint_gradient_heterogeneity_audit_v1.json`

Contract SHA256:
`0905b11467e982cdff72c624f2c61c2849e4e2dd90cbee860b9996c9588d945b`

## Question

The discarded single-artifact proposal failed because artifact 512 has a
strong 7-day versus 30-day gradient conflict that was absent from the eight
coefficient-calibration artifacts. This audit asks whether that conflict is
an isolated batch effect and, specifically, whether accumulating more
artifacts per optimizer update produces a stable three-source direction.

## Frozen Data

The audit uses train/train artifacts `512, 548, 584, 620, 656, 692, 728, 763`.
They are equally spaced across the planned joint-training interval and exclude
the calibration and endpoint-evaluation artifacts. Each contributes 208
direct Teacher-GPP days, eight recursive 7-day windows, and four recursive
30-day windows.

No new Teacher data is generated. Historical v5, validation, and sealed-test
values remain excluded.

## Analysis

Every artifact supplies the same three specialist gradients used by the
completed calibration and preflight. The coefficients remain fixed at `1.0`,
`5.065197169020285`, and `4.594433600967992`; they are not recalibrated.

The frozen artifact order is partitioned into contiguous, nonoverlapping
groups at effective batch sizes 1, 2, 4, and 8. Within each group, each source
gradient is the arithmetic mean across artifacts, and the combined gradient
uses the frozen coefficients. A group has common descent only when direct
GPP, 7-day science, and 30-day science each have strictly positive cosine with
that combined gradient.

The classification is the smallest tested batch size for which every group
at that size passes. Individual and grouped pairwise cosines, norms, and
source-to-combined cosines are all reported. The classification is not an
execution gate: a complete finite audit can validly conclude that no tested
size is consistent.

## Acceptance And Boundary

All 24 source gradients and every grouped gradient must be finite, nonzero,
cover all `12,879,363` specialist values, and reach all specialist leaves.
All objectives must be finite and positive; all physical hard-failure counts
must be zero; the parent and warm-start checkpoint must remain unchanged.

No optimizer is created, no update is executed, and no checkpoint is written.
Final inference remains a true daily transition without 48 forcing records or
a subdaily state scan.

If a consistent size exists, the next gate is one discarded accumulated
AdamW proposal at the minimum size. If none exists, paid training remains
closed while the objective or multi-objective update rule is redesigned.

Contract, grouping, classification, host-gradient aggregation, runner
negative-design, Ruff, Python compilation, CLI import, and Bash syntax checks
pass locally.
