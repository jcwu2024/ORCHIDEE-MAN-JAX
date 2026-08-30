# Structured-GPP Broad-Data Preflight v1

Date frozen: 2026-08-30

Machine contract:
`manifests/coarse_graining/target_oriented_structured_gpp_broad_preflight_v1.json`

Canonical contract SHA256:
`f0e50031715b26898d581d3d99a17ae3c01a6a27e693f2e0130fb9e6c63096a3`

## Purpose

The step-498 model fit its repeated train-only artifacts but failed every cell
of the independent short-horizon science screen. The exact audit showed that
only 103,399 unique direct-GPP days, 1.203494% of train/train, had been exposed.
This contract tests the resulting broad-data hypothesis without changing the
coordinate-aligned Attention parent or structured-GPP mechanism and without
authorizing paid training.

## Target-Blind Training Schedule

Build at most 5,000 updates from existing v6 train/train parent shards. Every
update contains exactly 52 distinct landpoints, one point-year per landpoint,
and four unique days per point-year, for 208 days. The full proposed stage has
1,040,000 unique days. Exclude row zero, all 103,399 prior-exposure day keys,
and all point-years in the accepted 128-point-year monitor.

Landpoints, point-years, and eligible rows have separate SHA256 rank domains
under seed `20260830`. Updates rotate through the ranked landpoint list;
point-years rotate within each landpoint; rows advance without replacement in
their ranked order. The schedule reads no target values. A free-CPU audit must
rebuild it, freeze its canonical digest, and prove the exact counts and
disjointness before GPU preflight.

## Model And Input Boundary

Start from the selected step-498 specialist parameters, SHA256
`77cb405f...95f2b`. Trainable scope remains the 12,879,363 structured-GPP
specialist coordinates; all 93,504,164 parent coordinates remain fixed. Use
`DailyProcessTrainingAdapter.assemble_direct_gpp`, which supplies canonical
daily inputs plus only `daily.fast_gpp`. It reads no typed sidecar and carries
no retained-tail, other process target, next-state, or 7/30-day payload.

## Free-GPU Preflight

On 16 broad schedule batches, compute the exact 208-day sufficient statistics
and raw specialist gradients. All loss statistics and all specialist gradient
elements must be finite, with every specialist leaf reached. Derive the future
clip threshold as the p95 raw norm across these 16 batches.

For four of those batches, independently initialize fresh AdamW moments at the
same step-498 parameters and propose one `4e-6` update. These are not chained
training updates and write no optimizer checkpoint. Every proposal must be
finite, nonzero, reduce its same-batch direct-GPP loss, preserve the retained
parent exactly, and pass the one-day physical hard gate.

## Proposed Paid Stage

A passing preflight may freeze, but does not itself authorize, a first paid
stage of 1,000 updates and 208,000 unique days. The fixed train-only monitor is
evaluated at updates 0, 250, 500, and 1,000. Only after that stage demonstrates
broad-data monitor improvement may training extend toward the 5,000-update,
1,040,000-day maximum. Optimizer moments start fresh because the old moments
were accumulated on the repeated artifact distribution.

No outcome here opens the consumed 96-window panel for tuning, 365-day rollout,
Gate D2, sealed test, or product acceptance.
