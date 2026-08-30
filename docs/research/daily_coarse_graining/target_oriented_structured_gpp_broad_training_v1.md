# Structured-GPP Broad Training Stage 1 v1

Date frozen: 2026-08-30

Machine contract:
`manifests/coarse_graining/target_oriented_structured_gpp_broad_training_v1.json`

Canonical contract SHA256:
`c54233c118e206200d1e8e5ec827355a469228c5a5e383786a720dd4af5340e1`

## Purpose

Run the first meaningful broad-data test of the unchanged coordinate-aligned
Attention plus structured-GPP mechanism. This is one model, not an Attention
versus control comparison. It tests whether replacing 1.2% shallow repeated
exposure with globally balanced unseen days repairs GPP generalization.

## Training Boundary

Load the rejected-for-promotion but train-capable step-498 specialist and reset
AdamW to fresh zero moments. Train only its 12,879,363 coordinates; keep the
93,504,164 parent coordinates exactly fixed. Execute schedule updates 1-1,000:
208,000 unique train/train days, 52 distinct landpoints per update, four days
per point-year, and no overlap with old exposure or monitor point-years.

Use exact direct-GPP sufficient statistics, constant learning rate `4e-6`, and
global norm clip `20.98786557796449`, the accepted broad-preflight p95. The
training path carries no typed sidecar, retained-tail target, next-state target,
or 7/30-day payload.

## Monitor And Stopping

Evaluate the fixed 128-point-year, 2,048-day train-only monitor before training
and at updates 250, 500, and 1,000. If update 500 has not improved aggregate
monitor loss from baseline, stop before update 1,000. Select the completed
checkpoint with minimum monitor loss.

Stage acceptance requires at least 5% aggregate monitor-loss reduction, at
least 96 of 128 point-years improved, and no increase in absolute aggregate
normalized bias. The monitor is for GPP model selection only; it is not the
consumed 96-window science panel and no validation or sealed-test value is
read. Every resulting checkpoint remains non-promotable until a fresh
independent 7/30-day science screen is frozen and passed.

## Recovery, Physical Gate, And Cost

Write hash-bound recovery every 25 updates. Before each checkpoint, run the
existing one-day physical gate on prepared artifact 251; budget, nonfinite,
unexpected-mask, discrete, and negative-stock hard counts must remain zero.

The proposed paid envelope is one `gnall` node, eight CPU cores, one V100,
32 GiB host memory, and five hours. At additive published rates, CPU is
`8 * 5 * 0.07 = CNY 2.80` and GPU is `1 * 5 * 2.20 = CNY 11.00`, for a
worst-case `CNY 13.80`. This document does not authorize submission.
