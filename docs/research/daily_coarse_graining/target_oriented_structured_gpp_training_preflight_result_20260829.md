# Structured GPP Training Preflight Result

Date: 2026-08-29

Status: accepted; opens the bounded direct-GPP warm-start contract

Execution commit: `d852e698171d380ed6ab841762f8140202e8ced2`

Result SHA256:
`d531f8dcc6d684c1cba6c471d28e640a82114d0967849bea5bb5d7f47a2cbccc`

## Execution

The free-`gln01` GPU-1 preflight completed in `1059.82 s` on a V100. It used
the existing Teacher-v6 prepared artifacts and update-250 distribution-aware
parent checkpoint. It regenerated no Teacher data, read no validation or test
values, retained no optimizer proposal, and wrote no checkpoint or recovery.

The 7- and 30-day recursive paths completed forward-only compatibility checks.
Their science objectives were finite (`0.25311244` and `0.11525403`), with no
budget, nonfinite, discrete, negative-stock, or unexpected-mask hard failure.
No horizon gradient was compiled.

## Direct GPP Calibration

All eight train-only direct-GPP gradients were finite and covered all
`12,879,363` specialist elements and all 129 leaves. Their norms ranged from
`6.40936` to `19.08827`, with median `12.26476`. The accepted p90 warm-stage
clip threshold is `17.256861560753983`.

A fresh AdamW proposal at learning rate `4e-6` used artifact 252. Its raw
gradient norm was `18.63911`, clip scale was `0.925842`, and every specialist
component changed. The same-artifact direct-GPP loss decreased from
`4.04517568` to `4.04308744`. All `93,504,164` retained parent parameters were
the same array objects before and after the proposal.

## Decision

The specialist-only forward, loss, gradient, clipping, optimizer, ownership,
and daily physical boundaries are admitted. The next action is a separately
frozen 250-update direct-GPP warm start. That stage is not scientific
promotion: after it completes, 7/30-day gradients must be calibrated from
train-only artifacts at the retained endpoint before any joint training.
