# Target-Oriented State-Anchor Amendment

Status: **frozen after the first zero-update full-step diagnostic; paid training
remains closed**.

Date: 2026-08-21.

## Trigger

The first eight-batch real-v6 full-objective preflight completed without a
state, budget, inventory, or learned-output failure, but all eight complete
gradients were nonfinite. Component and one-day audits localized the failure
before the Attention backbone:

- one exact/diagnostic field, `roughheight_pft`, is structurally undefined and
  must not be counted as a learned-output failure;
- all observed state-mask mismatches belong to the already declared dynamic
  `diffuco_previous_step_state.rveget` status;
- one `slowproc_stomate_previous_step_state.DOC` coordinate has Teacher values
  around `1e-14` to `1e-11`, empirical scale
  `7.960966184890271e-12`, and an initialization prediction around `0.01`;
- that coordinate is counted in both the carbon group and the complete
  `other` view, producing state loss around `1e14` and state-head gradients up
  to about `6e14` before the masked-nonfinite backward failure.

This is a loss-conditioning defect, not evidence that the 83.8-million-
parameter Attention architecture lacks capacity.

## Frozen Repair

The machine amendment is
`manifests/coarse_graining/target_oriented_state_anchor_amendment_v1.json`,
SHA256 `840089cfdf288c77453b0e7be0f4eeb0345b9225ad950e7b70b332791a77382c`.

It changes only the one-day state anchor and preflight status accounting:

1. The seven named groups remain, but `other` is now the residual state view.
   Coordinates already represented by water, carbon, or thermal groups are
   excluded from `other`; the deliberate canopy/soil/snow emphasis inside the
   water hierarchy remains.
2. Existing Teacher-v6 train/train next-state standard deviations remain the
   scale source. Empirical scales at or below `1e-6` fall back to `1.0`, so the
   coordinate is trained in native units instead of amplifying unresolved
   variation. This is the existing semantic-input normalization convention.
   It affects 46 of 3,854 continuous-state coordinates, including 40 of 1,222
   carbon coordinates.
3. MSE plus squared signed bias is retained. Huber or another robust loss is
   deliberately not added in the same amendment, so the scale repair can be
   evaluated independently.
4. `rveget` defined status remains BCE-supervised and explicitly reported but
   does not veto an optimizer proposal. Any undeclared status mismatch remains
   a zero-tolerance hard failure.
5. Nonfinite process-output counts cover only the 161 learned outputs. Exact
   or diagnostic structural undefined values remain reported separately.

The network, output boundary, conservative update, retained exact tail,
optimizer, effective-batch selection, Teacher data, and absolute 7/30/365-day
science gates are unchanged. Existing prepared batches are reusable because
their selected rows and retained inputs do not depend on this loss policy.

## Verification And Next Gate

Focused local tests pass `25/25`; Ruff, compileall, and diff checks pass. The
next operation is a free-node real-v6 scale/count audit followed by one
zero-retained-update representative component-gradient probe. It must show
finite one-day/process, 7-day, and 30-day gradients, zero undeclared hard
failures, and a state anchor no longer dominated by one DOC coordinate before
the full eight-batch preflight is repeated.

No paid training, validation tuning, Teacher regeneration, historical-v5
access, sealed-test read, 365-day rollout, Gate D2, or product acceptance is
authorized by this amendment.
