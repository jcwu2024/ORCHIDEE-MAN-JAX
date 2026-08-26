# Target-Oriented Maintenance Head Conditioning Audit v1

Date: 2026-08-27

Status: frozen before zero-training execution

Machine contract:
`manifests/coarse_graining/target_oriented_maintenance_head_conditioning_audit_v1.json`

Contract SHA256:
`b354c4e8309e878899ce61c83f851b08279977a9984eccf0b9a60bc7089fc369`

## Question

After the accepted snow-thermal causal cuts, the remaining extreme 30-day
maintenance gradient lies almost entirely in the direct
`daily.fast_maintenance_respiration` final head. Is that existing 769-parameter
head locally capable of realizing a useful 7/30-day maintenance correction,
or do the required groups and derived carbon objectives conflict even before
the shared Attention body is changed?

This is not a new candidate or a candidate/control comparison. The matched
reference is the unchanged update-1000 model with a zero step.

## Frozen Direction

Use the eight established train/train updates `251, 358, 465, 572, 679, 786,
893, 1000`. For every group, aggregate its eight 7-day windows and four
30-day windows separately. Differentiate each aggregate maintenance field
score only with respect to the existing final head kernel and bias:

```text
8 groups x 2 horizons = 16 gradients x 769 head parameters
```

All other `83,754,404` parameters remain at the exact parent values and are
not differentiated. Normalize each head gradient, then solve the same
minimum-norm simplex common-descent problem used by the earlier full-model
geometry work. The realized direction must have cosine at least `0.01` with
all 16 gradients.

This isolation is specific to the current target-oriented model, whose GPP
and maintenance final heads are independent. It does not repeat the rejected
post-training projection of the older shared-decoder model.

## Frozen Step Calibration

An arbitrary full-model parameter norm is not reused for the much smaller
head. Let `s_i` be each parent maintenance score and `d_i` its positive
directional derivative along the common head direction. Define

```text
base norm = median_i(s_i / d_i)
```

and evaluate predicted median relative-decrease fractions
`0.001, 0.003, 0.01, 0.03, 0.1`. Each candidate norm is its fraction times
the base norm. A candidate above head norm `0.01` is reported but not
evaluated. The grid is not extended after seeing results.

## Exact Protection Gate

For every admissible candidate, rerun all eight complete group evaluations.
A pass requires:

- maintenance improves in every group within `1e-6`, with at least `0.1%`
  median improvement at both horizons;
- GPP has nonpositive median change and no group regresses by more than `1%`;
  its own final-head parameters remain exactly unchanged, while later-day GPP
  may still respond physically to maintenance-driven state changes;
- growth respiration and every derived flux have nonpositive median change
  and no group regresses by more than `1%`;
- one-day state/process median does not regress and no group regresses by
  more than `0.1%`;
- every stock median does not regress and no group regresses by more than
  `1%`;
- parent-passing absolute keys remain passing, parent-failing stock RMSE does
  not increase by more than `1%`, and all physical/nonfinite counts are zero.

Select the largest passing predicted fraction. Every proposal is discarded.

## Decision Boundary

A missing common head direction means the maintenance objective or sampling
unit must be redesigned before training. A common direction whose finite step
cannot protect the other targets points to output-head parameterization and
requires architecture discussion before an edit. A protected exact step
opens design of bounded head-conditioned training with the architecture still
unchanged; it does not itself authorize training.

The audit uses only accepted v6 train/train prepared data. It does not
regenerate Teacher data, read v5, validation values, or sealed test, invoke an
optimizer, retain parameters, or write a checkpoint. It is restricted to free
`gln01`. Input masks, source routing, architecture, 365-day rollout, Gate D2,
and product work remain closed.
