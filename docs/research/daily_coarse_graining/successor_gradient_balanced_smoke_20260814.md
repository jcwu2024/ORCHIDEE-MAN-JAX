# Gradient-Balanced 10-Update Smoke Decision, 2026-08-14

## Decision

Reject and close the current raw-gradient minimum-norm mechanism. Do not run
the conditional 100-update A/B and do not tune this mechanism with more
training updates.

This is not an Attention-architecture rejection. Both objectives used the
same `source_routed_process_attention_v2` architecture, update-1000
parameters and AdamW state, four-point-year groups, learning rates, and
validation subset. The rejected mechanism combines unit component gradients
before passing the result to the inherited AdamW optimizer.

## Frozen Protocol

- smoke contract SHA256:
  `e0980930a466c87716e5e022937817014f370060ba0fd105e102f85e012547af`;
- accepted zero-update feasibility contract SHA256:
  `cf173aee52cfff64d1462e5437e95d39666645ce34552cc0c9941577a9a5f400`;
- control: accepted staged-v3 objective;
- candidate: per-update minimum-norm convex combination of unit gradients for
  one-day anchor, NPP, AGB tendency, and BGB tendency;
- training: optimizer updates 1001-1010, with 32 one-day anchors and four
  30-day windows from four distinct point-years per update;
- validation: first and last ordered prepared artifact in each temporal,
  spatial, and joint slice, six artifacts and 12 windows total;
- no checkpoint, recovery write, Teacher regeneration, historical-v5 data,
  paid node, sealed test, or architecture change.

## Result

The corrected free-`gln01` run at commit `e22a235` completed all 10 updates
for both objectives and all six validation artifacts. Report:

- runtime path:
  `outputs/training/successor-gradient-balanced-smoke-e22a235/report.json`;
- file SHA256:
  `d84a6a764efdbd3b4d8cb74d5ddd0965c301b1ea162671a3db1b7a465bb83834`;
- elapsed Python time: `1548.89 s`;
- peak JAX device bytes: `13,304,877,568`;
- training and validation hard failures: zero.

The candidate passes both training-geometry gates: all `10/10` directions
have a minimum component cosine above `0.05`, with observed range
`0.0701-0.9064`, and candidate gradient clipping occurs on `0/10` updates.
It nevertheless fails every long-horizon scientific gate.

| Metric | Control | Candidate | Candidate change |
|---|---:|---:|---:|
| Four-key composite | 0.822531 | 0.891558 | +8.39% |
| NPP | 1.644505 | 1.782559 | +8.39% |
| AGB | 0.028044 | 0.028470 | +1.52% |
| BGB | 0.012848 | 0.014286 | +11.19% |
| Total live biomass | 0.029662 | 0.031178 | +5.11% |
| Maintenance respiration | 0.517933 | 0.358433 | -30.80% |

The candidate improves `0/6` research-primary slice/horizon cells. Relative
regressions are `+12.47%` and `+5.16%` for temporal 7/30 day, `+28.63%` and
`+12.46%` for spatial, and `+8.42%` and `+2.37%` for joint. Maintenance
passes both guards and improves in aggregate, but does not offset systematic
NPP and live-biomass damage.

## Attribution

The minimum-norm solution frequently lies on the simplex boundary. The
one-day anchor has nonzero normalized simplex weight on all 10 updates, while
NPP has nonzero direct weight on only three, AGB on five, and BGB on eight.
Zero direct weight does not violate the raw-gradient common-descent gate
because cross-component gradient correlations can still make the directional
derivative positive. It does show that the optimizer is not preserving an
explicit contribution from every scientific component on every update.

More importantly, the feasibility gate and smoke training records diagnose
the raw combined gradient. The actual parameter displacement is produced by
AdamW using first- and second-moment state inherited from the old staged-v3
objective. The current evidence does not establish that this preconditioned,
momentum-bearing displacement remains a common descent direction, even on the
training batch. Independent 7/30-day validation establishes that the complete
mechanism does not generalize after 10 updates.

## Operational Record

The first run at commit `ac5e31f` completed updates 1-9 and failed during the
update-10 host-side simplex solve because projected-gradient convergence at
`1e-13` exceeded 20,000 iterations for a near-degenerate Gram matrix. It
wrote no report or recovery. Commit `e22a235` adds a deterministic active-set
fallback over the 15 nonempty faces of the four-component simplex and a
forced-fallback regression test. The corrected run then completed from the
same immutable update-1000 recovery. Both runs used only free `gln01`.

## Next Gate

Do not continue gradient-balanced training. Before another optimizer update,
freeze a zero-update realized-step audit that compares, on matched prepared
groups:

1. raw component-gradient cosines with the selected combined gradient;
2. component-gradient cosines with the actual proposed AdamW parameter
   displacement using inherited moments;
3. the same proposed displacement after a matched optimizer-state reset or
   another explicitly preregistered control, if included;
4. before/after component losses for a virtual step, without retaining the
   proposed parameters.

If inherited AdamW destroys common descent, redesign the optimizer handoff
before training. If the realized step remains locally improving, the next
problem is training-batch versus independent-rollout generalization, and the
audit should stop without an optimizer experiment. Neither outcome changes
the network architecture without a separate architecture discussion.
