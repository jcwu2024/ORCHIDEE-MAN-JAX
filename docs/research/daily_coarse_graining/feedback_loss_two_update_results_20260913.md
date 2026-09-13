# Feedback loss: completed two-update technical admission

Source497fe80, free gln01 GPU0, PID65767/Python66443, completed with exit0.
Both processes are absent. Report `outputs/feedback-loss-updates-v1.json`,
remote prefix `runtime/outputs/diagnostics/feedback-loss-updates-v1`.
Actual probe elapsed2050.326s (34.17m); preceding runtime check29.314s.
No paid allocation, inherited neural checkpoint or saved neural checkpoint.

## What passed

Two ordinary AdamW updates of all109,464,375 Attention parameters, fresh
seed20260911, first two epoch0 batch32 draws, microbatch4 accumulation,
unchanged v6 TRAIN pool and sampler seed20260912. Fixed weights were declared
before execution: base1, canopy stock-change3e-9, litterpart input-scale2.5e-5.
The same64 TRAIN monitor rows were evaluated at updates0/1/2.

The first combined batch4 parameter gradient is finite across all109,464,375
coordinates, norm1.865276. Independent native JVP comparisons of the actual
combined scalar pass in both declared parameter directions. Projections are
2.3690044e-4 and7.8039387e-5; absolute discrepancies1.6869e-11 and7.0030e-11.
Both projections are resolved and JVP/reverse primals agree. This is bounded
first-microbatch/one-day/two-direction evidence, not an all-coordinate or
changed multi-day objective certificate.

All update and monitor physical/mask/discrete/budget/optional-validity hard
counts are zero; parameters and optimizer states remain finite. Batch32 raw
gradient norms are2.801982/3.590722, clipped at1 with scales0.356890/0.278495.
The report declares `technical_admission_passed=true`, completed updates2.

## Same-panel scientific readout

Flux RMSE is gC m-2 day-1. Biomass rows are one-day change RMSE in gC m-2;
canopy is daily land-area organic-carbon stock-change RMSE. Litterpart is
coordinate residual RMSE in frozen neural-input standard deviations.

| Quantity | Before | After two updates | RMSE change |
| --- | ---: | ---: | ---: |
| GPP | 3.064839 | 3.061182 | -0.119% |
| NPP | 2.613462 | 2.605221 | -0.315% |
| Maintenance respiration | 0.154736 | 0.151020 | -2.401% |
| Growth respiration | 0.601660 | 0.601477 | -0.031% |
| Total autotrophic respiration | 0.521047 | 0.524724 | +0.706% |
| AGB change | 2.019488 | 2.013337 | -0.305% |
| BGB change | 0.647342 | 0.644950 | -0.369% |
| Canopy carbon change | 43396.249 | 43313.475 | -0.191% |
| Litterpart input-scale residual | 44.798011 | 44.600136 | -0.442% |

Combined fixed-panel loss0.79497748 ->0.79238661 (-0.3259%); all three
component losses decrease. The two training-batch losses must not be used
as a learning curve because they contain different rows. Heterotrophic
respiration target/prediction remain zero on this panel; that is no evidence
of learning nonzero respiration.

The total-autotrophic regression is real: maintenance overprediction decreases
while growth remains underpredicted, reducing cancellation in their sum.
Its bias changes-0.310771 ->-0.314872. Individual component improvements
therefore do not imply improvement of their sum. This warrants monitoring,
not an architecture change or a success claim after only two updates.

All science numbers are near-fresh, TRAIN-only one-day evidence. Canopy and
litterpart errors remain large; modest changes do not establish successful
fit, GPP recurrent repair, independent generalization or7/30-day acceptance.

## Runtime and next operation

Reverse compilation928.747s; independent JVP compilation658.033s. Together
these account for26.45m of the34.17m probe, plus separate initial forward
preparation. Actual first optimizer update loop22.309s including12.008s
optimizer compilation/execution; second update loop7.303s with optimizer
execution0.082s. Fixed64 monitor after each update takes about3s. This small
sample is not a production-throughput benchmark across the whole data pool.

JAX allocator peak in use9.862GiB, reserved pool20.630GiB; no OOM. Neither
number is a claimed multi-day training memory bound.

The technical admission is complete. Next consolidate the bounded amended-loss
learning run: declare update budget, fixed monitor/checkpoint/stopping rule,
wire the frozen coefficients through the existing pool trainer and retained
checkpoint method, then evaluate free T7/T30 science. Do not repeat these
completed JVP/gradient checks for unchanged objective/AD boundaries. Reuse
job15048786 as unchanged-loss reference within the scope documented in
`feedback_loss_learning_acceptance_20260913.md`; its final1998-update weights
are not a matched shorter-prefix checkpoint. Keep Attention and normalization
fixed. Do not change weights post hoc from this monitor. No new paid run has
been proposed or submitted by this result review.
