# Diagnosed recurrent-state and canopy-change loss amendment

Date: 2026-09-13. Proposed objective; no training coefficient selected yet.

## Mechanism

The completed fixed-weight intervention identifies litterpart as the dominant
same-day input mediator of the observed recurrent GPP loss: replacing only
that field removes95.10% of the T30 free-minus-reset MSE gap. It does not
establish a repaired model. Actual input scaling amplifies prediction errors
that existing endpoint/state supervision scores in native fraction units.
Separately, the canopy inventory audit establishes large physical stock drain
with small outgoing-fraction loss. These are two measured supervision gaps.

Keep the complete109,464,375-parameter Attention architecture, current input
normalization, constrained updater, canonical retained tail and v6 release.
Do not force litterpart to a constant or substitute Teacher states at inference.
The source ratio contains increment/denominator branches; a subdaily formula
does not automatically provide an exact daily reconstruction. Any later exact
replacement requires source proof and target parity without48 steps.

Add two optional components to the shared one-day objective:

```
L = L_base + lambda_litterpart * L_litterpart
           + lambda_canopy * L_canopy_change
L_base = L_daily_flux + 0.1 L_state + 0.01 L_process
L_litterpart = mean_valid[((predicted_next - teacher_next) / input_scale)^2]
L_canopy_change = mean_valid[((predicted_stock_change - teacher_stock_change)
                             / train_canopy_change_scale)^2]
```

Litterpart uses only its declared compact state leaf, the head's actual frozen
continuous-input scale and the Teacher next-state label. All active compact
slots are eligible, including bare soil. Square individual coordinate errors
before averaging; do not weight by cover or allow opposite errors to cancel.
The PFT14 input scale is0.007890912518918275; bare fallback is1. This coefficient
is separately calibrated, so the large scaling factor is not silently added
at unit weight to training. No scale is fitted on diagnostic windows.

Canopy change follows `canopy_change_loss_proposal_20260913.md`: direct sum of
land-area carbon over active slots, no second cover factor; frozen train/train
PFT14 daily-change scale5.403926065530931. Bare contributions are reported.
Both terms preserve existing hard checks and explicitly reject invalid or
empty eligible supervision. Absent/zero optional weights preserve the previous
objective. Helpers are shared by training and later native-AD rollout losses.

## Bounded calibration and matched comparison

First use fresh initialization seed20260911 and the first16 epoch0 train rows
of the existing broad-pool sampler (seed20260912, batch32, microbatch4). Exclude
the pool's monitor days as before. No exploratory weights or diagnostic-window
rows enter this calculation. Reuse the existing prepared pool; no Teacher run.

Evaluate the three component gradients on four four-row batches using native
JAX AD. Supply loss-combination weights as dynamic arguments to one compiled
gradient graph. Report component loss/norm, full finiteness, cross-component
dot products/cosines, physical values, actual memory and elapsed time.
For each supplemental term, report weighted-gradient/base-gradient ratios for
the declared ladder `[1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2]`. This is calibration
evidence, not an automatic claim that a ratio threshold ensures learning.
The earlier canopy-only ratio rule remains a proposal; do not automatically
combine independently chosen weights without checking their joint effect.

The first remote run is calibration only, zero optimizer updates/checkpoints,
one idle free gln01 GPU, at most60 minutes, after real-array CPU shape checking
and the required once-per-new-graph deployed-runtime preflight. Preserve each
completed batch before further compilation/execution. No paid job is proposed.

Then record fixed coefficients and a two-update technical admission using the
first two32-row training batches from the same declared fresh initialization.
Verify the actual combined-loss gradient against bounded independent JVP
evidence and run the ordinary optimizer; report GPP/NPP/Ra/Rh, AGB/BGB changes,
litterpart input-unit errors, canopy change errors, physical validity, memory
and runtime. Two updates demonstrate execution/learning direction only.

For a subsequent bounded learning comparison, the matched control is the
unchanged objective from the identical seed and row order. Its purpose is to
attribute effects of the loss amendment, not to call the poor old checkpoint
a successful model. Keep current absolute productivity/respiration/biomass
acceptance requirements. Neither lower total loss nor better litterpart alone
admits the model. Declare update budget, coefficients and stopping rule before
launch; evaluate free7/30-day trajectories without Teacher resets. If useful,
test the two supplemental terms separately before claiming their individual
effects. No365-day/long-holdout/Gate D2 promotion follows from calibration.

## Reproducibility and current state

The effective architecture is unchanged. The final scientific method still
starts fresh and optimizes the whole network; this amendment adds explicit
losses and a reproducible training-only coefficient-calibration stage. The
1998-update checkpoint supplies diagnosis only. Existing passed native T7/T30
AD evidence is reused within its scope; a changed actual training loss needs
its own relevant gradient/update checks, not a repeat of all historical probes.

Shared helpers and optional one-day integration passed13 focused/integration
regressions; a further runner test verifies dynamic component-gradient
selection. Ruff and diff checks pass. At ac5aa51, the real shape/gpu launcher
argument expansions pass on test02. The real batch4 full-objective and native
parameter-backward shape check passed in151.33s from the isolated checkout
`runtime/checkouts/trained-feedback-v1`. Both free GPUs were idle; the GPU0
runtime-preflight/calibration chain launched as PID61046, at the same ac5aa51
checkout. Keep it fixed while running. GPU calibration cap60m after the3m
runtime-preflight cap; no update/checkpoint. All report/log paths
share `runtime/outputs/diagnostics/feedback-loss-calibration-v1`.
No new trained weights or scientific improvement have yet been produced.
