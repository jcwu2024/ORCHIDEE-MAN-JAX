# Completed feedback-loss calibration and fixed coefficient decision

Date: 2026-09-13. Calibration source ac5aa51, free gln01 GPU0, PID61046
finished with exit0 in1063.644s. All four batch4 evaluations completed;
all component losses and gradients finite, all hard counts zero. There were
zero optimizer updates, inherited checkpoints or saved neural weights.
Fresh seed20260911, full109,464,375-parameter Attention, existing train-only
epoch0 sampler rows. No architecture, input scaling or Teacher-data changes.

Evidence: `outputs/feedback-loss-calibration-v1.json` and its host-only
`outputs/feedback-loss-calibration-v1-readout.json`. Remote prefix is
`runtime/outputs/diagnostics/feedback-loss-calibration-v1`. Runtime check
passed22.67s; CPU shapes passed151.33s. Compiled reverse estimated live
storage8,381,437,268 bytes (7.806GiB), not a measurement of whole-process peak.
The reported JAX allocator peak bytes in use across all batches is8.631GiB;
its reserved pool is20.630GiB. Reserved GPU memory is therefore substantially
larger than live allocations; neither value predicts the optimizer/JVP peak.

## Observed scale and decision

| Component | Raw loss range | Gradient norm range | Declared coefficient |
| --- | --- | --- | --- |
| Existing base | 0.1614–0.7866 | 1.1166–2.4606 | 1 |
| Canopy stock change | 9.32e6–5.89e7 | 1.36e8–7.85e8 | 3e-9 |
| Litterpart input-scale error | 2006.13–2006.97 | 50348–57502 | 2.5e-5 |

Unit coefficients would let the supplementary gradients overwhelm the base.
Choose the rounded fixed coefficients above to put the added gradient norms
in the same order as the base, not to equalize numerical loss values. Across
the four batches the weighted canopy/base norm ratio is0.365–1.033 and
litterpart/base is0.531–1.287. Combined raw norm is1.865–3.648. Pairwise cosines
have magnitude below2.8e-5: the measured full-parameter gradients are almost
orthogonal at this fresh initialization. This is not a claim that trained
objectives will stay independent.

The original reported coefficient ladder started at1e-5; even that canopy
weight would give a gradient thousands of times the base. The3e-9 choice
extends below that diagnostic ladder using saved raw norms/dots. It is a
recorded calibration-informed amendment before training, not a claim that
the original ladder contained a usable canopy value. No new GPU sweep is
needed because coefficient combination is linear in the saved gradients.

Saved Gram geometry gives negative directional derivatives for all three
components along the negative combined raw gradient in all four batches.
This supports this bounded trial but does not predict an AdamW finite step;
Adam preconditioning, clipping and later recurrence can change the outcome.
The weight choice uses only the fresh train calibration, not the development
rollout scores. It is recorded in
`manifests/coarse_graining/feedback_loss_update_admission_v1.json` before updates.
It is an engineering choice, not an automatic or optimal coefficient search.

## Next execution and parallel preparation

The native-AD two-update runner is implemented, with batch32 accumulated from
microbatch4, all parameters trainable and fixed64 TRAIN-row monitoring at
updates0/1/2. It preserves two reverse projections before independent JVP
compilation. It records clipping, physical/optional-loss diagnostics, memory
and timings, and writes no checkpoint. Actual target-runtime shape tracing
and GPU optimizer execution remain required. The bounded GPU cap is60m after
a3m runtime check for the added JVP/optimizer graph boundary, using a separate
checkout from the completed calibration.

Sol(high) implemented the update runner and Terra(high) prepared the future
learning/7/30-day acceptance plan; root reviewed both and added the host-only
gradient geometry reader. Local focused calibration/update/geometry tests
pass8 tests including preservation of opposite per-coordinate litterpart
errors. The report-only addition exposes coordinate statistics and their
contract PFT layout without changing the objective or AD graph. Real-model
deployment remains a separate evidence step.

`feedback_loss_parallel_preparation_20260913.md` describes this preparation.
`feedback_loss_learning_acceptance_20260913.md` describes the subsequent
bounded learning and rollout sequence; its training budget and checkpoint
rule still need declaration. The old16-window panel is development evidence,
not final independent validation. Two updates cannot establish improved
long-term GPP, respiration or biomass and are not a repaired trained model.

## CPU admission attempt and bounded correction

At f364c44, actual Bash shape/gpu expansions pass on test02. The CPU shape
attempt then exited with SIGSEGV after an XLA slow-compile alarm for
`jit__lambda_`, before the trace stage. The report remains at bootstrap and
is preserved as `feedback-loss-updates-v1-cpu-shape-f364c44-interrupted.json`;
its stale running field is not evidence of a live task. There is no full-model
AD discrepancy or scientific result from this attempt.

The update runner introduced a separately jitted full-tree finiteness
reduction before its shape branch. Remove that unnecessary compilation and
use ordinary host-side finite checks for the bounded real update probe.
Shape mode traces the new objective/JVP/optimizer interfaces; actual GPU
execution retains the parameter/optimizer finite checks. This is a localized
diagnostic correction, not a relaxation of numerical or physical criteria.
The failure location implicates the added check, but the original log alone
does not prove the native compiler's internal fault. A successful rerun will
establish usability of the simplified path, not an exhaustive compiler cause.

The corrected497fe80 path passed actual CPU objective/reverse/JVP/optimizer
shape tracing on test02 in204.898s, exit0. Report
`feedback-loss-updates-v1-cpu-shape.json` declares zero executed updates;
the synthetic-shape label denotes no numerical science execution even though
the supplied weights are the reviewed fixed trial weights. Launcher expansion
evidence is reused because the shell/CLI did not change.

Both free GPUs were idle. The GPU0 two-update/runtime chain launched as
PID65767 from `runtime/checkouts/feedback-updates-v1` at497fe80. Keep that
checkout fixed. Prefix `runtime/outputs/diagnostics/feedback-loss-updates-v1`;
pipeline log `-pipeline.log`, runtime `-runtime.json`, final `.json`. Caps are
3m runtime preflight then60m actual probe. No checkpoint or paid submission.
Read the actual directional checks, completed update count, fixed-panel
science and hard counts before calling this technical admission a pass.
The deployed GPU runtime preflight passed29.314s (JAX0.4.38/NumPy2.4.6,
GPU backend); the actual-model runner has reached bootstrap. This small
preflight does not certify the pending full-model update result.
