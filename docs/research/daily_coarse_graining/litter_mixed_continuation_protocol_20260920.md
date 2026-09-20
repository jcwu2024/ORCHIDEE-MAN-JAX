# Selected litter supervision: bounded continuation to100 updates

Frozen before execution2026-09-20. User authorized continued exploration after
reviewing the completed matched20-update result. This is one free single-GPU
diagnostic continuation, not paid training or a final scientific recipe.

## Question and fixed method

Does the newly supervised carbon loss continue to shrink with more updates,
and do canopy, production and biomass tradeoffs persist or grow? The existing
20-update result improved nonliving-carbon change RMSE60%T7/44%T30 versus
matched old loss, but all16 windows still have the wrong change sign.

Start selected20 from immutable1391a75, full parameters and AdamW step4020.
Keep Attention/decoder, all109,464,375 trainable parameters, FP32 execution
policy, admitted native AD, LR1e-5, clip1 and original daily/recurrent losses.
Hold the already TRAIN-selected litter coefficient5.221716892123007e-5 fixed.
No new weight calibration, teacher regeneration or old-arm training.
The update objective remains D_old + lambda*L_litter + .5R7 + .5R30.
Graph instrumentation remains .1; the dynamic fourth coefficient applies
lambda once to the unweighted litter loss.

Execute80 further updates, relative21..100, absolute4021..4100. Consume
existing prepared schedule batches4020..4099, daily64/micro8 plus oneT7
and oneT30 each update. Additional consumption is5120 daily draws and80
windows of each horizon, not5120 guaranteed unique new examples. Including
the first20, this diagnostic has6400 daily draws plus100T7/100T30 windows.
The data sources/splits/seeds remain the original plan; do not reconstruct
or reshuffle the pool. An expanded training pass is not yet being proposed.

## Monitoring and stopping

At relative40/60/80/100, save full model/optimizer before evaluating fixed8
and excluded8 daily TRAIN panels and unchanged8T7/8T30 development windows.
Reuse selected20's saved metrics as the baseline; do not rerun it or retune
the coefficient based on development outcomes. Preserve each milestone's
physical-unit errors and per-window signed changes, including wrong-sign
counts; report normalized screens as secondary evidence.

Compare cumulative GPP/NPP/autotrophic-respiration RMSE and change RMSE for
above/below biomass, canopy carbon and nonliving carbon, separately atT7/T30.
RH remains reported by the full monitor; its all-zero Teacher targets do not
establish nonzero-response learning. These panels are development evidence,
not a sealed test or long-run acceptance. The output's scientific acceptance
remains false even if every existing normalized screen passes.

Stop at100 updates,60min wall limit, or actual numerical/physical failure.
Also stop cleanly if the SAME physical metric at TWO consecutive milestones
exceeds2x its fixed selected20 RMSE AND the absolute RMSE increase exceeds:
1gC/m2 for cumulative flux/biomass change,5 for canopy change,10 for nonliving
change. These deliberately broad diagnostic ceilings detect sustained large
regression, not ordinary batch noise or a final accuracy standard. They are
specified before execution, not tuned to observed continuation performance.
Single-monitor regressions and smaller tradeoffs are reported without an
automatic stop. Do not choose or promote a best checkpoint automatically.

## Runtime and recovery

Use one idle free gln01 GPU,60min maximum, no automatic paid fallback. Previous
recovery took28m38s including18 updates and three full monitoring passes;
80 warm updates alone are expected around17min at12.5s/update. Plan roughly
40-55min total, subject to cache/compile variability;60min is a ceiling, not
a completion guarantee. Reuse the checked graphs/cache without new JVP tests
because no loss/state/AD boundary changes. Target-shell dry-run and actual
deployed CLI parsing plus focused local recovery/schedule tests before launch.

Four milestone containers plus one atomically replaced recovery container
(every5 updates) limit additional storage to about6.6GB and under10 output
files. Retain all existing source artifacts. Resume from the saved optimizer
and next exact batch; if interruption followed checkpoint but preceded its
monitor, finish that milestone's evaluation before advancing. Preserve report
history. A stopped or incomplete run is never relabeled a completed100 updates.

Source report/checkpoint:
`runtime/outputs/diagnostics/litter-mixed-admission-v1.json` and
`litter-mixed-admission-v1.selected-update-000020.checkpoint.pkl`.
New output prefix `runtime/outputs/diagnostics/litter-mixed-continuation-v1`.
Manifest `manifests/coarse_graining/litter_mixed_continuation_v1.json`;
launcher `scripts/hpc/run_litter_mixed_continuation.sh`.

This inherited exploratory fork remains diagnostic. The paper's final method
must be independently specified and executed from its declared initialization;
these ad hoc checkpoint continuations do not satisfy that requirement.

## Local implementation evidence

22 focused tests pass across continuation metadata/recovery alignment,
physical-unit stopping, prior admission, shared checkpoint state and monitor.
Ruff/diff checks pass. Actual saved selected20 report validates the parent
method and fixed coefficient; self-comparison yields14 unit-consistent
metrics with ratio1 and no stop. Code review fixed report-history preservation
and checkpoint-before-pending-monitor recovery. Prior admission numerical
body remains unchanged; continuation invokes the same shared loss/AD/update
functions. Full reports flush every5 updates and lifecycle boundaries,
with compact progress each update. No new gradient graph validation is
required for this unchanged objective.

## Launch

Started2026-09-20 18:25:23CST on gln01 (ibc13b02n04) idleGPU1 from immutable
5fdfd5e in `runtime/checkouts/litter-mixed-continuation-v1`. Actual target-shell
dry-run and deployed GPU-container CLI parsing/import check pass.
Wrapper72153/timeout72165/container72166/Python72207;60min ceiling about19:25.
Log confirms metadata validation and selected relative20 start; model loading
in progress at last check. Output prefix as above. No paid submission.
Sol/high implemented the continuation branch and focused tests; lead reviewed
recovery/monitor behavior, ran the combined22-test check and deployed.
