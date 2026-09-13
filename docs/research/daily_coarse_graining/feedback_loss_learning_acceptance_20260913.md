# Feedback-loss bounded learning and future 7/30-day acceptance

Date: 2026-09-13.

Status: prepared learning/acceptance plan. Trial coefficients are declared in `feedback_loss_calibration_results_20260913.md`; the subsequent learning-run budget remains pending. Preparation and free-node work are already authorized by the user; paid jobs retain the existing resource/cost approval rule.

## Claim boundary

The completed calibration is zero-update, fresh-initialization, train-only gradient evidence. Its four microbatches establish finite, connected gradients for the three objective components and inform a fixed weight choice. They cannot establish scientific learning, free recurrence, or acceptance.

The following two-update check is likewise a technical admission only. It shows that the selected, already-declared objective runs through ordinary native JAX AD and the optimizer on real train data. It is not a scientifically accepted model, a checkpoint-selection event, or evidence that the diagnostic update-1998 checkpoint has been repaired.

The completed 16-window screen is a train-domain, day-disjoint diagnostic panel: eight T7 and eight T30 windows, 14 sites, seven years, and 296 mutually disjoint Teacher days. It excludes one-day and monitor dates, but it is neither held-out sites nor a spatial/temporal/joint test. The update-1998 checkpoint from job `15048786` was diagnosed on this panel. Because the feedback losses and their candidate coefficients were developed using its intervention and inventory findings, that panel is a development panel, not independent final validation for the amended method.

No result here authorizes Teacher regeneration, v5 access, sealed-test access, a new 48-step inference path, a Teacher state reset during rollout, Gate D2, a 365-day claim, a paid submission, or a change to the 109,464,375-parameter Attention architecture.

## Invariants

The amended run keeps the frozen v6 release, input normalization, constrained daily update, canonical retained tail, complete trainable network, and shared science/rollout interfaces. The two supplemental losses are the declared `litterpart_feedback_mse` and `canopy_carbon_stock_change` terms from `feedback_and_canopy_loss_amendment_20260913.md`; they do not substitute Teacher values at inference or create a second implementation of an adjoint or observation cotangent.

The acceptance basis remains the target-oriented contract's 13 report keys: GPP; maintenance, growth, and total autotrophic respiration; NPP; heterotrophic respiration; NEP; NEE; AGB; BGB; total live biomass; nonliving ecosystem carbon; and ecosystem carbon. A lower total loss, litterpart loss, or canopy-change loss does not offset a failed primary GPP, NPP, respiration, AGB, BGB, or nonliving-carbon key.

Every reported free rollout retains the existing physical gates with required count zero: unexpected defined-status/mask mismatch, discrete mismatch, nonfinite learned or declared-defined values, negative declared stocks, and water/carbon budget failure. The budget tolerance remains `2e-12`. The existing declared-dynamic `rveget` reporting policy remains unchanged; it is reported but is not silently recast as an ordinary hard failure. Optional-loss validity is also fail-closed: litterpart invalid or empty eligibility, and canopy invalid rows or scale applicability, must be reported and must not be hidden by a zero-valued loss.

## Sequence after calibration

1. Root inspects the immutable calibration report. It must contain all four train-only microbatches, finite component gradients, component dot products and cosines, existing hard counts, physical diagnostics, memory/runtime, and the declared coefficient ladder. Do not choose either coefficient from the 16-window development panel or from a later 7/30-day screen.

2. Root records a coefficient decision before updates: the saved calibration report and source commit, both fixed coefficients, their joint-gradient rationale, fresh seed `20260911`, pool/sampler seed `20260912`, first two epoch-0 32-row batches, unchanged architecture/protocol, ordinary optimizer settings, and output location. Coefficients are an engineering choice made from the evidence; the user need not supply them. No new hash inventory is required.

3. Run the predeclared two-update technical admission from the declared fresh initialization. Use the shared one-day objective and ordinary native JAX AD; retain the bounded independent JVP check of the combined selected loss. The run does not inherit `15048786` parameters or resume from its output.

4. Admit a subsequent matched bounded-learning test only after its update budget, checkpoint/stop rule, monitor cadence, and evaluation protocol are written before launch. Both arms use the larger existing TRAIN pool, the same fresh seed, row order, optimizer, trainable scope, and stopping rule. The control is the unchanged objective; the treatment is exactly the frozen pair of selected supplemental coefficients. This comparison estimates the loss amendment's effect, not whether the old poor checkpoint is good. Do not pick coefficients after looking at this comparison.

5. Evaluate resulting checkpoints in free T7/T30 recurrence with only the declared Teacher initial state and actual daily forcing/retained inputs. First emit the reporting payload and hard-gate status for every frozen window; then classify science results. Do not replace poor windows or reset a recurrent state from Teacher data. A development-panel result is labelled development evidence even when all cells pass.

6. Any claim beyond this bounded learning comparison requires a separately frozen, independently selected non-test validation design. Final-method acceptance still requires the executable fresh-start protocol and the target-oriented contract's all-slice 7/30-day and later 365-day conditions.

## Required two-update record

For the fixed TRAIN panel at updates0/1/2, record the base, litterpart, canopy,
and combined losses and shared physical diagnostics. Record the combined
gradient norm, clipping, parameter/optimizer finiteness, memory and timing
for each actual update, with first-microbatch combined-loss JVP/reverse evidence.
Separate component gradient norms come from the completed calibration; do
not repeat three reverses per update merely for logging. Retain physical-unit
one-day productivity, respiration and biomass changes from shared diagnostics;
the fuller stock/cumulative payload below belongs to the subsequent rollout.

Litterpart reporting is coordinate-level. It must give predicted and Teacher next-state values, residual in the field's native fraction units, residual in actual neural-input standard-deviation units, valid/expected/invalid counts, and separate active bare/PFT14 counts. Do not label this fraction as carbon or cover-weight it.

Canopy reporting is source-direct land-area carbon: predicted and Teacher stock change, signed residual, RMSE, bias, normalized MSE, active/bare slot and row counts, and bare contribution. Its physical unit is `gC m-2 day-1`; no PFT cover factor is applied to `interception_storage`. Report its existing typed incoming/outgoing endpoints only as supporting diagnostics, never as a replacement for the realized stock-change quantity.

## Future free T7/T30 report payload

The acceptance report must keep the current normalized 13-key science results and add the following physical summary alongside, from the same shared trajectory extractors and rollout observations.

| Family | Required physical fields | Unit / aggregation |
| --- | --- | --- |
| Fluxes | daily prediction/Teacher/error; complete-window cumulative prediction/Teacher/error for GPP, maintenance, growth, autotrophic, NPP, heterotrophic, NEP, NEE | `gC m-2 day-1`; cumulative `gC m-2` |
| Primary stocks | start, terminal prediction/Teacher, terminal error, and start-to-terminal change for AGB, BGB, total live, nonliving, ecosystem carbon | `gC m-2` |
| Canopy feedback | daily predicted/Teacher change, residual, RMSE, bias, normalized MSE; valid/active/bare counts and bare residual | `gC m-2 day-1` |
| Litterpart feedback | per-coordinate next-state residual, input-standardized residual, RMSE/bias, eligibility and bare/PFT14 counts | native fraction and input standard deviations |
| Integrity | all physical hard counts, declared-dynamic status count, source NPP identity error, nonfinite/missing metric population | counts; `0` required where the established hard gate applies |

For fluxes, mark zero-Teacher totals separately: an exact heterotrophic match with a zero target is an integrity observation, not fitting evidence. For all stocks, endpoint agreement cannot be presented as correct stock change. For canopy and litterpart, report both the feedback diagnostic and the primary outcomes; neither is a substitute primary outcome.

## Absolute classification

For each complete T7 and T30 development or later validation panel, classify separately:

- `physical_gate_passed`: every retained hard count is zero and every required metric population is finite and complete.
- `absolute_science_passed`: all 13 target-oriented keys meet the frozen per-key normalized-RMSE and absolute-normalized-bias limits: T7 `<=0.35` and `<=0.10`; T30 `<=0.25` and `<=0.07`.
- `feedback_diagnostics_passed`: a descriptive, predeclared diagnostic status only. It reports canopy/litterpart values and coverage; it cannot promote a model or relax either preceding classification.
- `evidence_scope`: one of `train_domain_day_disjoint_development`, declared independent non-test validation, or later formal acceptance. The old 16-window panel must always use the first label.

A missing observation, incomplete window population, nonfinite value, or unreported optional-loss applicability is a failed or incomplete record, never a pass by omission. A two-update report has no `absolute_science_passed` field; it instead states `scientific_acceptance=false`.

## Existing-screen compatibility audit

The current files encode the completed `15048786` diagnosis. Future reuse should make the differing run metadata explicit while preserving that historical default, rather than copying the evaluator for each experiment.

| Current boundary | Current fail-closed assumption | Required later work for a candidate checkpoint |
| --- | --- | --- |
| `scripts/hpc/trained_screen_common.py::load_model` | completed report/checkpoint at update1998, matching report method and parameter count | Reuse the existing checkpoint format. Pass the declared run's expected update count and method explicitly while preserving the historical1998 default; ensure optional loss coefficients in the report describe the loaded model. No new checkpoint format is needed merely because the loss changes. |
| `scripts/hpc/trained_screen_common.py::validate_selection` and `WindowPool` | 16 windows, eight per horizon, mutually disjoint days and known source | Reuse the same development panel when its existing exclusions still cover the proposed training/monitor dates. Only broader data coverage that overlaps it requires a new selection; do not repeat unchanged selection work. |
| `scripts/hpc/run_trained_forward_screen.py::run` | imports the shared canonical rollout, shared science extractor, physical checks, and target-oriented statistics; emits `scope=train_domain_disjoint_days_not_formal_spatial_temporal_acceptance` | Reuse these components. Extend the report through source-owned/shared extractor APIs to include canopy stock-change and litterpart diagnostics, then make population-completeness check the declared new panel shape. Do not rebuild rollout loss, state cotangents, or science fields in a runner. |
| `research/daily_coarse_graining/daily_operator/forward_screen_metrics.py` | physical summary provides flux daily/cumulative values and terminal/change stock values; `complete_metric_gate` requires 8 windows and its existing flux/stock registry | Add optional feedback diagnostics using the shared helpers, retaining existing primary keys and thresholds. Test masks, units and population completeness for the changed report path. |
| `research/daily_coarse_graining/daily_operator/successor_training.py::extract_biomass_productivity_trajectories` | owns GPP/respiration/biomass/nonliving/ecosystem science extraction and already includes canopy interception carbon in ecosystem carbon without a second cover factor | Expose a named canopy-interception diagnostic trajectory or shared reduction, and obtain litterpart through the existing contract leaf/layout plus `feedback_state_mse` geometry. The future reporter consumes those APIs; it must not duplicate flat offsets, carbon reductions, or loss arithmetic. |

The current `run_trained_forward_screen.py` depends on `run_fresh_start_one_day` argument/bootstrap conventions. Candidate loading should therefore be audited against the future training artifact before any runner implementation: command arguments, report protocol schema, checkpoint `expected_method` identity, prepared-source identity, model protocol/parameter count, and dynamic optional-loss configuration. Until that comparison exists, the correct action is to leave the old screen strict and describe the compatibility work above.

## Remaining implementation

### Existing unchanged-loss control can be reused

Terra(high) static audit confirms job15048786 uses the same fresh seed,
full parameter scope, architecture/input conditioning/scales, prepared source,
sampler row order, batch32/micro4 and AdamW settings. The new canopy area
correction changes physical stock reporting/hard diagnostics, not the base
flux/state/process scalar. Disabled supplemental terms preserve that scalar.
Default to reusing this completed control, without a bitwise equivalence gate
or an expensive duplicate control training.

Its stored report must match the declared pool/method identity. Per-update
scalars and monitors at0/100/500/1000/1500/final support report-level prefix
comparison. The runner overwrites one checkpoint; arbitrary prefix weights
are not retained by its source. The final1998-update checkpoint supports a
matched full-epoch checkpoint comparison, not a shorter treatment budget.
Old physical stock summaries need corrected re-evaluation or explicit scope
labels; they cannot be assumed numerically identical to corrected accounting.
Later bounded training should distinguish these available report-level and
checkpoint-level comparisons instead of automatically rerunning a control.

### Implementation work

1. Freeze the coefficient decision and bounded-learning manifest; verify its training pool is the existing TRAIN pool and its monitor exclusions remain.
2. Complete the two-update technical report with native AD/JVP and all physical fields above. Preserve it as technical evidence only.
3. Reuse the existing checkpoint and panel formats; expose the specific expected run metadata at the shared loader boundary and test that path.
4. Extend shared reporting extraction once, then use it in training diagnostics and free-rollout reports. Add small unit tests for units, masks, population completeness, no second canopy cover factor, and strict candidate identity.
5. Run the declared matched bounded-learning comparison and its free T7/T30 development evaluation. Report treatment/control separately and retain the evidence-scope label.
6. Do not claim final scientific acceptance until the separately declared independent non-test validation and full reproducible fresh-start protocol are complete.
