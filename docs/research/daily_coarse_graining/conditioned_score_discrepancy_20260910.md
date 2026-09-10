# Update-42 single-window score localization

Completed localization following job 14991510; input-conditioning repair remains open.
Retained update-41 weights,
artifact 883, zero-based thirty-day window 3. The failed callback ordinal 239
is resolved against the stopped attempt's first science group and four windows
per artifact, using the same ordering as `_measure_source_gradients` and
`_horizon_gradient`. No train/test reselection or parameter update.

First reproduce the four-window production totals and the saved failed pair.
Then compare compact trace observations, their postprocessed statistics, and
the existing production trajectory evaluator. Returning trajectories changes
the compiled outputs; therefore its agreement with the original production
statistics is measured, not assumed. Save the trace/microbatch/totals for a
follow-up without redoing initialization. Locate flux/stock/day discrepancies
before extending to state replay or derivative tests.

Runner: `scripts/hpc/audit_conditioned_score_discrepancy.py` and `.sh`.
The standalone copies in `runtime/transfer` reuse the admitted environment and
argument setup at ce7cd79 without updating the checkout used by CPU job 14992138.
Free gln01 GPU 0 was idle; one process, 45-minute cap, allocator fraction 0.55.
No paid job, retained gradient step or full-window reverse compilation.
Python PID at launch: 111626 on ibc13b02n04.
Report: `runtime/outputs/diagnostics/conditioned-score-discrepancy-14991510.json`.
Log: `runtime/logs/conditioned-score-discrepancy-14991510.log`.
Trace output uses the report path with suffix `.trace.pkl` and has no weights.

Local module import, actual-report window resolution, Ruff, shell syntax and
whitespace checks pass.

## First completed localization

The free-GPU run completed in 787.6 seconds. Both original scores reproduced
exactly: production 0.024385581647048654, compact observation
0.02459177191733883. All production physical hard counts are zero. Recomputing
statistics from the separately returned production trajectories reproduces
every production statistic exactly; postprocessing the compact trajectory
reproduces its different observation score to roundoff. All defined masks
match. The difference is already present in forward trajectories, before any
daily reverse/VJP propagation, not merely in loss aggregation.

GPP differs most on day 10: production 5.3011891841 versus compact
5.5908235837 gC m-2 day-1, a difference of 0.2896343996. Days 1-9 differ by
at most about 3.8e-7, and days 11-30 by at most about 9.2e-6. Day-10 NPP and
growth respiration differences peak at 0.2085367932 and 0.0810976418;
maintenance differences appear later and are much smaller. Some nonliving
carbon differences predate day 10; do not claim all states first diverge then.

This narrows the next work to the day-10 GPP jump; it does not identify a root
cause or prove the streamed reverse accumulation defective.

## Completed full-state and crossed replay

The same runner's `--trace-localization` mode reuses the saved microbatch and
statistics, emits full canonical state traces, checks their statistics against
the original production path, and replays day 10 using crossed continuous and
discrete carries. It then replaces at most 32 differing state leaves one at a
time, ranked by normalized day-9 discrepancy. These mixed states are causal
diagnostics, not physical acceptance trajectories. No parameter gradients or
training are computed in this phase.

Free GPU 0, same 45-minute bound, new report/log prefix
`conditioned-score-day10-14991510`. Source checkout remains ce7cd79.
This run completed in 604.8s. Its full trace score is 0.024385579840638782,
close to the original production 0.024385581647048654; maximum sufficient-
statistic difference is 5.01e-8. Unlike the compact trace it preserves the
production day-10 GPP result, within about 3.5e-7.

Crossing day-start discrete carries does not change either result. Using the
compact continuous state gives GPP 5.590823583681096; the full continuous state
gives 5.301188835179558, irrespective of the discrete carry source.
Of the 32 individual state-leaf interventions, only
`slowproc_stomate_previous_step_state.DOC` changes GPP. Replacing just that
leaf in the compact day-start state gives 5.301189184116634, matching the
original production GPP. The other 31 leave compact GPP unchanged.
This establishes a causal DOC-input explanation for this forward discrepancy,
not a global all-regime validation of the reverse implementation.

## Frozen input-scale audit

`scripts/hpc/audit_conditioned_doc_input_scale.py` completed on free test02,
using saved traces and the actual frozen process statistics, with no model call.
The runtime's input normalizer is `process_statistics.input_normalization()`;
`continuous` statistics inherit base state moments, not the semantic-domain
minimum-scale rule. `strong_baseline_architecture._standardize` casts value,
mean and scale to float32, then divides by any positive finite scale. It
does not lower-bound a tiny positive continuous scale or bound standardized
features. These features feed field and axis summaries; canopy/GPP source
routing admits the `state_carbon` family.

The most sensitive normalized DOC coordinate is global index 1785 (DOC-local
458). Frozen mean is 1.28283896e-12 and scale 7.96027116e-12. Day-10 input is
0.04795535427446734 on the compact trace versus 0.04795534980054048 on the full
trace. Their roughly 4.47e-9 raw difference becomes 512 after float32
standardization: 6,024,336,384 versus 6,024,335,872. Thus predicted rollout
state lies extremely far outside that coordinate's Teacher scale; tiny
forward roundoff is amplified before reaching the neural layers.

The largest raw DOC-leaf difference is 4.38e-7 at another coordinate; distinguish
that from the above normalization-dominant coordinate. Causality has been tested
for the whole DOC leaf, not yet for coordinate 1785 alone or a specific internal
attention operation. Do not overstate that finer localization.

Local report: `outputs/conditioned-doc-input-scale-14991510.json`.
All three diagnostics completed without parameter updates or paid GPU work.

## Next implementation gate

Keep the Attention topology and retained trained weights. First quantify tiny-
scale/large-standardized-input behavior on active training coordinates, and
design an explicit input-conditioning correction that remains meaningful for
predicted states outside the Teacher distribution. The immutable v6 data and
statistics are not to be rewritten; any transform/scale amendment belongs to
a versioned model-input protocol and must be reproducible in final training.

Use unchanged input handling as the matched reference. Before any training,
test the implicated DOC coordinate/leaf perturbations and paired production,
trace, one-day VJP/JVP and finite differences on this window plus ordinary
training windows. Check actual forward GPP/NPP, state and budgets: simply
making two scores agree is insufficient, and changing inputs can disturb
inherited trained modules. Do not relax the score tolerance, zero DOC states,
silently remove DOC from the network, or claim a repaired/accepted model yet.
If correction changes the effective model, rerun required free-node admission
and amend the coverage comparison so both data pools use the same correction.
