# Combined backbone and decoder execution admission

Status: diagnostic implementation, free resources only. The completed
expert128 real daily result is the predecessor, not scientific model acceptance.

FreeGPU0 launcher92971 is running from approximately23:47CST2026-09-25,
30minute main-probe ceiling after the deployed runtime check. JAX0.4.38 runtime
check passed in24.49s; original result reuse completed in3.1s and candidate
lowering began at3.6s. Both GPUs were empty before dispatch. Overlay:
`runtime/transfer/shared-combined-real-daily-20260925`; report/log:
`runtime/outputs/diagnostics/shared-combined-real-daily-20260925.{json,log}`.
No new paid task. Eight focused probe tests pass, including scoped restoration,
explicit helper loading and external-result identity checks.

The local context output/mask comparison and assembled state.other gradient
pass; strict old-FP32 full-parameter comparison has sparse discrepancies.
A bounded true-FP64 two-block backbone calculation at the first failing
block0 key-kernel coordinate[27,15] yields0.0107432357843, versus original
FP32 0.0107447504997 and reused FP32 0.0107435565442. The reused value is
closer at this coordinate (errors3.208e-7 versus1.515e-6). This is independent
evidence for rounding sensitivity at that coordinate, not proof that all
discrepancies are harmless. Keep the actual GPU comparison strict. Tiny
FP64 attribution and the bounded real-loss diagnostic proceed in parallel;
neither is production promotion or scientific acceptance.

The completed local adapter suite passes3tests (75.84s), and Ruff passes.
The independent dtype-preserving FP64 oracle compares both scalar value and
every backbone-parameter gradient leaf for two blocks: full-bank and
shared-source formulas agree at rtol=atol=1e-11. Projected inputs are frozen
for that oracle; this is not a full-network or eight-block proof.
The separate strict FP32 full-context diagnostic finds27/72,417 parameter
coordinates outside the unchanged1e-5/1e-6 tolerance across16leaves:
4backbone key/value,2input-projection and21upstream-tokenizer coordinates.
Assembled state.other has zero failing gradient coordinates. Reproduce with
`conda run -n ORCJAX python -m tests.unit.test_shared_backbone_execution`.
This diagnostic reports failures explicitly; default unit tests do not
silently mark full-FP32 parameter parity as passed or expected-failed.

The current backbone projects the same393 source tokens in each of seven
expert banks for each of eight blocks. The proposed execution projects common
source keys/values once per block, then uses those values with the original
distinct expert-axis banks, queries and masks. Attention weights, normalization,
FFN, residuals and all weights remain. All derivatives remain native JAX.
This is compute reuse, not another architecture or a detached state cache.

The source formula estimates22.25billion forward MACs avoided per sample-day.
That estimate and the isolated scalar helper's1.914x speedup are not a full
training performance prediction. Synthetic eight-block strict comparisons had
output/gradient discrepancies, retained in the backbone diagnostic report.
Independent one-block source/axis projections agree, and the one-block output
comparison passes. This permits a bounded diagnostic with the actual loss;
it does not waive that diagnostic or promote the helper.

## Fixed comparison

Use the same saved fresh update0, artifact742 first four daily rows, compact
head/template/scales context, canonical daily objective and expert128 decoder
as the previous passed real daily test. Keep GPP's original decoder. Compare
combined execution to the saved original full result at unchanged1e-5/1e-6
and exact zero physical hard counts. Also report the difference to the saved
expert128 result to distinguish the incremental change. Timing from saved
reports is explicitly cross-run evidence; separately replay admitted binaries
for a matched profile without new gradient compilation.

The optional context-assembler binding exists only while lowering/compiling
the candidate. Restore both assembler and raw-output bindings even on exceptions.
Retain reference results and save the candidate executable/result before
comparison. Do not alter the passed decoder report or overwrite its artifacts.

Local canonical tests cover returned context and input gradients, including
masks/inactive PFTs; parameter evidence has the limitations above.
Deployed runtime gate precedes the new graph.
One free GPU, at most30minutes after at most3minutes runtime checking; reuse
the14.47MB prepared context instead of another Teacher/runtime preparation.

## Decision and limits

Loss/auxiliary/full parameter-gradient or physical failure remains failure.
No tolerance relaxation, loss adjustment, hidden freezing, lower precision,
capacity reduction or shorter recurrence is included. If this route has a
material actual daily gain and passes, candidate T7/T30 native gradients and
complete same-work mixed updates are next. If it fails, preserve attribution
against both references and diagnose the actual discrepancy before another
compile. Complete training throughput and card-hours, not fragment wins or
changed sampling mix, determine whether the speed objective is being met.

In parallel, `profile_shared_saved_daily.py` replays the original and admitted
expert128 daily binaries at exactly these weights/inputs. It collects warm
times and one device trace each, with a10minute free-node bound and no model
gradient compilation. Profiler traces locate remaining work; their instrumented
times are not the warm throughput result. No paid job or formal training.
