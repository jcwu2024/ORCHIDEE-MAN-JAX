# Public rollout component: first real-model two-day probe

Freeze before execution. The public component's toy T1/2/3/7 matrix and legacy
streamed regressions pass; root's canonical binding tests verify actual PFT
fraction reconstruction and its state derivative. This does not yet admit
current-model multi-day gradients.

Use the current fresh109,464,375-parameter Attention tree, seed20260911, DOC
conditioning and exact bare-soil gsmean output. Reuse artifact322, seven-day
window1, prefixT=2, with its actual initial continuous/discrete state and daily
retained inputs. No later Teacher state enters the candidate recurrence.
No training, checkpoints inherited, validation/test access or v6 regeneration.

Bind the canonical daily operator to `RolloutLoss`. Observations include current
state-derived PFT fraction, GPP, next continuous/discrete state and validity,
active PFTs and budgets. The probe loss is the active-field mean of four
cumulative normalized carbon-flux MSEs and final AGB/BGB normalized MSEs,
using existing train scales and stock_weight1. This horizon-agnostic diagnostic
loss is not a declaration of the final batch-coupled training objective.

Compare ordinary JAX scan AD, checkpointed scan AD and the public streamed
operation for both all neural parameters and initial continuous state. Report
global norms and per-leaf/field differences (atol1e-6,rtol1e-4) so a dominant
state sensitivity cannot hide a missing smaller route. Compare scalar primals
at atol1e-6,rtol1e-5. The native reference retains JVP; compare its parameter-
direction projection with streamed reverse and central rounded finite
differences at3e-4,1e-4,3e-5, using atol1e-6,rtol1%. Check primal state masks
and discrete values at displacements, plus finite states/observations and
budgets. This is fixed-branch local AD evidence, not Gate D2 or 7/30 acceptance.

Compile and inspect each ordinary/checkpointed gradient's memory before
execution. Use an18GiB compiled buffer envelope, leaving room for other live
arrays within the launcher's observed20.63GiB JAX allocator pool; oversized reference
is reported, never blindly executed. The streamed core also measures one-day
VJP memory. No 30-day monolithic backward allocation is attempted.

Local tests and CPU abstract tracing first. One free gln01 GPU, at most90minutes,
single two-day trajectory. Most cost is compilation of three independent AD
paths, not long data processing. Save reports incrementally. Any mismatch
blocks promotion and larger multi-day training; do not infer current-model
acceptance from toy tests or the earlier15.96M parameter route. After this
bounded probe, extend the measured matrix to T3/7 where resources allow.

## Integration preparation

The generic component and prior streamed suite pass21 tests. Root's actual
PFT-fraction binding/science loss and bounded data-cache tests also pass.
First real-data CPU preflight at e3aa80f caught a runner layout error before
any GPU work: prepared `window.day_inputs` has a window axis and contains a
shared annual retained bundle; it does not have a second day axis to slice.
The existing production rollout reuses that bundle with each actual day's
canonical input. The adapter now broadcasts the selected per-window bundle
along the prefix day axis explicitly. A scalar/annual-array layout regression
joins the canonical binding tests;3 tests pass in13.83s.
The original failed shape report remains `public-rollout-shapes-v1.json`.
The corrected CPU shape run passed at b7fd1bc in477.36s, report
`public-rollout-shapes-v2.json`. Before GPU execution, integration review adds
explicit Teacher GPP validity, Teacher-defined state finiteness, declared-stock
and learned-output checks, and coordinate-wise gradient agreement at the same
atol1e-6/rtol1e-4, retaining the declared per-leaf L2 gate. Updated CPU tracing
is required for the revised observation structure.

The revised canonical/probe/batch focused tests pass7 cases, and the generic
public/legacy streamed suites pass21 cases. Updated real CPU shapes passed on
test02 at4410c41 in275.3s, `public-rollout-shapes-v3.json`.
The free GPU1 T2 numerical probe started2026-09-12 13:10 CST at4410c41,
launcher PID110317,90-minute limit. Report/log:
`runtime/outputs/diagnostics/public-rollout-T2-v1.json` and `.log`.
The run exited at its90-minute limit before the checkpointed compile finished.
No final JSON was written; the report's running/compile_checkpointed state is
stale, confirmed by absent launcher/Python PIDs after the deadline. No exit
code was retained by the old wrapper, so timeout classification uses the
configured limit and process observation rather than a captured code124.

Partial result: direct AD completed at4126.9s, after about62.8minutes of direct
compilation. All109,464,375 parameter gradients and3854 initial-state gradients
are finite; every physical count is zero. Loss0.00384148725, parameter gradient
norm0.24276435, initial-state norm1473.4141; compiled buffer estimate5.31GiB.
The checkpointed/streamed and JVP/FD comparisons are absent. This does not
admit multi-day training and does not show a derivative disagreement.
Next revise execution ordering/reference persistence so the critical
streamed/native comparison is not placed after two expensive compilations
within one90-minute cap. The independent broader one-day training proceeds.
