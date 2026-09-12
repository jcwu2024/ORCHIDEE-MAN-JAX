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
execution. The same24GiB compiled buffer envelope applies; oversized reference
is reported, never blindly executed. The streamed core also measures one-day
VJP memory. No 30-day monolithic backward allocation is attempted.

Local tests and CPU abstract tracing first. One free gln01 GPU, at most90minutes,
single two-day trajectory. Most cost is compilation of three independent AD
paths, not long data processing. Save reports incrementally. Any mismatch
blocks promotion and larger multi-day training; do not infer current-model
acceptance from toy tests or the earlier15.96M parameter route. After this
bounded probe, extend the measured matrix to T3/7 where resources allow.
