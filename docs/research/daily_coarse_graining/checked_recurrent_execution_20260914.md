# Shared recurrent loss and compact checks

## September15 execution revision

V1 stopped at its55min ceiling (wrapper exit124); its JSON still contains
the last running stage because timeout interrupted the report writer. Both
processes exited. Original timing and the T7 gradient were preserved; no
checked equivalence or speed result exists. The last stage combined compile,
first execution and host comparison, so it is not evidence that compilation
alone caused the timeout.

The next bounded attempt uses `--separate-checked-validation`: native
`value_and_grad(..., has_aux=True)` returns observations and science counters
from the same primal rollout; a separate parameter-free JIT reduces physical
and feedback counters. Inputs and observations stay on GPU, and only the scalar
and final compact count vector cross to the host during normal execution.
The ordinary full-gradient host copy is diagnostic-only. No second rollout,
loss change, precision change, custom cotangents or relaxed gates are introduced.
Canonical T30 observations are approximately1MiB, small compared with the
measured16GiB live requirement. This removes unnecessary compiler coupling;
it is a candidate optimization, not an established cause/fix for the timeout.

The wrapper records raw AD compile, first execution, validator compile/memory,
observation bytes and full-coordinate comparison separately. Warm calls add
no new diagnostic synchronization. Original profile and saved T7 reference
are reused; any missing reference is computed/preserved at its own horizon.
Daily compilation is delayed until both recurrent comparisons pass. Source
runtime is unchanged and its78.45s passing synthetic check is reused.

Free attempt: independent checkout `runtime/checkouts/mixed-learning-checked-reuse-v1`,
new output `mixed-learning-checked-profile-v2.json`,80min diagnostic ceiling.
Use `CHECKED_REFERENCE_REPORT` pointing to the v1 report and shell mode
`checked-profile --separate-checked-validation`. Reused baseline timing is
from another process and must be labelled accordingly. Training stays opt-in;
no paid job is authorized. The inline variant below remains diagnostic.

Local split-route verification:38 focused tests pass across public rollout,
compact aux, invalid physical counters, split wrapper, mixed runner and
profiling; Ruff and diff whitespace checks pass. The wrapper test checks exact
inline/split loss/gradient/count equality, one primal rollout, dynamic expected
discrete values, no full-observation host transfer and one validator compilation.

Deployment: target-shell dry-run passed; free gln01 GPU0 launch uses immutable
fec83e0, Bash89392/Python89436 (parent89390), with both GPUs idle before launch.
The actual runner reached metadata. Results pending; keep this checkout immutable.

V2 outcome: wrapper exited124 at80min; both task processes exited. T7 native
gradient/loss/full-counter comparison passes (zero failed gradient coordinates,
all hard counts zero). Original/new losses121.8003196598/121.8003208346;
gradient error/reference L2=.0002910945/2634.8539. Changed T7 compilation
took1980.81s, including an XLA slow operation lasting23m58s. First raw execution
15.87s, validator compilation1.656s, full host comparison1.961s; observations
243418bytes. These first-call timings are not warm throughput evidence.
Original T30 compiled221.61s and its full reference was saved before changed
T30 compilation, which remained incomplete at timeout. No numerical failure
was demonstrated, but T30 equivalence and complete-update speed remain unknown.
The raw JSON retains its interrupted running stage; the log exit124 controls
completion status. No active job or paid submission. Reuse both references.

## Initial inline implementation

User authorized implementing the first performance optimization after the
fixed-checkpoint profile. Keep architecture, FP32 neural parameters/FP64
physics, objective/scales, sampling, native checkpointed AD and AdamW unchanged.

`RolloutLoss.bind_checkpointed_loss_with_aux` runs one existing checkpointed
rollout and calls a scalar-plus-auxiliary reducer. `recurrent_validation.py`
implements the existing physical and feedback counter semantics in pure JAX.
`checked_recurrent_execution.bind_checked_backward` evaluates shared science
terms once, adds those counters and returns one compact integer vector beside
the scalar under native `value_and_grad(..., has_aux=True)`. It returns no
observation trajectory and creates no manually assembled cotangents. The
gradient path remains through the same recurrent scalar and retained tail.

The mixed runner exposes opt-in `--checked-recurrent` for training, recording
the execution choice in checkpoint method identity. Its original execution
remains available. Do not use the new path for paid learning before the
bounded real-runtime comparison below passes.

## Bounded admission

1. Local T1/T2/T3/T7 loss and parameter/initial-state gradient comparison;
   dynamic inputs, masks/discrete carry and auxiliary NaN isolation. Compare
   pure-JAX counters against the prior NumPy implementations on deliberately
   invalid cases, not only all-zero real cases.
2. Once on the deployed GPU runtime, run the existing small synthetic runtime
   check (3min ceiling), then use the trained mixed-learning-v1 update200
   checkpoint on free gln01. No paid job or Teacher regeneration.
3. Lower the changed T7/T30 graph before timing work, exposing trace-time
   argument/type failures early. Measure the original fixed daily64/micro8 and
   first T7/T30 pair, warmups plus three repeats, as in the accepted profile.
4. Preserve each full original parameter gradient and counters before compiling
   its checked replacement. Compare all109M gradient coordinates using the
   existing `gradient_comparison` tolerance (1e-6 +1e-4*abs(reference), plus
   existing per-leaf L2 check); scalar atol1e-6/rtol1e-5; identical named hard
   counters and all physical/finite counts zero. Do not loosen on failure.
5. Repeat complete mixed-update proposals through the checked executables with
   the same unchanged parameters/optimizer and data. Compare losses/component
   norms/combined norm/clip with original replays; proposals are computed and
   discarded, not scientific continuation. Record warm total/component times
   and memory. Compiled live bound remains18GiB. Target at least5% median time
   reduction; observed regression requires investigation, not automatic adoption.

Launcher: `scripts/hpc/run_mixed_learning.sh checked-profile`, runtime check3min
then diagnostic55min ceilings. Isolated checkout:
`runtime/checkouts/mixed-learning-checked-v1`. Report/log:
`runtime/outputs/diagnostics/mixed-learning-checked-profile-v1.{json,log}`.
Two reference gradient pickle files (~876MB total) are retained alongside it
for discrepancy diagnosis. No new sample/shard materialization and no trace
export (prior CUPTI coverage was incomplete). Original scientific training
and profiling checkouts remain immutable.

Local validation:50 relevant tests pass, including public rollout, compact aux,
invalid physical/feedback counters, mixed profile/update and checkpoint tests.
Target-shell dry-run passes. Free gln01 GPU0 chain started at immutable45da5f9,
Bash81582, with both GPUs idle at launch. Pending runtime and real-model result;
no paid job submitted.
Deployed synthetic runtime check passes78.45s, exit0. Actual-model Python82371
has lowered both checked horizons successfully and is preparing the original
reference executables. Local follow-up rejects `--shape-only --checked-recurrent`
because that legacy shape path cannot certify the checked branch; the active
checked-profile already exercises its real lowerings and is unaffected.

The original daily compilation took965.81s in this run (prior profile~103s),
including an XLA slow-compilation warning; no cause has been established.
Original complete replay did finish (~14.2s/update), and the T7 reference was
preserved before checked compilation. At50min the checked T7 comparison had
not completed. This is not a demonstrated numerical failure or speed benefit.

Reference-reuse support now accepts `--checked-reference-report` (shell
`CHECKED_REFERENCE_REPORT`). It reuses a matching completed original profile
and preserved gradient files, including paths inherited through attempts,
and postpones daily compilation until checked gradient comparisons finish.
It keeps the same model/loss/AD graph; reuse the already passed deployed
runtime check. If continuation is needed after the55min ceiling, use a new
isolated checkout/output and an80min free diagnostic ceiling informed by the
observed compilation cost. Reused timings are explicitly labelled as coming
from another process, so do not claim a contemporaneous paired speed test.
Reference reuse/checked profiling tests:11 pass. No paid work is authorized.

Acceptance establishes execution equivalence and speed for a bounded panel;
it neither improves model accuracy by itself nor establishes long-run science
acceptance. Larger data exposure and the RA/canopy imbalance remain subsequent
research work. Do not make full closure of historical streamed AD discrepancies
a prerequisite for this native-AD execution improvement.
