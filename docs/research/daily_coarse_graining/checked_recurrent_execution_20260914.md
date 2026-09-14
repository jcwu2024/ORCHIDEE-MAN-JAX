# Shared recurrent loss and compact checks

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

Acceptance establishes execution equivalence and speed for a bounded panel;
it neither improves model accuracy by itself nor establishes long-run science
acceptance. Larger data exposure and the RA/canopy imbalance remain subsequent
research work. Do not make full closure of historical streamed AD discrepancies
a prerequisite for this native-AD execution improvement.
