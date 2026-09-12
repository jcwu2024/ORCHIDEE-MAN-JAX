# Complete the T2 gradient probe with durable stages

User authorized completion after the first90-minute test timed out during the
checkpointed compile. Preserve its partial report. No architecture, loss,
sample, initialization, derivative tolerance or physical gate is changed.

## Frozen case and acceptance

Same fresh109,464,375-parameter v3 tree, seed20260911, artifact322/window1/T2,
canonical binding, full initial continuous/discrete state, current PFT fraction,
Teacher validity, retained inputs, four cumulative fluxes and final AGB/BGB.
Only daily transitions are differentiated. No training, v6 regeneration or
validation/test access. Native reference and streamed parameters/initial-state
gradients must agree per coordinate and per leaf at atol1e-6/rtol1e-4; scalar
loss at atol1e-6/rtol1e-5. Rounded parameter-direction JVP/FD checks retain
epsilons3e-4,1e-4,3e-5, atol1e-6/rtol1%, and all physical/mask-stability gates.

## Executable sequence and persistence

`scripts/hpc/probe_public_rollout_staged.py` uses the unchanged public component
and canonical graph. The historical monolithic runner remains reproducible.

1. `core`: compute/reuse the ordinary direct reference, save its complete
   parameter and initial-state gradient immediately, then compute/save streamed
   gradients and compare them. Never compile the checkpointed reference here.
2. `derivatives`: only after the core comparison passes, load both gradients and
   perform JVP/rounded central FD. Save each completed epsilon independently.
3. `checkpointed`: separately compare the native rematerialized graph against
   the saved ordinary reference. This is last, after the critical evidence.

Every stage has a separate JSON report. Atomic pickle artifacts contain the
case identity, complete gradient/value and numeric summary. Case identity uses
the protocol, actual artifact schedule, selected window/days, and explicit
loss/component versions; no bulk data hashing. A later failed/limited stage
cannot erase an earlier complete reference. Each invocation reconstructs the
fresh model, checks the canonical physical forward, and checks reference scalar
replay before using saved derivatives. Source commits are recorded per run.
No exploratory model checkpoint is inherited; these artifacts are gradients.

The existing persistent compilation cache remains at the canonical runtime
path. Cache hits are measured, not assumed. The ordinary reference graph and
dynamic argument structure are unchanged; it still must be evaluated again
because the previous process did not save its gradient arrays. Native compiled
memory is checked before execution at18GiB; the public streamed core retains
the same one-day VJP envelope. Stages preserve host-side gradient arrays, so
subsequent comparisons do not depend on keeping the original process alive.

## Execution bounds

One idle free gln01 GPU after local staged-resume tests; use the existing
CPU-traced numerical graph. The new orchestration gets a small local actual-JAX
test covering durable reference, interrupted continuation, streamed comparison,
directional checks and cached-stage reuse. No additional real-data full gradient
tracing is needed merely for persistence/ordering; the admitted graph is unchanged.

The runner chain gives core90minutes, derivatives60minutes and the separate
checkpointed check90minutes. These are stop limits, not predicted durations.
Only a completed, passed preceding stage starts the next stage. A failure or
time limit stops the chain and retains all complete artifacts; explicit rerun
resumes rather than discarding them. This is one bounded T2 diagnostic chain,
not paid training. No paid submission is authorized.

The broader one-day two-update smoke completed independently with exit0. Its
scientific/throughput review is separate from this AD completion task. Neither
T2 AD success nor two one-day updates establishes7/30-day scientific accuracy,
365-day stability, Gate D2 or product acceptance.

## Launch record

Seven staged/canonical/physical regression tests pass; independent review found
no numerical or persistence blocker. The existing direct jit_loss cache file
is present (26MB), but its reuse is not assumed. The staged chain started on
free gln01 GPU1 at4d128c3, launcher PID19594. Pipeline log:
`runtime/outputs/diagnostics/public-rollout-T2-v2-pipeline.log`.
Reports: `public-rollout-T2-v2-core.json`, `-derivatives.json`,
`-checkpointed.json`; shared stage directory `public-rollout-T2-v2-stages/`,
all under the same diagnostics directory. The chain is currently in core
bootstrap. Do not update its checkout or start a duplicate run. No paid job.

The first staged attempt saved the full418MB reference, then failed after
compiling streamed reverse: JAX0.4.38 rejects numpy.void scalar arguments.
The host slice of a scalar bool/int observation's float0 cotangent produced
that NumPy scalar. The fix at8e522ce retains a0-D array with the same float0
dtype. No biological equation, forward value, reference gradient, or derivative
rule changes. Seventeen local public/staged tests pass. The tiny GPU diagnostic
`probe_rollout_float0_compatibility.py` reproduced the exact old failure under
JAX0.4.38 and passed fixed streamed/direct agreement in a few seconds, before
relaunching the real model. Result `rollout-float0-compatibility-v1.json`.

The original report/log are archived as `public-rollout-T2-v2-core-attempt1.json`
and `public-rollout-T2-v2-pipeline-attempt1.log`. The chain resumed on free GPU1
at8e522ce, launcher PID31604; existing reference.pkl is reused. Numerical
cross-method admission remains pending. Single-day direct-AD training does not
use this streamed interface and its isolated12bac94 checkout remains unchanged.
