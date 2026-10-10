# Bounded V4/V7 learning sensitivity decision

Approved by the user on2026-10-04 for this one bounded execution, now completed.
The [actual results](shared_v4_learning_sensitivity_results_20261004.md) record
48 updates/580 model calls, exit0,1192.757057s/0.331321405 free card-hours/CNY0
and released GPU1. The1800s window is closed. The original specification below
is preserved; no acceptance threshold, production default or paid fallback
changed. Candidate throughput rose16.56–16.72% during the observed trajectories,
which differ beyond original-repeat drift and reverse their loss ordering
between steps8 and16. Strict failures remain and the candidate is not adopted.

## Why this is the remaining useful question

The saved same-input complete-update candidate takes11.186190s versus
13.103010s original:1.171356x throughput over two warm pairs. Pair2 mixed
gradients fail977 coordinates in12 leaves at the unchanged1e-5/1e-6 gate,
while loss, actual updated parameters, AdamW and physics pass. Fixed-weight
repeats reproduce a T7 version effect; daily parameter repeats pass.

[Saved-array numerical effects](shared_v4_numerical_effect_results_20261004.md)
quantify a mixed-gradient relative L2 difference of2.3967957e-5 and cosine
0.999999999721. The actual rounded step differs by1.1714730e-4 relative to
the original step norm, cosine0.999999993155. Local step and moment differences
remain consequential evidence: a single backbone coordinate differs by12.81%
of its original step despite an absolute difference of1.58325e-7, and second
moment difference energy is concentrated in the shared final readout. Neither
global smallness nor strict failure counts measure learning damage.

The bounded source audit preserves formulas, masks, complete recurrent
history, precision and native AD. It gives no evidence of an omitted
mathematical term and does not establish the actual compiled floating-point
cause. There is no verified small repair that retains the measured gain.
Repeated local precision/cancellation probes would not answer whether the
current discrepancy changes learning. A short independent-trajectory test
can measure that question, while strict substitution remains unadmitted.

## Executable diagnostic

- Three independent arms: `reference_a`, `candidate`, `reference_b`. Both
  reference arms use the same original executables. The candidate uses saved
  V4 daily and V7 T7 binaries with the **same original T30** binary.
- Start all arms from the saved pair1-reference actual parameters and AdamW
  state (optimizer step2), artifact742, source `c599236`. This diagnostic
  initialization is not the final fresh-initialization training protocol.
- Run16 actual full-network updates per arm,48 total, rotating execution
  order. Each update uses the same declared daily32/eight distinct micro4,
  T7 and T30 inputs and the original loss mix, leaf slicer, clipping, learning
  rate and AdamW. Neural FP32 and physical FP64 remain unchanged.
- Use the **original evaluator** to score every arm at updates0,4,8,16.
  Evaluate the identical initial state once; later milestones evaluate three
  distinct trajectories. This separates update effects from scoring-path
  changes. Two reference trajectories show bounded original repeat variation;
  their drift does not waive candidate strict parity.
- Exactly480 update model calls plus100 monitor calls:580 total. Load the
  five existing model executables; any new model compilation stops. The
  unchanged official slicing/norm/AdamW helpers may compile normally.
- Save complete parameters, AdamW, mixed gradients, all input-state gradients,
  losses/auxiliaries and physical evidence at updates1,4,8,16 for all arms
  (12 consolidated snapshots). Preserve ten raw monitor records and all48
  update rows. About12.5GB raw snapshots; require20GiB free before execution.
  No one-file-per-variable/day layout is introduced.

Scripts: `scripts/hpc/probe_shared_v4_learning_sensitivity.py` and
`run_shared_v4_learning_sensitivity.py/.sh`. `--dry-run`, `--protocol-check`
and CPU `--metadata-check` do not execute a model or optimizer. Both launcher
and probe reject learning execution without `--execute-approved-diagnostic`;
that flag is now authorized for this one1800s execution.

## Measurements and interpretation

Retain the unchanged strict comparison at the first common-input update for
loss/all auxiliaries, all parameter gradients, actual parameters, full AdamW,
norm/clip and all state derivatives. Subsequent arm differences are trajectory
differences, not same-input AD comparisons. Monitor joint/daily/T7/T30 losses
and the public flux/stock/living/nonliving terms through the common original
evaluator. Report each arm's change from initialization, candidate differences
from both references, and reference-repeat differences without inventing a
new scientific pass threshold. Actual state, finite, source-defined zero,
physical and24GiB memory gates remain hard stops. Strict parity failure is
preserved and does not terminate the proposed learning observation if hard
conditions pass; this continuation itself is the requested scientific choice.

The result can show observable short-run learning sensitivity at this fixed
diagnostic initialization and data. It cannot establish generalization,
multi-year accuracy, long-run stability or universal absence of learning
damage. A comparable16-step outcome would not authorize adoption or relax
1e-5/1e-6. A larger candidate trajectory change would motivate rejection or
a specific new investigation rather than automatic tolerance change.

## Resource budget and exact command

One free gln01 V100 GPU1, OMP4/OpenBLAS1,24GiB active-memory ceiling. Inspect
occupancy immediately before execution. One1800s cumulative window including
entry, setup, imports, ordinary pre-model repairs, loading, all calls, saving
and up to10s kill grace; maximum0.5 free card-hours, CNY0, no paid fallback.

Budget basis: the completed saved-binary diagnostic loads in382.39s. Reserve
420s for that stage,48 updates×14s=672s,10 monitor evaluations×14s=140s,
200s for transfers/consolidated snapshots and180s for new helper/hard-check
overhead:1612s, leaving188s including kill/exit allowance. The14s planning
rate is above the measured13.103s warm original update and11.186s candidate.
This is a planning allowance, not a promised runtime. If loading/IO consumes
the window, save partial evidence and stop without extending it or calling
the candidate ineffective. No new model lowering/backend time is budgeted.

Working directory:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1`

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-v4-learning-sensitivity-20261004/run_shared_v4_learning_sensitivity.sh --gpu 1 --execute-approved-diagnostic
```

The shell records the original time before Python entry. Launcher records
real setup/phase exit status and elapsed time, enforces a derived remaining
timeout, and rejects a paid Slurm environment. A pre-model ordinary repair
may use `--resume` only within the original deadline, preserving failed logs
and source. Any started model call prohibits a trajectory reset or duplicate
learning run. No automatic retry, extra model call or new budget follows.

## Preparation evidence

Local ORCJAX syntax and fake-state protocol checks pass:48 fake callbacks,
ten monitors,12 snapshots and independent final trajectories; zero real
models/optimizers. On test01, target `bash -n` and actual shell
`--gpu 1 --dry-run` expansion pass. The prepared `--resume --dry-run` branch
also expands on the target shell without execution. Test02 CPU metadata checks bind the saved
original/candidate reports, five binary paths, actual imported learning
overlay, artifact742 and step2 initialization with zero executable loads,
model/GPU/optimizer calls or Teacher reads. Records are in
`outputs/diagnostics/shared-v4-learning-sensitivity-preparation-20261004/`.
Independent source review found three preparation gaps: failure arrays could
be lost before the shared update returned, derived comparisons were deferred
until the end, and reused preflight JAX version was not checked. All three
are repaired and independently accepted in `independent-source-review.md`.
Finished monitor raw is saved before a memory stop; available failed-update
operands are preserved before rethrow; first-step strict and completed
monitor effects are written incrementally. Post-repair syntax/protocol and
target argument checks pass. Execution and saved-array acceptance are complete;
actual results, preservation limits and accounting are in the linked results
report. This completed allowance authorizes no further learning calls.
