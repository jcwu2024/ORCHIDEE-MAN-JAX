# Two bounded routes before long training

User authorized implementation and free-node tests on2026-09-15. The prepared
5415-update/24h job is paused before submission. No paid job is authorized by
this protocol. Model architecture, Teacher release and derivative mathematics
remain the admitted implementation.

## Route1: two-device execution of the same update

At the fixed trained200 checkpoint, compare serial and concurrent computation
of the same64 daily samples, one T7 window and one T30 window. Each component
uses identical pre-update weights. GPU0 computes D64/micro8; GPU1 computes T7
and T30 sequentially. Transfer both recurrent gradients to GPU0 and combine in
the original order, D+.5T7+.5T30. Clip once, update AdamW once, and transfer the
updated weights back to GPU1. This final synchronization is part of timing.
No model partitioning, new AD implementation, enlarged batch or retained
training updates. Reuse the existing checked-gradient and physical counters.

First verify the orchestration on small two-device inputs. Then test the real
same-input complete gradient, optimizer proposal and physical counters against
serial execution, using existing comparison tolerances. Preserve a completed
serial reference before another compilation. Measure both branch durations,
end-to-end time including reads/checks/communication, live memory per GPU,
startup and cache costs separately. Interleave3-5 warmed serial/dual repeats.
Warm fixed-data throughput does not certify rotating-data throughput or training.

Report speedup S and GPU-hour ratio2/S. With8CPU on either run and additive
rates, total cost ratio=(4.40+.56)/(2.20+.56)/S=1.7971/S. If later CPU allocations
differ, recalculate. No arbitrary numerical-pass-to-paid-adoption shortcut:
present actual time saved and incremental cost. Poor balance may make dual GPU
unattractive. Free gln01 tests need both GPUs idle; no paid submission here.

## Route2: single-device staged learning versus joint learning

Both arms freshly initialize all109,464,375 parameters at seed20260911 and use
the admitted explicit Attention FP32/checked execution. Same daily64/micro8,
train pool, fixed sampling seeds, objective components, physical constraints,
development panels and AdamW lr1e-5/clip1. No exploratory weight inheritance.

Joint arm: D+.5R7+.5R30 on every update. Staged arm:400 D-only updates, then400
mixed updates. Daily-only must actually skip recurrent gradients, not multiply
an expensive computed gradient by zero. Both stages update all parameters;
preserve AdamW moments and global step at the transition. Consume daily batches
continuously; recurrent cursor advances only on mixed updates and starts at0
in the mixed stage. Record phases, sample counts, cursors and cumulative update
time; recovery must continue the declared phase and cursor exactly.

The initial cost estimate suggests roughly600 joint updates versus400+400
staged. Before paid learning, use bounded complete daily/mixed timings (not
learning outcomes) to fix the joint count:
N_joint=round_to_nearest10(400*(t_daily/t_mixed)+400).
Freeze both executable schedules before either learning arm. This is nominal
compute matching, not identical data exposure or update count; report actual
wall/update time and compare learning curves at common compute milestones.
Compilation and evaluation are recorded separately but still count in bills.

Do not add LR changes, freezing or alternative architectures to this first
comparison. A separate1daily+1mixed integration identity tests the transition,
physical/finite outcomes and checkpoint replay without entering either arm.
The existing native-AD evidence certifies the same objective components;
the new work must test composition/phase/optimizer behavior, not repeat a long
standalone adjoint study.

Report fixed-development physical-unit GPP,NPP,RA,RH,AGB,BGB and canopy/nonliving
carbon through T7/T30, daily TRAIN/excluded loss, full metric populations and
physical gates. Select each declared final checkpoint; show all milestones.
Scientific accuracy failure is an outcome, not a technical reason to abort.
This pilot decides whether staging merits expansion, not final-model acceptance.
Do not interpret a tiny difference or one seed/panel as decisive. Freeze the
paid resource proposal after free integration and timing; no long job now.

## Existing preparation reused

Epoch integration atafb7b5f completed1270.8s, one full fresh mixed update,
initial/final monitoring and checkpoint replay, wrapper exit0. Its weights
remain integration-only. Reports are under runtime diagnostics and locally at
`outputs/mixed-learning-epoch-v1-integration.json`. The existing5415 schedule
and346539-day pool supply both pilots; no Teacher regeneration or new sample
files are needed.

## Implementation and running checks

Implemented e5878fb/dea4083. Sol implemented the staged runner and recovery
extension; Terra reviewed dual-device placement and gradient lifetimes. Main
agent integrated the code and source review.37 focused stage/state/runner/
profiling/dual tests passed, Ruff and diff checks passed. Shell syntax and
actual target-shell argument expansion passed for dual profiling, stage
integration and both paid-pilot arms (dry-run only).

Staged checkpoints derive their allowed update range from the declared method
and validate the independent recurrent cursor. This also removes the historical
200-update checkpoint ceiling for the prepared5415 route. Tests actually save
and load a2500-update checkpoint and stage-boundary checkpoints. The accepted
one-update epoch integration alone did not exercise that historical ceiling.

Actual JAX0.4.38/CUDA preflight passed27.43s; dual-device tiny two-update check
passed2.04s. AOT per-device placement variant also passed1.97s atdea4083, zero
serial/dual parameter/optimizer discrepancies. No Teacher payload reads in
these runtime checks. They do not certify real-model speed or numerical parity.

Real-model dual profile is RUNNING on free gln01 GPU0+GPU1 at immutabledea4083,
Bash7920/Python7963,90min ceiling. Checkout:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/mixed-learning-parallel-routes-v1`.
Log/report: runtime diagnostics `mixed-learning-dual-profile-v1.{log,json}`.
Next collect full-gradient/AdamW/physical parity and warmed serial/dual/daily-only
timings. Then run the independent stage integration on GPU0 after the two-GPU
test releases resources; do not change a running checkout. Free integration:
`GPU_DEVICE=0 CODE_ROOT=<checkout> bash scripts/hpc/run_mixed_learning_stages.sh integration`.

Provisional paid pilot script `job_mixed_learning_stages_pilot.sh` runs joint
then staged on1V100/8CPU/64GiB with8h cap, maximum22.08yuan under additive
rates. Each arm has220min ceiling; no automatic requeue. Create its Slurm log
directory before submission. It is NOT approved, submitted or ready for paid
adoption until the free phase-transition check passes and the nominal matched
joint count is frozen from timing. The24h epoch job remains paused.
