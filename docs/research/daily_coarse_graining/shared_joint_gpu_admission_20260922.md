# Shared daily/T7/T30 numerical admission

Status: wrapper7359 COMPLETED0; see `shared_joint_results_20260922.md`.
The launch snapshots below are historical. Authorized by the user's
2026-09-22 continuation. No paid task or scientific acceptance.

Daily-v3 numerical admission100848 and recurrent CPU preparation45145 both
passed. Reuse those reports and existing TRAIN calibration; see
`shared_v3_admission_results_20260922.md`. Do not repeat their shape suite or
read/rebuild the complete Teacher release. Architecture, outputs, loss
weights and75,578,497 trainable parameters remain fixed.

## Bounded execution

- Free gln01, one idle V10032GiB (GPU1 idle at initial inspection), four CPU
  threads. Recheck occupancy immediately before launch.
- Main timeout240min, plus a separate5min deployed synthetic runtime check.
  This is a two-update technical test, not a four-hour learning experiment.
- The previous daily graph needed39.63min to compile its reverse and20.02min
  for first JVP. This test introduces joint-daily/T7/T30 reverse graphs and
  recurrent directional graphs; cold compilation motivates the upper bound.
  Actual runtime is unknown. Reuse the persistent shared-transformer-v1 cache,
  without claiming every graph is a cache hit.
- Separate immutable checkout, all outputs under project runtime/outputs.
  Save completed expensive stages before subsequent compilation. A timeout
  is a resource interruption, not a numerical rejection.

Protocol: `manifests/coarse_graining/shared_recurrent_admission_v1.json`.
Fresh daily-v3 seed initialization; do not inherit the two-update daily weights.
One daily row and window0 at both horizons from artifact819 form update1;
artifact982 supplies update2. Four window identities are listed in the
September22 result report. These TRAIN samples do not test generalization.

The joint loss is exactly:

```
process_day/3 + (daily_flux + daily_stock)/9
              + (T7_flux + T7_stock)/9
              + (T30_flux + T30_stock)/9
```

Evaluate the three component gradients at identical parameters, sum once,
then clip and perform one full-network AdamW update using daily-v3 optimizer
settings. Sequential evaluation limits memory; it does not make three
successive optimizer updates. No new handwritten reverse or cotangents.
Native JAX checkpointed daily scan differentiates the recurrent components.

## Order and interpretation

1. Cheap report/config prerequisites on free CPU and actual shell argument
   expansion; local toy orchestration tests. No bulk hashes or new calibration.
2. Fresh numerical initial daily and all four recurrent forwards. Require
   finite losses and all declared physical/science hard counters zero before
   compiling reverse graphs. Initial prediction accuracy is not a hard gate.
3. Compile and measure each reverse graph before execution (24GiB ceiling);
   check finite gradients and compare AD primals to corresponding forwards.
   Primal comparison uses rtol2e-5/atol2e-6 for the FP32 neural execution;
   directional comparison retains the frozen rtol0.01/atol1e-6. Report actual
   differences rather than only the pass flag; do not tune these after a failure.
   Preserve completed initial gradients. Check native JVP/reverse directional
   agreement for T7/T30. Group participation is judged on the combined
   gradient, not each component's unused output heads.
4. Two complete joint updates, post-update physical/finite checks, diagnostic
   checkpoint and final cross-sample replay. Record group losses and memory;
   do not require all groups to improve after each update.

A pass permits preparation of bounded learning with independent development
panels. It does not admit365-day/long-term accuracy or make these technical
checkpoints the formal initialization. A failure should identify physical,
numerical, execution or resource cause; do not automatically change losses,
backbone, tolerances or Teacher data.

## Local integration evidence

The new runner's12 tests pass, including two current-parameter AdamW updates,
physical and nonfinite-gradient rejection, metadata prerequisites, and recovery
after saving the first component. Existing recurrent/preparation12 and
daily/trajectory14 tests also pass (38 unique tests total). Ruff, CLI help
and shell syntax pass. A Sol/high worker implemented the runner and focused
tests; parent reviewed the integration and owns launch/acceptance. Forward and
reverse executables remain cached across updates; no per-component cache purge.
Actual deployment prerequisite validation passed on free test02 in0.1s,
as did CPU/GPU target-shell argument expansion. The full-model GPU result
remains pending.

## Launch

Code c2498e4, independent checkout`runtime/checkouts/shared-joint-c2498e4-full`.
Wrapper7359 on free gln01 GPU1; rechecked idle immediately before launch.
Command: `MODE=gpu GPU_DEVICE=1 RUN_TIMEOUT=240m` with this checkout's
`scripts/hpc/run_shared_recurrent_admission.sh` and output below.
Report/log prefix`runtime/outputs/diagnostics/shared-joint-c2498e4-gpu`.
Prerequisite evidence:`outputs/shared-joint-c2498e4-prerequisites.json`.
No paid job was submitted. The separate deployed GPU runtime check passed
in36.10s. At wrapper elapsed1m06s, the main runner had passed prerequisites
and was loading existing TRAIN artifacts. Full-model numerical results are
still pending.

Deployment note: cloning another shared checkout hit the old server Git's
alternate-object-store nesting limit before any computation. A9.44MB full
single-branch bundle and independent clone resolved it. The incomplete
`shared-joint-c2498e4` directory is not active; no existing checkout/data was
removed. Avoid further chains of shared clones; use the independent checkout
or a complete bundle as the source for later immutable checkouts.
