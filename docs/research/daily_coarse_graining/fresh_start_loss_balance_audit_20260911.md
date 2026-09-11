# Fresh-start loss localization before any new training

The cb3c454 corrected run completed all 12 full-network updates. Its initial
loss is 22.951749, with 0.1*state_loss contributing 99.97534%. Initial sample-0
combined gradient norm is 758.4517; state heads alone have norm about 758.45.
Clipping at 1 occurs on all 12 updates. Loss magnitude alone does not establish
gradient dominance; the parameter-family evidence motivates this localization.

Freeze the same seed 20260911, architecture, DOC input rule, three train
samples, scales and original objective. No optimizer updates, validation,
Teacher regeneration or new paid work. The existing update-12 checkpoint may
be read only to explain the completed prototype; it is not a formal starting
weight source. Matched numerical references are the initial and final values
already recorded in `fresh-start-one-day-v2.json`.

Add default-off returned-state diagnostics to the complete objective. For each
sample at initialization and update 12, reproduce the recorded loss/hard counts
and report per-state-group MSE/bias/count, per-coordinate predicted/Teacher/
scale/mask values, and canonical named-field attribution. Recompute group loss
from the exported arrays on CPU and require agreement with the actual loss.
Attribute signed-bias contributions using residual_i * group_mean_residual;
these can be negative and must not be misreported as squared-error shares.
Report repeated physical fields across grouped views explicitly. Grouped axes
must be mapped using the actual assembler layout, not guessed flat offsets.

Reuse the ordinary full-gradient evaluator for the initial first sample and
rank every state-output head by gradient norm, attaching its learned field
name. Verify its loss and global norm reproduce the recorded initial probe.
Do not change loss weights, clipping or initialization inside this audit.
The audit exports a small state-array artifact for subsequent CPU analysis,
not a new parameter checkpoint. Initial/final forward output agreement uses
atol=1e-6, rtol=1e-5; the gradient norm comparison uses rtol=1e-4.

First run local attribution tests and CPU abstract tracing of the diagnostic
output. Then one free GPU for at most 45 minutes, sharing the existing compile
cache. Once a concrete dominant field/mechanism is identified, write the
resulting normalization/initialization/loss amendment before training; do not
blindly reduce the entire state coefficient or equate frequent clipping with
proof of an incorrect clipping threshold. Resource and sample expansion remain
part of the subsequent reproducible training plan.

## Implementation and execution record

`fresh_start_training.make_objective(..., state_diagnostics=True)` returns
grouped predicted/expected states, scales, eligibility and exact original
group statistics. Default false leaves ordinary training outputs unchanged.
`state_loss_attribution.py` maps the actual canonical inventory/thermal layouts
and reconstructs loss including signed-bias cross terms. Three focused tests
pass in 12.29s (attribution, actual group layout, existing physical extractor).
Ruff and shell syntax pass.

Free test02 abstract tracing of the complete diagnostic objective passed in
201.1s, report `runtime/outputs/diagnostics/fresh-start-loss-balance-shapes.json`
(also copied to local outputs). No new model weights were trained.

The GPU diagnostic started around 19:36 on 2026-09-11, free gln01 GPU 1,
source **9a6edb6**, Python PID **49644**. The GPU was idle (3 MiB, 0%); GPU 0
was occupied and left untouched. One V100, 45-minute cap, same compile cache
as the completed prototype. Working directory:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`.

Command:
`GPU_DEVICE=1 PYTHON_MODULE=scripts.hpc.audit_fresh_start_loss_balance RUN_TIMEOUT=45m OUTPUT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/fresh-start-loss-balance-v1.json bash scripts/hpc/run_fresh_start_one_day.sh`.

Report: `runtime/outputs/diagnostics/fresh-start-loss-balance-v1.json`.
Raw state export: same prefix with `.states.pkl` suffix (no parameters).
Log: `runtime/logs/fresh-start-loss-balance-v1.log`.
The diagnostic completed in 572.14s with no parameter updates or paid work.
All six initial/update-12 forward comparisons reproduce the completed probe,
all hard counts are zero, and the original first-sample gradient norm agrees.

## Completed localization

The dominant state field is `diffuco_previous_step_state.gsmean`, specifically
global coordinate 10, compact bare-soil slot 0 (source PFT1). The neighboring
coordinate 11 is PFT14. Initial bare-soil predictions are 2769.587, 2767.579
and 2768.925, with Teacher zero and state-loss scale 1.0. The neighboring
vegetation coordinate has scale 360508.623. Gsmean supplies 99.75-99.77% of the
entire weighted objective and 99.79% of the unweighted grouped state loss.
Its 513-parameter output head has gradient norm 758.44758, versus the full
758.45172: 99.99891% of the squared norm.

After 12 updates the erroneous bare-soil predictions are still about 2656-2659.
Most reported loss decrease was this output moving toward zero. The next
largest fields are fuel_1hr, dead_leaves and fuel_1000hr, also with suspicious
bare-soil predictions, but their reset/carry semantics need source auditing
before deciding any constraint. This is output applicability, not evidence
to delete the state objective or to further compress the DOC input.

`diffuco.f90::diffuco_trans_co2` initializes `gsmean(:,:)=zero` at line 2289,
loops over vegetation `jv=2,nvm` at 2299 and assigns `gsmean` within that loop
at 2885. The canonical catalog identifies compact source index0 as bare soil.
Source-backed applicability must use that identity/trait rather than a fixed
array position. Separate source and normalization reviews are being prepared.

Root's offline accounting probe sets only exported state index10 to zero,
recomputes MSE and the group signed-bias cross term, and leaves all other
coordinates/scales unchanged. Initial state losses become 0.48457, 0.48618,
0.47089 versus 229.648, 229.308, 229.427. This is an accounting counterfactual,
not a forward acceptance result. Files:
`scripts/dev/analyze_gsmean_zero_counterfactual.py`,
`outputs/gsmean-zero-counterfactual.json` and the copied diagnostic states.
Next implement source-backed bare-soil gsmean zero and rerun the unchanged
fresh-start learning protocol with explicit applicability enabled.
