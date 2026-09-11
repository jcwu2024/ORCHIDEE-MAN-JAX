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
