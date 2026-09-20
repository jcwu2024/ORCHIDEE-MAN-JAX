# Focused physical litter-change admission

Status: protocol fixed before learning on 2026-09-20; implementation complete,
46 focused local tests and Ruff pass. Actual GPU runtime interface check
passes in23.01s. Free real-model admission RUNNING at22e7fb9 since12:58:13 CST,
GPU0/gln01, Python89790, timeout89756, wrapper89740;60min ceiling (~13:58).
Report/log `runtime/outputs/diagnostics/litter-change-admission-v1.{json,log}`.
No new paid training is authorized. Launch is not acceptance.

## Mechanism and scope

The completed 16-input update4000 endpoint probe identifies excessive learned
litter outgoing amounts, not a retained-tail loss, as the dominant one-day
litter error. The first intervention adds a direct loss on each of aboveground
and belowground litter carbon change. Attention, decoder, parameter count,
teacher data, train splits, optimizer and existing recurrent losses stay fixed.
This tests a supervision gap; it does not presume that the decoder is sufficient
or that all T30 carbon/GPP error has the same cause.

For each owner, select ICARBON, sum litter types/soil layers and weight active
PFTs by land fraction. Compare predicted and Teacher next-day changes from the
same Teacher initial state. Do not let opposite above/below errors cancel.
Use existing defined masks, report missing/nonfinite active coordinates, and
keep PFT-fraction derivatives. This is stock supervision, not an assertion that
endpoint outgoing bookkeeping is respiration or ecosystem carbon closure.

New daily loss is D_new = D_old + 0.1 L_litter. For each of two owners,
r = (predicted_change - Teacher_change)/s and h(r)=sqrt(1+r*r)-1,
implemented in its numerically stable pseudo-Huber form. L_litter is the mean
of the two owner penalties. The linear large-error regime prevents squaring
the very large initial daily carbon error. No new loss enters R7 or R30 in
this candidate. The later mixed objective remains D_new + 0.5 R7 + 0.5 R30.

## Frozen scales

Use existing full TRAIN-only statistics, not development outputs. Dataset
`pft14-daily-teacher-669-1961-2010`, spatial/temporal split train/train,
8,591,565 observations per included carbon coordinate. Source metadata:
`runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json`
and its existing `training_statistics.npz`.

For source PFT index13, s=max(sqrt(sum(coordinate state_delta variances)),0.01)
in gC m-2 day-1, separately by owner. This coordinate-variance scale does not
claim to be the measured aggregate standard deviation: covariance is absent
from the stored statistics. It is deterministic and independent of this
model's errors. The 0.01 physical floor avoids an arbitrarily amplified loss
for near-constant owners and is inactive for both current owners.

| Owner | Coordinates | Sum of variances | Effective scale |
|---|---:|---:|---:|
| Aboveground | 2 | 0.009859231592171273 | 0.09929366340392155 |
| Belowground | 64 | 0.0035517143253187044 | approximately 0.0595963 |

All coordinates have complete TRAIN counts. Crucially, 42 belowground
coordinates have zero variance but the general normalizer stores scale=1
for them. RSS of those general-purpose scales would give 6.481 and strongly
underweight this owner. Use the original variance, not normalizer scale.

## Matched bounded admission

Executable declaration: `manifests/coarse_graining/litter_change_admission_v1.json`.
Load exact parameters AND AdamW state from original mixed-learning update4000,
retaining original checkpoint identity validation. This is a diagnostic fork;
it does not resume job15279139 or become the final paper training recipe.

Both arms start from those same saved arrays and take two full-parameter
DAILY-ONLY AdamW updates, learning rate1e-5 and clip1 unchanged. Use original
TRAIN schedule batch indices4000 and4001, batch64/micro8. The only differing
coefficient is new litter loss0 versus0.1. Four dynamic coefficients represent
base/canopy/litterpart/litter, with arrays [1,.1,.1,0] and [1,.1,.1,.1].
These are NOT full joint D/R7/R30 updates: this bounded test isolates the
changed daily term before paying for a larger joint experiment.

Before updates, check the actual deployed JAX/CUDA runtime once, then finite
forward/physical counters, finite nonzero new-term gradient, and native reverse
versus JVP for the new complete daily scalar on the first microbatch. Two
declared seeds20260920/20260921; existing rtol0.01, atol1e-6, with projection
magnitude>1e-5. Do not loosen tolerance or count unresolved zero projections
as useful evidence. Preserve partial reports if a check fails.

Initial/final fixed TRAIN and excluded-TRAIN panels each use the first8
previously declared rows, with separate owner bias/RMSE and GPP/NPP/RA in
physical units. These are reused development inputs, not independent final
validation. Record actual update losses, gradient norm/clipping, hard counts,
memory/runtime and completed-update count. Finite full parameter/optimizer
states and zero physical hard counters are required to admit the daily path.
Any carbon improvement after two updates is only a local direction check.
Report lack of improvement or productivity tradeoffs without redefining success.

Use one idle free GPU with a60min ceiling initially. No paid job or automatic
retry/extension. If daily admission passes, next freeze a short matched joint
learning experiment and evaluate unchanged7/30-day development panels; only
then consider wider data/longer learning. This document does not admit365-day
or final-product use. The reproducible final method still starts from a
declared fresh initialization with all neural parameters trainable.
