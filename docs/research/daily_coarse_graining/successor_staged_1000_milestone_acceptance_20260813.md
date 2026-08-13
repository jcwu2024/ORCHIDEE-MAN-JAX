# Successor Staged 1000-Update Milestone Acceptance

Status: **accepted second staged milestone; continue-to-2000 is eligible but
not yet authorized**.

## Decision

The matched staged-v3 pair resumed the accepted update-500 recovery, completed
updates `501-1000`, and re-evaluated the unchanged frozen 48-window validation
panel. Both arms pass the predeclared update-1000 gate: each improves its own
update-500 primary composite by more than 1% and has zero validation hard
failures.

This is the first evidence within the staged regime that exposing the models
to more distinct training point-years improves held-out 7/30-day behavior. It
supports continued bounded training through update 2000. It does not promote
either arm, establish convergence, or open the 365-day, long-held-point, Gate
D2, sealed-test, or product gates.

## Incremental Prepared Input

The accepted incremental product is under
`runtime/outputs/training/successor-staged-prepared-0501-1000-ea21f8a/`:

- updates `501-1000`, exactly 500 gzip artifacts;
- total artifact bytes: `1,660,059,042`;
- manifest SHA256:
  `fe39325a3c18d6a7240e0fe5c2c23c14324b5abd9a88377a1ba380112196f076`;
- producer commit: `ea21f8aa67c00e9c70ec5fede617aab946af694c`;
- full schedule SHA256:
  `8b14d538516dbad953a30e0be2928d2b1dd2474adf7bf940f4707d177b0d2626`.

The first approved preparation attempt, `cnmix` job `14527233`, reached its
four-hour limit after writing 56 resumable artifacts. Several ordinarily
minute-scale point-years were delayed for tens of minutes. Free-node re-runs
of updates 501 and 549 completed their core preparation in 90 and 85 seconds
and reproduced the paid-run artifacts bit for bit, attributing the timeout to
execution performance rather than data semantics. Approved `cnall` job
`14532991` then reused the 56 artifacts and completed the remaining work plus
finalization in `01:47:12` on four one-core workers.

A separate free `test01` process rebuilt the frozen 10,000-update schedule,
checked exact `501-1000` coverage and policy flags, recomputed every artifact
hash, and decompressed and deserialized all 500 artifacts. No Teacher data
were regenerated, and no historical-v5 or sealed-test data were used.

## Training Execution

Paid `gnall` job `14535746` completed with exit code `0:0` in `00:52:54`
using four CPUs and one V100. The accepted products are under
`runtime/outputs/training/successor-staged-fit-1000-ea21f8a/`:

- stage-report SHA256:
  `e379e3e700ebc93b558ddefa054af36854470b002a8224bf261211aebbd68ce1`;
- update-1000 recovery SHA256:
  `d49222d4f9f64941e555d1c8b0536c785cc89942d59fb41b3a5d5ea291a7cae7`;
- best-checkpoint SHA256:
  `8a3967e3d22d7e5db341b43f79dcda1cce0ffb49bcff7b02e231333308f50d0a`;
- maximum host RSS: `9,457,532` KiB;
- parameter count: `8,722,689` per arm.

All 500 incremental updates were applied to both arms. Every recorded loss,
component, gradient norm, and parameter value is finite. All training and
validation nonfinite, mask, discrete-state, negative-stock, and budget hard
counts are zero. A separate `test01` process loaded the recovery through the
production loader, loaded both best parameter trees, verified update
`501-1000` coverage and 7,000 recorded scalar values, and confirmed both best
milestones are 1000.

Pre-clip gradient norm exceeded the threshold on 290/500 incremental attention
updates and 402/500 incremental control updates. This is lower than 482/500
and 474/500 in the first stage, especially for attention, but large gradient
spikes remain. The result supports an improving optimization trajectory; it
does not establish convergence or validate the clip threshold.

## Validation Result

The aggregate primary composites are:

| Arm | Update 500 | Update 1000 | Relative improvement |
| --- | ---: | ---: | ---: |
| Graph-MLP control | 1.0457081708 | 1.0060782200 | 3.79% |
| Process attention | 1.0523784009 | 1.0145714634 | 3.59% |

All six temporal/spatial/joint by 7/30-day composites improve for each arm.
Control improvements range from 2.42% to 13.43%; attention improvements range
from 1.98% to 13.79%. Thus the aggregate improvement is not hiding a regressed
validation cell.

Control remains 0.84% lower (better) in the aggregate and is lower in five of
six cells. Attention is only 0.02% lower in the 30-day temporal cell. This is
retained as evidence, not used as an arm-selection gate. Attention has
slightly lower GPP error in most cells but generally higher NPP and live-stock
errors; process-axis attention has not demonstrated a decisive benefit.

The scientific objective remains open. In the 30-day temporal cell,
normalized GPP/NPP RMSE remains approximately `2.27/2.34` for control and
`2.27/2.36` for attention. Total-live-biomass RMSE remains much smaller at
approximately `0.0611/0.0615`. More data have improved the fixed-panel score,
but productivity is still the dominant error and neither arm is ready for
long-horizon acceptance.

## Cost And Next Gate

Approximate observed additive cost for this interval is:

- timed-out preparation: `4 * 4.004 h * CNY 0.07 = CNY 1.12`;
- successful continuation: `4 * 1.787 h * CNY 0.07 = CNY 0.50`;
- GPU training: `4 * 0.882 h * CNY 0.07 + 1 * 0.882 h * CNY 2.20 = CNY 2.19`;
- total including the infrastructure timeout: approximately `CNY 3.81`.

The contract makes `1001-2000` the next eligible stage, using 1,000 new
point-years and the accepted update-1000 recovery. At update 2000 each arm is
again compared with its own current best composite and must improve by at
least 1% with zero hard failures. Preparing or submitting that continuation
requires a new resource estimate and explicit approval. No architecture,
objective, coefficient, validation-panel, or arm-selection change is
authorized by this acceptance.
