# Shared decoder execution: memory passes, admission incomplete

Code094dfd2, free gln01 GPU0. Wrapper10651/Python11462 have exited.
The shell log ends with `run_exit_status=124`: the60min main-test limit
interrupted directional-JVP compilation after a full reverse gradient had
completed and been saved. The report JSON still says running/directional_jvp
because external timeout bypassed its normal final emit; that is stale state.
No optimizer update or scientific acceptance was completed. No paid job ran.

## What the execution optimization achieved

The original and optimized architectures retain75,578,497 trainable parameters,
161 fields and39006 output coordinates. Static PFT gather computes8550 eligible
coordinates; K/V reuse preserves capacity and differentiability. Decoder
rematerialization is disabled.

| Compiled one-day gradient | Original4eeb168 | Optimized094dfd2 |
| --- | ---: | ---: |
| Estimated live memory | 36.1747GiB | 6.67245GiB |
| Numerical reverse gradient executed | No | Yes |

Estimated live memory falls81.5549%. Optimized temporary bytes are6559419040,
arguments302755227, outputs302318880, no aliases. This is compiler memory
analysis, not a measured reverse-pass allocator peak. The successful execution
does establish that this one-day gradient fits the V100. T7/T30 joint execution
is still untested; do not infer that all recurrent training fits or add decoder
rematerialization without relevant memory evidence.

Both initial canonical forwards pass all hard/physical counters. Compared with
the saved original initial reports, process-label, state-anchor, flux and stock
loss relative differences are at most1.83e-7 across the same two TRAIN rows.
These are forward-loss comparisons, not a full original-gradient reference.

All75,578,497 gradient values are finite. Both tokenizer and shared-Transformer
groups have nonzero norm; total L2 norm8.1859734e9,295/298 nonzero leaves.
Finite gradients do not certify correctness: JVP/reverse agreement remains
incomplete. The final two-update/checkpoint-replay stages did not run.

Main-test stages: initial forwards695.2s; gradient compilation3260.4s;
gradient completion3265.8s; gradient saved3266.6s; JVP started3274.6s.
Thus only~325s of the declared hour remained for JVP and updates. The budget
was insufficient for this cold compilation sequence; the stop was neither an
OOM nor a numerical-gradient failure. Do not present the~5.4s gradient stage
as a warm production update benchmark. Separate environment check passed70.95s.

## State-anchor attribution

Host decomposition agrees with the unchanged objective on both rows.
The `ptn` and `tdeep` field contributions together account for98.95% of the
state anchor. Remaining large contributions include both `cgrnd` records.
This identifies the dominant thermal path; it does not yet prove an erroneous
normalizer or justify removing these states from supervision.

For row819 (307.0-095.0,1961,start86), one deep-soil `ptn` coordinate predicts
389.1448K versus301.7850K, with daily-change scale0.000118618K, giving normalized
error736481.4. Row982 predicts380.2010K versus299.9147K at the same coordinate.
There is a substantial physical prediction error as well as small-scale
amplification. The corresponding `tdeep` report repeats the same endpoint
values; audit canonical thermal ownership, decoder initialization/transforms,
and state-loss accounting before changing losses. Carbon-flux/stock objectives
are overwhelmed in the total scalar, so their directional evidence should be
examined separately when qualifying a revised objective.

## Next work and retained artifacts

1. Reuse saved fresh parameters/gradient and successful compilation cache to
   finish bounded JVP/reverse and two-update checks. Prepare a resume path
   before launching; do not repeat the whole completed initial compilation.
2. Trace the dominant thermal outputs and their day-change scaling/duplicate
   ownership before selecting a correction. Do not merely lower a weight to
   conceal the80-87K error, or reduce auxiliary decoder capacity.
3. Then qualify the effective joint T7/T30 objective and execution before a
   bounded learning contract; no paid submission is authorized here.

Server prefix: `runtime/outputs/diagnostics/shared-decoder-094dfd2-gpu`.
Saved `.gradient.pkl` is604649034bytes and contains fresh parameters, the
completed full gradient, value and protocol/source identity. Keep it remote.
Local evidence: `outputs/shared-decoder-094dfd2-gpu.json` and `.log`.
