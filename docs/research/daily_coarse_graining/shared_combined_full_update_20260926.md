# Combined saved execution: complete-update admission

Historical result (2026-09-26): completed0/passed, free gln01 launcher41280. Local11tests and actual test02 CPU inspection passed.
Actual CPU/GPU shell dry-runs passed; both free GPUs idle at13:23CST before
dispatch. No paid job or formal learning. Daily, T7 and T30 components have passed unchanged loss/full
parameter-gradient/physical checks. This next gate measures whole updates.
Free gln01 launcher41280 started approximately13:24CST September26,
GPU0 parent/GPU1 recurrent worker,30minute ceiling. Main probe completed
649.48s (10.82min); no model-gradient compilation.

## Completed result — 2026-09-26 historical scope

Three interleaved complete fixed-weight proposals per mode, identical
daily32+T7+T30 workload, including IPC/checks/AdamW:

| Execution | Previous original run | Combined execution | Historical time reduction |
| --- | ---: | ---: | ---: |
| One V100 | 14.851248s | 12.553775s | 15.47% |
| Two V100s | 11.192650s | 10.270496s | 8.24% |

Original-vs-combined columns are different runs. Within the combined run,
dual gives1.222314x throughput /18.19%less wall time than serial, but uses
1.636240x GPU-seconds per update. Under the existing illustrative8CPU,
0.07yuan/core-hour and2.20yuan/GPU-hour accounting, equal-work charge ratio
is1.470245. This free test incurred no paid charge. Three repetitions are
bounded measurements, not confidence intervals or sustained-training rates.

Both changing updates pass64,884,481-coordinate mixed-gradient/parameter
and129,768,963-coordinate AdamW serial/dual comparisons at unchanged
rtol1e-5/atol1e-6. Largest serial/dual gradient difference5.960464478e-8;
largest updated-parameter difference1.192092896e-7. Final checkpoint vs
original also passes: maximum parameter difference9.685754776e-8 and full
AdamW difference1.396983862e-9. All48technical loss/scalar coordinates pass.
Physical checks pass. No tolerance or method changes, no formal weights.

Graphs ready406.2s after the main timer; parent daily load~12.1s and recurrent
loads~93.9/95.9s, worker recurrent loads98.49/99.40s. These are deserialization,
not gradient compilation. Parent peak live allocations14.998GiB while JAX
pool25.391GiB: the pool is not actual live memory; no worker peak captured.

The second changing dual update records daily4.326s, recurrentGPU8.732s,
branches including IPC10.842s and final merge/check/AdamW0.136s. This points
to the recurrent branch and transport as the remaining dual critical path;
these are one changing update's measurements, not the fixed-proposal medians.
Do not label the whole difference as pure transfer without finer evidence.
Update-report outer wall times include extra reference/comparison work and
must not be reported as training update throughput.

The exact-execution optimization now has bounded complete-update admission.
It has not met the10x target. Keep these reusable binaries as an admitted
diagnostic reference; do not launch another blind tile/microbatch sweep.
Next frame a concrete larger throughput proposal against the observed
recurrent cost, with numerical/architecture/method implications discussed
before changing them. Dual32 is available but has a poor card-hour tradeoff.
The passed two-window batching experiment remains a separate throughput
option requiring an explicit production sampling decision.

## Fixed method and comparison

Use the original c599236 source profile, compact input bundle, fresh update0
and existing artifacts742/275. Two successive diagnostic updates each use
32daily rows (micro4), oneT7 window and oneT30 window, with unchanged loss
weights, native recurrent feedback, gradient clipping and AdamW. All64,884,481
parameters remain trainable. No new Teacher preparation or changed precision.
Do not substitute the two-window batched executable: that would increase
recurrent counts for this workload. Its passed throughput evidence is separate.

The candidate uses expert128 decoder scheduling plus common-source backbone
K/V reuse for ALL three branches. Supply explicit component admission reports
to the existing serialized dual runner. Keep original source profile/data/
method identity intact and record executable provenance separately; do not
manufacture a profile pretending these are the original binaries.

One GPU executes all branches serially. Two GPUs place daily work onGPU0,
T7+T30 onGPU1, with parameter/gradient transfer, summation, physical checks
and AdamW included in the wall clock. Verify full mixed gradient, updated
parameters and optimizer state between serial and dual execution for both
changing-artifact updates. Compare final parameters/full AdamW and numerical
update reports with the saved original two-update reference at1e-5/1e-6.
This uses component all-gradient admission plus final original checkpoint
evidence; it does not separately reload original kernels to compare every
original mixed-gradient coordinate at each intermediate update.

At fresh weights perform at least three interleaved serial/dual complete
proposals over the SAME daily32+T7+T30 work, discarding each proposal. Skip
the old64daily-row experiment. Report one/two-GPU wall time, speed ratio,
GPU-seconds per update, peak device memory and startup separately. Historical
original timings are a separate-run reference, not a contemporaneous paired
original-vs-candidate benchmark. No scientific accuracy claim follows.

## Bounded execution and recovery

First run lightweight CPU inspection of the existing bundle, reports and
admitted paths, then actual target-shell CPU/GPU dry-runs. Reuse already
passed deployed runtime checks and worker transport evidence: no new model
gradient, state, loss or AD boundary is introduced by choosing saved binaries.
Unit-test override propagation and invalid admission cases before dispatch.

Free test-node inspection has a5minute ceiling. Free gln01 replay uses both
GPUs with a30minute ceiling after checking both are available. Model-gradient
tracing/lowering/compilation is prohibited; small tree/AdamW helpers may
compile. If loading or comparison fails, retain stage/timing evidence and
diagnostic checkpoints. Do not quietly change tolerances or spend another
full-gradient compile. Checkpoints remain diagnostic and cannot become the
formal initialization. No production defaults change.

Remote overlay: `runtime/transfer/shared-combined-full-update-20260926`.
Launcher: `run_shared_combined_full_update.sh`, delegated base wrapper
`run_shared_serialized_dual.sh`. Report/log stem:
`runtime/outputs/diagnostics/shared-combined-full-update-20260926`.

## Decision after completion

If complete-update parity passes, quantify measured total gain and the
remaining recurrent/transport/optimizer costs before selecting more work.
Do not present component gains or increased GPU count as a10x same-resource
result. If the total remains modest, that is evidence that this exact-FP32
execution route alone has not solved the throughput requirement; discuss
any proposed precision, capacity or training-method change explicitly.

Historical complete-update reference is
`outputs/shared-split-dual-compact-20260925.json`: daily32 serial14.851248s,
dual11.192650s. Do not use the earlier abandoned `shared-split-serialized-dual`
reports with stale running status as the completed baseline.

## 2026-10-04: fixed-proposal comparison of two dual schedules

Completed and released, launcher exit0. This is a new engineering scheduling
comparison, separate from the September26 changing-update result above.
The same two V100 GPUs, same run, original update0 parameters/full AdamW,
artifact742 inputs, admitted expert128/source-reuse executables, architecture,
FP32 neural/FP64 physics, masks, losses and native AD were retained. Work was
daily32/eight distinct micro4, one complete T7 and one complete T30.
Artifact275 remained in the validated compact bundle but was not a second
training sample in this comparison. No precision or sampling change occurred.

Old scheduling placed daily32 on parent GPU0 and T7 then T30 on worker GPU1.
New scheduling placed daily32 then T7 on parent GPU0 and T30 on worker GPU1.
The parent already loaded the saved T7 binary on its own logical GPU0;
no executable was retargeted across devices. Recurrent loss/gradients were
already weighted by the saved objectives. Both schedules used the public
`make_component_combiner` in original daily-mean/T7/T30 order, without a
second T7 weight or a prematurely merged local component.

`--comparison-only` directly loaded the original compact update0 checkpoint.
It bypassed both `learning.run` and the historical two changing updates.
One warmup per schedule preceded three alternating pairs; all eight complete
proposals were discarded, no checkpoint was saved, and AdamW step remained
0→0. Source-schedule call accounting is64 daily micro4 gradients plus8 T7
and8 T30 gradients, zero model-gradient compilations. Small tree/AdamW
helpers were allowed. These are complete optimizer proposals at fixed
weights, not demonstrated learning or changing-weight throughput admission.

| Pair | Order | Old wall (s) | New wall (s) | Old/new ratio | Full comparison |
|---|---|---:|---:|---:|---|
|1|old/new|10.038619|7.650988|1.312068|passed|
|2|new/old|9.592130|7.735377|1.240034|passed|
|3|old/new|9.883401|7.692416|1.284824|passed|
|Median|—|9.883401|7.692416|1.284824|—|

New scheduling improves measured same-workload **dual-GPU proposal throughput
28.48%**, reducing wall time22.17%. Two-card elapsed occupancy is
19.766803→15.384831 card-seconds/proposal, equivalently
0.00549078→0.00427356 card-hours/proposal. Both use two cards; this is not
a same-card10x result or a scientific-model accuracy result. Historical
single-card and dual results above are separate-run evidence.

Every pair compares64,884,481 mixed-gradient coordinates,64,884,481 proposed
parameter coordinates and129,768,963 full AdamW coordinates, plus all three
component gradients to prevent cancellation hiding a discrepancy.
All pass at unchanged `rtol=1e-5, atol=1e-6`; maxima across pairs are
5.960464478e-8 mixed gradient,5.960464478e-8 proposed parameters and
9.313225746e-10 AdamW. The largest component-gradient difference is
2.980232239e-8. Component loss, norm, joint loss, clipping/optimizer scalars,
physical counters and retained auxiliary trees also pass. All hard physical
counters are zero. Declared-mask-change counts49/14/60 (daily/T7/T30) are
identical between arms; these declared changes are retained, not hard errors.
Joint loss is identical
at0.8864569111603594 in every measured arm.

The actual auxiliary trees are retained locally and remotely. Daily covers
160 numeric coordinates: process labels/day, science, flux/stock,
state-change anchor, declared mask changes and physical counts. Each T7/T30
tree contains19 scalar coordinates: flux/stock/living/nonliving losses,
declared masks and physical counts. Auxiliary differences are zero.
These parameter-gradient binaries return no full next-state trajectory or
initial-state gradient arrays. The result preserves their existing
state-related/science diagnostics; it does **not** supply new independent
state AD certification or waive previously recorded state-reference limits.

### Measured critical path and timing limits

The following are the actual third pair's synchronized intervals, which also
has the median complete wall values. They are not component medians added
to construct a complete step. Nested and overlapping intervals must not be
summed indiscriminately.

| Pair3 interval | Old (s) | New (s) |
|---|---:|---:|
|Parent daily32|4.289311|4.253692|
|Parent T7|0|1.603767|
|Parent synchronized branch work, daily+local T7|4.289311|5.857459|
|Worker synchronized recurrent branch, old T7+T30/new T30|8.343703|6.718832|
|Branches elapsed including request/response IPC|9.759383|7.606943|
|Parent branch wait or host work|5.470072|1.749484|
|Worker other or wait within complete proposal|1.539698|0.973584|
|Returned-gradient device upload and synchronization|0.082039|0.047606|
|Original-order component merge and synchronization|0.011423|0.010344|
|AdamW and synchronization|0.020084|0.017050|
|Final upload/merge/check/AdamW group|0.123968|0.085426|
|Complete proposal|9.883401|7.692416|

Moving T7 reduces the overloaded worker interval and parent waiting.
The new worker/T30 and IPC interval still determines the observed critical
path; parent daily+T7 completes earlier. Branch work/wait values are blocked
wall proxies, not NVML kernel-active times. Two-card occupancy is2×complete
wall, not the sum of busy proxies. Parent wait/read/unpickle includes waiting
for worker compute and serialization, so it is not pure transfer. For pair3,
request/response elapsed is9.753881→7.601475s; worker result download is
0.256851→0.099080s, and request pickle/write is0.422138→0.356491s.
These support a combined scheduling/IPC improvement without isolating a
pure network or PCIe contribution.

The executed worker returned per-horizon times, but the parent persisted
only its combined recurrent interval. Old worker T7 and T30 cannot be split
after the fact. New worker has only T30, so its full6.718832s synchronized
interval is available. A post-run logging-only fix now retains
`component_seconds` for future diagnostics; it was not used to reconstruct
missing history and caused no further GPU execution.

### Runtime, checks and artifacts

Main timer553.014322s; graphs ready432.159741s. Parent binary load stages
included daily about12.3s and T7/T30 about89.1/89.1s. Worker T7/T30
deserialization took95.990/96.060s. These are saved-executable loading costs,
not new gradient compilation; startup and out-of-timing array comparisons
are included in the overall window, not the complete-proposal medians.
Parent/worker peak live allocations13.019365/3.503345GiB pass24GiB;
each JAX pool is25.391GiB, which is distinct from live allocations.

The one persistent deadline was1800s including a30s TERM/KILL tail
(`timeout1770s`, no budget reset). Budget start22:30:45CST and last launcher
log write22:40:23CST give approximately578s whole-window use at1s timestamp
resolution: about1156 card-seconds /0.321111 two-card hours, CNY0.
The conservative579s timestamp bound is0.321667 card-hours, below1 GPU-hour.
Initial node load was0.00/0.01/0.05 on `ibc13b02n04`; driver525.85.12.
Before and after, both cards showed0MiB/0% and no compute processes.
Launcher exit0 and worker cleanup completed; this window is closed.

Local CPU tests:36 passed in3.99s, including actual CLI flag conflict,
thread overlap, exact horizons, same parameter/AdamW identity, weighted
component order, strict tree/dtype checks, cancellation and discarded
proposals. Actual test02 compact/admission inspection passed; CPU/GPU
target-shell dry-runs and `bash -n` passed. The immutable c599236 checkout
lacked the extracted JSON helper, so the seven-file overlay includes the
small shared `training_artifact_io.py`; checkout and data were preserved.

Executable entry: `scripts/hpc/run_shared_combined_full_update.sh` with
`COMPARISON_ONLY=1`, `OVERLAY=runtime/transfer/shared-dual-schedule-20261004`
and `OUTPUT=runtime/outputs/diagnostics/shared-dual-schedule-20261004.json`.
The shell requires absolute paths as recorded by the retained GPU dry-run.
Source changes retain the old runner defaults; the new mode is explicit.

Local artifacts under `outputs/diagnostics/shared-dual-schedule-20261004/`:
[full paired JSON](../../../outputs/diagnostics/shared-dual-schedule-20261004/shared-dual-schedule-20261004.json),
[actual launcher log](../../../outputs/diagnostics/shared-dual-schedule-20261004/shared-dual-schedule-20261004.log),
[retained auxiliary pickle](../../../outputs/diagnostics/shared-dual-schedule-20261004/shared-dual-schedule-20261004.auxiliary.pkl),
[readable actual auxiliary](../../../outputs/diagnostics/shared-dual-schedule-20261004/actual-auxiliary.json),
[resource/check evidence](../../../outputs/diagnostics/shared-dual-schedule-20261004/verification.json),
`cpu-dry-run.txt`, `gpu-dry-run.txt` and the executed seven-file `overlay/`
snapshot. Remote artifacts use the matching diagnostic stem; the original
bundle/checkpoint/admissions remain unchanged. No formal training,
architecture selection, latent implementation or production adoption follows.
