# Combined saved execution: complete-update admission

Status: implemented; local11tests and actual test02 CPU inspection passed.
Actual CPU/GPU shell dry-runs passed; both free GPUs idle at13:23CST before
dispatch. No paid job or formal learning. Daily, T7 and T30 components have passed unchanged loss/full
parameter-gradient/physical checks. This next gate measures whole updates.
Free gln01 launcher41280 started approximately13:24CST September26,
GPU0 parent/GPU1 recurrent worker,30minute ceiling. Results pending.

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
