# Combined execution: daily64 dual load balance

User requested this test on September26 after complete-update41280 passed.
Status: free gln01 two-GPU launcher47778 running,30minute ceiling. Local
11tests and actual test02 inspection passed (daily_proposal_counts32/64);
target-shell GPU argument dry-run passed. Both GPUs idle before dispatch.
Larger daily count is a distinct
sampling proposal, not a same-work speedup or an accepted training recipe.

## Fixed calculation

Reuse admitted combined daily-micro4/T7/T30 executables, fresh update0 and
existing compact artifacts742/275. Compare daily32 (artifact742) with daily64
(both distinct32row artifacts) while retaining oneT7 and oneT30 from742.
Each daily branch is the MEAN of its rows; the recurrent loss weights and
gradients are unchanged. Daily64 does not double the daily loss coefficient.
It does change relative data coverage and gradient variance, so runtime and
parity alone cannot establish learning quality at this sampling recipe.

Run the existing two changing daily32 diagnostic updates to retain the
original checkpoint/optimizer equivalence check, then discard all fixed-
weight proposals. For each32/64count compare serial and dual at identical
weights, data and AdamW state with at least3interleaved repeats. Preserve
full-gradient/parameter/AdamW, loss/scalar and physical checks at1e-5/1e-6.
No new model-gradient trace/lower/compile or Teacher preparation. Daily64
uses16micro4calls instead of8; it does not enlarge microbatch.

Record component durations for EVERY timed proposal: serial daily/T7/T30/
optimizer and dual daily/recurrentGPU/recurrentWorker/branches-with-IPC/
merge-check-optimizer. Report whole wall times separately, card-seconds,
daily rows/s, and the32-to64throughput ratio. Do not add overlapping branches
or infer pure transfer duration from their difference. Source runner's old
`daily_32_microbatch_4` timing label is normalized to `daily_microbatch` in
the diagnostic timing report because it also measures64rows.

## Bounded execution

Use a separate immutable overlay `runtime/transfer/shared-combined-daily64-20260926`
and report stem `runtime/outputs/diagnostics/shared-combined-daily64-20260926`.
Existing `run_shared_combined_full_update.sh` uses `OVERLAY` and `OUTPUT`
overrides, with trailing `--daily-counts 32 64`. CPU inspection5minutes,
freeGPU0+GPU1 replay30minutes. Check target-shell argument expansion and both
GPU occupancies before dispatch. Reuse graph/runtime evidence from41280;
this edit changes proposal selection and reporting, not AD or physics.

## Decision

If daily64 brings branch durations closer and materially raises rows/s at
small extra wall cost, it is a useful load-balance option. It does not make
one sample cheaper and cannot alone solve the large absolute compute cost.
Do not use this result to claim10x or silently replace the final training
sampling protocol. Larger structural speed research proceeds separately.
