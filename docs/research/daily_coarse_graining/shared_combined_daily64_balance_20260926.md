# Combined execution: daily64 dual load balance

User requested this test on September26 after complete-update41280 passed.
Status: free gln01 two-GPU launcher47778 completed0/passed in785.34s,
zero model-gradient compilations, within30minute ceiling. Local
11tests and actual test02 inspection passed (daily_proposal_counts32/64);
target-shell GPU argument dry-run passed. Both GPUs idle before dispatch.
Larger daily count is a distinct
sampling proposal, not a same-work speedup or an accepted training recipe.

## Completed measurements

| Daily count + oneT7 + oneT30 | Serial median | Dual median | Dual daily rows/s |
| --- | ---: | ---: | ---: |
| 32 | 12.582875s | 9.859359s | 3.245647 |
| 64 | 16.879002s | 9.769036s | 6.551312 |

Daily64 doubles the daily count at effectively unchanged dual wall time:
2.018492x daily rows/s. The0.9%lower64wall time is within the small-repeat
variation, not evidence that64intrinsically executes faster than32. Serial
and dual were interleaved within each count;32/64count groups were sequential,
three repetitions each. This is not a unique-data epoch or same-work2x gain.

Median dual64component durations: daily8.3131s, recurrentGPU8.3262s,
recurrentworker8.5975s, branches includingIPC9.6181s, merge/check/AdamW0.1098s.
Corresponding daily32medians: daily4.2617s and recurrentGPU8.3463s. Thus the
daily card's earlier idle interval is effectively filled by64rows. Summing
overlapping branches is invalid, and remaining branch overhead is not pure
transfer attribution. Individual timing medians need not sum to wall median.

For identical64row work, dual beats serial1.727806x (42.12%less wall), using
1.157537x GPU-seconds. Illustrative8CPU tariff charge ratio1.040106 under
existing0.07yuan/core-hour plus2.20yuan/GPU-hour assumptions; actual test free.
Parent measured peak live allocations15.055GiB; JAX pool25.391GiB. Peak is
process-wide across both counts, not isolated64memory or a worker measurement.

Both32/64proposals pass full64,884,481-coordinate gradient and parameter,
129,768,963-coordinate optimizer, loss/scalar and physical checks at unchanged
1e-5/1e-6.64gradient maximum difference5.960464478e-8, parameter2.980232239e-8,
optimizer4.656612873e-10. The two changing32updates and final original-reference
checkpoint comparisons also pass. This does not claim64learning acceptance.
Model-gradient compilation count0; graphs ready386.9s, main785.34s.

The measured load-balance hypothesis succeeded. Use64as the next explicit
daily-count candidate when designing training; retain oneT7/oneT30 and the
mean daily loss unless the sampling protocol is separately revised. Actual
learning with64is still untested. Shared per-day computation remains slow:
the balanced branches each cost~8.3s. Next larger-speed proposal is in
`shared_next_structural_speed_route_20260926.md`; precision/capacity changes
are discussion items, not implicit consequences of this load-balance pass.

## Fixed calculation (executed)

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
