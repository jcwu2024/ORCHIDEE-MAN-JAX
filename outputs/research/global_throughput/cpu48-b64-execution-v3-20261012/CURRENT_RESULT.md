# CPU48 / FULL64 B64 v3 Terminal Result

Job `15820467` is terminal `FAILED 1:0` after `00:36:37` on `ibc11b09n32`.
The sole authorized submission was accepted once; the registry records
`submitted=true`, invocation count `1`, and this job ID. That grant is closed:
there was no retry, requeue, or second submission. The allocation was one
`cnall` node, one task, 48 CPUs, zero GPUs. See [authorization](AUTHORIZATION.md),
[submission audit](SUBMISSION_AUDIT.md), and the [terminal evidence manifest](TERMINAL_EVIDENCE_MANIFEST.md).

The run was pinned to execution commit
`03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8` (`codex/global-cpu48-b64-v3`),
base `b1b5cf0835da298bddb388e61d754feeb6c14cf2`, certified science SHA
`11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`, actual
science-source SHA `b33450cae9cd8c9b2d5f7c6075fe270f461887edc09c77417c5e820a5c18fab8`,
baseline ref `133e35aec7b83b9d79888c46092f64d518e97299`, and frozen prepared
input SHA `be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7`.
The FULL64 GPU-reference SHA was
`a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6`.
The declared workload was 64 unique points, Days 2-8 and 9-15, 48 half-hours
per day, 896 point-days, and 760 native output leaves.

The one-CPU scalar FULL64 step returned `0` after `2190.0347s`. Its saved
exploratory report completed all required stages, but strict numerical status
is false: 61 of 64 GPU-reference lane gates passed; lanes 6, 11, and 37 had
finite tolerance failures (four failing leaves total). The scalar frozen-reset
repeat passed. The report therefore supplies diagnostic scalar timing and
signature evidence, not an accepted CPU/GPU numerical result. Its scalar AOT
lower/compile signatures each report 536 caller/public/internal input leaves,
417 retained leaves, equal public shape/dtype/weak-type signatures, and equal
internal signatures. This verifies the scalar binding path only.

Five scalar warm-core samples measured a median `127.075s` (range
`116.764-133.899s`) over 896 point-days, or `7.051 point-days/s`. Five
preprepared-wrapper samples measured a median `121.382s` (range
`118.899-133.709s`), or `7.382 point-days/s`. These are one-CPU scalar
measurements. Scalar lowering was `55.877s`, compilation `684.266s`, and first
call `137.362s`.

The 48-CPU batch child failed about `0.42s` after dispatch in
`global_server_resources.py` with `RuntimeError: batch48 cannot inherit a
different OMP thread budget`, before JAX initialization, lowering, compilation,
or model evaluation. The command requested `OMP_NUM_THREADS=48`, but the
failed child's actual OMP value was not logged; no specific observed value is
asserted. This run has no B64 compilation or timing samples, CPU48 throughput,
same-workload CPU/GPU ratio, or scientific/physical acceptance.

Terminal parent `sacct` accounting is `FAILED 1:0`, `ElapsedRaw=2197`,
`AllocCPUS=48`, node `ibc11b09n32`, `TotalCPU=35:47.671`, and
`AllocTRES=billing=48,cpu=48,node=1`. The one parent allocation is counted
once, including the scalar step: `29.2933` allocated CPU core-hours, estimated
CNY `1.4647` primary / `2.0505` conservative; billing rates remain unverified.
The coordinator-only interval estimate (`2190.526s`, CNY `1.4604` / `2.0445`)
excludes submission/startup and is not the whole-job cost. Approved worst-case
ceilings were CNY `3.60` / `5.04`.

RSS sources are kept distinct: report sampled peak `21,320,556,544` bytes;
Linux `ru_maxrss`/`/usr/bin/time` high-water mark `21,406,240,768` bytes
(`20,904,532 KiB`, about `19.94 GiB`); checkpoint-event current RSS
`5,585,903,616` bytes; Slurm scalar-step `MaxRSS=5,351,876K`. These are
different measurements and are not interchangeable.

The bracketed login/workstation snapshots showed a stable clock offset of
about 8m20s; allocation accounting uses Slurm elapsed time. A terminal
`scontrol show job` query now returns `Invalid job id specified`; the earlier
live `scontrol` receipts and terminal `sacct` rows are preserved. The bounded
collector ran on `test01` with the canonical project Python and returned
`incomplete_run_failure_preserved`, `scientific_acceptance=false`, and
`failure accounting only; no throughput claim`.
That early failure path did not independently admit the scalar arrays, gates,
or throughput. The scalar values above come from the saved raw report and
stage manifest and remain exploratory records.
