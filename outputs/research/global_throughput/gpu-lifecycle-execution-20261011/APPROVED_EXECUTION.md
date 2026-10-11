# Approved B1024 Lifecycle Execution

Status: **failed during initial validation**. This record captures the user's `批准` in supervisor thread `01a128d2-3ec2-77a1-8d51-1665c220d2ca`, conveyed through the explicit delegation. The machine-readable ledger is [PAID_EXECUTION_LEDGER.json](PAID_EXECUTION_LEDGER.json).

The approval covers exactly one `gnall` allocation: 1 node, 1 task, 4 Slurm CPUs, 1 V100 32GB, 7200 seconds, no requeue and no exclusive-node request. The approved worst-case cost is CNY 4.80 primary and CNY 4.96 conservative. No resource escalation is authorized. The parent execution owner retains exclusive responsibility for connection, deployment, submission, monitoring, retrieval, result reporting and commit/push.

The immutable execution pin is `3b4ae35e695f85480ecb72461e4ba93f4534bbd8`. Use this exact checkout and environment:

```text
CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-lifecycle-v1
EXPECTED_CODE_SHA=3b4ae35e695f85480ecb72461e4ba93f4534bbd8
```

The historical [EXECUTION_CANDIDATE.json](../gpu-lifecycle-preparation-20261011/EXECUTION_CANDIDATE.json) remains unchanged with its pre-authorization status, even though it records the same pin. Do not regenerate that candidate after accounting commits, and do not substitute the local worktree HEAD.

The only approved output root is a new, single-use path:

```text
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-lifecycle-20261011-${SLURM_JOB_ID}
```

The two child stages run sequentially in fresh processes: `phased_resident_control`, then `phased_release`. Each is B1024 x 2 chunks, using 64 FULL64 points repeated 32 times (2048 lanes, 28672 lane-days). Each runs five core samples followed by five wrapper samples. The common identity must retain `sampling_order=five_core_then_five_wrapper` and match across arms.

Inputs are the canonical prepared 64-point pickle, frozen FULL64 model references, historical B64 memory baseline, fixed-B1024 evidence, the pinned checkout's ORCHIDEE config, and selected64 assets; exact paths are in the ledger. Keep x64 enabled, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, the default allocator, the original physical-free memory guard, `atol=1e-8`, `rtol=1e-10`, exact discrete schema/dtypes, and only the declared bare-soil roughheight NaN mask.

Each arm must exit successfully, seal its outputs and pass the independent CPU/NumPy saved-array validator before the next arm is dispatched. The validator checks 2051 gates, 760 native leaves, 2048 reference comparisons, reset/permutation/replacement outputs, and the two-chunk state handoffs. At least 3600 seconds must remain immediately before dispatching the second arm. Publish same-job core and wrapper release/control ratios only when both arms pass and their common identity matches.

Any failure stops the sequence and preserves available artifacts. There is no retry, second job, requeue, wall-time extension, allocator or tolerance change, larger shape, or scale-up. A failure produces no throughput ratio or memory-benefit claim. Stop and report to the root execution owner if the approved allocation cannot be honored.

The exact submission argv and receipts are recorded in the ledger. Job `15769234` was submitted once with the dedicated launcher, `--partition=gnall --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:1 --time=02:00:00 --no-requeue`, no `--exclusive`, the fixed `CODE_DIR` as `--chdir`, and the shared absolute log file `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/global_gpu_lifecycle_15769234.txt`.

The selected node was `ibc13b03n02`. Immediately before submission, the recorded scheduler snapshot showed `State=IDLE`, `CPUAlloc=0/112`, and `CPULoad=0.01`; the job request pinned that node. The initial `scontrol show job` receipt reported `PENDING`; that remains the historical submission state. Startup evidence then confirmed allocation on this node with 4 CPUs, 1 V100, and the approved execution SHA. The selected UUID was `GPU-18a4abb8-43d5-5142-c32f-a1898866d9bd`; captured CPU affinity was `0,1,56,57` and initial host load was `0.08/0.03/0.05`. Startup orchestration reported the coordinator running and the first `phased_resident_control` child dispatching. The GPU receipt is metadata-only and does not claim CUDA selector UUID resolution or model acceptance.

The earlier `13-monitor-initial.stdout.txt` receipt recorded `RUNNING` at Slurm elapsed 1106 seconds. The final receipt, `14-monitor-initial-gate.stdout.txt`, reports job `15769234` as `FAILED` with exit code `1:0`, `ElapsedRaw=1140`, start `2026-10-11T16:47:51`, and end `2026-10-11T17:06:51` on the server/Slurm clock. The first `phased_resident_control` arm stopped during initial validation: independent saved-array FULL64 tolerance failed at lane 45. The failed initial array `failed_initial_arrays.pkl` is 880,948,565 bytes with SHA-256 `698d3dece201424778d9cf4b0f10c1ac3b1802a1ce35b7d231f73b751641ddc5`; the recovered collection manifest independently matches that hash. During the original terminal metadata recovery, the scientific arrays were not retrieved or loaded into this checkout; they remained preserved at the remote run path.

Compilation completed before collection, and the original preallocation guard permitted proceeding. The failure occurred before warm samples; the stage was not sealed and did not pass independent stage acceptance. Only the control arm was dispatched. The release arm was never dispatched, no throughput ratio was published, and there is no root-release evidence. The terminal metadata archive was recovered and hash-verified; its terminal checkout receipt confirms the exact pinned SHA and a clean checkout. The failure ends this approved allocation; there is no retry, requeue, extension, extra job, or residual-budget reuse.

Target admission receipts show GNU Bash 4.2.46 on `ibc13b02n04`; direct and spooled launcher expansion matched and both reached the real coordinator dry-run. Source admission at the pinned checkout reported the expected scientific SHA `8c8b1dfd0b207e96bbb81c2d3eb85ae6edccb9ed3c6b77fb470dabc58c5fa77d` without importing JAX or executing the model. JAX/jaxlib 0.4.38 and CUDA/NVML library bindings were imported on the target, with no device enumeration, GPU initialization, model, or JIT execution. The metadata snapshot showed both V100 GPUs on that host occupied at approximately 31.7 GB each; no extra GPU probe was run.

The terminal monitor command receipt started at Windows UTC `2026-10-11T09:18:38.262542+00:00`; the server-side observation was `2026-10-11T09:10:17Z`. Keep those clock values separate and derive runtime and cost from Slurm parent elapsed on the server/job clock. The parent allocation used 1,140 seconds, or 1,140 GPU-card-seconds and 4,560 CPU-count-seconds. Prorating the approved two-hour estimates gives CNY 0.76 primary and CNY 0.785333333 conservative; these are estimates, not a verified bill.

Under the supervisor's one-time follow-up authorization in source thread `01a128d2-3ec2-77a1-8d51-1665c220d2ca`, a free `test03` CPU/NumPy comparison later read the preserved arrays at their remote path. This was one comparison, with no new scheduler job, model, JAX, or GPU execution, and no pickle was downloaded into this checkout. The `20-compare-saved-arrays` preflight stopped before input verification or array loading because the old Git rejected `-C`; its archived stdin SHA-256 is `e31c61efb8ebc14b9e48127afe8123ed30f36806774ad36ffc7da23de0450c24`. `21-compare-saved-arrays` is the sole actual comparison. Its [summary](saved-array-explanation/comparison/summary.json) checked all 760 leaves across 2,048 lanes and localized the one failure to `[0][0][0][7]`, `diffuco_previous_step_state.rveget`, a float64 Day 8 end state. At the original `atol=1e-8`, `rtol=1e-10` thresholds, the 32 failing lanes are `45 + 64*k` for `k=0..31`, all repeats of point `283.0-091.0`; the other unique points passed. The supervisor accepted this difference localization. Scientific acceptance remains false: this result gives no causal attribution and is not evidence of the old ABI failure. Script time was 22.631719 seconds, outer elapsed time 29.54 seconds, and peak RSS 1,008,807,936 bytes. The paid grant remains closed with no retry or residual-budget reuse.
