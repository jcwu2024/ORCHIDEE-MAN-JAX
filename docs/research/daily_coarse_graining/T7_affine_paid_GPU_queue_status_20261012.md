# Paid affine v2: submitted, pending — 2026-10-12

The one approved unchanged job was submitted once and accepted as **15782177**.
Its first snapshot is **PENDING / Resources**. Slurm estimates StartTime
`2026-10-12T14:41:06` and EndTime `15:01:06`, about five hours after that
cluster snapshot. These are pending-job predictions, not an actual allocation
or a guaranteed start. This monitoring turn stops immediately; Slurm keeps the
job queued. There is no second sbatch, availability loop, heartbeat or automatic
LLM monitoring. Future retrieval must use this same job ID.

The [earlier resource non-admission report](T7_affine_paid_GPU_execution_results_20261012.md)
remains the accepted no-submission/CNY0 historical stage. Subsequent supervisor
queue steering lifted the wait-for-an-empty-card stop without changing the
human-approved resources, command, source or CNY1.00 ceiling.

## Version, approval and single submission

The seven accepted earlier documents were normally committed/pushed as
`77afd8033fd63984277fa398867ebf28ef958eb9`, local/origin agreeing. That is a
documentation identity. **Execution remains the frozen**
`f4ae942ae2bf806337d6518d6683d73a1de4ffa8` checkout; it was not updated or
refrozen from the documentation SHA. The
[documentation receipt](../../../outputs/T7-affine-paid-queue-20261012/accepted-docs-version.v1.json)
and ordinary commit/push logs retain the exact seven paths.

Human approved the exact paid-v2 resource/cost contract; the supervisor
subsequently directed queue submission within that unchanged approval. Human
CNY1.00 approval was directly verified by the parent from supervisor message
`01a12da6-1e1b-73c3-b582-dbc95cada758`; the contemporaneous approval and queue
steering are recorded in
[submission-intent.v1.json](../../../outputs/T7-affine-paid-queue-20261012/submission-intent.v1.json).
Before submission, CPU-only identity and bounded known-free admission verified
the clean frozen HEAD, exact source/input hashes, b676 HEAD and8 absent free
evidence paths, with no JAX/GPU initialization. No resource refresh loop or
CPU numerical/dry-run/test rerun occurred. The earlier gnall
UP/OverSubscribe=NO policy evidence was reused as authorized.

The runtime scope is now `open/execution_authorized=true`, explicitly binding
that approval and queue steering. The original false scope bytes, local v6
proposal/manifest and old df8/v1 remain preserved. Original false scope SHA256
is `99e3f32ef0ebac209fca077157c3d09186271dc15f6429c51fa4aa1ae5966e22`;
opened scope SHA256 is
`eec2cd5c8e0a7f6916149246460b7c710f9f64ebc95463a938a5d551d1b2af89`.
[Submission admission](../../../outputs/T7-affine-paid-queue-20261012/submission-admission.v1.json)
records those bindings and the separate read-only free-state result.

| Approved/submitted item | Exact value |
|---|---|
| Partition / fixed node | gnall / `ibc13b03n01` |
| Resources / policy |1node/1task/4 actual Slurm CPUs/1 V10032GB/nonexclusive/no-requeue |
| Wall / cost |20min from actual allocation StartTime; maximum0.8266667 CNY/upward0.83; approved ceiling1.00 |
| cwd | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-20261011-v2` |
| Command | `sbatch /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-20261011-v2/scripts/hpc/slurm_shared_latent_T7_batch_numerics_paid.sh` |
| Endpoint |gln01, actual host `ibc13b02n04`, existing identity through192.168.11.1 |
| sbatch response | `Submitted batch job 15782177`; exit0 |

Raw [submission stdout](../../../outputs/T7-affine-paid-queue-20261012/remote.submission.sbatch.v1.stdout.txt),
[stderr](../../../outputs/T7-affine-paid-queue-20261012/remote.submission.sbatch.v1.stderr.txt)
and [call receipt](../../../outputs/T7-affine-paid-queue-20261012/sbatch-once.v1.receipt.json)
are preserved. Acceptance consumes the unique scheduler submission attempt
even while PENDING. Ordinary submission-call/accepted receipts record it; the
launcher's first-write `slurm-attempt.v1.json` was not precreated.

## First snapshot and limits

Only job15782177 was queried with `scontrol show job --details -o` and
`squeue --start --jobs=15782177`. The
[raw snapshot](../../../outputs/T7-affine-paid-queue-20261012/first-job-status.v1.stdout.txt)
records PENDING/Resources, RunTime00:00:00, fixed ReqNodeList n01 and empty
allocated NodeList. Its `TRES`/NumCPUs fields describe the pending request;
they do not prove allocated resources. Cluster stdout date is
`2026-10-12T01:41:30Z`; local receipt start is separately
`2026-10-12T01:49:51.006197+00:00`, recorded in the
[command receipt](../../../outputs/T7-affine-paid-queue-20261012/first-job-status.v1.receipt.json).
No clock reconciliation or confirmed future start is inferred.

Thirteen declared runtime artifact paths were absent at the snapshot, including
Slurm job stdout/stderr, runtime attempt, ledger/supervisor/preflight, result
arrays and HLO. Actual allocation start/end/terminal state and release NVML
are not yet applicable. Allocation seconds/card-hours/estimated charge **as of
this snapshot** are0/0/CNY0; this does not make the accepted future job free.
The approved maximum cost remains in force when Slurm allocates it.

No GPU preflight or numerical comparison is evidenced as of this snapshot. Planned sample3/raw434,
scalar/vmap/lax.map once each plus serial merge once, repeats3/12 measurements/
91 arrays, FP32 neural/FP64 oracle, strict `rtol=1e-5, atol=1e-6`, all primal/
JVP/input/parameter derivatives/masks/isolation/repeats remain unchanged.
The diagnostic cotangent is not the T7 loss. Runtime must still enforce actual
Slurm/GPU/UUID/cgroup/container/CUDA/PJRT identity,1200s total/240s preflight/
780s probe/≥180s release/24GiB/OMP4/OpenBLAS1. Predictions do not start or
renew that allocation clock. Historical CPU12/91 and136 compatibility cases
remain separate evidence. Free b676 dispatch stays paused and its approval
unconsumed. No old input/checkout is modified, and no scientific repair,
precision/architecture change or training acceptance follows this queue status.
