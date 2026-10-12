# Single allocated-GPU paid terminal result

2026-10-12. The one approved `T7_affine_allocated_single_gpu_v1` attempt is
closed and consumed. Sole job **15812634** on `ibc13b03n04` ended **FAILED 2:0**
after **6 allocated seconds**. Its first GPU CSV query failed before Supervisor,
preflight, CUDA/PJRT or the affine probe. Numerical status is `not_executed`;
no numerical failure, speed improvement, contract acceptance or T7 repair follows.
No second submission, retry, node change, mapping query, repair/redeployment,
free fallback, training or background monitoring is authorized.

## Approved identity and sole submission

The parent relayed the human's explicit approval for the exact new contract,
immutable code `c8a44c3e5ba203f57da5f6b44aae47587af31d31` and documentation
`650116f3aa0eda51d684ddf79b194fff8dc48d91`. The
[preparation/proposal](T7_paid_single_GPU_preparation_results_20261012.md) and
[contract](T7_paid_single_GPU_execution_contract_20261012.md) retain the method
and original false preparation snapshot. Approval bound scope
`T7-summary-affine-paid-single-gpu-20261012-v1`, gnall, fixed n04,
1 node/1 task/4 actual Slurm CPUs/1 V100, nonexclusive/no-requeue, 20 minutes,
CNY 1 ceiling and 24 GiB. Startup/preflight cap 240s, unchanged affine cap 780s
and protected release reserve 180s share that same 1200s allocation. Original
12 measurements/91 arrays/4 small compiles, FP32/FP64 host oracle and strict
`rtol=1e-5, atol=1e-6` were unchanged.

Working/code directory:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-single-gpu-20261012-v1`.
Remote evidence directory:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-paid-single-gpu-20261012-v1`.
The full argv in local `approval.v1.json`, `submission-intent.v1.json` and
`sbatch-invoked.v1.json` matches tested `proposal.v2.json`; the explicit resource,
wall-time, node, cwd and output/error options were used, never bare `sbatch`.
Submission-shell inherited `SBATCH_*` recording was empty and all inherited
names were cleared. The sole submission receipt records job 15812634.

The fresh scheduler snapshot at `2026-10-12T12:58:53Z` showed n04 MIXED,
CPUAlloc 48/112, CPULoad 2.26 and GPUAlloc 3/8. It had sufficient unallocated
resources and was neither idle nor reserved. Slurm site timestamps are submit
20:58:53, start 20:59:11, end 20:59:17 on 2026-10-12. Actual allocation:
4 CPUs and 1 GPU, live `GRES=gpu:v100:1(IDX:5)`, scheduler `CPU_IDs=72–75`.
Saved startup reports actual OS process affinity `[36, 37, 92, 93]` and original
`CUDA_VISIBLE_DEVICES=0`; those are distinct from the scheduler identity fields.
The startup record's RUNNING/end-of-time-limit snapshot is preserved separately
from the authoritative terminal FAILED/6s record.

## First GPU command and unreached stages

Allocation, clock, code/source/input and read-only free-management admission
passed. Exactly one GPU observation command ran:

```text
nvidia-smi -i 5 --query-gpu=index,uuid,name --format=csv,noheader,nounits
returncode: 6
stdout: No devices were found\n
stderr: empty
```

The full JSON rejection retains the command, raw output and preceding startup
observations. No XML, UUID/minor/kernel check, actual-job cgroup devices rules,
OS opening-denial assessment, Supervisor, runtime preflight, CUDA driver/PJRT
identity or actual affine identity was reached. There were zero affine
measurements, saved affine arrays, affine compilations and optimizer updates.
JAX was not imported. No results/partial NPZ or owned runtime ledger exists.
The OS field is `not_assessed_stage_not_reached`; this run does not verify the
new contract's separately reported OS-permission policy.

Slurm IDX 5, original CVD 0 and the failed `nvidia-smi -i 5` are retained facts.
They do not establish a driver/configuration cause or an alternate NVML mapping.
No additional mapping query ran. Old job 15806646's devices.list remains missing.

## Closure, accounting and preservation

The shared cooperative mutex was released before any child and verified absent
at closeout. Slurm's allocation ended. No live post-exit GPU memory/process
query was performed, so hardware cleanup and peak GPU memory are unobserved.
Mutex absence is not live GPU cleanup evidence. No Supervisor occupied-time
ledger was produced; science/child execution seconds are zero, distinct from
six allocated GPU seconds.

Additive estimate using the approved rates:
CPU `4×6/3600×0.07=CNY 0.0004666667`,
GPU `1×6/3600×2.20=CNY 0.0036666667`, total **CNY 0.0041333333**.
Allocated GPU card-hours: **0.0016666667**. Actual billing is unknown.

| Preserved identity | SHA256 |
|---|---|
| Original false scope | `59bd008466c9d01aedadd53e3342add952a204caa449ad7754a3045542bae4b5` |
| Approved/open scope | `d0b5fb0c17622cc022d64c39f43f33b209b586a43a22c29717e78297ce79a215` |
| Closed scope (`execution_authorized=false`) | `85c369bd1e3c5274ddcf49404b03a89261cbeeb98b29580fdaea65a84944509c` |
| Frozen deployment manifest | `539e6da0e0fa0cdb2811f00e1f37f60c017cdc90a264078a1fa61d282ddfa2a6` |
| Terminal 28-file archive | `65449f087fe17f8137d7653fb0e544f3aa4ac636a571584ad5f39ceb41c093f9` |

The first closeout helper's test01 `scontrol` query exited1 before scope
mutation; the exact underlying reason is unproved because its command stderr
was not retained. Its original script/stdout/stderr/receipt remain. The
successful evidence-only recovery supplied the exact terminal
`scontrol`/`sacct` output already saved by the login-node observation, then
validated and closed the scope on test01. No new scheduler/GPU query or scheduler
repair was used. Archive retrieval and saved-only local analysis passed; archive
SHA and nine bounded raw evidence anchors match the extraction inventory.
No code retest or numerical rerun was needed for this documentation stage.

Local approval recording is `2026-10-12T13:07:03.964421+00:00`, while the remote
submission snapshot is `2026-10-12T12:58:53Z`. Local and remote wall clocks differ.
Original timestamps are preserved; cross-host clock values do not establish
approval ordering. Terminal remote Slurm start/end control allocation accounting.

Frozen c8a44c3 and old 58a/f4ae deployments, grants, recovery inputs and results
remain untouched. Free b676 is still approved/paused/unconsumed, unopened and
unreserved. No Teacher data, weights or scientific results were removed or
regenerated. Closed scopes grant no further work.

The local [execution package](../../../outputs/T7-paid-single-gpu-execution-20261012/)
contains [saved-only summary](../../../outputs/T7-paid-single-gpu-execution-20261012/execution-summary.v1.json),
full extracted rejection, first-query raw files, all command receipts, safely
extracted 28 raw terminal files, closed scope/closeout, immutable original
archive, analysis script and documentation diff/check/versioning receipts.
