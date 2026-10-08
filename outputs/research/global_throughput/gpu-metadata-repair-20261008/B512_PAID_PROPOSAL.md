# Separate paid proposal: B512 only, approval pending

Status: concrete immutable deployment and interface checks passed; **not approved or submitted**. Accepted deployed code SHA is `ad6c0e1faa7e22b590627844e9cb0ad492e9deb3`; the checkout is clean before/after all 9 actual declared-checkout checks. See [deployment receipt](deployment-receipt.json) and [CLI/Bash/spool receipt](deployment-interface-receipt.json). Direct and spool expansion agree, JAX was not imported, and no model preparation/lowering/compilation/invocation or GPU initialization occurred. This is a new allocation and additive cost envelope; previous paid approvals are consumed.

Deployed Linux actual full-source SHA is `b33450cae9cd8c9b2d5f7c6075fe270f461887edc09c77417c5e820a5c18fab8`, with active exact execution-only cache certificate and admitted baseline science SHA `11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`. Native-output equivalence and canonical real lowering evidence remain the bounded checks described in [current result](CURRENT_RESULT.md); the unchanged CUDA entry and revised real GPU executable are still runtime-pending.

Resources: Explore1000 `gnall`,1 node,1 task,4 auxiliary CPUs,1 V100,45-minute hard wall (`00:45:00`), no requeue. Maximum0.75GPU card-hours and3CPU-hours. Additive planning cost isCNY1.80 atGPU2.20/card-hour+CPU0.05/core-hour, orCNY1.86 withCPU0.07/core-hour; neither is verified billing.

Exact scheduler working directory: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`.
New immutable `CODE_DIR`: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-metadata-v1`.
New run-root template: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-capacity-512-${SLURM_JOB_ID}`. It must not already exist at launch.

Proposed submission command, **for approval only**:

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
/rmprog/slurm/v22.05.7/bin/sbatch --nodelist=ibc13b03n02 --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX --export=ALL,CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-metadata-v1 /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-metadata-v1/scripts/hpc/slurm_global_gpu_capacity_512.sh
```

`ibc13b03n02` is only the provisional 2026-10-08 11:21:20UTC [snapshot candidate](proposal-node-snapshot-approval.txt): MIXED, CPUAlloc34/112, CPULoad3.07, FreeMem323680MiB, 7/8 GPUs unallocated. The prior10:24 candidate `ibc13b02n02` now has CPU108/112 and load12.75. No node was idle, and this is not a reservation or whole-node exclusive request. Refresh eligible occupancy/load at separately approved submission; if the selected node changes, state it explicitly. The launcher records actual node, CPU allocation/affinity/load and visible GPU processes at startup.

The wrapper resolves the shared helper from `CODE_DIR` even in Slurm spool and expands the actual command through `srun --ntasks=1 --cpus-per-task=4 --cpu-bind=cores`, Singularity CUDA runtime, and the canonical GPU environment. It invokes `global_batch_gpu_entry.py` with `ORCHJAX_GLOBAL_BATCH_ENTRY=benchmark_global_gpu_capacity.py`, `--continue-512`, `--budget-seconds 2700`, five repeats, and these sources:

- prepared input: `runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl`;
- FULL64 reference: `runtime/outputs/research/global_throughput/gpu-round4/model`;
- old B64 host memory origin: `runtime/outputs/research/global_throughput/server-benchmark/global-throughput-v2-2b14506/gpu-b64/candidate-first`;
- sealed B128: `runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v1-load512/capacity`;
- sealed B256: `runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512/capacity`.

All relative source paths above are under the exact working directory. Configuration is the new checkout's `configs/orchidee_man_250919.yaml`; asset root is `runtime/outputs/research/global_throughput/round4-inputs/selected64-assets-v2`. Output is `${RUN_ROOT}/capacity`; logs are `runtime/logs/global_gpu_capacity_512_%j.txt`.

Workload: same ordered512 lanes=64 genuine points×8, continuous Days2–8 then9–15,48 steps/day, full native outputs and physical masks. Import sealed B128/B256 origins without compiling or measuring them; B256 compiler temporary bytes1,151,609,352 provide the device estimate, and old B64 RSS supplies the host estimate. Keep2GiB device/8GiB host slack and the1800-second remaining-window admission gate. One full synchronized B512 warmup precedes5 core+5 prepared-wrapper collections and complete strict reference comparisons. No fake B1024 comparison is made.

45 minutes is a bounded completion attempt informed by prior B256 lower93.37s, compile686.84s and warmup27.17s plus validation/collection time. It is not a completion guarantee for B512. Memory, host or remaining-time denial must stop with no valid B512 claim. GPU compilation, invocation, full numerical gates and timing of the revised executable are unresolved. The physics baseline, native dtypes, masks, outputs and tolerances remain fixed; the exact-byte certificate transparently distinguishes the host-cache repair's actual source digest. No training, AD, science-capacity change, old-budget reuse or automatic retry is authorized by this proposal.

The read-only [B512 saved-result collector](collect_b512_result.py) is prepared and deployed before approval, with 41 lightweight tests and Ruff passing. Independent review found no blocking issue. Its deployed raw SHA is `36e8971fd94f9883c579d1fedb4c155f27298db595742b5aae2bcb347b6a6926`. It remains external to immutable execution checkout `ad6c0e1`; no runner redeployment was needed. Exact HEAD and clean status are required before importing executed helpers; bytecode writes are disabled. It pins Linux source `b33450ca...` and the active exact certificate, binds report/manifest/checkpoint identity, environment, allocation, gates and common startup timings, and admits only a newly measured B512 stage. B128/B256 origins are revalidated without changing their original hashes, jobs, code, raw samples or memory. [Actual historical-origin verification](b512-collector-origin-verification.json) passes and contains no B512 timing.

The [canonical collector interface receipt](collector-interface-receipt.json) verifies the deployed tool, actual checkout/source certificate, exact legacy dependencies and real CLI help without JAX, model work, GPU initialization or repeated saved-array reads. The [first attempt](collector-interface-first-attempt.json) found an absent prior helper; its exact source bytes were copied only after confirming absence. The existing legacy helper was hash-verified and left unchanged. Both SSH-inclusive attempts total 2.9559588s; cumulative conservative canonical wall is 367.0582004s of 600s. This verifies collection readiness, not a terminal B512 result or paid approval.

After approval and terminal completion, obtain exactly this raw9 parent/step schema (untruncated fields):

```bash
/rmprog/slurm/v22.05.7/bin/sacct -n -P -j "$APPROVED_JOB_ID" --format=JobIDRaw,JobName%64,Partition%64,State%64,ElapsedRaw,AllocCPUS,AllocTRES%512,NodeList%128,ExitCode > "$SACCT_RAW9"
```

The approved JobID/node, observed allocated GPU identity, admission file and terminal accounting file must be supplied; none exists by implication of this proposal. Collection command template, after filling those verified variables:

```bash
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/gpu-metadata-repair-20261008/collect_b512_result.py \
  --code-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-metadata-v1 \
  --stage-dir "$NEW_B512_STAGE_DIR" --admission "$ADMISSION_JSON" --sacct-file "$SACCT_RAW9" \
  --job-id "$APPROVED_JOB_ID" --job-name orchjax-gpu-b512 --node "$APPROVED_NODE" \
  --gpu-uuid "$OBSERVED_ALLOCATED_GPU_UUID" --slurm-global-gpu-id "$OBSERVED_SLURM_GLOBAL_GPU_ID" --driver-version "$OBSERVED_DRIVER_VERSION" \
  --approved-nodes 1 --approved-tasks 1 --approved-cpus 4 --approved-gpus 1 --approved-wall-seconds 2700 \
  --b128-baseline-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v1-load512/capacity \
  --b256-baseline-dir /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512/capacity \
  --output "$NEW_B512_COLLECTION_JSON"
```

Run saved-array collection on an allowed CPU test node, never on a login node. The collector checks one raw parent row for `gnall`, approved name/JobID/node, CPU4/GPU1/node1 and elapsed≤2700s; child rows are not added to cost. Resource denial or incomplete/tolerance-failed evidence remains withheld, including an empty denied-stage manifest. A terminal failed parent retains its actual failure and cost; a separately complete stage is retained only if every strict stage/report gate passes. Core/wrapper warm cost, startup, checkpoint IO and whole-job cost remain separate. No scientific acceptance, CPU/GPU ratio, K-amortization, annual Fortran acceptance or10× claim is produced.
