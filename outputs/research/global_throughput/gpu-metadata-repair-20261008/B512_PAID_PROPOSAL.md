# Separate paid proposal: B512 only, approval pending

Status: prepared locally; **not approved or submitted**. Final code SHA and actual immutable-checkout interface receipt are pending root commit/deployment. This is a new allocation and additive cost envelope; previous paid approvals are consumed.

Resources: Explore1000 `gnall`,1 node,1 task,4 auxiliary CPUs,1 V100,45-minute hard wall (`00:45:00`), no requeue. Maximum0.75GPU card-hours and3CPU-hours. Additive planning cost isCNY1.80 atGPU2.20/card-hour+CPU0.05/core-hour, orCNY1.86 withCPU0.07/core-hour; neither is verified billing.

Exact scheduler working directory: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`.
New immutable `CODE_DIR`: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-metadata-v1`.
New run-root template: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-capacity-512-${SLURM_JOB_ID}`. It must not already exist at launch.

Proposed submission command, **for approval only**:

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
/rmprog/slurm/v22.05.7/bin/sbatch --nodelist=ibc13b02n02 --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX --export=ALL,CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-metadata-v1 /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-metadata-v1/scripts/hpc/slurm_global_gpu_capacity_512.sh
```

`ibc13b02n02` is only the provisional10:24UTC snapshot candidate: MIXED,CPUAlloc8/112,CPULoad1.81,6/8 GPUs unallocated. No node was idle, and this is not a reservation or whole-node exclusive request. Refresh eligible occupancy/load before approval/submission; if the selected node changes, state it explicitly. The launcher records actual node, CPU allocation/affinity/load and visible GPU processes at startup.

The wrapper resolves the shared helper from `CODE_DIR` even in Slurm spool and expands the actual command through `srun --ntasks=1 --cpus-per-task=4 --cpu-bind=cores`, Singularity CUDA runtime, and the canonical GPU environment. It invokes `global_batch_gpu_entry.py` with `ORCHJAX_GLOBAL_BATCH_ENTRY=benchmark_global_gpu_capacity.py`, `--continue-512`, `--budget-seconds 2700`, five repeats, and these sources:

- prepared input: `runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl`;
- FULL64 reference: `runtime/outputs/research/global_throughput/gpu-round4/model`;
- old B64 host memory origin: `runtime/outputs/research/global_throughput/server-benchmark/global-throughput-v2-2b14506/gpu-b64/candidate-first`;
- sealed B128: `runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v1-load512/capacity`;
- sealed B256: `runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512/capacity`.

All relative source paths above are under the exact working directory. Configuration is the new checkout's `configs/orchidee_man_250919.yaml`; asset root is `runtime/outputs/research/global_throughput/round4-inputs/selected64-assets-v2`. Output is `${RUN_ROOT}/capacity`; logs are `runtime/logs/global_gpu_capacity_512_%j.txt`.

Workload: same ordered512 lanes=64 genuine points×8, continuous Days2–8 then9–15,48 steps/day, full native outputs and physical masks. Import sealed B128/B256 origins without compiling or measuring them; B256 compiler temporary bytes1,151,609,352 provide the device estimate, and old B64 RSS supplies the host estimate. Keep2GiB device/8GiB host slack and the1800-second remaining-window admission gate. One full synchronized B512 warmup precedes5 core+5 prepared-wrapper collections and complete strict reference comparisons. No fake B1024 comparison is made.

45 minutes is a bounded completion attempt informed by prior B256 lower93.37s, compile686.84s and warmup27.17s plus validation/collection time. It is not a completion guarantee for B512. Memory, host or remaining-time denial must stop with no valid B512 claim. GPU compilation, invocation, full numerical gates and timing of the revised executable are unresolved. The physics baseline, native dtypes, masks, outputs and tolerances remain fixed; the exact-byte certificate transparently distinguishes the host-cache repair's actual source digest. No training, AD, science-capacity change, old-budget reuse or automatic retry is authorized by this proposal.
