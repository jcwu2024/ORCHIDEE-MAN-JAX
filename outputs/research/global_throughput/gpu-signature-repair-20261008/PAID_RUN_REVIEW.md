# B256/B512 v3 Continuation Proposal

Status: prepared and verified; awaiting fresh explicit paid approval. The user approved one v2 submission, job15643327, which failed in wrapper startup before model/GPU initialization. That approval is consumed and does not authorize this corrected v3 attempt. Historical v2 command provenance remains in [APPROVAL_COMMAND.sh](APPROVAL_COMMAND.sh); its terminal record and one-second accounting are in [CURRENT_RESULT.md](CURRENT_RESULT.md) and [preparation-ledger.json](preparation-ledger.json). Do not automatically resubmit.

## Proposed execution

| Item | Proposal |
| --- | --- |
| Working directory | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX` |
| Immutable code checkout | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3` |
| Code commit | `12b495b9425de3ad1dc16efa3a5a0e69d7e89222` |
| Run root | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512` (unused) |
| Partition and allocation | `gnall`, 1 node, 1 task, 4 Slurm CPUs, 1 Tesla V100 32GiB |
| Wall time | `01:30:00`, no requeue, no exclusive-node request |
| Candidate node | `ibc13b03n02`, subject to a fresh occupancy check; not reserved |
| Primary cost cap | `(2.20 + 4 × 0.05) × 1.5 = CNY3.60` |
| Conservative cost cap | `(2.20 + 4 × 0.07) × 1.5 = CNY3.72` |

Costs are additive rate scenarios, not verified billing. The refreshed snapshot at2026-10-08T06:50:43Z found no idle eligible node. `ibc13b03n02` was mixed with CPUAlloc12/112, load2.85, freeMem323601MiB, and AllocTRES `cpu=12,gres/gpu=1`; it had the lowest load among eligible nodes but remains a candidate, not a reservation. Refresh occupancy before submission. No exclusive-node or full-node allocation is requested.

The v3 checkout is a clean independent clone without hardlinks or alternates, fast-forwarded from the verified spool-fix bundle. The v2 checkout remains clean and unchanged. Physics diff from `a33b82d` is empty; benchmark/AOT/entry modules are unchanged from validated v2. The wrapper spool-path regression failed before repair with exit127 and passed after repair. Twenty-one local capacity tests passed. On target `test02`, running the copied wrapper alone without a sibling child script passed actual Bash-to-Python argument-expansion dry-run with explicit and default `CODE_DIR`, selected only B256/B512, and initialized no JAX/model/GPU; unexpected argument rejection passed. The independent review found no blocking launcher issue. Allocated Slurm/container startup and real GPU B256/B512 remain unverified. Evidence: [spool regression tests](spool-regression-tests.txt), [target verification](spool-target-verification.txt), [deployment log](spool-fix-deployment-log.txt), [deployment verification](spool-fix-deployment-verification.json), and [node snapshot](spool-fix-node-snapshot.txt).

## Workload and safeguards

Run B256 then B512 only against the unchanged ordered512 lanes: 64 real FULL64 points repeated eight times, Days2–8 and9–15 at48 half-hour steps/day. Each admitted shape requires a synchronized full warmup, five core and five prepared-wrapper collections, full output/reference checks, strict repeat/permutation/replacement gates, and atomic stage sealing. Reuse the original sealed B128 checkpoint and timing/provenance without compiling or measuring B128 again. Report scientific acceptance separately; these repeated lanes are capacity evidence, not 512 independent grids.

B512 requires actual B256 compiler-memory evidence, 2GiB GPU slack, 8GiB host slack and at least1800 seconds remaining. It does not require a B256 speedup; this permits measuring a plateau or decline. Keep original physics, dtype, full outputs and tolerances. Preserve complete earlier stages on failure, and do not resubmit automatically.

## Exact command

The following command is the v3 proposal. It has not been submitted:

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3 \
RUN_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512 \
B128_BASELINE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v1-load512/capacity \
/rmprog/slurm/v22.05.7/bin/sbatch --export=ALL --time=01:30:00 --nodelist=ibc13b03n02 \
  --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3/scripts/hpc/slurm_global_gpu_capacity_continue.sh
```

The wrapper's Bash-to-Python argument expansion passed a dry-run; the exact `sbatch` command itself was not executed. No paid job or monitoring automation is authorized yet. Obtain new explicit approval for the resources, wall time, worst-case cost, working directory, and exact command before every `sbatch`, as required by the [Explore1000 skill](C:/Users/admin/.codex/skills/explore1000-job-script/SKILL.md): “Before every `sbatch`, show the proposed resources, requested wall time, worst-case estimated charge, command, and working directory. Submit only after the user explicitly approves that estimate.”
