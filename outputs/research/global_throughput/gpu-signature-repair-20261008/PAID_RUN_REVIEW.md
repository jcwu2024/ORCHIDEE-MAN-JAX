# B256/B512 v3 Continuation Proposal

Status: v3 approved and submitted once; job15643868 is RUNNING. The user separately approved this v3 allocation after the earlier v2 submission, job15643327, failed during wrapper startup before model/GPU initialization. The v2 approval is consumed and the old command remains historical provenance in [APPROVAL_COMMAND.sh](APPROVAL_COMMAND.sh). The exact v3 command is [APPROVAL_COMMAND_V3.sh](APPROVAL_COMMAND_V3.sh). Current job details and terminal one-row accounting policy are in [CURRENT_RESULT.md](CURRENT_RESULT.md) and [preparation-ledger.json](preparation-ledger.json). No automatic retry, requeue or monitoring automation was created.

Job15643868 was submitted from `gln01` at2026-10-08 15:31:05 Asia/Shanghai and started at15:31:39 on `ibc13b03n02`, with a 17:01:39 bound end. At07:32:08Z it was RUNNING with four CPUs and one GPU allocated, no restart/requeue, and the expected Tesla V100 identity passed startup checks. No saved B256/B512 outputs, gates or samples are yet observed; throughput and final parent charge remain pending. The CNY3.60/CNY3.72 figures below are cost scenarios for the full allocation, not final billing. See [v3 receipt](v3-submission-receipt.txt), [node snapshot](v3-submission-node-snapshot.json), and [initial status](v3-job-15643868-initial-status.txt).

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

The following exact v3 command was submitted once after approval:

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3 \
RUN_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512 \
B128_BASELINE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v1-load512/capacity \
/rmprog/slurm/v22.05.7/bin/sbatch --export=ALL --time=01:30:00 --nodelist=ibc13b03n02 \
  --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3/scripts/hpc/slurm_global_gpu_capacity_continue.sh
```

The wrapper's Bash-to-Python argument expansion passed a dry-run before submission. The exact `sbatch` command was later submitted once after fresh approval; no retry, requeue or automation is active. The approved resources were one `gnall` node, one task, four Slurm CPUs, one Tesla V100, 90 minutes and no exclusive allocation. Price only the final terminal parent accounting once; the CNY3.60/CNY3.72 numbers are scenarios, not actual final billing. The [Explore1000 skill](C:/Users/admin/.codex/skills/explore1000-job-script/SKILL.md) requires explicit approval before each paid submission: “Before every `sbatch`, show the proposed resources, requested wall time, worst-case estimated charge, command, and working directory. Submit only after the user explicitly approves that estimate.”
