# B256/B512 continuation preparation

Status: code repair, free bounded verification and immutable deployment passed; no paid approval or submission. This proposal replaces neither job15600405's closed accounting nor its sealed B128 evidence.

The proposed workload is the same ordered 512 lanes: 64 genuine FULL64 points repeated eight times, with Days2–8 and9–15 at48 half-hour steps/day. Each admitted shape gets a synchronized full warmup, five core and five prepared-wrapper collections, full output gates and an atomic checkpoint before the next shape. These are capacity measurements, with scientific acceptance reported separately. No B128 model compilation or experiment is repeated; its original checkpoint, raw samples and execution origin remain baseline evidence from job15600405.

Proposed allocation: one `gnall` node, one task, one Tesla V10032GiB, four auxiliary allocated Slurm CPUs, `01:30:00` hard limit and no requeue. No whole-node exclusivity or memory reservation is requested. Four Slurm CPUs can correspond to two physical cores through SMT; startup records must preserve the actual affinity and topology.

Worst-case price scenarios assume GPU and allocated CPU charges add:

- Primary: `(2.20 + 4 × 0.05) × 1.5 = CNY3.60`.
- Conservative: `(2.20 + 4 × 0.07) × 1.5 = CNY3.72`.

These rates are not a verified bill. The previous job's unused envelope is not authorization for this job.

Verified clean independent immutable checkout: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v2`, commit `9d19f0760b93e1186eaf796db4903ddfd16a806f`. New run root: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v2-continue256512`, still unused after final verification. Working directory: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`. Canonical GPU Python, existing Singularity runtime and driver binds are unchanged; no installation is proposed.

The [final scheduler snapshot](node-snapshot-final.txt), clusterUTC2026-10-08T00:15:23Z, found no idle eligible node. Selected node `ibc13b03n03` was mixed, with32/112 CPUs allocated, load4.85,330551MiB reported free and no allocated GPUs. It had the lowest load among nodes with available CPUs and GPUs. This is not a reservation; refresh occupancy before an approved submission. The launcher records actual node, allocation, CPU affinity/load, selected physical GPU UUID/driver and GPU process diagnostics.

B256 runs first. B512 follows when actual B256 compiler memory analysis, selected-card free memory plus2GiB slack, conservative host memory admission and at least1800 seconds remaining permit it. The continuation measures B512 independently of whether B256 improves over B128 by5%, so a plateau or decline can be observed. Hard input/schema/dtype/static/nonfinite/discrete failures stop the affected attempt. Preserve finite reference-tolerance failures as failed exploratory gates with the fixed tolerances. A partial first call or unsaved stage is not throughput.

Exact proposed command, to run from the GPU submission endpoint only after approval:

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v2 \
RUN_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v2-continue256512 \
B128_BASELINE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v1-load512/capacity \
/rmprog/slurm/v22.05.7/bin/sbatch --export=ALL --time=01:30:00 --nodelist=ibc13b03n03 \
  --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v2/scripts/hpc/slurm_global_gpu_capacity_continue.sh
```

Expected outputs are complete new B256/B512 stages with raw5+5 samples, full outputs/checks, signature audits, compiler buffers, RSS and GPU boundary snapshots, separate lower/compile/transfer/checkpoint IO and terminal parent-only job accounting. If a stage fails or B512 resource/time admission fails, preserve the complete earlier stage and diagnostic evidence without resubmission. No K-amortization is promised without full observed setup accounting. After both measurements, assess a plateau/decline before finer sizes; the final candidate subsequently needs different real points.

[130-test/focused check record](checks.json), [actual deployed-module canonical evidence](final-module-canonical-diagnostic.jsonl), [Bash/Python argument verification](final-launcher-verification.txt) and [deployment verification](deployment-verification.json) passed. Historical trigger remains unresolved; this is a defensively bound interface validated by small canonical checks, not real GPU B256/B512 acceptance. Old B128 and new shapes use different host dispatch paths; report complete timing/cost observations with that limitation.

Explicit approval is required by the [Explore1000 skill](C:/Users/admin/.codex/skills/explore1000-job-script/SKILL.md): “Before every `sbatch`, show the proposed resources, requested wall time, worst-case estimated charge, command, and working directory. Submit only after the user explicitly approves that estimate.” The user resumed repair and preparation only. No `sbatch` or monitoring automation is authorized by preparation alone.
