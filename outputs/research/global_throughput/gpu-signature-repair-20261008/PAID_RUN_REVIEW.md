# B256/B512 v3 Continuation Proposal

Status: approved v3 attempt job15643868 is closed as `FAILED`, exit1:0. The user separately approved this v3 allocation after the earlier v2 submission, job15643327, failed during wrapper startup before model/GPU initialization. The exact v3 command submitted once is [APPROVAL_COMMAND_V3.sh](APPROVAL_COMMAND_V3.sh); v2 command provenance remains in [APPROVAL_COMMAND.sh](APPROVAL_COMMAND.sh). Current result and parent-only charge accounting are in [CURRENT_RESULT.md](CURRENT_RESULT.md) and [preparation-ledger.json](preparation-ledger.json). No automatic retry, requeue or monitoring automation was created.

Job15643868 was submitted from `gln01` at2026-10-08 15:31:05 Asia/Shanghai, ran15:31:39–15:53:15 on `ibc13b03n02`, and ended `FAILED` exit1:0. B256 completed and was sealed; its 760×512 native leaves across two seven-day blocks, 515 strict gates, and 512 independent saved-reference comparisons passed. Checkpoint SHA256 is `ee7210f2c83ea11346e9aa752320e8760560b2a6c1e377aa2058aa53da16b08c` (220429675 bytes). Five core samples: median27.024156772s, range27.022209791–27.044812379s, 265.244168785 lane-days/s. Five prepared-wrapper samples: median30.468557430s, range30.319014769–30.538280688s, 235.258922788 lane-days/s. B512 failed during lower-time AOT signature audit with expected536/actual673 metadata, before compile/run; no B512 outputs/gates/samples/checkpoint exist. Parent-only accounting: 1296 wall seconds, 5184 CPU-seconds, 1296 GPU-card-seconds (1.44 CPU-count-hours and0.36 card-hours), estimated CNY0.864 primary/CNY0.8928 conservative; billing is unverified. The CNY3.60/CNY3.72 in the resource table are full-window cost scenarios, not the terminal charge. Evidence: [v3 receipt](v3-submission-receipt.txt), [terminal parent record](job-15643868-terminal-parent-final.txt), [terminal status log](job-15643868-terminal-status-log.txt), [failure diagnostics](job-15643868-failure-diagnostics.json), and [B256 capture](job-15643868-B256-evidence/capture.json).

For future point×parameter/scenario members, use prepared-wrapper unit trajectory-day cost/throughput as the primary single-card concurrency curve; core is explanatory, while startup, compile, checkpoint IO and whole-job cost remain separate. On the same fixed512 repeated-input work, B128 wrapper was33.838030240s/211.832661 lane-days/s at CNY3.147138229 per million lane-days primary; B256 was30.468557430s/235.258922788 lane-days/s at CNY2.833757201 per million, a measured +11.0589% throughput/−9.9577% unit-cost difference. B128's lazy-call and B256's Python-bound dispatch differ, so this is not a batch-size causal effect. A true669-point 14-day scenario arithmetic example is3×256=768 lanes: 157 real points plus99 padded tail slots, 45.702836s and CNY0.030468557/CNY0.031484176 at the B256 wrapper rate. This assumes identical per-chunk time and unverified heterogeneity; it is not an end-to-end measurement and excludes Day1/initialization, annual aggregation, setup and checkpoint. 669 is not a batch cap and the arithmetic is not a 50-year projection. Next work is a free bounded B512 diagnosis; any paid retry requires separate approval. Larger batches need a matching larger real shared workload (not padding the fixed512 measurement to pretend B1024). If the curve plateaus, prefer near-peak throughput with lower memory/compiler cost.

## Proposed execution

| Item | Proposal |
| --- | --- |
| Working directory | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX` |
| Immutable code checkout | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3` |
| Code commit | `12b495b9425de3ad1dc16efa3a5a0e69d7e89222` |
| Run root | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512` (used by terminal job15643868) |
| Partition and allocation | `gnall`, 1 node, 1 task, 4 Slurm CPUs, 1 Tesla V100 32GiB |
| Wall time | `01:30:00`, no requeue, no exclusive-node request |
| Actual node | `ibc13b03n02` (the pre-submission snapshot was not a reservation) |
| Primary cost cap | `(2.20 + 4 × 0.05) × 1.5 = CNY3.60` |
| Conservative cost cap | `(2.20 + 4 × 0.07) × 1.5 = CNY3.72` |

Costs are additive rate scenarios, not verified billing. The 2026-10-08T06:50:43Z pre-submission snapshot found no idle eligible node. `ibc13b03n02` was mixed with CPUAlloc12/112, load2.85, freeMem323601MiB, and AllocTRES `cpu=12,gres/gpu=1`; that snapshot alone was not a reservation. The approved job subsequently started on that node. No exclusive-node or full-node allocation was requested.

The v3 checkout is a clean independent clone without hardlinks or alternates, fast-forwarded from the verified spool-fix bundle. The v2 checkout remains clean and unchanged. Physics diff from `a33b82d` is empty; benchmark/AOT/entry modules are unchanged from validated v2. The wrapper spool-path regression failed before repair with exit127 and passed after repair. Twenty-one local capacity tests passed. On target `test02`, copied-wrapper-only Bash-to-Python argument-expansion dry-run passed with explicit/default `CODE_DIR`, selected only B256/B512 and initialized no JAX/model/GPU; unexpected-argument rejection passed. Subsequent allocated Slurm startup and GPU identity checks passed in job15643868. That job's B256 stage passed and was sealed; B512 hit a real AOT metadata mismatch during lower-time audit before compile/run. See [spool regression tests](spool-regression-tests.txt), [target verification](spool-target-verification.txt), [deployment log](spool-fix-deployment-log.txt), [deployment verification](spool-fix-deployment-verification.json), [v3 terminal record](job-15643868-terminal-parent-final.txt), [failure diagnostics](job-15643868-failure-diagnostics.json), and [B256 capture](job-15643868-B256-evidence/capture.json).

## Workload and safeguards

Run B256 then B512 only against the unchanged ordered512 lanes: 64 real FULL64 points repeated eight times, Days2–8 and9–15 at48 half-hour steps/day. Each admitted shape requires a synchronized full warmup, five core and five prepared-wrapper collections, full output/reference checks, strict repeat/permutation/replacement gates, and atomic stage sealing. Reuse the original sealed B128 checkpoint and timing/provenance without compiling or measuring B128 again. Report scientific acceptance separately; these repeated lanes are capacity evidence, not 512 independent grids.

B512 requires actual B256 compiler-memory evidence, 2GiB GPU slack, 8GiB host slack and at least1800 seconds remaining. It does not require a B256 speedup; this permits measuring a plateau or decline. Keep original physics, dtype, full outputs and tolerances. Preserve complete earlier stages on failure, and do not resubmit automatically.

## Exact command

The following exact v3 command was submitted once after approval; the resulting attempt is terminal and closed:

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3 \
RUN_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v3-continue256512 \
B128_BASELINE_DIR=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/server-benchmark/gpu-sweetspot-v1-load512/capacity \
/rmprog/slurm/v22.05.7/bin/sbatch --export=ALL --time=01:30:00 --nodelist=ibc13b03n02 \
  --chdir=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3/scripts/hpc/slurm_global_gpu_capacity_continue.sh
```

The wrapper's Bash-to-Python argument expansion passed a dry-run before submission. The exact `sbatch` command was submitted once after fresh approval, and this attempt is closed; no retry, requeue or automation is active. The approved resources were one `gnall` node, one task, four Slurm CPUs, one Tesla V100, 90 minutes and no exclusive allocation. Price only the terminal parent accounting once; CNY3.60/CNY3.72 are full-window scenarios, while the parent-only one-time estimate is CNY0.864/CNY0.8928 and billing is unverified. The [Explore1000 skill](C:/Users/admin/.codex/skills/explore1000-job-script/SKILL.md) requires explicit approval before each paid submission: “Before every `sbatch`, show the proposed resources, requested wall time, worst-case estimated charge, command, and working directory. Submit only after the user explicitly approves that estimate.”
