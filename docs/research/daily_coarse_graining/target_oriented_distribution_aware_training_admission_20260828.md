# Distribution-Aware Adapter Training Admission

Date: 2026-08-28

Status: completed admission; Slurm job 14856147 executed the accepted envelope

Implementation commit:
`ad5c11b27d8c3e92bca88a22181fbf3fc6159a02`

Bounded training contract SHA256:
`0c60924ee12d0ea555b082f76990ca55e34f86f67effb052f5eec44c01210051`

One-update admission report SHA256:
`a32b2c175b153d8ac7c732b70a1bf9b2124fbe49a122ab7414f16a7842c7734e`

## Frozen Execution

The run reconstructs the complete `93,504,933`-parameter candidate from the
exact update-1000 Attention parent and deterministic adapter initialization.
Only the `9,750,529` adapter parameters enter direct autodiff and fresh AdamW;
all `83,754,404` retained parameters remain fixed and are checked after
training.

Optimizer updates `1..250` consume the 250 preregistered prepared artifacts in
ascending order from `252..511` after excluding calibration and evaluation
groups. Every update combines the existing free objective gradient with `0.1`
times the Teacher-day-start guard gradient and uses the frozen schedule and
clip threshold. Atomic adapter-only recovery replaces itself every 25 updates.

The same eight train/train groups are evaluated at initialization and update
250. The final checkpoint contains only the adapter, fresh optimizer state,
update records, and both screens. It is written with
`promotion_eligible=false`; promotion is determined only after report review.

Teacher v6 is not regenerated, historical v5 is not used, and validation and
sealed-test values are not read. The candidate reconstructs no 48-record
forcing sequence and executes no 48-step state scan.

## Local Qualification

The bounded-contract loader, disjoint 250-artifact selection, combined-
gradient rule, relative plus aggregate-absolute science classifier, ephemeral
admission policy, adapter-only recovery/checkpoint round trip, production
runner policy, launcher, and Slurm resource declarations pass focused tests.
The candidate, predecessor, and training suite passes 22 tests. Ruff, Python
compilation, Bash syntax, `git diff --check`, and CLI import also pass.

The free one-update run populated the shared persistent cache for the complete
free and Teacher-guard objective graphs. Its live JAX peak was about 2.66 GB;
the Slurm job requests 32 GiB host memory and one 32-GiB V100.

## Resource Envelope

Seven hot calibration batches measured approximately 44 seconds for the
combined free and guard gradient work. This projects about 3.1 hours for 250
updates. The two eight-group screens add approximately 50 minutes, giving an
expected runtime near four hours. A five-hour hard limit covers initialization,
checkpoint writes, and cache-load variance.

```text
Partition: gnall
Resources: 1 node, 8 CPU cores, 1 V100 GPU, 32 GiB host memory
Wall-time limit: 05:00:00
CPU cost: 1 * 8 * 5 * CNY 0.07 = CNY 2.80
GPU cost: 1 * 1 * 5 * CNY 2.20 = CNY 11.00
Worst-case additive estimate: CNY 13.80
Working directory: /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
Command: /rmprog/slurm/v22.05.7/bin/sbatch scripts/hpc/slurm_target_oriented_distribution_aware_training_250.sh
```

Live inspection found `gnall` up with nine mixed nodes, one idle 8-V100 node,
and an empty queue. Availability can change before submission and is not part
of the scientific gate.

## Decision Boundary

Submission requires explicit approval of the exact resource envelope above.
After completion, review the frozen update-0/update-250 relative and absolute
7/30-day gates before any validation read, 365-day rollout, long held point,
Gate D2, sealed test, or product work.

## Outcome

Job `14856147` completed all 250 updates with exit code zero in `03:38:12` at
commit `90c6636`. The accepted result is recorded in
[`target_oriented_distribution_aware_training_result_20260829.md`](target_oriented_distribution_aware_training_result_20260829.md).
