# Target-Oriented Source-Shaped Branch Training Admission

Date: 2026-08-27

Status: completed by Slurm job `14701638`; scientific result is recorded in
`target_oriented_source_shaped_branch_training_result_20260827.md`

Implementation commit:
`9224398f6740a8008f06f112194e7f0e2d37d153`

Screen contract SHA256:
`a06beb815834142f44939022eea6ebd044e0a096b5d3c443754a9d87fb0f8cee`

Direct-autodiff admission report SHA256:
`5852d996f2baa8afa5e68d0446d30d09bff64b5ac7996836f448a27535e58538`

## Purpose

This is the first bounded training stage for the accepted source-shaped
maintenance mechanism. It does not compare two weak architectures. It asks
whether 100 updates to the two new physical correction branches retain the
large zero-update maintenance gain while repairing the frozen 7/30-day GPP,
respiration, NPP, biomass, and ecosystem-carbon protection failures.

## Frozen Execution

The run reconstructs the complete 84,148,646-parameter candidate from the
exact update-1000 parent and deterministic source-shaped initialization. The
complete forward objective remains unchanged, but only the 394,242 branch
parameters are differentiated and updated. All retained parameters remain
outside the optimizer.

The accepted zero-panel report supplies update-0 screen evidence without
re-execution. The accepted one-update report supplies the frozen p90 clip
threshold `0.1973866669`; calibration is not repeated. Training uses prepared
artifacts `252..351` exactly once and maps them to optimizer updates `1..100`.
A fresh AdamW state follows the frozen ten-update warmup and cosine-decay
schedule.

Every update must have finite nonzero branch gradients, a finite nonzero
parameter step, zero frozen hard failures, and a valid clip scale in `(0, 1]`.
An atomic branch-only recovery is replaced every 10 updates. It contains no
retained parent coordinates and is never promotion eligible.

After update 100, the run evaluates exactly the eight frozen train/train
groups `251, 358, 465, 572, 679, 786, 893, 1000`, each with eight 7-day and
four 30-day windows. Only after that complete finite zero-hard screen is a
branch-only update-100 checkpoint written. The checkpoint stores the two
branches, fresh optimizer state, update records, final screen, and immutable
parent/data identities. It remains `promotion_eligible=False` even if the
science gate passes, pending review.

Teacher v6 is not regenerated, historical v5 is not used, and validation and
sealed-test values are not read. The candidate reconstructs no 48-record
forcing sequence and performs no 48-step state scan.

## Local Qualification

The checkpoint round-trip, identity, update-coverage, direct-gradient,
schedule, policy, launcher, and Slurm resource tests pass. The combined
focused and related suite passes 51 tests. Python compilation, CLI import,
Ruff lint and format, Bash syntax, and `git diff --check` also pass.

## Resource Envelope

The accepted direct-autodiff hot effective batch takes `25.18 s`, implying
about 42 minutes for 100 updates. The prior frozen eight-group screen took
about 25.5 minutes. Initialization, artifact loading, recovery writes, and
final checkpointing make approximately 70--90 minutes the expected hot-cache
wall time. A three-hour limit covers a persistent-cache miss without turning
the estimate into a routine target.

```text
Partition: gnall
Resources: 1 node, 8 CPU cores, 1 V100 GPU, 32 GiB memory
Wall-time limit: 03:00:00
CPU cost: 1 * 8 * 3 * CNY 0.07 = CNY 1.68
GPU cost: 1 * 1 * 3 * CNY 2.20 = CNY 6.60
Worst-case additive estimate: CNY 8.28
Working directory: /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
Command: /rmprog/slurm/v22.05.7/bin/sbatch scripts/hpc/slurm_target_oriented_source_shaped_branch_training_100.sh
```

Live scheduler inspection found `gnall` up with ten 8-V100 nodes, one idle
node, and an empty queue. Availability can change before submission and is
not part of the scientific gate.

## Decision Boundary

The user approved the exact resource block above and Slurm job `14701638`
completed with exit code zero in `00:58:51`. The engineering run passed, but
the frozen scientific promotion gate failed. The update-100 branch checkpoint
is diagnostic only. See
`target_oriented_source_shaped_branch_training_result_20260827.md` for the
accepted result and next gate. Validation, broader training, 365-day rollout,
Gate D2, sealed test, and product acceptance remain closed.
