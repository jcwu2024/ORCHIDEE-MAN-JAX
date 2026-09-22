# Parallel micro8 and micro16 execution extension

## Approved launch

User approved the incremental11.04yuan pair. Submitted15329519 micro8 and
15329520 micro16; both RUNNING from2026-09-22 23:20:43CST onibc13b03n05/06.
Scheduler hard end2026-09-23 01:20:43CST,110min main timeout, each1V100,
8CPU,64GiB. Working checkout remains immutableef8c23f, as declared below.
Node checks before submission showedIDLE,CPUAlloc0/112,CPULoad0.01;
startup logs show host load0.00 and no other visible GPU compute processes.
Scheduler records confirm expected nodes/resources/workdir/time limits.
Log/report prefixes:`runtime/outputs/diagnostics/shared-staged-15329519` and
`shared-staged-15329520`. Reference6597 completed1387.6s,exit0, so neither
new job needs a hold. Existing2/4 pair still runs. Total cap for all four is
22.08yuan. Preparation-phase approval-pending text below is historical.

The user requested larger microbatches while micro2/4 jobs15329444/15329445
run. Prepare two additional independent probes, not new scientific training.
Both reuse the completed source100/artifact742 scalar32 reference in
`runtime/outputs/diagnostics/shared-staged-source100-v1`. Reference6597 has
completed and automatically released the original paid pair. At the latest
check both paid jobs had entered their candidate compilation without errors.

This explicitly extends the execution selection of
`shared_execution_benchmark_v1.json` with named stages micro8 and micro16.
The original `all` selection remains [2,4]; the numerical reference identity,
32 daily rows, T7/T30 cases, objective, precision, tolerances, AdamW and24GiB
compiled-live ceiling remain unchanged. Named larger stages require the same
complete saved reference. No old reference metadata is rewritten or loosened.
Record the selected sizes in each report. Candidate states have distinct names.

Each job loads the shared reference, retimes scalar32 locally, compiles one
candidate, checks daily loss/gradient/hard counts and full parameter/optimizer
parity, then times three warmed complete updates. A capacity failure, timeout,
or numerical mismatch is distinct from a measured lack of speedup. Increasing
microbatch size is not assumed to increase speed.32 remains the effective daily
batch size; micro8 uses4 groups and micro16 uses2 groups. Do not increase the
effective training batch or retain diagnostic optimizer proposals.

Proposed resources per job: gnall, one V10032GiB,8CPU,64GiB host memory,
2h scheduler maximum,110min main limit. Each costs at most5.52yuan,
combined incremental cap11.04yuan. Including the already authorized2/4 pair,
all four would have a combined ceiling22.08yuan. Prefer separate idle nodes
ibc13b03n05 andibc13b03n06 after a current occupancy check. No new paid
submission is authorized by this document; request the incremental approval
after preparation. Running2/4 checkouts remain immutable.

Choose the fastest numerically admitted full-update configuration that fits
memory, considering actual compiled/live allocator memory rather than JAX's
reserved pool. Compilation cost is reported separately. A single fixed-batch
probe is execution admission, not a learning/scientific accuracy result.

## Prepared deployment

Source ef8c23f is deployed at
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-staged-large-v1`.
27 local tests pass;11 relevant stage tests pass in deployed GPU Python with
CPU backend (4.65s). Both actual Slurm-entry dry-run expansions select the
requested micro8/micro16 stage and110min main limit. At preparation, nodes
ibc13b03n05/06 were bothIDLE,CPUAlloc0/112,CPULoad0.01.

From the deployed checkout, prepared commands are:

```bash
STAGE=micro8 CODE_ROOT=$PWD /rmprog/slurm/v22.05.7/bin/sbatch --job-name=shared-micro8 --nodelist=ibc13b03n05 scripts/hpc/job_shared_staged_execution.sh
STAGE=micro16 CODE_ROOT=$PWD /rmprog/slurm/v22.05.7/bin/sbatch --job-name=shared-micro16 --nodelist=ibc13b03n06 scripts/hpc/job_shared_staged_execution.sh
```

Reference is already complete; no dependency hold is required. Recheck node
availability and submit only after incremental11.04yuan approval. No paid8/16
jobs submitted yet. Existing2/4 approval does not cover this additional pair.
