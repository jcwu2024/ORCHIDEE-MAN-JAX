# Repaired current-model T7 precision admission: in-flight execution

Updated: 2026-10-16. Status: **RUNNING snapshot, job15895454**, not a terminal
scientific result. Fresh explicit human approval authorized this single repaired
attempt independently of the consumed job15866925 authorization.

## Exact deployment and submission

Execution SHA: a833f987ac65e12748f50d4ee9cfd05cbc69d839.
Proposal SHA256: af1cce1067b55941243028d4778380195f003a09d8d40fca481293d6ed10c1b6.
Probe SHA256:52eb808dd0a7896277604095fc6d2bde0bb5e2abf4a4f54cbdc30779695ba709.
Fresh approval SHA256:
5e769bdf86732c1b56bf7ed34000b95d6117df1b208f5363060d82c38e7aaefb.
Checkout:
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-current-model-precision-admission-repaired-20261016.
Output directory:
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-current-model-precision-admission-repaired-20261016.

The actual test01 Bash expansion and generic-wrapper CPU dry-run match the
approved argv; the fresh approval check passed without GPU initialization or
real-model preparation. Deployment preserves the exact committed source.
Earlier target-validation logs record a pre-submission CRLF transfer error;
the corrected successful test01 evidence is target-shell-approval-check.json.
The new checkout shares Git objects with the old checkout: retain the old
T7-current-model-precision-admission-20261014-v2 directory and its dependencies.

The submission receipt records one sbatch invocation, exit0 and raw stdout
Submitted batch job 15895454. Submission/start were2026-10-16T10:34:48+08:00,
actual node ibc13b02n01, four allocated CPUs/one V100/one node/one task,
5400-second limit, nonexclusive, Requeue=0 and automatic node placement.
The approved ceiling is24GiB/CNY3.72; at most one real-model FP64 compile and
52 planned calls (16 native state-VJP/JVP,36 scalar FD). This is an execution
bound, not an observed compile/call count. No retry, training or new experiment.

## Startup evidence at10:36:12+08:00

The retained raw scheduler response validates RUNNING, NodeList/hostname n01,
legacy allocated TRES cpu4/gpu1/node1, CPU_IDs4–7 and allocated GPU detail IDX1.
This real response is retained before validation, unlike the unsaved response
from the prior failed attempt. GPU discovery passed with runtime/driver count1,
PCI0000:1B:00.0 and UUID GPU-a6321ffd-0e5a-8be8-1703-86713168c895.

Synthetic runtime preflight passed in60.71884106658399 seconds on the GPU.
Its scope is synthetic_runtime_interfaces_only: real_model_checked=false,
scientific_acceptance=false, teacher_data_read=false and real_training_updates0.
The supervisor phase elapsed65.53589138761163 seconds includes child startup;
this differs from the preflight's own internal timer.

The supervisor snapshot records passed discovery/preflight and a started probe
child with cap4980 seconds, without a terminal child exit/result yet. The
discovery phase completed in3.011349592357874 seconds; probe PID49486 is a
snapshot identifier. Supervisor/ledger status reserved is in-flight, not terminal.
Supervisor CPU affinity is[2,3,58,59], host load[3.38,3.26,2.78]. Its
GPU_executed=true records discovery/preflight GPU work, not model acceptance.
The startup inventory has no result.v2.json. No full-model precision/gradient/FD,
repeat/full3854, physics, memory admission, real-model compile count or complete
52-call outcome is established. Historical state20 failure remains unwaived.

## Interpretation and follow-up boundary

The sole job is in flight at this saved snapshot; no terminal cost, release,
scientific admission or failure is inferred. A later scientific child exit3
must be interpreted with result.v2.json and supervisor records. A generic outer
runtime_failed label alone cannot distinguish an expected scientific gate
failure from a runtime exception; the scientific JSON owns classification.

No automatic watcher, waiting loop or repeated query is installed. A manual
follow-up may read squeue/scontrol/sacct and the new output directory. Do not
resubmit this one-shot job, launch a new experiment or start training.

Manual status/accounting for the existing job only:

```powershell
ssh gln01 '/rmprog/slurm/v22.05.7/bin/squeue -j 15895454; /rmprog/slurm/v22.05.7/bin/scontrol show job -dd 15895454'
ssh gln01 '/rmprog/slurm/v22.05.7/bin/sacct -j 15895454 --format=JobID,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,NodeList,Start,End -P'
```

The [old job15866925 report](T7_current_model_precision_admission_execution_results_20261015.md)
remains the closed8-second launcher-failure history; its authorization was not
reused. B4 repair/adoption and derivative-reference claims remain unaccepted.

## Local evidence

outputs/T7-current-model-precision-admission-repaired-execution-20261016/ holds
submission.receipt.json, deploy.stdout.txt, target-shell-approval-check.json,
first-scheduler-snapshot.txt and startup-check.txt. Its startup/ snapshots include
approval.v2.json, result.v2.allocation-diagnostic.json, result.v2.discovery.json,
result.v2.ledger.json, result.v2.supervisor.json and runtime-preflight.v2.json.
deployment-submission-startup-receipt.json binds11 key evidence hashes;
MANUAL_HANDOFF.md retains the existing-job-only manual follow-up boundary.
[NEXT_STEPS](../../NEXT_STEPS.md) owns the current execution boundary.
