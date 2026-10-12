# Single paid affine GPU preparation — 2026-10-11

The supervisor initially authorized preparation, local regressions, ordinary
commit/push, an independent frozen checkout and target CPU-only dry-run.
The later human specifically approved the exact paid-v2 job/CNY1.00 contract.
The [earlier resource result](T7_affine_paid_GPU_execution_results_20261012.md)
remains accepted historical no-submission/CNY0 evidence. Later explicit steering
authorized the unchanged single queue submission without waiting for a free GPU.
The [current queue status](T7_affine_paid_GPU_queue_status_20261012.md) records
accepted job15782177, first snapshot PENDING/Resources, scope open and sole
scheduler attempt used. Predictions do not prove allocation/start. Stop LLM
monitoring; Slurm continues queueing. Future retrieval uses this same job,
never another sbatch. Execution remains immutable f4ae942, independently of
accepted historical-docs77afd803. Original false scope bytes remain preserved.

The free b676 one-shot remains approved/unconsumed, but this execution owner
temporarily dispatches paid only. Old code, scope, ledger and approval remain
untouched. Earlier candidate diffs/logs remain historical; directory isolation
and restoration machinery have been removed from the accepted preparation.

## Exact proposed job and quote

| Item | Proposed value |
|---|---|
| Partition / preferred node | gnall / `ibc13b03n01` |
| Resources | 1node,1task,4 actual allocated Slurm CPUs,1 V10032GB, nonexclusive |
| Wall / attempt policy | `00:20:00`, no-requeue, one scheduler attempt, no automatic retry |
| Caps | 1200s from allocation StartTime; preflight240s/probe780s/release≥180s;24GiB; OMP4/OpenBLAS1 |
| Frozen cwd | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-20261011-v2` |
| Command | `sbatch /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-20261011-v2/scripts/hpc/slurm_shared_latent_T7_batch_numerics_paid.sh` |
| Independent output/scope/ledger | `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-paid-gpu-20261011-v2/` |

Current additive rates: CPU CNY0.07/allocatedCPU-hour, GPU CNY2.20/GPU-hour.
CPU `4×20/60×0.07=0.0933333`; GPU `20/60×2.20=0.7333333`; total
`0.8266666667`, rounded upward **CNY0.83**. The later human specifically approved
ceiling **CNY1.00**. No extra resource, whole-node exclusivity or paid fallback.
CR_CORE with2 threads/core means4 Slurm CPUs are2 physical cores. Do not add
`--hint=nomultithread`; actual `NumCPUs=4` and allocated TRES cpu=4/gpu=1 are
mandatory, and8 allocated CPUs reject admission.

[Saved scheduler selection](../../../outputs/T7-affine-paid-preparation-20261011/scheduler-selection.v1.json)
records server UTC `2026-10-11T12:43:36Z`, gnall UP, cons_tres/CR_CORE,
cgroup plugins,56physical/112Slurm CPUs and8V100s/node; n01 was IDLE with
CPUAlloc0/GPUAlloc0/load0.01. It is a selection snapshot, not a reservation.
The later single resource refresh passed gnall policy but found all8 GPUs
allocated. Subsequent explicit queue steering authorized submission without
waiting for availability; job15782177 is accepted. No availability loop or
further monitoring follows the first saved job snapshot.

## Scientific scope and execution admission

Reuse the two small inputs already under
`runtime/transfer/T7-summary-recovery-20261011-v1/`: operand
`bf36a098b4e6a89d555f8d18521555f2da66135425d9435cfe01878adeec6129`, receipt
`f863a2d964f626cec327981d86a8832e73aa45af62d708a342805c4aa4220e00`.
Capture source `70619b0a6445cde87eccaeb31feac46bbb34cdc0d7a6f68c68938c4d09901072`
and probe `05e2ee66b1905fd238722d3a4cf645e74639f7134de4af5a700cf0b657305ee0`
remain unchanged. No recapture, real CPU numerical probe, Teacher/checkpoint
transfer or numerical-reference rerun is needed. The earlier lost original
file bytes, six matching recovered numeric hashes and historical CPU12/91
remain separate evidence.

Fixed sample3/raw434, `(4,257,4) @ (4,384)`; scalar/vmap/lax.map each compile
once plus serialmerge once; repeats3,12 measurements/91arrays. FP32 neural,
independent FP64 oracle, strict `rtol=1e-5, atol=1e-6`; preserve primal/JVP/
input/parameter derivatives, masks/isolation/repeats, raw arrays and HLO.
Cotangent is diagnostic, not T7loss. No extra layers/directions/T7/T30/full
network/precision/loss/training/search or numeric-failure extension follows.

Paid scope must be specifically approved/open and bind the genuine clean
accepted HEAD and exact source manifest. Candidate and target dry-run compare
source hashes; a nonnull HEAD must equal actual Git HEAD. Remote metadata uses
POSIX paths; executable local argv uses real local paths. Slurm spool location
cannot choose code: the script fixes canonical checkout plus `#SBATCH --chdir`.
Numeric job-ID admission is followed by exclusive `slurm-attempt.v1.json` as
the first filesystem action, before code/scope/directory checks. Missing code/
scope or later startup failure consumes the attempt. Unwritable/missing output
or earlier persistence failure means unknown/fail-closed state: retain the
pre-submission receipt and never automatically retry. The job creates no output
directory. Python records a separate job attempt and independent paid ledger.

Require actual node/partition/one task/4CPU/one GPU/20min/no-requeue, allocation
StartTime/EndTime/currentclock; late Python startup does not reset the1200s budget.
`scontrol show job --details -o` supplies allocated `GRES` in22.05.7, or
`GresDetail` in the supported alternate schema. Physical Slurm GPU index,
NVML UUID/V100/minor_number, actual character-device major/minor and exact
`job_<SLURM_JOB_ID>` cgroup-v1 membership must agree. Read/write permissions
must isolate that single GPU data device, excluding observed driver control
devices; wrong minor/job, ambiguous mapping or wildcard access including
`c *:N` fails closed. Host index is never inferred from CUDA visible ordinal.

The child selects the admitted UUID, explicitly passed into the container.
The optional paid hook within the existing preflight independently observes
CUDA-driver count/UUID and the already initialized PJRT single GPU at local
hardware ordinal0. Both must agree; inability to prove binding rejects. No
new phase/JAX compile is added, and preflight240s/total1200s/memory remain.
Public preflight behavior without this paid environment is unchanged. The
Supervisor requires the identity report before the numerical probe. Existing
initial/release0MiB/0%/noapps, report/count/precision/memory and owned-group
deadline gates remain. Unverified cleanup retains the lock and disallows retry.

## Management-chain exclusion

The authorized execution owner is the sole dispatcher for this bounded stage:
paid only; free dispatch paused; existing free approval preserved. The new
launcher uses a persistent atomic mkdir mutex at
`runtime/locks/T7-summary-affine.shared.lock`, with host/PID/scope/nonce.
No automatic stale-lock takeover occurs. Own verified cleanup is required to
release it. This is a cooperative boundary for new entries; the unchanged
b676 launcher and arbitrary old copies do not use it and are not mechanically
constrained by it. The paid dispatch relies on the declared management chain.

Before any GPU observation, reservation or preflight, read only the known free
checkout HEAD and its bounded scope/ledger/result paths. The declared b676
checkout must remain present at `runtime/checkouts/T7-batch-numerics-20261011-v1`.
Its output directory is `runtime/outputs/diagnostics/T7-summary-affine-gpu-20261011`.
The known `scope.v1.json`, `T7-summary-affine-gpu-20261011-v1.ledger.json`, result,
preflight, supervisor, outer-timeout and result/partial NPZ must remain absent:
that grant was never opened/reserved. Any existing evidence, changed HEAD,
unavailable path or permission/unknown state stops paid dispatch. There is no
full-tree/process scan, directory rename/chmod, restoration journal, manual
lock-takeover framework or modification of the old entry/scope/ledger.

## Versioned preparation delivery and current approval

The [prior v1 delivery receipt](../../../outputs/T7-affine-paid-preparation-20261011/final-delivery.v1.json)
records exact final files/diff, current test logs, real local/origin/remote HEAD,
false scope/manifest and actual test-node Bash→CPU parser/import/argv evidence.
The final source stage is committed/pushed normally on
`research/daily-coarse-graining`, then frozen from that genuinely clean HEAD
with `freeze_shared_affine_paid_scope.py`. The independent checkout is created
from a verified Git bundle if needed; no old/main/shared checkout is updated.
Two existing inputs are reused. New output/log directory and false scope are
prepared without opening the grant or consuming a job/ledger. Actual target
CPU dry-run on test01–test04 proves parser/source/import/argv admission only.

Prior v1 local regressions include the existing30 launcher and14 shell cases
against the shared code, retained paid admission/mutex/identity tests,
and directly related optional public-preflight tests. Actual logs/counts are
in the delivery receipt; prior candidates/failures remain historical evidence,
and old44 passes do not substitute for changed paid-code regression. Negative tests
write temporary fixtures only. Graph Verify/project `E-ORCHIDEE-MAN-JAX`,
generation `2026-10-10T08:46:50Z`, ready: relevant daily paths are excluded/
not_tracked under `runtime/worktrees`; exact source fallback was used, with
no reindex or completeness claim.

Actual paid Slurm allocation, device isolation, container/PJRT/GPU runtime and
numerical comparison remain unverified. Target CPU readiness does not certify
those gates or scientific accuracy. The unchanged command/cwd/resources/wall/
CNY1.00 contract was submitted exactly once as15782177 under later explicit
queue steering. The scope is open; the sole scheduler attempt is consumed.
The first snapshot is PENDING/Resources with13 runtime paths absent. No paid
allocation/preflight/GPU/numeric execution is evidenced as of that snapshot;
cost/card-hours0 are snapshot-only. No terminal/release claim follows. Original
false scope bytes and earlier no-submission report remain preserved history.
Stop LLM monitoring; future retrieval must use the same job, never resubmit.

## Slurm22.05.7 compatibility revision

The active preparation now uses independent `T7-affine-paid-20261011-v2`
checkout and `T7-summary-affine-paid-gpu-20261011-v2` output identities. The
old df8cfac/v5 false scope, v1 checkout and receipts remain unchanged.

One actual visible RUNNING CPU job15779975 was read once with installed
`scontrol --details -o`; the preceding gnall-only selection returned empty and
was corrected to any visible RUNNING job. This is schema evidence, not a
GPU allocation or availability check. Public trimmed fields/provenance are
in `tests/fixtures/slurm22/`; full private metadata and command receipts are
under `outputs/T7-affine-paid-preparation-20261011/slurm22-compatibility-v1/`.
The CPU record emits `TRES`, `OverSubscribe=OK`, `Nodes`, `CPU_IDs`, `GRES`
and scheduler `StartTime`/`EndTime`. It has no allocated GPU detail and proves
no GPU index/UUID/cgroup or actual batch-environment values.

The installed binary reports22.05.7; its prefix has no manpages. Exact
[22.05.7 job formatter](https://github.com/SchedMD/slurm/blob/slurm-22-05-7-1/src/api/job_info.c#L696)
prints `TRES` from allocated TRES, with a requested-TRES fallback when no
allocation exists. Therefore legacy `TRES` admission additionally requires
RUNNING, the actual single-node detail, four distinct allocated CPU IDs and
allocated `GRES` physical index. `ReqTRES` and `JOB_GRES` never prove allocation.
If both TRES schemas exist they must agree; unknown/conflicting details reject.
The detailed formatter takes `GRES` from allocated `gres_detail_str`; the
alternate `GresDetail` field remains supported with the same strict index gate.

[Job sharing strings](https://github.com/SchedMD/slurm/blob/slurm-22-05-7-1/src/common/slurm_protocol_defs.c#L2807)
map default to `OK`, explicit no-sharing to `NO`, requested oversubscription to
`YES`. Admission requires `OK` for the declared default nonexclusive job.
Partition-level `OverSubscribe=NO` is a separate saved policy and is not the
job flag. `OK` alone does not prove that the partition forbids oversubscription.
The one necessary pre-submission refresh after specific approval rechecked
gnall's partition `OverSubscribe=NO` policy; it passed, but fixed n01 GPUs8/8
were allocated. No subsequent resource query or availability loop follows.
Neither exclusivity nor requested oversubscription is introduced.

The tagged `env_array_for_batch_job` and sbatch OUTPUT ENVIRONMENT documentation
do not promise `SLURM_JOB_START_TIME`/`SLURM_JOB_END_TIME`. A bounded CPU-only
helper resolves actual scheduler StartTime/EndTime before outer deadlines;
any original env token, when present, must match exactly. The supervisor reads
the allocation independently and matches the shell's start epoch. End minus
start must be exactly1200s; elapsed runtime/current clock and release reserve
remain strict. Missing/unparseable time fails closed; no shell-clock fallback,
deadline extension or synthetic replacement env token is allowed.

This revision changes admission only. Resources, costs, scientific inputs,
precision, numeric gates and phase caps are unchanged. Real paid batch/GPU
boundaries remain unverified; later explicit queue steering submitted the
unchanged job once as15782177, first snapshot PENDING/Resources.

Current affected paid Python/shell verification passes136 distinct cases;
the independent shell32-case run overlaps and is not counted again. The
successful combined receipt is `slurm22-compatibility-v2/paid-tests.v2.receipt.json`
under the preparation output directory; prior failed fixture and shell runs
remain preserved. The consolidated v2 diff, commit/deployment binding, false
scope and target CPU argv/import evidence are recorded in
`slurm22-compatibility-v1/final-delivery.v1.json`. Those CPU checks certify
preparation only; no actual paid allocation or GPU acceptance is implied.
