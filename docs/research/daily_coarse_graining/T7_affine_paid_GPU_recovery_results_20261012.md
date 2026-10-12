# T7 affine paid v3 recovery result — 2026-10-12

The authorized single v3 job **15806646** ended **FAILED, exit2:0, after7s**
on the approved `ibc13b03n01`. Supported CSV and UUID-bound XML queries both
succeeded. Startup then rejected wildcard GPU access in the actual job device
cgroup: single-device isolation was not admitted. No Supervisor, runtime
preflight, CUDA/PJRT verification, affine graph, measurement or optimizer update
ran. This is a startup admission failure, not a numerical failure.

The independent v3 scope is now closed/authorization exhausted; scheduler and
Python attempt markers are consumed. No repair-and-rerun, replacement, card
change, free fallback, training or automatic monitoring follows. The free b676
approval remains preserved, paused and unconsumed. All old v2 evidence remains.

## Authorization and frozen identities

The human replied “批准。” in execution chat
`01a128db-3508-7100-8f6d-0f9a2584d46f` after delivery of the
[complete v3 proposal](T7_affine_paid_GPU_recovery_proposal_20261012.md).
The parent relayed and verified this direct approval. The saved provenance states
that fact; no user-message identifier was supplied or invented.

Approved contract: gnall/fixed n01,1node/1task/**4 actual Slurm CPUs**/1V10032GiB,
nonexclusive,20min,no-requeue,one scheduler attempt. CPU0.07 and GPU2.20 CNY/hour
are additive: worst-case quote0.8266667/upward0.83, approved ceiling **CNY1.00**.
The1200s clock begins at actual scheduler StartTime, with preflight240s/probe780s/
release reserve≥180s,24GiB,OMP4/OpenBLAS1. Startup then unchanged affine12/91 were
authorized within this one allocation; failed startup ends the attempt.

Execution SHA remains **58a2d93885e991fc8e5e751f8a52b27aaade08e0**.
Local documentation HEAD at dispatch was `00de1304520511a656d1ac2844a039dfcb3e6758`;
later documentation commits do not refreeze or redeploy execution.

```text
cwd: /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-recovery-20261012-v3
command: sbatch /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-recovery-20261012-v3/scripts/hpc/slurm_shared_latent_T7_batch_numerics_paid.sh
output: /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/T7-summary-affine-paid-recovery-gpu-20261012-v3
scope ID: T7-summary-affine-paid-recovery-gpu-20261012-v3
```

The original false scope was preserved locally and remotely before opening:
`1237f75b88dce058d66aacb7986cfb18a9695a351aee203bb7a9ab81ef5c11df`.
Approved open scope SHA is
`8c3a4fcc3f58f9781abee11da7e3889a7517d5278c4997cb126ac04f09d37c1f`;
closed scope SHA is
`5b6b24bbb54c98879d74033e6d1b21ed9bc9d4e9f77dafe5ae14fa7092c077d4`.
Source manifest, fresh recovery NPZ `bf36a098...6129`, receipt `f863a2d9...0e00`
and capture source `70619b0a...901072` matched before opening and dispatch.
Frozen code was clean at admission and terminal closeout.

## Sole dispatch and actual allocation

One pre-submit snapshot at2026-10-12T08:53:38Z found gnall UP/OverSubscribe=NO;
n01 IDLE,CPUAlloc0,CPULoad0.01,AllocTRES empty,configured8V100. This is a
pre-submit snapshot, not a whole-node reservation or proof of later load.
CPU-only gln01 metadata admission imported no JAX, initialized no GPU, checked
exact HEAD/source/input/scope identities, absent attempt/ledger/results/mutex,
and known free-state/sole paid ownership. No preparation tests were rerun.
The inherited `SBATCH_*` capture was empty; the submit shell nevertheless
cleared such overrides before the exact approved sbatch, preserving directives.

Exclusive local intent/invoked and remote call/accepted receipts surround the
**one** successful scheduler invocation. Acceptance consumes the attempt even
if PENDING. Scheduler records subsequently showed:

| Field | Actual evidence |
|---|---|
| Job/state/exit |15806646 / FAILED /2:0 /Requeue0 /Restarts0 |
| Submit |2026-10-12T16:56:11+08:00 |
| Allocation Start/End |16:56:31→16:56:38+08:00 |
| Node/resources |ibc13b03n01 /1node /1task /NumCPUs4 /CPUsPerTask4 /gres/gpu1 |
| Request |gnall /fixed n01 /TimeLimit00:20:00 /no exclusion |
| Allocation duration |7s /0.0019444444 GPU card-hours |
| Estimated elapsed charge |CPU0.0005444444 +GPU0.0042777778 =CNY0.0048222222 |

Elapsed estimates use reported allocation duration and current additive rates;
the actual bill/minimum billing granularity is unknown. GPU-child compute
occupation is0 because admission failed before child creation. It is not0 paid
allocation cost. Historical v2's separate8s/CNY0.0055111111 remains separate.

## Observed startup boundary and limitations

Both retained query commands exit0 with empty stderr:

```text
nvidia-smi -i 0 --query-gpu=index,uuid,name --format=csv,noheader,nounits
nvidia-smi -i GPU-3d780420-3659-2d22-3ffa-c6860e4cad9c -q -x
```

CSV selected index0/UUID `GPU-3d780420-3659-2d22-3ffa-c6860e4cad9c`/
TeslaV100-SXM2-32GB. UUID-bound XML supplied the same UUID/name and Linux minor0,
driver525.85.12 and total32768MiB. The current n01 observation confirms this
compatibility route works for this dispatch. It does not recover old job15782177's
missing child stderr or establish its exact cause.

The raw stdout contains `paid_admission_rejected`, `GPU_executed=false`, and
`wildcard GPU access does not prove Slurm device isolation`. In frozen source,
`admit_gpu_mapping` rejects wildcard read/write access involving the selected
GPU's major, or all-device access; this follows kernel character-device/minor
checks and actual-job cgroup membership lookup. The refusal remains unchanged.
The rejection retains complete CSV/XML outputs, but does **not** retain the
actual `devices.list`, kernel stat rows or cgroup path. Therefore the exact
wildcard line, its origin and any scheduler configuration remedy are unproved.
Single-device isolation, CUDA/PJRT identity and science remain unaccepted.

No Supervisor/preflight/ledger/result/partial/resultsNPZ/outer-timeout exists at
retrieval and closeout. No graph/HLO/binary was created. The planned FP32 affine
scalar/vmap/lax.map plus serial merge, independent FP64 oracle,3 repeats,
12measurements/91arrays and strict `rtol=1e-5,atol=1e-6` were **not executed**.
Full primal/JVP/input/parameter gradients, masks/isolation/repeats remain gates;
the diagnostic cotangent is not the public T7 training loss. No repair,
throughput gain, scientific accuracy or architecture adoption follows.

Scheduler allocation ended and batch/extern accounting are terminal. Mutex was
absent at terminal retrieval and closeout. Since rejection preceded Supervisor
creation, source releases the acquired cooperative mutex without GPU-child
cleanup. No post-exit live GPU/process observation was performed; the query's
0MiB/0% values are admission-time observations, not a later cleanup receipt.
No claim of live post-exit GPU release validation is made.

Terminal closeout v1's CPU-only `scontrol` lookup failed because the completed
job had aged out of live controller records, before scope mutation. Its raw
exception is preserved. CPU metadata closeout v2 used durable `sacct`, verified
the same FAILED2:0/7s and absence gates, then closed only this canonical v3 scope.
That bookkeeping correction is not a scheduler/GPU retry.

## Saved artifacts and verification

Local evidence root is
[`outputs/T7-affine-paid-recovery-execution-20261012/`](../../../outputs/T7-affine-paid-recovery-execution-20261012/).
The terminal archive contains the complete v3 directory at pre-close retrieval,
including raw Slurm stdout/stderr and both consumed attempts; SHA
`28f582fdd0ab13b522c37be5dc43d340733281dfab242a773b42d62e5e41c1dc`.
Original false/open snapshots remain; separate closed scope/closeout files are saved.

- [Saved-only summary](../../../outputs/T7-affine-paid-recovery-execution-20261012/execution-summary.v1.json)
  and [structured rejection](../../../outputs/T7-affine-paid-recovery-execution-20261012/admission-rejection.saved.v1.json)
  bind the observed error, commands, timings, costs and non-execution limits.
- [Sole submission receipt](../../../outputs/T7-affine-paid-recovery-execution-20261012/sbatch-once.v1.receipt.json),
  [scheduler status](../../../outputs/T7-affine-paid-recovery-execution-20261012/first-job-status.v1.stdout.txt),
  [terminal extraction](../../../outputs/T7-affine-paid-recovery-execution-20261012/terminal-extraction.v1.json)
  preserve actual invocation, allocation and complete saved evidence.
- [Closed scope receipt](../../../outputs/T7-affine-paid-recovery-execution-20261012/closed.terminal-closeout.v1.json)
  binds exhausted authorization, absent mutex/ledger/results and unchanged free state.
- [Raw query1](../../../outputs/T7-affine-paid-recovery-execution-20261012/gpu-query-1.raw.v1.stdout.txt)
  and [query2 XML](../../../outputs/T7-affine-paid-recovery-execution-20261012/gpu-query-2.raw.v1.stdout.txt)
  retain complete metadata, with separate empty stderr files.

Saved-only local CPU parsing independently checked one structured rejection,
both exit0 queries, exact XML UUID/minor/driver, absent numerical arrays and
resource arithmetic. No GPU probe or historical numerical result was rerun.
The preparation's Python30/shell10 and actual test01 Bash/parser evidence remains
valid for unchanged source. Six documentation files constitute this result stage;
commit/push/clean receipts and final artifact package reside beside the summary.

Stop after this terminal result. A new live runtime isolation experiment or GPU
allocation requires its own bounded authorization; this consumed grant and its
unused time/ceiling authorize no new allocation or GPU execution. Saved-evidence
and CPU static review remain distinct from a new live run.
