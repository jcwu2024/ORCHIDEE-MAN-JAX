# Paid affine v2: startup admission failed — 2026-10-12

The sole paid job **15782177** ended **FAILED / 2:0** after an eight-second
allocation on `ibc13b03n01`. The first GPU metadata command returned2 before
GPU mapping, Supervisor creation, preflight or the affine probe. This is a
startup admission failure; no neural numerical result exists. The scheduler
attempt is consumed. There is no new submission or GPU diagnostic in this
follow-up.

## Terminal evidence and accounting

The supervisor supplied the terminal sacct rows and combined visible log.
They are preserved with explicit provenance in
[supervisor-terminal-provenance.v1.json](../../../outputs/T7-affine-paid-failure-20261012/supervisor-terminal-provenance.v1.json),
with separate [raw sacct](../../../outputs/T7-affine-paid-failure-20261012/supervisor-supplied-sacct.v1.txt)
and [supplied log](../../../outputs/T7-affine-paid-failure-20261012/supervisor-supplied-combined-log.v1.txt).
No fresh scheduler query or invented remote-query receipt was produced.

| Supplied actual scheduler field | Value |
|---|---|
| Job / batch |15782177 /15782177.batch: FAILED, exit2:0 |
| Extern step |15782177.extern: COMPLETED, exit0:0 |
| Start / end |2026-10-12T10:11:21 /2026-10-12T10:11:29; timezone not supplied |
| Elapsed / allocation |8 seconds /4 Slurm CPUs /1GPU /fixed n01 |
| TRES |billing=4,cpu=4,gres/gpu=1,node=1 |
| Estimated additive charge |CPU0.0006222222 +GPU0.0048888889 = CNY0.0055111111 |

CPU is `4×8/3600×0.07`; GPU is `8/3600×2.20`. One GPU used
`8/3600 = 1/450` card-hours. These rate estimates are not an invoice;
minimum charges and invoice rounding remain unverified. The approved20min
maximum0.8266667/upward0.83/ceiling1.00 contract was unchanged. Scheduler
terminal evidence ends the allocation, but does not prove a live GPU cleanup
observation. Earlier PENDING14:41 predictions and CNY0 apply only to the
[historical queue snapshot](T7_affine_paid_GPU_queue_status_20261012.md).
The earlier [zero-submission resource result](T7_affine_paid_GPU_execution_results_20261012.md)
remains unchanged historical evidence.

## Read-only retrieval and cleanup limits

The bounded retrieval preserved10 existing known files byte-for-byte, including
scope/current-original approval snapshots, scheduler/Python attempts, own Slurm
stdout/stderr and original submission receipts. The
[retrieval manifest](../../../outputs/T7-affine-paid-failure-20261012/retrieval-manifest.v1.json)
links each remote/local path, size and matching SHA256. Original bytes are under
`outputs/T7-affine-paid-failure-20261012/retrieved/`; inventory and transfer
stdout/stderr/UTC command receipts are retained beside the manifest. This was
file retrieval only, without any scheduler or GPU query.

The Slurm attempt says `scheduler_started/retry_allowed=false`; the Python
attempt says `scheduler_attempt_consumed`, job15782177, `GPU_executed=false`.
Ledger, Supervisor, preflight, outer-timeout, result/partial arrays and all3 HLO
files are absent. No missing artifact was fabricated. The known shared mutex
directory and owner file are absent at retrieval. No lock takeover, scope
reset, ledger mutation or remote cleanup was performed.

Frozen `run_live` acquires the cooperative lock, reads the known free admission,
then performs metadata observation. Its `finally` marks lock release safe when
no Supervisor exists and releases only its matching nonce/scope owner; it does
not launch a cleanup GPU command on this path. This source path and current
lock absence are consistent with early owned-lock release. They do not supply
a historical lock-release receipt or live NVML0MiB/0%/no-process evidence.
Scope JSON remains `open/execution_authorized=true` as preserved approval bytes;
the attempt receipts consume the grant and forbid reuse.

## Diagnosis and smallest local repair

The rejected command was exactly
`nvidia-smi -i 0 --query-gpu=index,uuid,name,minor_number --format=csv,noheader,nounits`.
Frozen source captures its stdout/stderr with `check=True`; the historical
main handler prints only `str(CalledProcessError)`. Retrieved Slurm stdout is
420bytes containing that rejection; stderr is0bytes. The command's underlying
stderr is therefore unavailable in the declared saved artifacts. Exit2 alone
does not prove unsupported `minor_number`, a wrong GPU/index, permission,
contention or faulty hardware. No compatibility or device-selection repair
has been selected.

The local change preserves failed subprocess command, return code and captured
stdout/stderr, including timeout output, in the existing rejection JSON. It
keeps exit2, `GPU_executed=false`, no-retry and all allocation/device/identity/
physical gates. It neither reruns nor repairs this historical failure. Two new CPU/mock cases exercise the real rejection path, retained command/
returncode/stdout/stderr, no Supervisor, owned lock release, consumed-attempt
reentry rejection and timeout partial bytes. The affected paid Python test
module passed106 cases, including2 new cases, in1.77s, with one existing
`cache_dir` configuration warning.
The [test receipt](../../../outputs/T7-affine-paid-failure-20261012/tests/focused.v1.receipt.txt)
and [actual stdout](../../../outputs/T7-affine-paid-failure-20261012/tests/focused.v1.stdout.txt)
retain command/date/exit0 and counts. The receipt supplies a date only, not
precise UTC timing; the elapsed1.77s comes from pytest stdout.
Actual deployment/GPU behavior remains unverified.

The execution checkout remains immutable **f4ae942ae2bf806337d6518d6683d73a1de4ffa8**;
accepted resource docs77afd803 and queue docs6886d26f are separate identities.
This follow-up changes only local source/tests/current documentation. Graph
Verify evidence supplied by the parent marks relevant worktree paths excluded/
not_tracked at generation2026-10-10T08:46:50Z. Fresh graph calls returned
transport-closed; exact frozen source fallback was used, without reindex or
completeness claims.

Stop after this package. The smallest future diagnostic would preserve exact
metadata-command stderr under separately authorized resources; it is not an
authorized next allocation or resubmission. No card change, isolation or
tolerance relaxation, new preflight/probe, original-input search, training or
automatic monitoring follows. Existing recovery identities and strict science
remain fixed; free b676 dispatch stays paused and its approval unconsumed.
