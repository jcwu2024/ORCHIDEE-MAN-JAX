# Current-model T7 precision/FD execution outcome

Updated: 2026-10-15. Status: **approved, resource-blocked, stopped without submission**.
The human explicitly approved the exact current-model proposal. No eligible GPU
was available at the bounded resource refresh, so this task stopped before
creating an approval file, submitting `sbatch`, or writing a mutex or ledger.
This is resource non-admission, not a runtime or numerical failure.

The accepted [preparation](T7_current_model_precision_admission_preparation_20261014.md)
remains valid within its CPU/source scope. It establishes the actual public
B1/artifact299/window0/T7 callable and exact input/parameter identities, target
CPU dry-run, saved-only full3854 endpoint inspection and shell→parser expansion.
It does not establish new full-model FP64 execution or derivative truth.

## Approved identity and bounds

The immutable remote checkout is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-current-model-precision-admission-20261014-v2`,
at exact execution source `30d7eb7675485fe3ce6bc3d877851feaea9854d6`.
The exact proposal SHA256 is
`59b09320d19470d2486ec7efe0e654cc749536a6fe2227debeba378b6ff2aa0f`.
The refresh confirmed the checkout clean and proposal hash unchanged.

The approved model is the existing 66,123,713-parameter latent32/private-reader64
candidate, B1 artifact299/window0/T7 public loss. The bound is one V100, four
allocated Slurm CPUs, 5400 seconds, 24 GiB and CNY3.72, with at most one real
FP64 compilation and 52 complete calls: 16 native state-VJP/JVP calls and 36
scalar FD calls through the same executable. No retry or training is included.
The four directions are 434, 2382, 2621 and control200; coordinate2621 remains
an exposed computational partial, without a physical-manifold claim.

## Resource refresh and stop

The captured scheduler snapshot reported:

| Nodes | State/resource fact | Eligibility at this snapshot |
|---|---|---|
| b02n02, b02n03, b03n01–b03n05, b03n07 | UP and not drained; each allocated 8 of 8 GPUs | No available GPU |
| b02n01 | MIXED+DRAIN; allocated 6 of 8 GPUs | Drained; not eligible |
| b03n06 | DOWN+NOT_RESPONDING; allocated GPUs 0 | Down; not eligible |

b03n04 had 32 allocated CPUs/load8.04 and b03n05 had 32/load8.01, but both
had all eight GPUs allocated. b03n07 was CPU-IDLE with zero allocated CPUs and
load0.01, while all eight GPUs were allocated; CPU idleness did not provide a
GPU slot. The ordinary and `--all` gnall queue views were empty for this user,
but `PrivateData=jobs` hides other users' job end times. There was no provable
short release window. These are snapshot facts, not a claim that resources can
never become available.

cln01 and gln01 were reachable. No GPU discovery, GPU command, preflight or
runtime child was started. The accepted stop ends this precision/FD diagnostic
line under this grant: no automatic resume, retry, budget extension, layer
graph, training, automation or further resource polling follows.

## Measurements and accounting

| Direction | New native FP64 JVP/VJP | New repeat/full-state/physics gates | Independent FD |
|---|---|---|---|
| 434 | Not measured | Not measured | Not measured; inconclusive |
| 2382 | Not measured | Not measured | Not measured; inconclusive |
| 2621 | Not measured | Not measured | Not measured; inconclusive |
| control200 | Not measured | Not measured | Not measured; inconclusive |

Full3854 FP64-versus-saved-FP32 comparison, precision admission, memory/runtime
gates and all other new scientific outcomes are unmeasured/inconclusive. The
saved FP32 projections and scalar-resolution bounds remain historical/preparation
evidence, not new FP64 results. The separate strict state20 nonfinite failure
remains unresolved; no value is filled in or exempted by this stop.

Actual submissions, job IDs, allocations, model compilations and model calls are
all zero. No approval file, mutex or ledger was written. GPU allocation/card-hours
and allocated CPU-hours are zero; estimated allocation cost is CNY0. Actual
billing was not queried. Release is **not applicable**, because no allocation or
owned GPU process existed. The paid GPU execution attempt was not consumed,
but that fact does not authorize continuation after the accepted stop.

The resource-blocked receipt and captured raw snapshot are retained under
`outputs/T7-current-model-precision-admission-execution-20261015/resource-blocked/`.
The exact receipt is `resource-blocked.json`; captured evidence is
`scheduler-nodes.captured-stdout.txt`, `partition.captured-stdout.txt`,
`private-data.captured-stdout.txt`,
`queue-gnall.user-visible.captured-stdout.txt` (empty), and
`connection-identity.captured-stdout.txt`, with command provenance in
`read-command-provenance.json`. These files transcribe actual captured tool
stdout; creating them involved no additional resource queries. The receipt
records session-clock 2026-10-14 16:02:45 UTC and cluster-clock
2026-10-14T15:56:58Z after the PrivateData query as separate observations,
without inferring clock synchronization; the task date is 2026-10-15.
CPU preparation evidence remains under
`outputs/T7-current-model-precision-admission-preparation-20261014/v2/server-preparation/`.
Original source/assets and historical results are preserved. No reference,
precision adoption, B4 repair, complete-update speed gain or training acceptance
follows. [NEXT_STEPS](../../NEXT_STEPS.md) owns the current stop boundary.
