# Paid GPU identity: completed local v2 repair candidate

2026-10-12. Local code, mock validation and current documentation repair only.
No remote connection, deployment, resource refresh, scope opening, submission,
GPU initialization, preflight, affine probe, training or fallback ran in this
stage. This is a candidate identity contract requiring acceptance before any
new execution. The [sole v1 terminal result](T7_paid_single_GPU_execution_results_20261012.md)
remains FAILED2:0/6s/not_executed, closed/consumed. Frozen c8a44c3, 58a/f4ae,
their scopes, recovery inputs and scientific evidence remain untouched.

## Confirmed engineering error and saved replay

Frozen c8a44c3 `observe_gpu_mapping` passed `SLURM_JOB_GPUS` directly to
`nvidia-smi -i`, then required NVML index equality in observation/admission.
Job15812634 had Slurm GRES IDX5, original CVD0; the sole `nvidia-smi -i 5`
CSV query exited6 with `No devices were found\n`, empty stderr. A saved-only
replay of that exact failure against the frozen function matches the raw
command/exit/stdout/stderr and its executed-manifest source SHA
`7bb3fd798463ad7c77722e2baaee76adf01ac6e497950c7d9d96ced9b240e143`.
This proves the old path rejects that observation, not a5→0 remapping,
driver/configuration cause or an actual usable replacement index.

Slurm's GRES index, NVML's displayed index, Linux character-device minor and
CUDA-visible ordinal are separate identity spaces. The local v2 route never
passes a Slurm integer to NVML or assumes index equals minor. It never chooses
the first card from a multi-visible device list.

## Explicit v2 identity change and association gate

Candidate token: `T7_affine_allocated_single_gpu_v2`; candidate code/output
path constants end in `T7-affine-paid-single-gpu-20261012-v2` and
`T7-summary-affine-paid-single-gpu-20261012-v2`. No deployment or scope was
created. Old v1 approval cannot satisfy the new scope/token. The legacy strict
mapping helper/default and v1 driver-count-one branch remain for compatibility.

The CPU management process imports pure helper definitions and never loads
CUDA or JAX. Its Supervisor launches an owned discovery child in the same
accepted container, interpreter, host driver binds and toolchain used by the
later preflight/probe. Discovery preserves the shell-captured original Slurm
CVD and CUDA_DEVICE_ORDER. It does not set a new ordering to reinterpret an
integer mask. Missing/multiple/invalid visibility rejects before library load.
Parent ordering and the child's observed ordering must agree; removing a stale
Singularity export alone does not prove the image adds no ordering value.

Documented CUDA-runtime `cudaGetDeviceCount` must return1. Ordinal0 then supplies
the physical PCI address via `cudaDeviceGetPCIBusId`. Driver
`cuDeviceGetByPCIBusId` resolves that measured address to a physical handle;
`cuDeviceGetPCIBusId` must round-trip to the same address and `cuDeviceGetUuid`
provides its UUID. The UUID is explicitly driver-derived through runtime PCI,
not an invented independent runtime UUID API. The implementation avoids an
undocumented `cudaDeviceGetUuid` runtime symbol and guessed cudaDeviceProp ABI.

**V2 changes the identity contract:** runtime-visible count1 and PJRT one GPU
at local hardware ordinal0 enforce execution visibility. Total driver
enumeration is recorded separately and may exceed1; it is not assumed to
obey CVD. Physical driver UUID/PCI binding must still match the selected
runtime device. Fresh preflight and the actual affine process repeat this
binding before affine upload/compilation. This is an explicit candidate change
from v1's driver-count1 predicate, requiring acceptance; science is unchanged.

Host CSV and XML queries select only that UUID. CSV UUID/name, XML UUID/name,
physical PCI address, XML minor and actual character-device major/minor must
agree. Actual-job devices-v1 membership and complete raw rules are retained.
The physical allocation association additionally requires one of:

| Original scheduler mask | Required association evidence |
|---|---|
| One full GPU UUID | Exact match to the PCI-selected driver UUID and UUID-bound NVML/kernel identity. |
| One integer token | Actual-job rules permit read/write on the unique selected GPU-data major/minor pair, excluding recorded control pairs; UUID-bound XML/kernel pair matches. Conservative broader/ambiguous data-access rules reject. |
| Integer token with broad/ambiguous access or missing identity | `physical_allocation_mapping_unknown`; reject before preflight/affine. Runtime/driver/NVML agreement alone can identify the same wrong physical card and cannot close this gap. |

These bases use trusted scheduler mask provenance and bounded actual-job
device evidence. They do not independently prove the node's GRES File↔UUID
configuration or OS denial of opening every other card. Broad rules do not
prove another GPU computation; they also do not supply a missing integer-mask
allocation association. `os_other_gpu_open_denial` remains `not_verified`.
Missing, conflicting or unsupported evidence fails closed.

## Lifecycle and environment audit

Discovery uses the existing Supervisor reservation, phase process groups,
watchdog, shell deadman and cleanup. Discovery and preflight share the remaining
240s startup window measured from the scheduler allocation start; no additional
240s is added. The unchanged original probe remains capped at780s, protected
release at least180s within1200s, 24GiB memory gate and strict source/input/math
checks. There are no separate paid requests per discovery/query command.

Discovery requires only the explicit v2 token and original mask, with no
expected UUID yet. The paid shell exports those values into the accepted
container and removes stale UUID/order exports. After identity/association
admission, both preflight and probe receive the explicit contract, expected
UUID and UUID CVD. Allowlisted entry/interpreter/argv checks stay mandatory.
The paid monitor also selects UUID for every memory/process/cleanup query;
NVML indices are retained observations, not secondary selectors.

Discovery errors retain raw child stdout/stderr/report and completed host
query outputs. Discovery is a GPU-runtime child even though it allocates no
scientific arrays or compiles JAX. CPU management cannot release the mutex
merely because an affine child was never reached. Owned groups are cleaned up;
an unknown UUID cannot certify live GPU cleanup, so the mutex remains and the
existing conservative full1200s reservation charge is retained. Known-UUID
cleanup still requires zero observed memory/processes/utilization. Failed
preflight/association remains `not_executed` numerically.

Transient GPU memory during identity discovery is unobserved until UUID is
known; the phase records that limitation. Later monitored peak/release samples
must not be presented as a complete measured discovery peak. Local mocks
establish no real container/API support, GPU memory or release acceptance.

## Local validation and next evidence

Independent API/association/full-lifecycle fixtures pass49 cases: nonzero
Slurm ID with job-local NVML/CUDA0, NVML index≠kernel minor, PCI-selected driver
handle6 with total enumeration8, allocated-only visibility, multi-visible
rejection before ordinal selection, UUID/PCI/environment conflicts, broad
integer-mask rejection, preserved query failure and owned-group cleanup.
Actual copied Bash runtime-child fixtures pass47 cases, including set/unset/
empty/stale ordering and container contract/UUID passthrough; no real container
or CUDA interpreter runs. Paid launcher/startup-evidence regressions pass130
cases. The first shared integration run retained92 passes and one failure in
an existing private Bash deadman timing fixture. Its readiness/clock repair
preserves actual owned-child signals, TERM→KILL ordering and the independent
blocked-supervisor check; production deadman code is unchanged. The repaired
shared-shell suite passes14 cases. The logs cannot distinguish trap-readiness
from wall-clock jitter as the original cause, and the corrected mock epochs
do not certify real Linux process-group/scheduler timing. Reusing unchanged
passing suites gives270 distinct passing cases across seven affected files,
without a further all-suite rerun. Bounded results and the retained first failure are in the
[validation summary](../../../outputs/T7-paid-single-gpu-execution-20261012/local-repair/validation-summary.v1.json).
Existing pytest `cache_dir` warning is retained; fixtures use system TEMP,
bytecode/cache generation disabled. No historical scientific probe was rerun.

CUDA12.6.2 [runtime Device API](https://docs.nvidia.com/cuda/archive/12.6.2/cuda-runtime-api/group__CUDART__DEVICE.html)
documents count/PCI operations. The parent retained its read-only official
page in `outputs/T7-paid-single-gpu-contract-20261012/cuda12-runtime-device-api.parent.html`.
The [driver Device API](https://docs.nvidia.com/cuda/archive/12.6.2/cuda-driver-api/group__CUDA__DEVICE.html)
defines PCI handle and UUID operations. Local API fakes validate argument/handle
binding, not deployed driver enumeration, symbols or physical allocation.

Before proposing a new complete bounded allocation, resolve the smallest
pending association facts: authoritative node/job GRES File↔physical UUID
mapping when available, or a scheduler UUID mask/actual-job unique data-device
pair sufficient for the bounded association gate. Job15812634 did not retain
its actual-job rules, so no blind rerun is justified by the old failure alone.
Separately authorized CPU metadata may establish the configuration in advance;
otherwise any required actual-job observations belong to the one future
startup+original12/91/4 probe allocation. Real libcudart12/API/container support,
unchanged ordering, runtime/PJRT visibility, memory and cleanup remain unverified.
No new execution authorization follows from this local repair.

The [local package](../../../outputs/T7-paid-single-gpu-execution-20261012/local-repair/)
retains frozen-query replay, bounded test receipts/raw failures, actual diff,
preservation, final checks and commit/push delivery. Main implementation:
`shared_affine_cuda_identity.py`, paid launcher/shell, shared Supervisor/runtime
shell and paid Slurm entry under `scripts/hpc/`. Tests:
`test_shared_affine_gpu_discovery.py`, `test_shared_affine_single_gpu_contract.py`,
paid launcher/startup-evidence/shell tests and the shared launcher regressions.
No Fortran/Teacher/network/scientific model, weights, FP32/FP64 oracle, strict
`1e-5/1e-6`, original12/91/4 counts or optimizer/training method changed.
