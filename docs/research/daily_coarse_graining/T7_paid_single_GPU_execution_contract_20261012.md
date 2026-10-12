# Allocated single-GPU affine execution contract

2026-10-12. New preparation contract `T7_affine_allocated_single_gpu_v1`,
scope `T7-summary-affine-paid-single-gpu-20261012-v1`. Preparation is authorized;
live execution is not. The proposed scope must remain `status=proposed`,
`execution_authorized=false`. The [preparation result/proposal](T7_paid_single_GPU_preparation_results_20261012.md)
will bind the actual immutable commit, target CPU evidence and concrete request.

## Deliberate boundary

This contract requires computation on the one GPU actually allocated by Slurm.
It records stronger OS denial of opening other GPU devices independently.
Both fields are explicit in the scope and frozen manifest:
`gpu_execution_contract=T7_affine_allocated_single_gpu_v1` and
`os_device_open_isolation_required=false`. A missing/different token or an old
scope ID fails admission; there is no fallback or migration of an old approval.
The helper's default remains `T7_affine_strict_device_isolation_v1`, preserving
the old conservative wildcard rejection for callers of that explicit route.
Old immutable58a/f4ae deployments and their consumed grants remain unchanged.

Slurm CVD, host NVML index and kernel minor are separate identity spaces. The
[evidence remediation report](T7_paid_startup_evidence_remediation_20261012.md)
links the saved official Slurm/CUDA/Linux sources and their limits. Broad device
permissions do not prove another GPU computation. The hypothetical
`c *:255 rw` control-wildcard case is a code example; actual job15806646 rules
were not retained and cannot be reconstructed.

## Mandatory execution chain

| Gate | Required evidence and behavior |
|---|---|
| Allocation/cost/clock | Actual numeric job, gnall/n04,1node/1task/4allocated CPUs/1V100, exact1200s scheduler start/end, no requeue/nonexclusive, exact approved commit/source/inputs, quote and ceiling; launch attempt is one-shot even on rejection. |
| Host identity | Existing Slurm allocation detail→CSV physical index/UUID/V100→UUID-bound XML minor→actual character device major/minor; control paths cannot be confused with selected data device. |
| Actual-job cgroup | Read `/proc/self/cgroup`, require supported actual-job devices-v1 membership and contained resolved devices.list path; retain complete raw rules, parsing and selected-device read/write permission. Unsupported/malformed/unknown membership still rejects. |
| OS permission assessment | Retain selected/control pairs, parsed rules and broader/ambiguous permissions. Wildcard/broad permissions alone do not block this new contract. `os_other_gpu_open_denial=not_verified` is truthful even for a narrow list: no complete hardware inventory/open test was performed. Never label that as hardware isolation proved. |
| Child/container | Supervisor sets exact UUID and contract; paid runtime-child rejects either missing, explicitly exports both Singularity variables and sets child CVD to UUID. Phase ownership and independent deadlines remain. |
| Preflight and actual affine process | Both use the shared identity helper: CVD must equal expected UUID; independent CUDA driver count1/UUID must match; PJRT must expose exactly one GPU at local hardware ID0. Actual affine binding runs before device_put, affine upload/compilation/AD. Identity-unknown, mismatched UUID, multiple driver/PJRT GPUs or missing mandatory hook rejects before science. Its raw API/count/UUID/status evidence is separately retained. |
| Results/release | Supervisor requires the actual affine result's identity as well as separate preflight identity. Existing JAX0.4.38,12measurements/91arrays/4small compiles, strict tolerances/all masks,24GiB, original scalar/vmap/lax.map/independent oracle gates and clean release remain. Unverified release retains cooperative mutex and conservative charge. |

No algorithm, weights, precision, tolerance, data, masks, gradients, loss or
recurrent science is changed. The actual affine identity gate is startup
admission; it adds no scientific graph and does not establish original T7 truth.
No environment dump or foreign process PID/name retention is added.

## One proposed allocation

Fixed `ibc13b03n04`, gnall,1node/1task/4actual CPUs/1V100,20minutes,
nonexclusive/no-requeue; no whole-node allocation. Preflight cap240s, original
affine cap780s, protected release reserve180s, total1200s and24GiB.
Startup admission and the original12/91 comparison belong to this same single
allocation, not separate paid requests. Additive ceiling arithmetic:
CPU `4×(1200/3600)×0.07=CNY0.0933333333`,
GPU `1×(1200/3600)×2.20=CNY0.7333333333`, totalCNY0.8266666667
(upward quoted0.83), approved ceiling to requestCNY1.00.

At2026-10-12T12:21:16UTC scheduler metadata showed n04 MIXED,
CPUAlloc48/112,CPULoad4.21,CfgGPU8/AllocGPU3. It had enough unallocated
resources and the lower load among the two eligible mixed nodes; it was not
idle. This snapshot is not a reservation. Refresh eligibility immediately
before any separately approved submission; do not substitute another node or
submit a replacement automatically. No current paid submission is authorized.
Free b676 remains paused/approved/unconsumed. No free fallback, training,
configuration repair, repeated polling or automatic monitoring follows.
