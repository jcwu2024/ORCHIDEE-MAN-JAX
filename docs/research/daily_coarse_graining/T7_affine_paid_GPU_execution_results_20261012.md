# Paid affine v2: resource non-admission — 2026-10-12

The specifically approved single paid job was **not submitted**. The fixed
node `ibc13b03n01` had all8 GPUs allocated at the one required refresh. This
is resource non-admission, with no runtime or numerical result. Human approval
remains valid and unconsumed; this dispatch stops without polling, automatic
monitoring, a different node, additional diagnostics or training.

## Approval, identity and resource evidence

The human message `批准。` in supervisor chat
`01a128cb-74c2-7fd0-822a-e411d7bcc64e`, message
`01a12da6-1e1b-73c3-b582-dbc95cada758`, was directly verified by the parent
before relaying this exact executable authorization. The approved contract is
gnall/`ibc13b03n01`,1node/1task/4 actual Slurm CPUs/1 V10032GB, nonexclusive,
20min/no-requeue/one scheduler attempt. CPU0.0933333+GPU0.7333333 gives an
additive maximum0.8266667 CNY, upward0.83; the approved ceiling is **CNY1.00**.

The accepted execution identity remains
`f4ae942ae2bf806337d6518d6683d73a1de4ffa8`; no source or deployed checkout
was changed, and this report's eventual documentation commit is not execution
identity. Frozen cwd is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/T7-affine-paid-20261011-v2`.
Approved command is `sbatch` on that checkout's
`scripts/hpc/slurm_shared_latent_T7_batch_numerics_paid.sh`, submitted via
gln01 only if the fixed resources are admitted.

The [raw refresh](../../../outputs/T7-affine-paid-execution-20261012/one-policy-node-refresh.v1.stdout.txt)
and [command receipt](../../../outputs/T7-affine-paid-execution-20261012/one-policy-node-refresh.v1.receipt.json)
record one read of gnall partition and the fixed node through gln01, actual
submission-endpoint host `ibc13b02n04`. Cluster stdout date is
`2026-10-12T01:06:20Z`; local receipt start is
`2026-10-12T01:14:40.302113+00:00`. These are separate recorded clock sources,
not a single reconciled timestamp or evidence about any workload's timing.

| Observed field | Actual value / consequence |
|---|---|
| gnall state / partition sharing | UP / `OverSubscribe=NO`; policy admitted |
| Fixed node state | MIXED |
| CPU allocation / total / scheduler load | 8 /112 /8.02 |
| GPU configuration / allocation | `gpu:v100:8`; `AllocTRES gres/gpu=8` |
| Unallocated scheduler GPU count | 0; required1 GPU unavailable |
| Submission endpoint load |43.54/43.61/43.75; separate from fixed-node scheduler load |

Partition `NO` was read independently; job `OverSubscribe=OK` was not used to
infer partition policy. No node or GPU availability loop followed the failed
resource gate. The saved snapshot does not reserve resources.

## Scope, artifacts and accounting

The target false scope's original bytes were saved as
[scope.before-approval.v1.json](../../../outputs/T7-affine-paid-execution-20261012/scope.before-approval.v1.json).
The target remained false/proposed; its SHA256 is
`99e3f32ef0ebac209fca077157c3d09186271dc15f6429c51fa4aa1ae5966e22` before
and after this stage. Human approval is separately preserved in
[approval-and-resource-nonadmission.v1.json](../../../outputs/T7-affine-paid-execution-20261012/approval-and-resource-nonadmission.v1.json).
No runtime scope opening or scheduler-attempt consumption occurred.

The [bounded final evidence](../../../outputs/T7-affine-paid-execution-20261012/bounded-final-evidence.v1.stdout.txt)
confirms the deployed clean HEAD, unchanged scope/input identities and absence
of16 declared attempt/ledger/result/preflight/phase-log/HLO artifacts. It is
ordinary evidence retrieval, not another resource check or numerical probe.

| Requested execution evidence | This stage |
|---|---|
| sbatch / job ID / Slurm terminal state | Not submitted / none / not applicable |
| Allocation start/end, actual CPU/GPU allocation | Not applicable; no job allocation |
| Occupied GPU seconds / card-hours / estimated charge |0 /0 /CNY0.00 from no allocation; not an account-wide invoice audit |
| Attempt / ledger / mutex / release NVML | Not created / not reserved / not acquired / not applicable |
| GPU mapping, cgroup, container/CUDA/PJRT preflight | Not executed |
| Component stdout/stderr, raw arrays, optimized HLO | Absent; no component was launched |
|12 measurements/91 arrays and strict oracle/AD/isolation/repeats | Planned only; none executed or analyzed |

Historical CPU12/91 and the accepted136 compatibility tests remain prior
evidence; neither is relabeled as this stage's GPU or numerical success.

## Preserved scientific contract and stop

The bounded identity check retained the exact recovery inputs and sources:

| Asset | SHA256 |
|---|---|
| Recovery operand | `bf36a098b4e6a89d555f8d18521555f2da66135425d9435cfe01878adeec6129` |
| Recovery capture receipt | `f863a2d964f626cec327981d86a8832e73aa45af62d708a342805c4aa4220e00` |
| Capture source | `70619b0a6445cde87eccaeb31feac46bbb34cdc0d7a6f68c68938c4d09901072` |
| Affine probe source | `05e2ee66b1905fd238722d3a4cf645e74639f7134de4af5a700cf0b657305ee0` |

Sample3/raw434, scalar/vmap/lax.map once each plus serial merge once,
repeats3/12 measurements/91 arrays, FP32 neural/FP64 oracle and strict
`rtol=1e-5, atol=1e-6` remain unchanged. The cotangent is diagnostic, not the
original T7 loss. Allocation1200s/preflight240s/probe780s/release≥180s/24GiB
and OMP4/OpenBLAS1 were not consumed or relaxed.

The free b676 approval remains valid/unconsumed and its dispatch stays paused.
No old b676/df8/v1/main/shared checkout, historical input, or frozen false
proposal was changed or moved. This dispatch is stopped; there is no automatic
retry, future wakeup or monitoring. Renewed human approval is not required for
the unchanged grant, but a future dispatch needs explicit steering rather than
an unattended resource loop. Actual paid GPU/runtime/numerical gates remain
unverified and no scientific repair or training-method acceptance follows.
