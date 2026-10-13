# CPU48 v4 batch-only execution protocol

This unpaid preparation is fixed by [execution-code-pin.json](execution-code-pin.json),
[the bound admission](scalar-audit/scalar-admission.json), and the
[literal submission proposal](PROPOSED_SUBMISSION.sh). It authorizes no deployment
or paid submission. The supervisor reviews these artifacts under the user's
existing delegated authority; the v3 grant is consumed.

Execution ref `codex/global-cpu48-b64-v4` is
`60dccbddde10be4295aab7e7fe55fa7864d3c3f3`, based directly on
`03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8`.
The 15 declared harness/test blobs come from reviewed research checkpoint
`493252e418f3fd827af4a6f7748a63d3157e680d`.
All 104 physical files retain tree
`75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba`, certified science SHA
`11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5` and actual
source SHA `b33450cae9cd8c9b2d5f7c6075fe270f461887edc09c77417c5e820a5c18fab8`.
The research branch's carbon fixes and tree `62e90547ee8eadab387713397bb111c399c81b6c`
remain preserved. [Local verification](local-package-verification.json) compares
the actual raw blobs and all declared artifact hashes.

Let `P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`. Future immutable checkout:
`P/runtime/worktrees/global-throughput-cpu48-b64-v4`; future output:
`P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/<job-id>`.
Use `P/.venvs/orcjax_cpu/bin/python`, accepted JAX/jaxlib 0.4.38 CPU runtime,
x64, one native CPU JAX device and 48 distinct allocated physical cores.
Preparation checks use isolated staging and a real restricted one-CPU allocation;
declared future identities are not evidence of production deployment.
The exact-pin dry-run used a staging admission copy with identical bytes and
mapped only the emitted coordinator and `--entry` locations to staged code.
The persistent admission destination and future checkout were not installed;
their actual clean-source admission remains a requirement of approved execution.

The sole new model phase is native FULL64 B64 batch/vmap. External scalar directory:
`P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar`.
Checkpoint `stage_B64_server_scalar_v1.pkl`: 27,604,466 B, SHA256
`1cddeaf80c2fc36f173835a5ff1bd4d8d520b144d4c96289ef6fe86dde898dfa`.
Admission must be installed, after approval, at
`P/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v4-20261014/scalar-audit/scalar-admission.json`,
SHA256 `f71eb5db4e638615ef5e06be5783d193bbfc1bd7563dd30c2424e04b40c14a8d`.
It binds audit receipt SHA256
`cb1331edb7094e74ec1814c6813a0eee218c58ea4e89f1c0c1f630e981479431`
and the exact new candidate, preserving original job/code/input/reference/window/order
and checkpoint/manifest/report/accounting identities. Preallocation candidate job_id
is null; actual execution must record a distinct new job ID.

Independent saved-array admission recomputed all 64 prepared lane identities,
unique point order, native structure and GPU lane gates. It retains 61 passes and
three finite tolerance-only failures (four comparisons); strict science is false.
Frozen-reset arrays were not retained for recomputation; only the original sealed
reset proof is admitted. See [validation](SCALAR_ADMISSION_VALIDATION.md) and
[actual execution](scalar-audit/audit-execution-notes.md).
No scalar/reset model rerun, fallback or retry is authorized by reuse.

Before JAX/model import or physical lowering, child entry applies mode environment
after transport and records inherited conflicts, applied whitelist, affinity and
pre-JAX success/failure. Production Popen receives this environment; preflight
and production share the launch builder. Real allocation, 48 distinct physical
cores, memory and sealed scalar admission remain hard gates. Wrong environment,
allocation, affinity or reference stops and preserves the receipt.
Never emit a complete environment dump.

Keep 64 unique points, continuous Days 2–8 then 9–15, 48 half-hours/day,
896 point-days, all 760 native output leaves/masks/dtypes, and
`atol=1e-8`, `rtol=1e-10`. Pass full CPU internal whole-group, permutation,
replacement/isolation, frozen-reset, structure and nonfinite hard gates.
Permutation/replacement failures stop immediately after their calls, before hot
timing. Reset follows warm repeats, then actual validation arrays are sealed for
independent collection. Reference reuse skips none of these gates.
Finite tolerance-only cross-GPU differences may support exploratory timing;
scientific_acceptance remains false.

Collect five warm core and five preprepared wrapper samples with original work
counts. GPU B64 job `15579869` is the same 896 point-day comparator.
GPU B1024 job `15671732`, 299.1398250995909 lane-days/s, remains a different-workload
normalized reference. Preserve independent old/new job and code identities.
Reused scalar and new B64 are not a same-job measurement.

The proposal is `cnall`, one node/task, 48 CPUs, zero GPUs, 3600s,
nonexclusive, no requeue and no added memory or placement restriction.
Requested-time scenarios: CNY 2.40 at 0.05/core-hour and CNY 3.36 at
0.07/core-hour, both unverified. The hour preserves about 3209s of intended v3
candidate time plus entry allowance, not a measured compile/runtime forecast.
Failure or timeout closes the single attempt; preserve terminal artifacts and
do not retry. Count the new parent once from terminal sacct, including
entry/compile/diagnostics/collection time spent in that parent.

The old parent raw row is
`15820467|FAILED|00:36:37|billing=48,cpu=48,node=1`.
Its 2197s × 48 = 29.293333333 core-hours and CNY 1.464666667/2.050533333
scenarios are already incurred, remain separate and count once. Scalar one-CPU
hot timing is not its 48-CPU parent cost. Incomplete/failed new runs still
report both parents separately; no fictitious new scalar directory.

Before separately approved submission, deploy and verify the prerequisite bundle,
exact clean checkout and persistent admission, confirm runtime/data/log paths,
inspect fresh scheduler/occupancy facts, and use the reviewed literal command.
Preparation did not validate a future nonexistent production checkout.
Real B64 compilation, gates, hot throughput/RSS and CPU48/GPU ratios remain unmeasured.
