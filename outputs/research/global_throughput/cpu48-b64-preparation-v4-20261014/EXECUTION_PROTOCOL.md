# CPU48 v4 batch-only execution protocol

This is an unpaid preparation contract. The exact code/ref/bundle and external
scalar-admission artifact will be bound in `execution-code-pin.json` and the
single-job proposal before review. Preparation alone permits no submission.

The immutable execution base is
`03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8`. All 104 physical files must retain
tree `75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba`, certified science SHA
`11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5` and actual
source SHA `b33450cae9cd8c9b2d5f7c6075fe270f461887edc09c77417c5e820a5c18fab8`.
The research branch's later carbon fixes remain on that branch.

Let `P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`. The future immutable checkout is
`P/runtime/worktrees/global-throughput-cpu48-b64-v4`; the future output is
`P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/<job-id>`.
Use `P/.venvs/orcjax_cpu/bin/python`, the accepted JAX/jaxlib 0.4.38 CPU runtime,
x64 enabled, one native CPU JAX device and 48 distinct allocated physical cores.

The sole new model phase is native FULL64 B64 batch/vmap. Its external scalar
directory is unchanged:
`P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar`.
Its checkpoint `stage_B64_server_scalar_v1.pkl` is 27,604,466 bytes, SHA-256
`1cddeaf80c2fc36f173835a5ff1bd4d8d520b144d4c96289ef6fe86dde898dfa`.
The immutable admission artifact separately binds the original job, code,
science/input/reference/window/order identities and checkpoint/manifest/report
bytes. It is an exploratory CPU internal reference; its original 61/64 passing
GPU lane gates and three finite tolerance failures remain unchanged.

Do not replace old source/job identities with the new candidate's identity.
Reference loading checks both actual identities and their byte-identical
physical subtree. Frozen-reset admission distinguishes saved, sealed proof
from a fresh array comparison when repeat arrays were not preserved. No reset
or scalar model rerun is authorized by reference reuse.

Before JAX/model import or physical lowering, the actual child entry records
its whitelist thread/JAX environment, affinity and thread policy and validates
the actual 48-CPU allocation and distinct physical-core selection. Wrong
environment, allocation, affinity or sealed reference stops the attempt and
preserves the failure receipt. Preflight and production share the same launch
implementation. No complete environment dump is permitted.

Keep 64 unique points, continuous Days 2–8 then 9–15, 48 half-hours/day,
896 point-days, all 760 native output leaves/masks/dtypes, and the original
`atol=1e-8`, `rtol=1e-10` and hard gates. The candidate must still pass full CPU
internal whole-group, permutation, replacement/isolation, frozen-reset,
structure and nonfinite checks. Reference reuse skips none of these gates.
Finite tolerance-only cross-GPU differences may support exploratory timing;
they do not establish strict scientific acceptance.

Collect five warm core and five preprepared wrapper samples with original work
counts. GPU B64 job `15579869` is the same 896 point-day comparator. GPU B1024
job `15671732`, 299.1398250995909 lane-days/s, remains a different-workload
normalized reference. Never present reused scalar and new B64 as same-job
measurements.

The proposed new allocation is `cnall`, one node/task, 48 CPUs, zero GPUs,
3600 seconds, nonexclusive, no requeue and no added memory or node restriction.
The 60-minute bound retains the approximately 3209-second intended v3 candidate
window plus entry allowance; B64 compilation/runtime remains unmeasured.
Worst-case rate scenarios are CNY 2.40 at 0.05/core-hour and CNY 3.36 at
0.07/core-hour, with rates unverified. Failure or timeout closes the single
attempt; no retry or automatic scalar fallback.

Count the new parent allocation once from terminal `sacct`. The old v3 parent's
2197 seconds × 48 CPUs = 29.293333333 core-hours and CNY 1.464666667/2.050533333
rate scenarios are already incurred and remain separate. The old scalar's
one-CPU hot timing is not the cost of its original 48-CPU parent allocation.
