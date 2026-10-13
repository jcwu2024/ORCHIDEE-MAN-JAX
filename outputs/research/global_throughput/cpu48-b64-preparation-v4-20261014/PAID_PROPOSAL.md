# CPU48 / FULL64 B64 v4 single-run proposal

**Preparation in progress; no submission has occurred.** The supervisor must
review the final exact pin, scalar admission, target-shell receipt and command
under the user's existing delegated authority. The v3 attempt is consumed.
This document will bind the final code/admission hashes before delivery.

The candidate is B64 only. It reuses the independently admitted saved scalar
from job15820467. There is no new scalar phase or automatic scalar fallback.
Its results must retain distinct original scalar and new candidate job/code
identities; this is not a same-job scalar/B64 experiment.

Partition `cnall`; one node, one task, 48 allocated CPUs, zero GPUs; wall limit
3600 seconds; nonexclusive, no requeue, no additional memory, node pin, node
exclusion, GPU or overcommit request. Slurm chooses an eligible node. Inspect
current scheduler facts and occupancy before the separately approved attempt.
An earlier occupancy snapshot is neither a reservation nor proof of the cause
of a past slowdown. Submission cwd is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`, using a working `cln01` or `cln02`
endpoint. Test nodes are for small CPU checks, never submission.

Requested-time ceilings count the entire new parent allocation once:

- `48 CPUs * 1 hour * CNY 0.05/core-hour = CNY 2.40`.
- `48 CPUs * 1 hour * CNY 0.07/core-hour = CNY 3.36`.

Both rates are unverified. The 60-minute window retains approximately3209
seconds of the original v3 intended candidate allowance plus entry allowance;
it is an exploratory stopping bound, not a measured B64 compile/runtime forecast.
The previous v3 parent's2197 seconds at48 CPUs already incurred29.293333333
core-hours and CNY1.464666667/2.050533333 rate scenarios. Keep that old cost
separate and count it once; scalar one-CPU hot timing is not its parent cost.

The future immutable checkout is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v4`.
The execution base is `03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8`; all104
physical files and tree `75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba` stay identical.
Only reviewed harness and tests are packaged. Import the verified Git bundle
with its prerequisite, retain the exact ref/pin, and require a clean target
checkout. Preparation does not deploy this production version.

Use the canonical CPU Python
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python`, accepted
JAX/jaxlib0.4.38 and x64. The input is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl`,
SHA-256 `be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7`.
The GPU numerical reference is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/gpu-round4/model/stage_B4_group_references.pkl`,
SHA-256 `a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6`.

The unchanged scalar directory is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar`.
Its checkpoint SHA-256 is
`1cddeaf80c2fc36f173835a5ff1bd4d8d520b144d4c96289ef6fe86dde898dfa`.
The final admission must live at
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v4-20261014/scalar-audit/scalar-admission.json`.
The config is `<CODE_DIR>/configs/orchidee_man_250919.yaml`; results stay below
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/<job-id>`.

Before submission, stop for any pin/ref/bundle/source/input/reference/admission
checksum mismatch, dirty checkout, missing runtime/path, stale scheduler
inspection, absent supervisor approval, or change to the reviewed command and
resource/cost envelope. Clear inherited `SBATCH_*` and launcher overrides as
part of the final exact submission command. Do not use the old v3 grant.

At actual child entry, normalize the declared whitelist after Slurm transport;
record inherited conflicts, actual whitelist values and affinity on success
or failure. Wrong environment, allocation,48 distinct physical cores, memory
admission or sealed reference stops before physical lowering. Preserve all
64 unique points, Days2–8 then9–15,48 half-hours/day,896 point-days and760
native output leaves/masks/dtypes, with `atol=1e-8, rtol=1e-10`.

The candidate must pass all CPU-internal whole-group, permutation,
replacement/isolation, frozen-reset, structure and nonfinite hard gates.
Seal actual candidate validation arrays for independent collection. Obtain
five warm core and five preprepared wrapper samples with unchanged work
counts. Finite tolerance-only cross-GPU differences may support exploratory
timing; they do not establish strict scientific acceptance. GPU B64 job15579869
is the896 point-day comparator; B1024 job15671732 is a different-workload
normalized reference only.

Failure or timeout closes the single attempt. Preserve receipts and terminal
`sacct`, account the new parent once, and do not retry or regenerate scalar.
Real B64 compile time, hard-gate outcomes, hot throughput/RSS and same-workload
CPU48/GPU speed or cost ratios remain unmeasured in this preparation.
