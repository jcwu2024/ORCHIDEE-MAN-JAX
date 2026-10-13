# CPU48 / FULL64 B64 v4 single-run proposal

**Fixed proposal for supervisor review; no submission or production deployment occurred.**
The supervisor can decide on this exact single attempt under the user's existing
delegated authority. The v3 grant is consumed. Changes to code, command, resource
or cost envelope require renewed review; failure or timeout closes this attempt.

The new candidate is B64 only, reusing audited saved scalar from old job 15820467.
There is no scalar generation, fallback or automatic retry. Preserve distinct
old/new job and code identities; this is not a same-job scalar/B64 experiment.

| Resource | Exact proposal |
| --- | --- |
| Partition | cnall |
| Allocation | 1 node, 1 task, 48 allocated CPUs, 0 GPUs |
| Wall limit | 3600 seconds / 01:00:00 |
| Scheduling | Nonexclusive, no requeue, no added memory or node restriction |
| Submission cwd | /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX |
| Batch cwd | /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v4 |
| Python | /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python |

Let `P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`. Submit from a working cln01 or
cln02 endpoint only after approved deployment and fresh scheduler/occupancy
inspection. Slurm selects an eligible node; an old idle snapshot is not a
reservation or evidence of historical contention. Test nodes never submit jobs.
The canonical runtime remains accepted CPU JAX/jaxlib 0.4.38 and x64.

Worst-case whole-parent requested-time costs, with both rates unverified:

- 48 CPUs × 1 hour × CNY 0.05/core-hour = **CNY 2.40**.
- 48 CPUs × 1 hour × CNY 0.07/core-hour = **CNY 3.36**.

The hour preserves approximately 3209s of the original intended v3 candidate
allowance plus entry allowance. It is a stopping bound, not a runtime forecast.
Old parent job 15820467 used 2197s × 48 = 29.293333333 core-hours and already
incurred CNY 1.464666667/2.050533333 rate scenarios. Keep old cost separate and
count once; scalar one-CPU hot timing is not its original parent cost.
New job actual cost is counted separately from terminal sacct even on failure.

Execution ref `codex/global-cpu48-b64-v4` is pinned to
`60dccbddde10be4295aab7e7fe55fa7864d3c3f3`. It is based directly on
`03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8`, with 15 reviewed harness/test files
from research checkpoint `493252e418f3fd827af4a6f7748a63d3157e680d`.
All 104 physical blobs/tree `75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba`
remain identical to base. The research branch's carbon fixes are preserved.

The [pin manifest](execution-code-pin.json), SHA256
`e9e301aeab7237d59a0e204ffc93fc3eee96d28ab997d89be31383d735ec2769` (8,351 B),
and [local actual verification](local-package-verification.json) bind these files:

| Artifact | SHA256 | Bytes |
| --- | --- | ---: |
| execution-harness.patch | d2d1f6c6ae7d64eefe41dfd41d15bac99fb59a7e31d441b78207c0d508553682 | 143960 |
| cpu48-b64-60dccbddde10.bundle | 970fc5be9bf3f59061b27ad6647d457e824392ce14336a69ff0308e3c91b928e | 36282 |
| scalar-audit/scalar-audit-receipt-v3.json | cb1331edb7094e74ec1814c6813a0eee218c58ea4e89f1c0c1f630e981479431 | 146678 |
| scalar-audit/scalar-admission.json | f71eb5db4e638615ef5e06be5783d193bbfc1bd7563dd30c2424e04b40c14a8d | 145538 |

The bundle is locally verified and requires base
`03026d013626d3ed7379c2a5fd3ebb7f1c5fd9f8`. It remains a local ignored binary;
preparation did not install it in a new production checkout. After approval,
use the existing deployment registry/bare repository and prerequisite bundle to
create the exact clean immutable checkout at
`P/runtime/worktrees/global-throughput-cpu48-b64-v4`.
Do not overwrite historical or active checkouts.

Input:
`P/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl`,
SHA256 `be83763773c2e515bc15743befa3f26beee885ccbcb14f387e15b13c387a4dc7`.
GPU reference:
`P/runtime/outputs/research/global_throughput/gpu-round4/model/stage_B4_group_references.pkl`,
SHA256 `a3a871e69d8bcb06565f463da43f864938aa5cedeb998cd61c2f1224c14555b6`.
Config: `P/runtime/worktrees/global-throughput-cpu48-b64-v4/configs/orchidee_man_250919.yaml`.
Results: `P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/<new-job-id>`.
Shared logs must exist and be writable before sbatch opens them.

Unchanged scalar directory:
`P/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar`.
Checkpoint `stage_B64_server_scalar_v1.pkl`: 27,604,466 B, SHA256
`1cddeaf80c2fc36f173835a5ff1bd4d8d520b144d4c96289ef6fe86dde898dfa`.
Approved deployment must place admission at
`P/runtime/outputs/research/global_throughput/cpu48-b64-preparation-v4-20261014/scalar-audit/scalar-admission.json`,
with the SHA256 above. The record preserves old job/code and exact checkpoint,
manifest, report, input, point order/window, science and accounting identities.
Candidate job_id is null before allocation; the actual new job must be distinct.
Audit recomputed all 64 input identities and GPU gates: 61 pass, three finite
tolerance-only failures across four comparisons, strict science false.
Reset arrays were not retained; only sealed original reset proof is checked.

The **literal proposed submission command** is [PROPOSED_SUBMISSION.sh](PROPOSED_SUBMISSION.sh).
Its SHA256 is `b86068c62c96464319175135e12e5d84ce2e494a00875aceaa01ee0b1a48c160` (2,011 B).
It clears inherited `SBATCH_*`, `ORCHJAX_*` and explicit launcher overrides,
then sets only canonical code/Python/scalar/admission paths and exact SHAs.
The command fixes partition/resources/time/logs/chdir and uses the immutable
launcher's absolute path. No sbatch command in that file was executed here.
[Environment validation](environment/ENVIRONMENT_VALIDATION.md) distinguishes
earlier placeholder evidence from final exact-pin staged checks. Staged dry-runs
cannot prove actual admission of the future nonexistent production checkout.
The final one-CPU check used an identical-hash staging admission copy and mapped
only the emitted coordinator executable and `--entry` paths to staged sources;
future code/config/result paths remained declared. The literal proposal's
persistent admission destination was not created or exercised.

Before submission, stop for absent supervisor approval, any ref/bundle/source/
input/reference/admission mismatch, dirty checkout, missing path/runtime/log
permissions, stale scheduler inspection or a changed command/envelope.
At real child entry, post-transport mode environment normalization and whitelist
receipts precede JAX/model import. Actual allocation/48 distinct physical cores,
memory and sealed reference remain hard gates.

Keep 64 unique points, continuous Days 2–8 then 9–15, 48 half-hours/day,
896 point-days, all 760 native output leaves/masks/dtypes, `atol=1e-8, rtol=1e-10`.
Pass all CPU internal whole-group, permutation, replacement/isolation,
frozen-reset, structure and nonfinite gates; seal actual validation arrays for
independent collection. Collect five warm core and five preprepared wrapper samples.
GPU B64 job 15579869 is the same-workload comparator; B1024 job 15671732 is
different-workload normalized evidence only. Finite cross-GPU tolerance-only
differences may support exploratory timing, while strict science stays false.

On failure/timeout, preserve receipts and terminal sacct, count each parent once,
and stop. Real B64 compile time, hard-gate outcomes, warm throughput/RSS and
same-workload CPU48/GPU speed or cost ratios remain unmeasured in this preparation.
