# CPU48 v5 terminal result, 2026-10-14

The single approved paid attempt **job15857765 FAILED1:0 after819s** on
`ibc12b03n02`, 14:34:22–14:48:01 CST. The attempt and its approval are closed.
No repair, repeat submission, extension, cancellation or new model work followed.
Read `execution-summary.json`, `saved-failure-values.json`, `collector-result.json`,
`CPU_GPU_COMPARISON.json` and `runtime-evidence-manifest.json`; approval is retained
in `AUTHORIZATION.md`. Scientific acceptance remains **false**.

The exact sealed v5 proposal ran once at execution SHA
`0100c8b09be5137b822173e0a0be66cb64939727`, ref `codex/global-cpu48-b64-v5`,
from clean immutable checkout
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v5`.
The104 physical blobs/science identity and original precision/tolerances/gates were
unchanged. Actual48 distinct physical cores, 48CPU parent/step accounting, original
thread policy and JAX/jaxlib0.4.38 CPU/x64 were admitted. Host-memory admission saw
185,688,641,536 available bytes against64GiB conservative need, without a scheduler
whole-node memory reservation.179 post-compile observed threads had masks within
the declared48 cores; this is not proof of48-core utilization.

B64 physical lowering completed in90.675771s, compilation in256.662969s, and first
call in131.938299s. Sampled peakRSS was10,336,268,288B; Linux ru_maxrss peak was
10,354,053,120B. Both continuous seven-day blocks and native760-leaf output were
materialized and saved before parity checks. Compilation signatures retained
536 caller/public/internal inputs,417 kept inputs, and original JAX signature checks.

The first substantive stop was the **strict CPU internal whole-tree gate**
`B64_batch_cross_mode_server_scalar`, comparing saved B64 output to oldscalar
job15820467. It has three failing leaf comparisons at two points, all finite
tolerance failures at unchanged `atol1e-8/rtol1e-10`. This is a numerical failure
after real model execution, not a startup/provenance failure. Targeted NumPy
extraction from the saved arrays produced these exact worst-ratio witnesses:

| Leaf path | Days | Lane / point | Candidate | Scalar reference | Absolute difference | Bound ratio |
| --- | --- | --- | --- | --- | --- | --- |
| `[0][0][0][7]` | 2–8 | 6 / 299.0-115.0 | 592.8152488486185 | 592.8152487724788 | 7.613971320097335e-8 | 1.098990146880813 |
| `[1][0][5][48]` | 9–15 | 37 / 313.0-079.0 | 29.926931253713647 | 29.926931273231105 | 1.951745787209802e-8 | 1.5021872433093637 |
| `[1][0][6][0]` | 9–15 | 37 / 313.0-079.0 | 29.926931253713647 | 29.926931273231105 | 1.951745787209802e-8 | 1.5021872433093637 |

The report's `max_abs=8.463580161333084e-5` is a760-leaf whole-tree statistic,
**not** the error at those three failed positions. Separate GPU comparisons were
61/64 passing points, three finite failing points/four comparisons; no shared
cause with the new CPU internal failure has been established. The original scalar
GPU failure and sealed-reset-proof-only limitations remain.

Permutation/replacement physical calls did run before the scalar gate, and their
two recorded gates passed exactly. There is no completed/sealed batch stage or
independent sealed-array recomputation of those gates. Reset, five core/five wrapper
warm repeats, and validation-array sealing were not reached. No valid CPU48 warm
throughput, speedup or warm-unit-cost comparison is available. Same896point-days
GPU B64 job15579869 remains5.632416466s core /6.222332641s wrapper; its evidence
is referenced through `../cpu48-b64-execution-v4-20261014/GPU_COMPARATOR_EVIDENCE.md`.
B1024 is different-workload context only. Cold first-call time is not a warm sample.

The unchanged pinned collector SHA
`efcb32efa754bbdbb75936f358dc6cb5a286e6742948096bda3e3ba22d27a9e3`
ran read-only on free cntest job15864638, ibtest01, actual1CPU/affinity[0],300s
limit, exit0. Its result is `incomplete_run_failure_preserved`; no physical
recomputation or independent accepted batch-stage verification occurred.

Raw parent sacct is authoritative:48×819/3600=**10.92 allocated core-hours**,
unverified new whole-parent cost **CNY0.546/0.7644**. Oldscalar15820467's2197s,
29.293333core-hours andCNY1.464666667/2.050533333 remain a separate prior asset,
counted once and not regenerated or recharged. The immutable collector's generic
scope string mentioning a scalar step does not change this batch-only accounting.
Compilation/startup and full-parent costs are separate from warm unit costs;
CPU warm unit costs remain unavailable.

Full raw report (1,128,664B) is retained locally as `model-report.json` and remotely
at `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/15857765/batch/report.json`,
SHA`75b094ec1708b029b541717eb0328f101e2f42df3734ab48f47cfe56f4b20068`.
Raw candidate output27,562,300B is mirrored locally and retained remotely at the
same batch directory's `raw_B64_batch_before_parity.pkl`,
SHA`a135f9436afa868176bf8cc76f28134218e510c65b75909075ba24a8964810ce`.
The failed `stage_manifest.json` is mirrored locally, SHA
`44f9417385dad33f927cfcef603c7ef5f230f815fb4cf2dd454a1a3f658f5a3e`.
Raw arrays/full reports stay outside Git; compact hashes, witnesses, stages,
terminal accounting and traceable receipts are retained. Useful arrays can support
a separately dispatched investigation; this attempt does not diagnose or repair
the cause. No paid authority remains active. Wait for new supervisor dispatch.
