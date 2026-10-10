# Original-mix whole-tree slicing results

Job15470735 completed0:0 on `ibc13b03n04` from2026-09-29 13:15:19 to13:23:49
CST. The fixed-weight daily32 + oneT7 + oneT30 full-proposal comparison passed.
Warm full-proposal medians leaf13.475146s/tree11.535444s give **1.168152x
throughput and14.3947% lower wall** on the same one-V100 workload. This is
bounded execution evidence; the10x target remains unmet.

## Scope and parity

Both routes reuse artifact742 daily32 in micro4, the same T7/T30 cases, saved
admitted field/recurrent executables, checkpoint and full AdamW state. Only
daily slicing changes from per-leaf execution to the opt-in cached whole-tree
dynamic slicer. Three warm pairs ran in leaf/tree, tree/leaf, leaf/tree order.
All proposals were discarded; retained training updates0 and daily64 not run.
The completed20260926 changing-update evidence was reused, not rerun.

Warm-up and all three pairs pass `rtol=1e-5`, `atol=1e-6` with zero failed
coordinates. The table gives maximum absolute differences across these checks.

| Compared quantity | Coordinates | Maximum absolute difference |
| --- | ---: | ---: |
| Mixed parameter gradient | 64,884,481 | 5.9604645e-8 |
| Proposed updated parameters | 64,884,481 | 5.9604645e-8 |
| Full AdamW state | 129,768,963 | 9.3132257e-10 |
| Auxiliary | 198 | 0 |
| Loss/norm/clip/update scalars | 22 | 2.3841858e-7 |

Both captured mixed gradients also pass the independent existing
`_serial_mixed` contract over all64,884,481 coordinates. All39 physical hard
counts per proposal are zero: daily11, T7 fourteen and T30 fourteen. These
tolerance passes are not a bitwise-equality claim or new scientific acceptance.

## Timing and memory

Reported medians use three synchronized warm measurements per route. Full
wall includes existing runner checks and AdamW; additional host full-tree
comparison is outside the timed execution. Branch medians must not be added
to infer the independently measured full-proposal median.

| Measured part | Leaf median, s | Tree median, s |
| --- | ---: | ---: |
| Complete proposal wall | 13.475146 | 11.535444 |
| Daily32/micro4 branch | 5.109202 | 3.187482 |
| T7 branch | 1.590061 | 1.582322 |
| T30 branch | 6.713479 | 6.714158 |
| AdamW | 0.020316 | 0.020224 |

The daily branch improves; these T7/T30 timings do not demonstrate recurrent
speedup. Startup272.852439s covers initialization through the initial slicer,
including saved-executable loading205.850782s and initial slicer compilation
4.029458s. These durations overlap; do not add them. Startup does not cover
all later warm-up helper compilation and is not time-to-valid-measurement.
New model-gradient compilations0. Total internal process timer491.356108s;
six timed warm executions consumed75.046506 GPU card-seconds.

Cumulative active-buffer peak12,434,782,208bytes (11.580793GiB) passes the
24GiB active-memory gate. Pool19,325,255,680bytes (17.998047GiB) is reported
separately and is not the gated quantity. The peak is a process high-water
mark shared by both routes, not isolated variant memory evidence.
Large proposal trees are released between pairs.

## Runtime and accounting

Approved resources were gnall/n04, one node/task,8CPU,one V100,64GiB,
nonexclusive,20min, maximum additive CNY0.92. Working directory remained the
immutable `runtime/checkouts/shared-split-width-v1` at c599236. Actual launch
13:15:20CST recorded affinity0-3,56-59, host load0.01/0.02/0.05, visible GPU0,
0MiB/0% and no visible compute PID. This snapshot does not reserve the node.

Slurm allocation510s gives0.1416667 GPU card-hours and additive estimate
`(8*0.07 + 2.20)*510/3600 = CNY0.391`, not a bill. Use510s allocation for this
accounting, not the491.356108s internal timer (despite the raw report's
`allocated_gpu_card_seconds` field name). The approved diagnostic is complete;
no retry or further GPU run is authorized by its result.

## Evidence and next work

- [Measured JSON](../../../outputs/diagnostics/shared-update-tree-slice-15470735.json), [log](../../../outputs/diagnostics/shared-update-tree-slice-15470735.log), [Slurm accounting](../../../outputs/diagnostics/shared-update-tree-slice-accounting-15470735.json), and [submission/launch record](../../../outputs/diagnostics/shared-update-tree-slice-submission-20260929.json).
- [Preparation](../../../outputs/diagnostics/shared-update-tree-slice-preparation-20260929.json) and [actual test01 CPU asset check](../../../outputs/diagnostics/shared-update-tree-slice-assets-cpu-20260929.json). Six local tests/py_compile and target CPU/freeGPU1/SlurmGPU3 argument expansion passed before submission; they were not rerun for this report.
- [Probe](../../../scripts/hpc/probe_shared_update_tree_slice.py), [launcher](../../../scripts/hpc/run_shared_update_tree_slice.sh), and [Slurm wrapper](../../../scripts/hpc/slurm_shared_update_tree_slice.sh). Remote overlay is `runtime/transfer/shared-update-tree-slice-20260929`.
- Preserve its accepted `runtime/transfer/shared-combined-full-update-20260926/probe_shared_serialized_dual.py` support owner and adjacent modules, the unchanged c599236 checkout, saved binaries and remote `runtime/outputs/diagnostics/shared-update-tree-slice-15470735.{json,log}`.

Defaults, sampling, precision and architecture remain unchanged; slicing stays
opt-in. Continue joint T1/daily, T7 and T30 speed work, prioritizing shared
neural forward/backward and retained physical execution. First prepare
source-consistent sample capture and native per-unit cache-key evidence for
the inconclusive broader-neural screen, reusing preserved assets. No physical
IR replacement, cache-key bypass, new neural/GPU allocation or full training
is authorized. That preparation addresses startup/cache compatibility and
numerical-screen resumption, not measured steady-state throughput. A precision
pass still needs branch-specific T1/daily, T7/T30 and original-mix full-update
speed evaluation. This diagnostic does not establish scientific learning,
changing-weight sustained throughput, solved T7/T30 costs or the10x target.
