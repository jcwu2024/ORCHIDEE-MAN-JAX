# Changing-weight whole-tree slicing: complete-update comparison

Status: passed on the free Explore1000 `gln01` GPU0, 2026-09-30 19:05 CST.
The process exited 0. This is a bounded execution diagnostic, not formal
training or scientific accuracy acceptance.

## Workload and method

The same saved artifact 742 daily32 input was evaluated in eight micro4 calls,
with one T7 and one T30 case, the admitted public losses, full parameter
gradients and one full AdamW update per proposal. Both routes used the same
immutable `c599236bcf57b745e9bafed678298010f4c3b58f` source checkout,
update-0 checkpoint, saved gradient executables and AdamW rule. Only daily
slicing differed: per-leaf versus cached whole-tree dynamic slicing.

A fixed-weight warm pair passed first. The three timed pairs alternated order
`leaf/tree`, `tree/leaf`, `leaf/tree`. Within each pair, both proposals read the
same parameters and optimizer state. After full parity and physical checks,
the tree proposal supplied the next pair's state. AdamW steps advanced
`0 -> 1 -> 2 -> 3`; the parameter L2 changes were 0.08022466, 0.06740126 and
0.03823545. Updated weights lived in this process only. No checkpoint was
written or selected. The same saved input cases were reused at every step;
this probes changing-weight execution, not a prescribed training sequence.

## Gates and timing

Every warm and timed comparison passed unchanged `rtol=1e-5`, `atol=1e-6`,
with zero failed coordinates: 64,884,481 mixed parameter-gradient coordinates,
64,884,481 updated-parameter coordinates, 129,768,963 full AdamW coordinates,
198 auxiliary coordinates and 22 loss/norm/clip scalar coordinates. Across the
three timed pairs, maximum absolute differences were respectively
5.9604645e-8, 4.8428774e-8, 9.3132257e-10, zero and 9.5367432e-7.
All 39 physical hard counts per proposal were zero on both routes.

| Pair | Leaf wall, s | Tree wall, s |
| --- | ---: | ---: |
| 1 | 12.693149 | 11.575142 |
| 2 | 12.668700 | 11.776426 |
| 3 | 12.772927 | 11.657018 |
| Median | **12.693149** | **11.657018** |

The matched median ratio is **1.088885x tree throughput**, or **8.163% less
wall time**. Full wall includes the existing runner's finite/physical gates,
mixed gradient and AdamW; host transfer and full-tree parity comparison happen
outside the timed proposal. The prior fixed-weight medians were 13.475146 s
leaf and 11.535444 s tree (1.168152x), but those came from a different run.
The changing-weight paired comparison is the relevant observation here. Neither
result approaches the project's eventual 10x research target.

The report records 208.811 s startup through slicer setup and 419.335 s
internal process time (0.11648 one-GPU card-hours on a free node). Six timed
proposals consumed 73.143 GPU card-seconds. Startup reused the previously
admitted serialized executables and JAX cache; no model-gradient compilation
was requested. The process-cumulative active-buffer peak was 13,517,517,824
bytes (12.59 GiB), below the 24 GiB gate; pool allocation was 19,325,255,680
bytes (18.00 GiB). These are not isolated route peaks.

At launch, the node was `ibc13b02n04`, affinity `0-111`, and both free-node
GPUs showed 0 MiB/0% with no resident compute PID. The host load average was
6.71/4.91/4.43. This is a shared-node, bounded three-pair observation, not a
paid-node throughput guarantee or confidence interval. One GPU was used; no
paid Slurm job or production-default change occurred.

## Evidence

- [Measured JSON](../../../outputs/diagnostics/shared-update-changing-slice-20260930.json), [launch log](../../../outputs/diagnostics/shared-update-changing-slice-20260930.log), and [exit sidecar](../../../outputs/diagnostics/shared-update-changing-slice-20260930.exit).
- [Probe](../../../scripts/hpc/probe_shared_update_tree_slice.py) and [launcher](../../../scripts/hpc/run_shared_update_tree_slice.sh). Separate remote overlay: `runtime/transfer/shared-update-changing-slice-20260930`; immutable checkout and original support owner remained intact.
- The target `test01` saved-asset inspection passed. Local targeted tests: 11 passed; local `py_compile` passed. Actual `gln01` shell `--dry-run` expanded the complete changing-update command before launch.

The whole-tree slicer remains opt-in. This diagnostic supports a modest
changing-weight complete-update execution gain for this exact workload and
node; it does not establish general recurrent acceleration, long-run training
throughput, trained-model accuracy or a reason to change the formal method.

## Runner opt-in integration gate

`run_shared_learning.py` now accepts `--daily-slice-execution tree`; omitted
means `leaf`. Its daily accumulator construction uses the same explicit factory
for either route and leaves `execute_learning_update`, objectives, row selection,
saved model executables, optimizer, tolerances and production default unchanged.
The option is execution metadata in the run report, not a change to the
declared scientific method or checkpoint identity. It is valid only for the v2
micro4 method.

The candidate runner was staged as a separate source overlay and the full
proposal probe imported its accumulator factory and complete-update function.
It reused the admitted saved daily/T7/T30 gradient executables, so no new
model-gradient compilation or long training was needed. Local targeted tests
passed 24/24; target launcher `--dry-run` expanded `--daily-slice-execution tree`
and the probe's candidate-runner argument. The target test01 saved-asset
inspection passed. A second free `gln01` GPU0 execution on the integrated
factory exited 0 with three paired changing-weight updates:

| Pair | Leaf wall, s | Tree wall, s |
| --- | ---: | ---: |
| 1 | 12.724278 | 11.542475 |
| 2 | 12.572854 | 11.839031 |
| 3 | 12.700542 | 11.573968 |
| Median | **12.700542** | **11.573968** |

The integrated paired ratio was **1.097337x throughput**, or **8.870% less
wall time**. All three pairs passed the same full-tree `1e-5/1e-6` gates with
zero failed coordinates and zero physical hard counts. Maximum absolute
gradient, parameter, AdamW, auxiliary and scalar differences across pairs were
1.4156103e-7, 2.9802322e-8, 1.3969839e-9, zero and 1.5497208e-6.
AdamW advanced `0 -> 1 -> 2 -> 3`; parameter L2 changes were 0.08022466,
0.06740126 and 0.03823524. No diagnostic weights were checkpointed.

Startup through slicer setup was 206.896 s, internal process time 388.995 s
(0.10805 one-GPU card-hours), and six timed proposals 72.953 GPU card-seconds.
The cumulative active-buffer peak was again 13,517,517,824 bytes (12.59 GiB),
below 24 GiB. This second free-node run launched at 19:27:53 CST on
`ibc13b02n04`, affinity `0-111`, with 0 MiB/0% GPUs and host load
0.01/0.45/2.78. The difference from the first run is ordinary shared-node
variation, not a measured production change.

The [integrated JSON](../../../outputs/diagnostics/shared-update-changing-slice-integration-20260930.json),
[log](../../../outputs/diagnostics/shared-update-changing-slice-integration-20260930.log)
and [exit sidecar](../../../outputs/diagnostics/shared-update-changing-slice-integration-20260930.exit)
preserve this gate. The integration validates this bounded saved-executable
complete-update path; it does not execute the runner's data bootstrap,
from-scratch gradient compilation, checkpoint writing or science monitors.
The model-gradient executables and AD boundaries were unchanged. This update
interface returns and compares every parameter gradient, but does not
separately return or compare initial-state gradients; their prior admission
evidence is reused, not newly extended by this gate.
