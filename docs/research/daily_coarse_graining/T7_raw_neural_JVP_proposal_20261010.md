# T7 First-Day Raw Neural JVP Diagnostic

Updated: 2026-10-10. The explicitly approved scope finished numerical exit 3.
Both graphs/24 raw results are saved; GPU released/scope closed. The
[accepted v3-final result](T7_raw_neural_JVP_results_20261010.md) retains active
primal/tangent failures. Method authorization is consumed; handoff complete, stop.

## Purpose and selected boundary

The [completed first-day diagnostic](T7_first_day_JVP_localization_results_20261010.md)
reproduces strict, finite, defined next-state tangent differences for directions
434/2382/2621/control 200, before later recurrence and trajectory loss.
Accepted metadata confirms exact same-route repeats and exact masks. The
source-undefined roughness-slot NaN retains its separate strict admission
failure; no cause of the defined tangent differences is established. The 2.555005x
four-window full-AD component candidate remains unadopted under the state gate.

Measure the canonical neural prefix before physical transformations, state
updates and tail materialization:

`rollout_input_from_template` → `head.input_assembler.assemble` →
`head.assemble_context` → `shared_process_head.shared_transformer_raw_outputs`.

Call the same patched raw callable and arguments used by the selected public
head. Preserve **all declared raw learned-field tensors and their complete
exact active masks**. No process, physical transform, observation cotangent,
updater, next state or trajectory loss is reconstructed for this diagnostic.
This distinguishes the neural prefix from later computation, without selecting
a cause, precision/reference change or scientific adoption.

## Fixed method and gates

Reuse candidate latent32/private-reader64 weights, prepared TRAIN artifacts
742/275/482/299, full FP64 state and the same first-day inputs. B1 is sample 3,
artifact 299; B4 uses four distinct cases with one shared parameter tree.
Directions are the existing FP64 unit vectors at 434/2382/2621/control 200,
isolated to B4 sample 3. Neural arithmetic and conditioning remain unchanged;
the declared method is FP32 neural/FP64 physics/state directions. The boundary
ends before the physical transforms. No Teacher regeneration or weight update.

Compare B1 raw outputs with B4 sample 3 at natural per-case scale, **no /4**.
Run three alternating cycles of four directions per route: 24 JVP calls,
12 per route. Save every complete primal/tangent/mask result before checks.
Require exact field sets, full declared shapes/dtypes and Boolean active masks;
preserve inactive coordinates as well as potentially active ones.

Keep `rtol=1e-5, atol=1e-6`, finite raw values/tangents, exact masks and topology,
and cross-route/own-repeat comparisons. Check other-three-sample B4 tangents
are exactly zero for all twelve B4 calls. Record full per-field failure counts
and bounded failed positions; preserve full arrays remotely. Do not waive a
failure, apply a new denominator or reduce precision/capacity to obtain parity.

There is no VJP, parameter-gradient graph, optimizer update or saved executable
load. Raw output admission does not supply full T7 loss/state/parameter
gradients, fourteen full-horizon hard gates, scientific accuracy or training
throughput acceptance.

## Inference limits

B1/B4 are separately compiled new output boundaries. Kernel and fusion choices
may differ from the integrated full-day/T7 graphs. A failure proves occurrence
in this measured prefix, not direct causal attribution of the old integrated
graph. A strict pass cannot exclude smaller or compilation-dependent raw
differences that downstream transforms/updates/loss amplify. A pass therefore
does not adopt batching or authorize another boundary automatically.

Use the failed fields or passing-boundary evidence to specify the next bounded
localization. Mechanism, derivative reference, tolerances, precision, architecture,
capacity, recurrence, loss and sampling remain unchanged unless separately decided.

## Executable and prospective resources

Working directory: immutable c599236 checkout
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1`.
New isolated overlay: `runtime/transfer/T7-raw-neural-jvp-20261010`.
Reuse unchanged common exact sources/data and the existing prepared manifest
under `runtime/transfer/shared-latent-reader-20261004`.

- `scripts/hpc/shared_latent_T7_raw_neural_jvp.py`: canonical prefix composition,
  full raw field/mask schema, native JVPs, comparison and isolation gates.
- `scripts/hpc/probe_shared_latent_T7_raw_neural_jvp.py`: exact asset binding,
  first-day inputs, per-route compile, full preservation, timing/memory/result.
- `scripts/hpc/run_shared_latent_T7_raw_neural_jvp.py` and `.sh`: immutable path
  resolution, actual deployed preflight, independent approval/occupation,
  phase caps and actual child exit.

Authorized independent grant: one free gln01 V100, OMP4/OpenBLAS1, at most **5400
cumulative GPU seconds / 1.5 card-hours / CNY 0**, no paid fallback. Probe cap
4800 seconds, new deployed runtime preflight 240, protected exit reserve 360.
Each B1/B4 route: lowering 180, backend 1800, cumulative route 2100 seconds.
One backend attempt per route; two new graphs, 24 JVPs, zero VJPs/updates.
Startup, preflight, compilation, ordinary repairs, measurements, preservation
and exit count against the same outer cap. Compiled/JAX and sampled NVML
card-total memory must stay below 24 GiB; no unsampled peak claim.

The cap uses actual first-day lowering 111.63/108.66 and backend 830.37/901.12
seconds as planning evidence. It is a termination envelope, not a prediction
or success guarantee. Reuse no old approval or unused balance: the visible
first-day approval was consumed by its completed 2124.572662/5400-second scope.
Its remaining 3275.427338 seconds does not fund these graphs.

Preserve successful binaries and compressed optimized HLO before subsequent
stages; retain attempts and complete arrays on numerical failure. Settle actual
occupation and keep numerical exit 3 distinct from runtime/resource failure.
Do not retry a failed backend beyond the declared one attempt per route.

Authorized command launched from gln01 after the actual idle-card check:

```bash
T7_RAW_NEURAL_JVP_GPU_INDEX=1 bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-raw-neural-jvp-20261010/run_shared_latent_T7_raw_neural_jvp.sh execute
```

Funding scope is `latent_T7_raw_neural_jvp_20261010`. Result prefix is
`T7-raw-neural-jvp-20261010` under
`runtime/outputs/diagnostics/shared-latent-reader-20261004/`.
Entry is the isolated `probe_shared_latent_T7_raw_neural_jvp.py` above.
The false preparation template remains historical. A separate true approval
now binds this command/path/output/cap; no prior funding is reused.

## Execution and closing boundary

The [true approval](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.approved.json)
records the user's explicit “那你批准一下” delegated through coordinating chat
`019fdc7a-5f6e-7620-a3be-28c855800369`, for the unchanged one-card/5400-second
scope. The [launch receipt](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.launch.txt)
records remote-host 2026-10-10T13:35:32+0800, ibc13b02n04, GPU 1 idle at
0 MiB/0%, wrapper PID 91303 and CPU affinity 0–55. Host load
41.92/41.92/41.95 is a compilation-context caveat, not a reserved idle host.
The [actual exit](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.actual-exit.json)
is 3. The [settled ledger](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.occupation.json)
charges 2169.713307645172/5400 GPU seconds/0.602698141012548 card-hours/CNY 0;
the [release](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.release.json)
confirms GPU 1 empty, wrappers absent and scope closed. The
[parent acceptance](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.parent-acceptance.json)
selects the [NumPy v3-final audit](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.numpy-audit.v3-final.json):
saved evidence/reconciliation/resources pass, numerical gate fails. Earlier
audits remain superseded records. Unused balance grants no new work.

The [result](T7_raw_neural_JVP_results_20261010.md), complete raw arrays, current
entries and [handoff inventory](daily_speed_handoff_20261010.md) preserve the
accepted failure. Stop: no automatic next GPU diagnostic/long training, directory move,
root-branch switch or runtime cleanup. No scientific acceptance follows merely
from execution authorization.

## Preparation status

The [local focused log](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.local-tests.log)
records 25 tests passed in ORCJAX, 3.45 seconds; the existing disabled-cache
configuration warning is retained. Actual target CPU
readiness/dry-run/plan/false-approval rejection passed on test01/ibtest01.
Parent accepted the actual source review, target receipts and archive. The
[independent review](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.review.json)
records the source verdict; supplemental earlier-result metadata mapping is
separate from this executable/archive acceptance.

The [target verification](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.target-verification.json)
records Bash syntax, actual Bash argument expansion, Python argv and actual
child plan-only passes. False approval exits 1 before GPU preflight. The
[CPU readiness](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.cpu-readiness.json)
binds 215 actual public first-day argument leaves, sample 3 exact, full 3854-wide
FP64 state, and all 161 raw learned fields with exact provider masks. The
[Python dry-run](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.execute.python-dry-run.json)
and [child plan](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.execute.plan-only.json)
retain the command, phase caps, two planned graphs and 24 planned JVPs.

The [source package](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.source-package.json)
and [target stage receipt](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.target-stage.json)
bind the [nine-member archive](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.source.tar.gz):
four new scripts and five unchanged dependencies, all member hashes verified
on target. SHA-256 is
`f3dbcc7f62aed4f3afe478ae3f8f434d3d8a7efd1dba292c6c2aea62e1c2fd86`.

At the recorded preparation checkpoint, all actual GPU/preflight/model/
backend-compile/JVP/VJP/update counts were zero;
preflight, occupation, result, true approval and B1/B4 compile-attempt artifacts
did not exist. The [approval template](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.approval.json)
remains `approved=false`; the separate true approval above now owns execution.
CPU checks do not supply GPU/model acceptance.
