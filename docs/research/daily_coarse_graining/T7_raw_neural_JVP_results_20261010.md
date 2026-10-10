# T7 raw-neural JVP results

Updated: 2026-10-10. Both graphs compiled once and all 24 raw JVP results were
saved. Actual exit is 3, `numerical_failed`; the independent CPU audit confirms
finite, active cross-route primal/tangent failures. GPU released, scope closed.
This reproduces differences inside the measured canonical neural prefix before
physical output transforms/state updates. The first differing operation and the
cause of earlier integrated-graph failures remain unproved. Batching is unadopted.

## Method and accepted evidence

The [executed proposal](T7_raw_neural_JVP_proposal_20261010.md) fixes unchanged
latent32/private-reader64 weights, 66,123,713 parameters and TRAIN742/275/482/299.
B1 is sample 3/artifact299; B4 shares one parameter tree across four distinct
cases. FP32 neural arithmetic, FP64 initial-state directions, natural per-case
scale/no /4, `rtol=1e-5, atol=1e-6` and 24 GiB gates remain unchanged.

The boundary is canonical `rollout_input_from_template` → input assembler →
head context → `shared_process_head.shared_transformer_raw_outputs`. All 161
declared raw learned fields and exact Boolean active masks are retained before
physical transformations, state updates and tail materialization. Three
alternating cycles cover directions434/2382/2621/control200: 12 JVPs per route,
zero VJPs/optimizer updates/saved-executable loads, no retry or paid fallback.

The [parent acceptance](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.parent-acceptance.json)
selects the [independent NumPy v3-final audit](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.numpy-audit.v3-final.json).
It independently checks the saved arrays, complete fields/shapes/dtypes/masks,
receipt reconciliation and resource contract. Five negative guards pass. CPU
analysis took 0.875 seconds with zero model/JAX/GPU/compile/executable-load calls.
Earlier `numpy-audit.json`, `.v2.json` and initial `.v3.json` are preserved as
superseded: receipt-reconciliation and duplicate-timing-guard defects were
corrected using saved CPU arrays; original numerical failures were retained.

## Cross-route and repeat result

Each row repeats in all three cycles, comparing B1 with B4 sample 3. All
failed primal/tangent coordinates are **active**; inactive failure count is zero.

| Input direction | Tangent failed fields /161 | Tangent failed coordinates | Worst field and tensor index | Error / allowance |
| ---: | ---: | ---: | --- | ---: |
| 434 | 91 | 1769 | `typed.ok_leak.doc_advective_interface[0,18,4,0]` | 459.3841084780129 |
| 2382 | 10 | 10 | `endpoint.fast_state.diffuco_previous_step_state.z0h[0]` | 14.523089838896766 |
| 2621 | 10 | 36 | `endpoint.daily_summary.soilhum_daily[0,2]` | 19.57187936638879 |
| 200 | 14 | 21 | `endpoint.fast_state.diffuco_previous_step_state.rveget[0,0]` | 12.19810142903012 |

Every cross comparison also fails four primal coordinates in two fields:
`inventory.carbon.DOC.incoming_amount` and
`inventory.carbon.DOC.outgoing_fraction`; worst allowance ratio is 1.276019.
Complete per-field counts and bounded failed positions are in the selected
audit and [saved probe report](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.result.json).
For direction434's worst raw tangent, B1 is -0.0926513671875 and B4 sample 3
is -0.093536376953125, at the index shown above.

All compared B1/B4 sample-3 primals and tangents are finite, and all 161 masks
match exactly in every comparison. The 16 nontrivial within-route comparisons
in cycles2–3 have finite values that are exactly equal; eight cycle1 baseline-self checks are separate
sanity checks. B4's other three samples have exactly zero tangents in all12
calls, with no nonfinite/nonzero isolation coordinate. Other samples' complete
primal outputs are preserved; they are not the cross-route target of this audit.

The earlier source-undefined roughness-slot NaN remains a separate strict
first-day state admission failure. A finite raw-output boundary does not waive
that failure or close the integrated state/parameter-gradient training gates.

## Mechanism evidence and repair candidate for discussion

The accepted [saved HLO inspection](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.hlo-structure.json)
records 449508/462642 lines and 623/661 cuBLAS GEMM calls for B1/B4. An FP32
`value @ kernel + bias` at `strong_baseline_architecture.py:887` has a B1
contraction `[128,64] × [64,64]` selecting algorithm18, versus B4
`[512,64] × [64,64]` selecting algorithm19. Attention score/weighted-value
contractions at `shared_transformer.py:518/543` also differ in shapes/choices.
These are structural lowering facts, not exclusive runtime costs or a causal
intervention. They support batch-dependent floating-point contraction/reduction
as a hypothesis; earlier input/context assembly or normalization is not excluded.

A concrete function-preserving candidate for later discussion is a native JAX
mapped scalar body for the neural prefix or responsible stage, retaining scalar
sample shapes/arithmetic sequence, all fields/masks, weights/capacity/precision
and parameter/state AD. It could align contractions but might change fusion,
rounding or parallelism. Correctness and throughput are unmeasured. The inspected
physical-output transforms already promote raw outputs to FP64; this result does
not select a precision or tolerance change. No candidate implementation or new
graph was started. Any later adoption would require the relevant strict complete
outputs/gradients and complete-update evidence under a separately decided scope.

Separately compiled prefix graphs may choose different kernels/fusions from the
full-day/T7 graphs. A failure proves occurrence in this measured prefix, without
proving it explains every previous failure or making B1 derivative truth. Counts,
exact repeats and HLO metadata do not identify the first differing operation.

## Runtime, memory and closed occupation

| Measurement | B1 | B4 |
| --- | ---: | ---: |
| Lowering seconds | 68.81056314893067 | 88.18654748052359 |
| Backend compilation seconds | 818.3713918421417 | 1003.1592364348471 |
| Route occupation seconds | 908.0132009610534 | 1114.5235965903848 |
| Median synchronized JVP seconds,12 calls/route | 0.08639165479689837 | 0.11747290752828121 |
| Compiled estimated live bytes | 662196830 | 1114904796 |

The actual deployed [runtime preflight](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.runtime-preflight.json)
passed in 71.8682838883251 seconds, JAX0.4.38, no real model/Teacher data/real
training update. Probe elapsed time was 2063.0935589093715 seconds. JAX peak
bytes in use was 1,137,140,992; sampled NVML peak3525MiB across1046 samples.
Compiled/JAX/sampled NVML checks pass24GiB. The independent audit accepts the
probe's NVML summaries; it does not independently reparse the CSV or claim an
unsampled instantaneous peak. Diagnostic JVP times are not update throughput.

The [true approval](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.approved.json)
binds the user's delegated explicit approval to this independent free scope.
[Launch](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.launch.txt)
was remote-host 2026-10-10T13:35:32+0800, ibc13b02n04/GPU1, CPU affinity0–55,
OMP4/OpenBLAS1. Host load41.92/41.92/41.95 is context, not proof of the cause
of compilation time or a whole-node reservation.

[Actual exit](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.actual-exit.json)
and [settled ledger](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.occupation.json)
record exit3, 2169.713307645172/5400 GPU seconds,
0.602698141012548 card-hours/CNY0. Session is
`55e2d57d-a073-463f-8f44-b9222a95f0ba`. Unused3230.286692354828 seconds
grant no new work. The [release receipt](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.release.json)
confirms GPU1 0MiB/0%, no compute apps, wrappers91303/91318 absent and
queried scheduler header only. Eight old tail/bash log followers remain untouched.

## Preserved artifacts and stop

Local/remote diagnostic root ends in `outputs/diagnostics/shared-latent-reader-20261004/`;
remote root is under `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/`.
Prefix is `T7-raw-neural-jvp-20261010`:

- All24 raw arrays, local and remote:
  `<prefix>.result.{b1,b4}.cycle{1,2,3}.coordinate{434,2382,2621,200}.result.pkl`.
- Both successful remote binaries and identity receipts:
  `<prefix>.result.{b1,b4}.jvp.executable.pkl` and `.identity.json`.
- Compressed optimized HLO and identities, local/remote:
  `<prefix>.result.{b1,b4}.optimized-hlo.txt.gz` and `.identity.json`.
- Compile attempts, NVLink/execute logs, true/false approvals, exact argv,
  source/member identities and the nine-member archive are retained. Archive
  SHA-256 is `f3dbcc7f62aed4f3afe478ae3f8f434d3d8a7efd1dba292c6c2aea62e1c2fd86`.

The [handoff](daily_speed_handoff_20261010.md) indexes all pending local changes,
actual tasks, immutable c599236 checkout/overlays, successful reference graphs/
gradients, prepared/Teacher data, cache/environment/toolchain paths and exact
reproduction commands. The measured2.555005x four-T7 full-AD component gain
remains worth preserving; complete-update acceptance and scientific accuracy
are unestablished. Forward18.57%/additional AD-recomputation81.43% remain
component shares, not pure-reverse or complete-update partitions.

**Stop here**, as explicitly directed. No next GPU diagnostic/long training,
new scope question, directory move, root-branch switch, runtime cleanup,
staging, commit or push is part of this round. Open causes and the candidate
repair are handed over for a later user decision.
