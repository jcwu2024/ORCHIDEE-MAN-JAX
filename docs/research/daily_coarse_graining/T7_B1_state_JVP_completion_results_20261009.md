# T7 B1 state-JVP completion results — 2026-10-09

The independent completion finished all 32 JVP calls and four full VJP endpoint
calls. One new B1 JVP graph compiled successfully; B4 JVP and both VJP graphs
were loaded. Both routes' four JVP cycles repeat exactly, but B1/4 versus B4
strictly fails at coordinates 434 and 2382 in every cycle. Coordinates 2621 and
control 200 pass that cross-route JVP comparison. Same-route JVP/VJP and full
state-gradient self comparisons also retain failures. Passing primal, parameter
and physical gates does not waive them or choose a derivative reference.

The free single-card scope is completed, with GPU released. Its independent
ledger charges **1780.7436698693782 GPU seconds**, **0.4946510194081606 card-hours**,
CNY 0. Unused **1819.2563301306218 seconds** do not implicitly authorize another
graph. No optimizer update, Teacher generation or precision/science change ran.

## Fixed method and identities

Reuse candidate latent32/private-reader64 weights, public checkpointed T7 loss,
guarded turnover, TRAIN artifacts 742/275/482/299 and all five dynamic argument
trees. Neural arithmetic remains FP32; physics and initial-state unit directions
remain FP64. Sample 3 is artifact 299; coordinates are 434, 2382, 2621 and control
200. Keep `rtol=1e-5, atol=1e-6`, finite/physical/mask gates and 24 GiB.

B1 retains a native tangent from the single-case loss and separately divides it
by four exactly once; its returned primal loss is /4 and auxiliary is unchanged.
B4 differentiates the four-case mean. Cycle pickles retain both `raw_tangents`
and `tangents`: B1 has `tangent_scale=raw_and_div4`, B4 has
`tangent_scale=mean_objective_scale`. The independent NumPy analyzer verifies
metadata, FP64 scalar shape, native/scaled values in JSON versus pickle, and
exact B1 `tangents == raw_tangents/4`; B4 vectors are identical at both labels.
Saved B1 VJP states remain native/unscaled shape `(3854,)`; B4 states are mean
gradients shape `(4,3854)`.

The accepted GPU source archive is
`T7-B1-state-jvp-completion-20261009.source.accepted.tar.gz`, SHA256
`66c99a3928509c70d78037ee348a91edb56ebc31cfd23d1d7a6e4922e07caff0`.
The separate true approval and independent ledger bind scope
`latent_T7_B1_state_jvp_completion_20261009`, output, entry and 3600-second cap.
The false template, earlier startup errors, original failed B1 attempt and B4
recovery remain preserved. The [prior recovery report](T7_state_JVP_results_20261009.md)
owns its dated partial scope; its missing-B1 statement does not describe this run.

## JVP values and strict cross-route comparison

All four cycles give exactly the following values. There are 32 observations
and 24 nontrivial repetitions against each route's first cycle. Native and
mean-scaled differences against that first cycle are all exactly zero.

| Coordinate | Native B1 JVP | B1 JVP/4 | B4 mean JVP | Signed B1/4 − B4 | Allowance relative to B4 | Error / allowance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 434 | 6.144825223097045 | 1.5362063057742612 | 1.5365953930062533 | -3.890872319920469e-4 | 1.6365953930062536e-5 | 23.774185950587, fail |
| 2382 | 11.485809273858125 | 2.8714523184645313 | 2.871486881035844 | -3.456257131251306e-5 | 2.971486881035844e-5 | 1.163140632829068, fail |
| 2621 | 0.014724918825140258 | 0.0036812297062850646 | 0.0036811811224103375 | 4.858387472710113e-8 | 1.0368118112241032e-6 | 0.04685891325807813, pass |
| 200 | 0.8292021854184574 | 0.20730054635461434 | 0.20730058243676125 | -3.6082146909199864e-8 | 3.0730058243676123e-6 | 0.011741646118300194, pass |

Eight of the 16 same-cycle cross comparisons fail. This is a same-precision
forward-derivative difference in addition to the retained reverse-mode
differences; reverse repeat variability alone is insufficient to describe the
whole observed discrepancy. It does not establish either route as truth,
scientific harm, or a mechanism.

## Own JVP versus VJP endpoints

The table shows selected VJP projections on the mean-loss scale. Full B1 files
retain native values (each displayed B1 value multiplied by four); native and
scaled tests keep their own unchanged absolute allowance. Their pass/fail
results cannot be substituted for each other.

| Coordinate | B1 VJP/4 begin | B1 VJP/4 end | B4 VJP begin | B4 VJP end |
| --- | ---: | ---: | ---: | ---: |
| 434 | 1.5362364330794662 | 1.5362227985169739 | 1.5363445091061294 | 1.5363260912708938 |
| 2382 | 2.871502216821093 | 2.8715202471730965 | 2.871404509744317 | 2.871394376999068 |
| 2621 | 0.0036805756390094757 | 0.0036794058978557587 | 0.0036837831139564514 | 0.003682062029838562 |
| 200 | 0.20730075497639713 | 0.2073007092109288 | 0.20730083688185474 | 0.20730075734731168 |

Every cycle has the following own-route failures; control 200 passes every check.
The comparison uses JVP as actual and the route/phase VJP as reference.

| Route / endpoint | Native failed coordinates | Mean-scaled failed coordinates |
| --- | --- | --- |
| B1 / begin | 434, 2382, 2621 | 434, 2382 |
| B1 / end | 434, 2382, 2621 | 434, 2382, 2621 |
| B4 / begin | 434, 2382, 2621 | 434, 2382, 2621 |
| B4 / end | 434, 2382 | 434, 2382 |

Native checks pass 20/64; mean-scaled checks pass 24/64. At 2621, B1 native
JVP versus begin VJP fails (difference 2.6162691023554063e-6, allowance
1.147223025560379e-6), while its /4 check passes (difference
6.540672755888516e-7, allowance 1.0368057563900948e-6). The B1 end /4 check
fails at that coordinate. These retained distinctions prevent scaling from
hiding native failures. The JSON records every signed difference and allowance.

## Full endpoint, primal and physical validation

All 64 own-route primal/auxiliary bridges pass: two routes × four cycles × four
directions × begin/end. B1 loss bridges apply /4; auxiliary comparisons retain
the complete native tree. Both routes' endpoint loss and all 19 numeric
auxiliary leaves pass strict self-repeat and finite checks. All 14 hard physical
and mask gates are finite and zero at every endpoint and bridge.

Each parameter endpoint has **355 numeric FP32 leaves / 66,123,713 coordinates**.
Full own-route parameter-gradient self repeats pass with zero strict failures.
Two static `None` nodes are excluded from numeric counts and retained in topology:
`root.shared_transformer.output_axis_position_embedding[1]` and
`root.tokenizer.axis_position_embedding[1]`. Their omission or replacement,
container/leaf path drift, shape drift or dtype drift is rejected.

| Complete state comparison | Values | Strict failures | Coordinates |
| --- | ---: | ---: | --- |
| B1 native end versus begin | 3854 | 3 | 33, 180, 2621 |
| B1/4 end versus begin | 3854 | 2 | 33, 2621 |
| B4 end versus begin | 15416 | 2 | sample 3: 434, 2621 |
| B1/4 versus B4 sample 3, begin | 3854 | 5 | 33, 434, 515, 2382, 2621 |
| B1/4 versus B4 sample 3, end | 3854 | 5 | 33, 434, 515, 2382, 2621 |

All are finite. Cross-route complete-state comparisons use B1/4 as actual and
B4 sample 3 as reference. No cross-route full-parameter comparison or all-four
independent-B1 full-state comparison is claimed by this completion analysis.
The full-state and own JVP/VJP failures remain strict failures despite passing
parameters, primals and physical gates.

## Runtime, memory and occupation

Execution started on free gln01 GPU 1, actual host `ibc13b02n04`, at host
22:26:56+0800. Initial GPU state was 0 MiB/0%/no processes; host load was
42.35/43.30/43.95. This snapshot does not attribute compile time to contention.

| Measurement | Seconds |
| --- | ---: |
| B1 lowering | 143.99193419329822 |
| Single B1 backend compile | 992.9232328124344 |
| Saved B1 VJP load | 205.49664796888828 |
| Saved B4 VJP load | 252.0583925768733 |
| Saved B4 JVP load | 25.679603258147836 |
| B1 JVP cycles 2–4 median per synchronized direction call | 0.4556320495903492 |
| B4 JVP cycles 2–4 median per synchronized direction call | 0.6623320979997516 |

There are 16 timed calls per route; cycles 2–4 supply 12 calls per route. These
diagnostic JVP times do not measure complete optimizer updates, compare four
B1 cases against B4, or establish same-workload training throughput.

JAX observed peak is 3,650,118,656 bytes / 3.39943790435791 GiB. Independent
NVML sampled 871 rows, peak 10,811 MiB / 10.5576171875 GiB, mean utilization
1.3605051664753158%. Both memory observations pass 24 GiB. JAX allocator peak
and sampled card-total memory are distinct measurements.

Independent ledger session `5587be92-e15b-45e0-b746-4e15378e7b9d`, wrapper
10339, is `finished`, `within_session_cap=true`, child exit **3**, occupied
1780.7436698693782 GPU seconds. Cumulative charged/remaining values are
1780.7436698693782 / 1819.2563301306218 seconds in this separate 3600-second
scope. Parent verification reports GPU released. The parent-observed final SSH
exit **1** is recorded separately from numeric status `numerical_failed` and
the occupation ledger's child exit 3; this report does not infer its cause.
Reservations alone are rejected as finished actual occupation. No old grant
is borrowed or reset.

The later [native-exit and release check](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-B1-state-jvp-completion-20261009.exit-release.json)
observed GPU1 at 0 MiB/0%, no compute processes, and all three completed PIDs
absent. A separate test01 `exit 3` command returned native SSH code 3 through
PowerShell. This small check verifies transmission; the actual completed-model
exit remains independently supported by its result and settled ledger.

## Artifacts and independent CPU acceptance

Remote diagnostic root:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-latent-reader-20261004/`.
Prefix: `T7-B1-state-jvp-completion-20261009`.

- Result, separate true approval and ledger: `<prefix>.result.json`,
  `<prefix>.approved.json`, `<prefix>.occupation.json`.
- Successful B1 graph: `<prefix>.result.b1.jvp.executable.pkl` and identity sidecar.
- Four full endpoints: `<prefix>.result.{b1,b4}.vjp.{begin,end}.result.pkl`.
- Eight direction-cycle artifacts: `<prefix>.result.{b1,b4}.jvp.cycle{1,2,3,4}.result.pkl`.
- NVML CSV: `runtime/outputs/diagnostics/latent-reader-T7-20261005-T7-B1-state-jvp-completion-20261009.memory.csv`.
- [Independent analysis](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-B1-state-jvp-completion-20261009.independent-analysis.json),
  [CPU execution receipt](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-state-jvp-completion-analysis-CPU-execution-20261009.json),
  [six-guard CPU verification](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-state-jvp-completion-analysis-CPU-verification-20261009.json),
  [NumPy analyzer](../../../outputs/diagnostics/shared-latent-reader-20261004/analyze_T7_state_jvp_completion_20261009.py).

Actual CPU acceptance ran on test01 with the canonical CPU Python, completed
in 4.87128541059792 seconds, exit 0. Only the 521,844-byte compact JSON was
downloaded; the full parameter/state/primal pickles remain remote. No JAX/model
or GPU call was made by the analyzer. Its first persistence attempt exposed a
NumPy-int64 JSON index bug after computation; native integer serialization was
fixed and checked before the successful CPU rerun. Original GPU results stayed
unchanged. Six meaningful small CPU guards passed; the execution receipt records
the exact analyzer hash and paths.

```text
# test01; diagnostics=<remote diagnostic root above>; prefix=<declared prefix>
PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu/bin/python -B \
  /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/T7-state-repeats-20261009/analyze_T7_state_jvp_completion_20261009.py \
  --result <diagnostics>/<prefix>.result.json \
  --vjp-prefix <diagnostics>/<prefix>.result \
  --occupation <diagnostics>/<prefix>.occupation.json \
  --nvml-csv <runtime>/outputs/diagnostics/latent-reader-T7-20261005-T7-B1-state-jvp-completion-20261009.memory.csv \
  --output <diagnostics>/<prefix>.independent-analysis.json
```

Graph project `E-ORCHIDEE-MAN-JAX` index/search/coverage transport remained
closed and generation unavailable. This task uses supplied audited identities,
exact source fallback and direct saved-array evidence, without a graph
completeness claim. No array analysis is repeated for this documentation update.

## Current boundary and subsequent localization

All executed GPU scopes are completed. CPU preparation passed for selected
[first-day JVP localization](T7_first_day_JVP_localization_proposal_20261009.md):
canonical complete daily outputs and native tangents at natural per-case scale,
with no /4. Saved T7 JVP graphs return loss tangents and primal auxiliaries,
so per-day output tangents need the proposed component graphs. Accepted
[verification](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.verification.json)
records 27 focused CPU tests, actual test01
[readiness](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.cpu-readiness.json)
(all 215 input leaves exact B1/B4 sample 3, full 3854 FP64 state), target
gln01 Shell dry-run/Python child-plan checks and false-approval rejection before
GPU preflight. The accepted seven-member
[archive](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.source.tar.gz)
has four new scripts/three unchanged dependencies, SHA256
`c947e671e61e214234fa0258da6e5e26a34ecc1eb49000229fa58735ed0a74bf`.
The user approved the new free one-V100/5400-second scope on 2026-10-10;
the [first-day run](T7_first_day_JVP_localization_results_20261010.md) completed
both graphs/24 calls, reproducing finite tangent differences within the daily
transition and retaining a separate raw state-20 nonfinite failure. That scope
closed with exit 3/GPU released; no balance implicitly funds another graph.
This completed full-T7 run's evidence is unchanged.
The component does not certify full T7 loss, parameter VJP, fourteen full-horizon
hard gates or adoption. No precision/reference, architecture, loss, recurrence,
tolerance, sampling or scientific acceptance choice follows. Closed balances
do not authorize a new graph. The main task
remains faster unchanged-workload T1/daily, T7 and T30 training; no complete-update
optimization has passed every gate. [NEXT_STEPS](../../NEXT_STEPS.md) owns action.
