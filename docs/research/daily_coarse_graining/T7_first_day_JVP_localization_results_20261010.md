# T7 First-Day Forward-JVP Localization Results

Updated: 2026-10-10. The approved first-day diagnostic completed both new
graphs and all 24 JVP calls, with no retry, VJP or optimizer update. Finite,
defined next-state tangents differ strictly between B1 and B4 sample 3 for every
direction in every cycle, including control 200 whose full-T7 scalar JVP
comparison passed. The discrepancy is reproduced within the first canonical
daily transition, before later recurrence and trajectory loss. This locates
its occurrence; it does not establish its mechanism or choose a reference.

Finite primal values, canonical observation tangents and exact auxiliaries
match within their unchanged gates. Raw next-state coordinate 20 is nonfinite
in each case, a separate retained failure. The result is `numerical_failed`,
actual launcher/ledger exit 3. The independent scope closed at
**2124.572662062943 GPU seconds / 0.5901590727952619 card-hours / CNY 0**;
GPU released. Unused **3275.427337937057 seconds** grant no additional graph.

## Fixed computation and comparisons

The [proposal](T7_first_day_JVP_localization_proposal_20261009.md) owns the exact
approved command and method. Reuse latent32/private-reader64 candidate weights,
TRAIN artifacts 742/275/482/299 and the existing public recurrent objective's
canonical `RolloutLoss.step`, source-reuse adapters and guarded turnover.
First-day input admission verifies all 215 leaves: B1 equals B4 sample 3,
including the full `(3854,)` FP64 continuous state. Only sample 3/artifact 299
receives directions 434/2382/2621/control 200. Neural arithmetic remains FP32;
physics and state directions remain FP64.

B1 and B4 sample 3 compare at natural per-case transition scale, with **no /4**.
Three alternating cycles per route give 12 calls per route. Preserve full
next-state values, canonical observations, floating tangents and exact
integer/Boolean/static auxiliaries before comparisons. Numeric gates remain
`rtol=1e-5, atol=1e-6`, with finite, topology, shape, dtype and mask checks.
The first-day component does not measure full T7 loss, parameter VJP,
fourteen full-horizon hard gates or complete optimizer updates.

## Cross-route directional results

Each row is reproduced in all three cycles. Failed counts refer to the full
3854-coordinate next-state tangent. B1 is actual, B4 sample 3 is reference;
allowance is `1e-6 + 1e-5 * abs(reference)` at natural scale.

| Input direction | Failed next-state tangents | Worst output coordinate | B1 tangent at worst coordinate | B4 tangent at worst coordinate | Error / allowance |
| --- | ---: | ---: | ---: | ---: | ---: |
| 434 | 200 | 63 | -65.767334229276 | -65.78758738427223 | 30.738953724477216 |
| 2382 | 16 | 11 | -19.84559375563874 | -19.8474884128562 | 9.498224429305221 |
| 2621 | 14 | 3613 | -0.03673016159880458 | -0.03672409287296587 | 4.438666010638788 |
| 200 | 16 | 383 | -0.3148858206612724 | -0.31493323189871625 | 11.4262328970104 |

The [corrected supplemental review](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-raw-neural-jvp-20261010.review.json)
maps each exact worst output coordinate through the saved state-leaf contract:

| Output coordinate | State field | Tensor index | Local flat coordinate |
| ---: | --- | --- | ---: |
| 63 | `enerbil_previous_step_state.cgrnd` | `[0,4,1]` | 9 |
| 11 | `diffuco_previous_step_state.gsmean` | `[0,1]` | 1 |
| 3613 | `slowproc_stomate_previous_step_state.soilhum_month` | `[0,3]` | 3 |
| 383 | `hydrol_previous_step_state.nroot` | `[0,0,8]` | 8 |

All four coordinates are defined in both routes. Exact coordinate lookup
replaces the provisional group-first lookup; this mapping does not establish
a cause or change any saved numerical result.

All compared tangent values are finite. The three floating observation leaves
(budgets, fraction, GPP; six values) pass cross-route tangent comparisons.
Their passing does not waive the next-state failures. All twelve selected
cross-route auxiliary comparisons pass exactly: 20 numeric leaves / 3883
values plus three static `None` paths. B4's other three samples have zero
directional tangents in every call, preserving direction isolation.

Every finite cross-route primal coordinate passes; worst finite next-state
error/allowance is 0.5935057179923553 at output 737. The raw primal comparison
still fails because next-state coordinate 20 is nonfinite. Each B1 artifact
contains one such entry; each full B4 artifact contains four, one per sample.
This is a coordinate identifier, not a count of twenty nonfinite entries.

The accepted [v3 metadata audit](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.state20-inspection.v3.json)
confirms all sixteen nontrivial repeats (cycles 2/3 against 1): finite primal/
tangent values are bitwise equal, nonfinite kinds/positions repeat exactly,
and auxiliaries/masks match. Primal admissions still retain one B1/four B4
nonfinite failures. Exact repeatability does not waive those admissions.

All finite tangent failures are marked **defined in both routes**, with exact
full-state masks (3834 defined / 20 undefined coordinates at sample 3).
Across the twelve direction-cycle comparisons, DOC accounts for 225 failed
coordinate instances and `thermosoil_previous_step_state.pcapa_en` for 162;
these include repeated cycles and are not distinct-coordinate counts. The
field groups bind localization targets, not causes or capacity decisions.

State 20 binds `diffuco_previous_step_state.roughheight_pft[0,0]`, source
`condveg_roughness`. Its initial and saved primal are NaN with canonical and
saved defined masks false; its tangent is finite zero. The source writer is
`fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_z0cdrag_dyn`, lines
1519–1720, assignment 1715–1717: the loop starts at Fortran `jv=2`, so that
routine does not assign the first PFT slot. The JAX writer initializes NaN at
`jax_orchidee/sechiba/condveg.py:364`, write loop 366–418. The consumer is
`diffuco.f90::diffuco_aero`, lines 940–1144, roughness use 1067–1068. The v3
audit ran on test01 CPU in 3.6836320608854294 seconds with zero model, JAX
numeric/trace/compile or GPU calls. This source/mask classification explains
why the raw slot is marked undefined; it neither waives its strict nonfinite
admission nor explains the finite, defined tangent mismatches. V3 supersedes
the earlier metadata artifacts.

All 24 raw health reports retain their nonfinite admission failures. Canonical
learned-output nonfinite counts are zero; observed primal/tangent budget
residuals are finite and zero. These limited daily checks do not substitute
for the fourteen full-T7 hard gates.

## Runtime, memory and closed occupation

Actual host was `ibc13b02n04`, free GPU 1, CPU affinity 0–55, OMP4/OpenBLAS1.
The **remote-host timestamp** at launch was `2026-10-10T10:37:35+0800`, load
41.91/41.98/42.03. Host and workstation clocks differ. The result's resource
snapshot records GPU 1 at 0 MiB/0% and no processes, load 42.24/42.06/42.06.
These snapshots do not establish a contention explanation.

The actual deployed GPU synthetic runtime preflight passed in
71.88881254382432 seconds, JAX 0.4.38/NumPy 2.4.6. It reads no Teacher data,
checks no real model and records zero real training updates; its synthetic
AdamW step is not a scientific model update.

| Measurement | B1 seconds | B4 seconds |
| --- | ---: | ---: |
| Lowering | 111.63072948530316 | 108.65577997267246 |
| Single backend compile | 830.3672055434436 | 901.1166920885444 |
| Cycles 2–3 median synchronized direction call | 0.06907174549996853 | 0.10021123103797436 |

Each graph compiled once and was preserved before HLO export; there was no
retry or saved executable load. Probe elapsed time was 2027.3243876416236
seconds. Diagnostic call times compare one B1 case with a four-case B4 graph;
they do not demonstrate whole-update throughput or a newly accepted speed gain.

Compiled estimated live memory is 661,934,238 bytes for B1 and 1,113,578,220
bytes for B4. Observed JAX peak is 1,134,434,560 bytes. The independent NVML
summary covers 1038 samples, peak 4793 MiB. Compiled, JAX and sampled card-total
memory pass 24 GiB; they measure different scopes, and no unsampled peak is
claimed.

Independent ledger session `9120ab94-1cce-497d-a9a9-2bc44e9f7d9c`, wrapper
71060, is `finished`, `within_session_cap=true`, actual exit 3. Charged/remaining
values are 2124.572662062943 / 3275.427337937057 seconds in this independent
5400-second scope, with no prior funding reused. The post-run release receipt
observes GPU 1 at 0 MiB/0%, no compute processes, wrapper absent and no matching
model processes. Settled actual occupation is distinct from the original
5400-second reservation. All previous scopes and balances remain closed.

## Preserved artifacts and independent CPU evidence

Remote diagnostic root:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-latent-reader-20261004/`.
Prefix: `T7-first-day-jvp-20261009`, preserving its source-origin date.

- [Actual result](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.result.json),
  [run log](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.run.log),
  [independent analysis](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.independent-analysis.json).
- [Settled occupation](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.occupation.json),
  [release](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.release.json),
  [runtime preflight](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.runtime-preflight.json).
- [Independent NumPy analyzer](../../../outputs/diagnostics/shared-latent-reader-20261004/analyze_T7_first_day_jvp_20261009.py)
  and [seven target fixture guards](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.analysis-self-check.target.json).
- Remote binaries and optimized HLO: `<prefix>.result.{b1,b4}.jvp.executable.pkl`
  and `<prefix>.result.{b1,b4}.optimized-hlo.txt.gz`, with identity/attempt records.
- All 24 raw artifacts: `<prefix>.result.{b1,b4}.cycle{1,2,3}.coordinate{434,2382,2621,200}.result.pkl`.
- NVML CSV: `runtime/outputs/diagnostics/latent-reader-T7-20261005-T7-first-day-jvp-20261009.memory.csv`.

The independent saved-artifact analysis is NumPy-only: zero model/GPU calls,
no JAX import, CPU analysis 0.8706923983991146 seconds. Its status remains
`numerical_failed`, preserving observed failures. Seven small target CPU guards
passed: exact masks, natural-scale allowance, native JSON coordinates,
nonfinite failure/strict JSON, static `None` and dtype topology, B4 sample axis,
and settled actual occupation versus reserved-only receipt. Fixture success
does not waive production failures.

Accepted source archive SHA256 is
`c947e671e61e214234fa0258da6e5e26a34ecc1eb49000229fa58735ed0a74bf`, seven members
(four new scripts, three unchanged dependencies). The false template and
separate true approval, immutable c599236 checkout, input receipts, raw arrays,
both binaries and compressed HLO remain preserved. No Teacher regeneration.

The parent recovered graph project `E-ORCHIDEE-MAN-JAX`: earlier checks used
generation `2026-10-09T15:41:40Z`; latest coverage records generation
`2026-10-10T03:36:01Z` (recorded `03:39:19Z`). Relevant helper/probe/canonical
rollout/model and `successor_training.py` coverage reports `no_recorded_issue`
with `metadata_changed`; exact source was read.
The independent reviewer had `Transport closed` and uses supplied/source
fallback evidence. These bounded checks do not establish exhaustive codebase
coverage or full scientific acceptance.

## Saved-HLO inspection

The accepted [compact v2 summary](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.hlo-summary.json)
and [detailed inspection](../../../outputs/diagnostics/shared-latent-reader-20261004/T7-first-day-jvp-20261009.hlo-inspection.v2.json)
were produced on test01 CPU, with zero model/GPU calls and no JAX import.
The [inspection script](../../../outputs/diagnostics/shared-latent-reader-20261004/inspect_T7_first_day_hlo_20261010.py)
reads the preserved compressed HLO; provisional v1 is superseded by v2.

| Optimized-HLO fact | B1 | B4 |
| --- | ---: | ---: |
| Entry parameters / outputs | 386 / 28 | 386 / 28 |
| Literal `dot` instructions | 0 | 0 |
| FP32 BLAS custom calls | 563 | 601 |
| FP64 BLAS custom calls | 0 | 0 |
| Converts producing FP32 | 872 | 763 |
| Converts producing FP64 | 307 | 326 |

Counts include computation/fusion bodies and are not exclusive runtime counts
or timings. Both structural state-20 paths reach a gather/select boundary
at `successor_training.py:149`; the third operand broadcasts an actual FP64
NaN constant. The predicate remains unevaluated. This identifies a candidate
source path, not causality, a defect or a mask exemption. Nonfinite literals
and compiled-operation counts alone do not explain finite tangent differences.

## Current consequence and next action

First-day tangent differences are reproduced, including control 200. The
[completed raw-neural result](T7_raw_neural_JVP_results_20261010.md) now reproduces
active primal/tangent failures at canonical input rebuild/assembler/context →
all161 raw learned fields/exact masks before transforms/update. Accepted v3-final
audit confirms exact repeats/isolation, retaining numerical failure. The separately
approved one-V100 scope closed at2169.713308/5400 seconds/CNY0, exit3/released,
two graphs/24 JVPs, zero VJPs/updates. No old balance is reused.
The [handoff](daily_speed_handoff_20261010.md) is complete; stop. No cause,
harmlessness, mask exemption, precision/reference choice or adoption is inferred.

The raw prefix uses separately compiled B1/B4 output boundaries, whose kernel/
fusion choices can differ from full-day/T7 graphs. A failure proves occurrence
in that measured prefix, not direct causality in the older integrated graph.
A pass cannot exclude small or compilation-dependent raw differences that
later computation amplifies. No additional checks or acceptance waiver follows.

The measured baseline four-window full-AD component gain remains 2.555005x;
batching is not adopted and previous full-update/state failures remain.
No complete-update optimization has passed every gate. The main task remains
faster unchanged-workload T1/daily, T7 and T30 before formal training.
[NEXT_STEPS](../../NEXT_STEPS.md) owns current action and authorization.
