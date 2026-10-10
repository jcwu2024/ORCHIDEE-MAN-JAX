# T7 state JVP diagnostic results

## Current conclusion

The B4-only recovery completed real execution:16 native JVP calls and four
saved full VJP calls, numerical exit3. All four JVP repetitions are identical
at all four selected coordinates. Every B4 primal/auxiliary bridge and physical
gate passes. Nevertheless, B4's JVP and its own VJP disagree strictly at434 and
2621 at both endpoints, and2382 also disagrees with B4's VJP. Control200 passes.
This establishes same-route AD inconsistency at these inputs; it does not
identify the cause or make either derivative a scientific reference.

B1 native JVP remains unmeasured: its original900-second backend limit was
reached after140.530821 seconds of completed lowering. The same-grant recovery
compiled only B4 and did not retry B1. Completing B1 is the selected next
diagnostic, using the saved successful B4 graph; no precision, tolerance,
architecture, loss or sampling change is selected. Additional GPU execution
needs its own bounded grant and executable preparation.

The existing four-window fixed-weight T7 screen remains serial 5.025831 seconds
versus B4 1.967053 seconds, or 2.555005x. Loss, all parameter gradients,
auxiliary outputs, physical/mask gates and memory passed; strict state parity
failed. This diagnostic adds no complete-update throughput or adoption claim.

## Fixed diagnostic and original attempt

Use the same latent32/private-reader64 candidate parameters, TRAIN windows
742/275/482/299, guarded turnover and public checkpointed loss. Neural
arithmetic stays FP32 and physics/state ABI FP64. Unit directions at sample299
coordinates 434/2382/2621 and control200 retain `rtol=1e-5, atol=1e-6` and the
24GiB memory gate. No optimizer update or formal training is performed.

The original run started 2026-10-09 19:18:29+0800 on gln01's actual host
ibc13b02n04, GPU1, initially 0MiB/0% with no processes. Host load was
43.67/43.85/44.05; this observation does not establish the cause of any slowdown.
Actual runtime preflight passed in 73.603 seconds. The raw report names
`deadline_phase=compile_B1_JVP`, completed lowering 140.5308211427182 seconds,
zero successful JVP graph compiles, zero JVP calls and zero saved VJP calls.
The occupied-card wrapper recorded exit124, 1149.8531614020467 GPU seconds and
2450.1468385979533 seconds remaining; its process group and GPU were released.

Preserve the raw report's `GPU_executed=false` and partial
`route_occupation_seconds.B1=140.530821` fields as historical logging defects.
Neither is an occupation total: the GPU preflight/backend and outer ledger
establish actual GPU use. The recovery records backend use separately from
successful graph creation and actual model evaluation.

## Same-grant recovery

Reserve at most 2400 additional GPU seconds in the same 3600-second ledger.
Keep the original approval, failed attempt and ledger history unchanged. Compile
only the unattempted B4 JVP once: lower180/backend1500/route1800 seconds,
saved-load300 seconds each, probe wall2100 seconds and outer session2400 seconds.
Reuse the passed runtime preflight. B1 backend is not retried and B1 native JVP
remains unmeasured.

The partial diagnostic runs 16 B4 JVPs and four saved full VJPs in
begin→JVP cycles→end order. Compare the B4 primal/auxiliary only with the saved
B4 result; the saved B1 sample loss and auxiliary tree are a different
objective. Compare every B4 tangent with both B4 and independently normalized
B1/4 saved VJP projections at both endpoints. Preserve self/cross failures,
physical/mask counts, full gradients, repeated values, call timing, memory and
actual exits. Neither AD route becomes an automatic scientific reference.

## Actual numerical, speed and memory result

| Coordinate | B4 JVP, all four cycles | B1 VJP/4 begin / end | B4 VJP begin / end | JVP versus B1 / B4 endpoints |
|---|---:|---|---|---|
|434|1.5365953930062533|1.536159818759188 /1.5361515411641449|1.5363727542571723 /1.5363381239585578|fail /fail|
|2382|2.871486881035844|2.8715123793769566 /2.871504064547615|2.8714192321311014 /2.8714472463598932|pass /fail|
|2621|0.0036811811224103375|0.0036787651479244232 /0.003679465502500534|0.003682747483253479 /0.003686361014842987|fail /fail|
|200 control|0.20730058243676125|0.20730076638006145 /0.2073007343746384|0.20730077867170932 /0.20730088348373316|pass /pass|

B1/B4 saved projections differ strictly at the three discrepancy coordinates
at both endpoints; control200 agrees. The four selected B1 projections pass
begin/end repeat checks, while B4 repeat checks fail434/2621. These projection
checks are not a whole3854-state repeat pass or complete two-route JVP result.
Independent CPU analysis of the saved full VJP arrays is now complete. Loss,
auxiliary outputs, all66,123,713 numeric parameter-gradient coordinates across
355 leaves, finite values and physical/mask gates pass begin/end repeats for
both routes. Two static `None` nodes are recorded separately; they are not
trainable coordinates. The initial analysis that counted them as scalars is
preserved as a raw artifact, and the corrected analysis replaces its receipt.

Unscaled full B1 state repeats fail1/3854 coordinates, at2621 (2.442x allowance).
B4 state repeats fail5/15416: sample/coordinate (3,2621), (1,509), (3,434),
(3,33), (2,514), worst3.485x allowance. The four B1/4 selected projections
passing repeats do not override the unscaled B1 failure: dividing by four
changes the relative-tolerance contribution while absolute tolerance stays
unchanged. Sample299's full3854 B1/4-versus-B4 comparison fails5 coordinates
at begin (33/434/515/2382/2621), and6 at end (also1882). Neither route supplies
an automatic reference.

B4 lowering176.972420s/backend967.551467s succeeded once. The actual967.55s
backend result exceeds the original900s subcap, supporting the internal budget
repair; it does not predict B1's missing completion time. Saved B1/B4 loads
took174.675832/217.958487s. B4 directional JVP calls are approximately0.66s
after the first0.873866s call. They are four-window directional derivatives,
not complete parameter/state updates or a measured new training speedup.

All JAX allocation snapshots pass24GiB, with observed peak3,650,118,912B.
Sampled NVML card-total peak is10,355MiB across812 observations. These memory
scopes differ; neither is compiler-planned live memory. After completed exit,
GPU1 is0MiB/0% with no compute process and wrapper110875 is absent.

Recovery occupied1659.6369441393763 GPU seconds. Original plus recovery totals
2809.490105541423 GPU seconds (0.780414 card-hours), CNY0. The same3600-second
grant has790.509894458577 seconds unused; completed diagnostic calls do not
authorize another graph attempt or borrowing another closed grant. The raw
numerical failure, complete graphs/results and finished ledger are retained.

Input-only CPU follow-up finds no selected coordinate on a field/axis extrema
tie at initial window0/day0, across65 axis memberships. Coordinates2621 and
control200 are unique field minima. This does not exclude later recurrent
ties or quantify derivative contributions; NumPy source arithmetic is not a
sampled GPU forward. See the saved-state report for masks and DOC caveats.

## Evidence assets

Recovery passed13 focused tests, actual target shell expansion and generated
full downstream `--plan-only` argument validation. Deployed scripts and repair
record match local artifacts. It launched with host timestamp
2026-10-09T20:26:06+0800 on ibc13b02n04/GPU1, initially0MiB/0%/no processes,
hostload44.92/44.62/44.68. Same-ledger session
`2cdfa103-6eae-40b6-8487-38a727b73c4f` reserved2400 GPU seconds, wrapper PID110875.
It is now finished at1659.636944s/exit3, with real model outputs preserved.

Local records are in `outputs/diagnostics/shared-latent-reader-20261004/`:

- `T7-state-jvp-20261009.approved.json`: original human approval.
- `T7-state-jvp-20261009.result.json`: preserved B1 backend timeout.
- `T7-state-jvp-20261009.result.b1.jvp.compile-attempt.json`: original attempt.
- `T7-state-jvp-20261009.runtime-preflight.json`: passed deployed GPU preflight.
- `T7-state-jvp-20261009.occupation.json`: cumulative occupied-card ledger.
- `T7-state-jvp-20261009-B4-recovery.recovery-budget.json`: same-grant repair.
- `T7-state-jvp-20261009-B4-recovery.result.json`: completed numerical comparisons.
- `T7-state-jvp-20261009-B4-recovery.result.b4.jvp.compile-attempt.json`: successful B4 phases.
- `T7-state-jvp-20261009-B4-recovery.result.b4.jvp.executable.pkl.identity.json`: saved graph identity; binary remains remote.
- `T7-state-jvp-20261009-B4-recovery.release.txt`: post-exit free GPU/absent wrapper.
- `T7-state-jvp-20261009-B4-recovery.independent-analysis.json`: corrected full-array CPU analysis, excluding static `None` nodes from numeric counts.
- `T7-state-jvp-20261009-B4-recovery.independent-analysis.raw-count-with-static-none.json`: preserved original counting defect; numerical comparisons are unchanged.
- `latent-reader-T7-20261005-state-jvp-20261009-B4-recovery.memory.csv`: card-total samples, plus `.processes.csv`.

Remote full arrays use recovery stem suffixes
`.result.{b1,b4}.vjp.{begin,end}.result.pkl` and
`.result.b4.jvp.cycle{1,2,3,4}.result.pkl`; the137MiB JVP binary uses
`.result.b4.jvp.executable.pkl`. Preserve these; no new B4 compilation is needed.

Remote records have the same names under
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-latent-reader-20261004/`.
The code runs from immutable c599236 checkout
`runtime/checkouts/shared-split-width-v1`, using the isolated
`runtime/transfer/T7-state-jvp-20261009` overlay and unchanged shared assets.

[NEXT_STEPS](../../NEXT_STEPS.md) owns current execution authority.
[Original proposal](T7_B1_B4_state_JVP_proposal_20261009.md) retains the original
two-route design; this recovery cannot claim its missing B1 JVP comparison.
[Saved-state analysis](T7_state_repeats_analysis_20261009.md) and
[T7 speed screen](shared_latent_reader_results_20261004.md) retain the preceding
numerical and speed evidence.
