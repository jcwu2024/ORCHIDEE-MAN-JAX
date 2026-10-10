# Bounded held-out scientific validation — 2026-10-04

Status: **completed, not adopted**. Actual diagnostic step8 and step16 weights
from original A / V4-V7 candidate / original B were evaluated through the same
admitted **original field expert128/backbone source-reuse canonical forward**.
All six panels and24 selected-window forwards completed with actual exit0.
This is descriptive scientific sensitivity, not a noninferiority test,
final-method acceptance or production adoption. Candidate strict
gradient978/state287 and original-repeat state7 remain unresolved.

The candidate's held-out effects are mixed at step8 and step16/T7. At
step16/T30, all five non-RH flux daily RMSEs and six of seven stock level RMSEs
are worse than both originals; DOC is better. This does not support waiving
the strict failures or claiming that the approximate V4/V7 update is harmless.
The evidence supports deferring adoption and retaining original execution as
the reference while structural speed routes are considered.

## Fixed panel and actual overlap

The following four prepared v6 records were frozen before any model prediction.
Temporal means train location / validation year; joint means validation
location / validation year. The data was historically available for method
development and is labelled **held_out_development**, not a sealed independent
test. Each represented split/horizon cell has one window; spatial-only is absent.

| Ordinal | Slice | Horizon | Location | Year | Zero-based Teacher rows |
|---:|---|---:|---|---:|---|
| 1 | temporal | T7 | 095.0-113.0 | 2007 | 167–173 |
| 9 | temporal | T30 | 321.0-081.0 | 2005 | 258–287 |
| 33 | joint | T7 | 189.0-095.0 | 2006 | 115–121 |
| 41 | joint | T30 | 281.0-091.0 | 2006 | 239–268 |

Panel SHA256:
`2198ff35f0daef2162e316a376e4708ab84d56515deb32abc912ff7c11bf67f8`.
The CPU audit read the actual compact training bundle and artifact742's
prepared schedule:32 daily rows plus the declared T7/T30 recurrent window
index,69 unique Teacher day keys, repeated throughout the16-update diagnostic.
All four validation locations are absent from that actual training-location
set; years2005/2006/2007 also differ from training1991. Exact day and exact
window intersections are zero. Prepared day_index is row+1; row keys and actual
day indices remain distinct. This checks those repeated inputs, not absence
of earlier exploratory checkpoint or method-development exposure. The
selection/overlap artifacts and independent source review retain that limit.

## Scientific results

The public monitor/extractors were reused. **NPP is source npp_daily**, with
GPP−(maintenance+growth respiration) retained only as an independent identity
diagnostic. Identity residual maxima were4.44e−16 for predictions and8.88e−16
for Teacher. Source NPP was never replaced by the identity formula.

The table shows pooled temporal+joint errors: two windows per horizon,
14 T7 or60 T30 defined daily values and two defined cumulative/endpoint values.
Daily values along each recurrent trajectory are not independent samples.
Bias is prediction−Teacher; positive candidate−original RMSE means worse
accuracy. Flux daily units are gC m⁻² day⁻¹; cumulative flux and stock units are
gC m⁻². Stock **level** uses the public prediction/Teacher/common-initial
valid mask; final uses endpoint and common-initial validity, not an all-day
complete gate. Daily stock change retains the preceding-day mask. The common
initial is shared, so endpoint stock error equals endpoint stock-change error.

| Milestone / horizon | GPP daily RMSE A / C / B | Source NPP daily RMSE A / C / B | AGB level RMSE A / C / B | BGB level RMSE A / C / B | Canopy C level RMSE A / C / B |
|---|---:|---:|---:|---:|---:|
| step8 / T7 | 3.32192 / 3.32372 / 3.32198 | 2.90523 / 2.90455 / 2.90524 | 10.7248 / 10.7230 / 10.7249 | 2.57575 / 2.57490 / 2.57575 | 19.2188 / 19.2838 / 19.2199 |
| step8 / T30 | 3.57740 / 3.57881 / 3.57744 | 3.37080 / 3.36967 / 3.37079 | 43.2468 / 43.2369 / 43.2467 | 17.1053 / 17.0941 / 17.1051 | 34.7854 / 34.9885 / 34.7895 |
| step16 / T7 | 3.27120 / 3.27216 / 3.27123 | 2.25975 / 2.25520 / 2.25971 | 8.47985 / 8.46365 / 8.47970 | 1.92898 / 1.92402 / 1.92892 | 29.4778 / 29.6158 / 29.4795 |
| step16 / T30 | 3.70590 / 3.71814 / 3.70587 | 2.90105 / 2.91148 / 2.90100 | 37.4834 / 37.6560 / 37.4827 | 13.6074 / 13.7273 / 13.6071 | 72.4119 / 72.7482 / 72.4157 |

C is candidate. At step8 both horizons improve maintenance-respiration and
source-NPP daily RMSE, plus AGB/BGB stock level and endpoint RMSE. GPP,
growth and total autotrophic respiration get worse. Canopy carbon, both litter
stocks and SOC get worse. DOC T30 improves; DOC T7 final improves slightly
while its daily level RMSE gets slightly worse.

At step16/T7, source NPP and AGB/BGB remain better, while the other non-RH
fluxes and other stock level RMSEs are worse. At step16/T30, GPP daily bias is
−3.61920 versus A−3.60233/B−3.60229; cumulative GPP RMSE is111.378 versus
A110.997/B110.996. Source-NPP cumulative RMSE is87.2347 versus
A86.9080/B86.9064. AGB final RMSE is63.7603 versus A63.5719/B63.5708;
BGB is23.0739 versus A22.9421/B22.9417; canopy is116.588 versus
A116.108/B116.114. DOC level RMSE improves to3.67755 versus
A3.70536/B3.70538. These are observed changes, not new scientific thresholds.

Original A/B repeats also drift. At step16/T30, candidate−A / candidate−B /
B−A direct daily GPP RMS differences are0.0539524 /0.0539617 /0.0000638093;
AGB level RMS differences are0.457399 /0.457716 /0.000688792; canopy level
RMS differences are0.345312 /0.341860 /0.00437471. The candidate effects on
these quantities exceed the observed original-repeat drift in this panel.
That comparison does not create a statistical acceptance threshold.

Every panel has all public hard counts0, including defined nonfinite
prediction/Teacher values, mask mismatches, negative shared stocks and missing
valid curves. All24 windows pass the physical gate; every reported flux/stock
metric has its expected bounded count and nonfinite_defined0. **RH Teacher and
prediction are zero on all74 daily values and all four window totals per
panel**. RH error0 therefore provides no evidence of a learned nonzero RH
response. Physics/mask passes do not establish trained scientific accuracy,
long-horizon stability, global branch coverage or noninferiority.

Full six-flux/seven-stock daily bias/RMSE, cumulative errors, stock final,
level-trajectory and daily-change errors, candidate−both-original metric
differences and B−A output drift are in
[`science-tables.md`](../../../outputs/diagnostics/shared-v4-heldout-science-20261004/science-tables.md).
Precision-preserving values, direct output differences and per-window
hard/RH/identity checks are in
[`science-comparison.json`](../../../outputs/diagnostics/shared-v4-heldout-science-20261004/science-comparison.json).
The six original `.science.json` files retain each window's public physical
curves; full rollout observations remain in remote `model/raw/`.

## Execution and verification

CPU preparation on test02 exited0 in **61.485419s /300s** with no model,
optimizer or Teacher regeneration. Seven parameter-only snapshots (common
initial plus six milestones) each passed300 finite numeric leaves /
64,884,481 coordinates /2 static None and the same shape/dtype descriptor.
The common initial was preserved for provenance; no step2 prediction or
training-loss bridge was added.

The free gln01 V100 GPU1 run exited0 in **570.202266s /1500s**, including
setup, actual runtime preflight81.782202s, loading, compiling,24 calls,
saving and process exit: **0.158389518 card-hours, CNY0**. Immutable checkout
`c599236bcf57b745e9bafed678298010f4c3b58f`; original adapters and public
monitor/pool overlay sources were preserved in source-attempt directories.
OMP4/OpenBLAS1, one visible GPU, JAX0.4.38, no paid job or Slurm fallback.
Start node load0.01/0.02/0.05, CPU affinity0–111, no visible GPU compute
processes. Affinity is observed availability, not a whole-node reservation
or exclusive allocation.

| Canonical forward ABI | Lowering seconds | Backend seconds | Estimated live GiB |
|---|---:|---:|---:|
| T7 | 61.557813 | 170.268365 | 0.3977 |
| T30 | 31.192068 | 154.275428 | 0.3986 |

Exactly two forward graph/backend attempts were saved, one per horizon,
shared by all six parameter trees. Runtime peak active bytes656,871,424
(0.612GiB), below24GiB. Calls started24/saved24, gradient compilations and
executions0, training updates0, optimizer executions0. Both V100 cards were
observed0MiB/0% after actual exit. No allowance balance was used for additional
model calls.

Remote persistent origin/deadline and performance durations are the resource
accounting authority. A local/remote wall-clock offset caused an interim
elapsed-time overestimate; remote date/ps and saved timings corrected it.
Timers and budgets were never reset. Tiny saved-JSON report reduction ran
locally without loading model/weights and took0.094368s; it is reporting
arithmetic, not another model execution or CPU preparation.

Canonical forward compilation totaled417.293674s, about73% of the entire GPU
window. This is measured turnaround evidence for a **forward-only** panel,
not proof of same-workload training speed or a10x result. Source-reuse science
probing fits the authorized1500s here; prospective complete-gradient route
probes need their own unchanged-workload budget and component/preflight
staging. This run does not authorize those probes.

An independent reviewer checked bounded extractor provenance, actual overlap
scope, fixed subset guard, endpoint/common-initial masks and sample reducer/
canonical numeric equality. Graph metadata was supplied by the parent at
Verify generation2026-10-04T09:48:48Z with metadata_changed/no_recorded_issue.
Worker MCP requests returned Transport closed; relevant exact sources were
read, and no exhaustive graph claim is made.

Artifacts: local `outputs/diagnostics/shared-v4-heldout-science-20261004` and
remote `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-v4-heldout-science-20261004`.
See `selection.json`, `training-overlap.json`, `CPU-preparation.json`,
`actual-cpu-exit.json`, `actual-gpu-exit.json`, `preflight.json`, `model/model.json`,
`model/compile-budget.json`, six `model/*.science.json`, `GPU-release.json`,
`science-source-review.md` and tables/comparison files. Reproduce small-JSON
report reduction with `conda run -n ORCJAX python -B -m scripts.hpc.summarize_shared_v4_science_validation outputs/diagnostics/shared-v4-heldout-science-20261004`; no model calls.
