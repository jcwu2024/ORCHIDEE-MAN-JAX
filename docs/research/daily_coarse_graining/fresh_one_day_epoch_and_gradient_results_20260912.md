# Completed one-day epoch and independent T2 derivative diagnostics

## One-day training: meaningful learning, limited acceptance scope

Slurm15048786 completed1998/1998 updates, exit0, scheduler elapsed02:24:59.
Runner elapsed8677.23s. It consumed63,936 row occurrences representing63,707
unique Teacher days, one complete declared prepared-pool epoch. Parameters
started fresh from seed20260911 and all109,464,375 neural parameters were
eligible for joint AdamW updates. The final checkpoint follows the declared
final-complete-update rule; update1500 is not retrospectively selected instead.

The existing physical/finite gates did not stop the run. The bounded learning
screen passes on both fixed panels. Final excluded-day monitor (64 dates,
44 landpoints and4 years; not independent sites or sealed validation):

| Metric | Initialization RMSE | Final RMSE | Final bias |
| --- | ---: | ---: | ---: |
| GPP, gC/m2/day | 3.06484 | 0.59343 | -0.157 |
| NPP, gC/m2/day | 2.61346 | 0.42091 | -0.099 |
| Maintenance respiration, gC/m2/day | 0.15474 | 0.05530 | -0.020 |
| Growth respiration, gC/m2/day | 0.60166 | 0.16191 | -0.038 |
| Autotrophic respiration, gC/m2/day | 0.52105 | 0.18036 | -0.058 |
| AGB one-day change, gC/m2 | 2.01949 | 0.30753 | -0.083 |
| BGB one-day change, gC/m2 | 0.64734 | 0.12010 | -0.016 |

These are Teacher-started individual days, not free-running7/30-day trajectories.
Heterotrophic respiration is identically zero on these small panels, so its
zero error provides no positive fitting evidence. The training monitor has
32 dates/28 locations/4 years. Neither panel represents all535 training points
and44 years covered by the pool, or proves spatial/temporal generalization.

The final stage is not monotonically improving: from1500 to1998, excluded
total loss rises3.87% and flux loss6.90%; five of the seven tabulated RMSEs
worsen. GPP/NPP biases drift negative. This is a limited-panel warning, not
proof of broad overfitting, and not a reason to erase the large initial-to-final
gain. Do not blindly continue the same fixed-rate run as the next main task.

Clipping occurs530/1998 times (26.53%). It falls from100% in the first100
updates to6.6% at1001--1500, then8.84% in the last498. The early maximum raw
gradient norm is about1.69e7; clipping protected those updates. Later stability
does not make the early outliers disappear from the evidence.

Mean update time excluding the first update is4.082s. Peak active GPU memory
is10.02GiB versus20.63GiB allocator pool; host peak reported by the runner is
7.84GiB. At the stated additive rates,02:24:59 costs approximatelyCNY6.67,
not an invoice, below the approvedCNY11.04 ceiling.

Report: `runtime/outputs/training/fresh-one-day-pool-15048786/report.json`.
Final checkpoint: `report.checkpoint.pkl` in that directory. Local report:
`outputs/one-day-pool-15048786-final.json`.

## T2 derivatives: execution completed, correctness gate unresolved

The native checkpointed test completed in2049.79s, with1800.02s compilation
and0.624s evaluation. Its compiled-buffer estimate is2.27GiB, versus5.31GiB
for the ordinary reference. All gradients are finite and primal loss agrees,
but the declared comparison fails:46 initial-state coordinates exceed tolerance,
and some parameter leaves fail their L2 gate despite zero parameter-coordinate
exceptions. This is not an exception or out-of-memory failure.

At the dominant reserve-carbon coordinate2250:

| Method | Derivative |
| --- | ---: |
| Ordinary direct reverse | 1469.15114 |
| Streamed reverse | 1470.69805 |
| Native checkpointed reverse | 1470.28568 |
| Forward JVP | 1470.29254 |

Checkpointed and JVP are especially close at that coordinate. Therefore the
discrepancy cannot simply be attributed to an omitted manual cotangent. This
is evidence to investigate execution/precision and favor a framework-owned
checkpointing route if it meets subsequent requirements; it does not prove
one method exact or admit all coordinates/horizons.

Replay plus directional diagnostics completed in1028.0s with no updates.
Stored streamed derivatives reproduce under the primal observer. Visible
one-day replays differ slightly from saved forward values; finite-field
differences are dominated by thermal heat capacity. Nominal JVP/reverse
projections are finite. FD values are not monotone and the reserve-coordinate
series has no convincing convergence plateau.

The original diagnostic report has two measurement defects: equal undefined
NaNs contaminate raw replay norms, and `(NaN-NaN)/(2 epsilon)` produces NaN
tangents on fixed undefined coordinates, poisoning per-scale reverse dot
products. These NaNs are not evidence of nonfinite model gradients. The fixes
keep only explicitly fixed coordinates' tangents zero, still reject genuine
nonfinite gradient/direction values, and respect defined masks in replay counts.
Six focused tests pass. Original reports remain unchanged. Projection recovery
reconstructs the same fresh parameters and recomputes only dot products from
saved gradients and epsilons; it does not rerun rollout/JVP/AD evaluations.

Raw reports are under `runtime/outputs/diagnostics/`:
`public-rollout-T2-v2-checkpointed-diagnosis.json` and
`public-rollout-T2-v2-directions-diagnosis.json`. Their local copies use
`outputs/public-rollout-T2-v2-{checkpointed,directions}-final.json`.

Projection recovery completed at7577899 in379.49s, exit0. Its result
`public-rollout-T2-v2-projections-recovered.json` contains finite per-scale dot
products for all three reverse methods, without rerunning rollout/JVP/AD.
At the nominal parameter-difference direction, direct/streamed/checkpointed/
JVP projections are0.0212038281/0.0212265513/0.0212203299/0.0212212202.
Checkpointed is closest to JVP there as well as at reserve coordinate2250.
The old finite FD/JVP values are preserved; they were not re-executed in this
recovery. The original replay aggregate NaN statistics cannot be fully recovered
without raw observations and are not used as finite-mismatch counts.

## Next research order

Use the final trained checkpoint for an independently specified7/30-day
forward-only screen: cumulative GPP/respiration/NPP, biomass change, physical
validity and actual inference cost. Forward evaluation does not depend on
closing the multi-day adjoint question. Select evaluation windows without
overlap with consumed training days and do not access sealed test data.

In parallel, consolidate numerical evidence and verify native checkpointed
AD for the intended longer training graph. Do not keep the original direct
reverse as an unquestioned exact oracle, relax thresholds merely to pass, or
restart a large training job before reviewing these results. Data/throughput
expansion remains necessary, with effective batch and learning-rate behavior
informed by the observed late bias drift. No further paid job is approved.
