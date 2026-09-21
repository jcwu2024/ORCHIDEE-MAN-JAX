# Shared admission recovery and thermal forward results

Both free gln01 tasks completed with exit0. No process remains active and no
paid task was submitted. This establishes bounded technical results, not
trained-model or long-horizon scientific acceptance.

## Original-transform admission: completed

Immutable8004a11, wrapper25784/Python26637, main test2079.96s. The saved
094dfd2 fresh parameters and full initial gradient were reused. The original
direct-tanh thermal transform was unchanged.

- Normalized-full-gradient direction: JVP8185972383.687063 versus reverse
  projection8185973620.733777; relative difference1.511e-7, PASS.
- Both full-network optimizer updates completed; all hard counters zero.
- AdamW save/load roundtrip equal; restored-parameter forwards pass finite and
  hard-counter checks on both samples. No separate before/after-save parameter
  equality or prediction-difference statistic was recorded.
- Measured allocator peak7.5221GiB; compiled gradient estimate6.67245GiB.
  The approximately26GiB device pool is not live tensor demand.
- The second update increased its sample loss2.733e6->8.283e6. Thus technical
  admission does not establish reliable descent or justify training this
  defective initialization further. Its checkpoint is a diagnostic asset.

Update timings431.72s/311.64s include compilation/loading work; they are not
warm training-throughput measurements. Source gradient/checkpoint remain remote.

## Corrected-transform forward: completed

Immutable6467453, wrapper30682, main test603.23s, zero optimizer updates.
Fresh seed20260921, identical two TRAIN rows and calibration/science scales;
no inherited trained weights. Only the shared thermal bounded transform changes.
All hard counters remain zero. No revised numerical gradient was evaluated.

| Quantity | Row819 original | Row819 centered | Row982 original | Row982 centered |
| --- | ---: | ---: | ---: | ---: |
| Total objective | 241893842.82 | 676.1371 | 221451720.12 | 434.2279 |
| State anchor | 1451363054.58 | 4054.4958 | 1328710318.03 | 2602.6893 |
| Thermal state group | 8466088283.45 | 0.318335 | 7750140626.00 | 0.070334 |
| Process labels | 1.35565 | 1.32728 | 0.658672 | 0.637545 |
| Carbon flux group | 0.0458455 | 0.0458455 | 0.161221 | 0.161221 |
| Carbon stock group | 0.454038 | 0.454038 | 0.858993 | 0.858993 |

The formerly worst deep coordinate ptn[0,27,13] had physical errors87.3598K
and80.2863K. It is no longer in the12 largest normalized thermal errors.
The reported maximum normalized errors2.85391/1.40889 and its unchanged
0.000118617833K scale give conservative upper bounds0.0003386K/0.0001672K
on that same coordinate's new absolute error. These are bounds from the
ranked report, not exact newly recorded errors at that coordinate.

Shallow soil is not accurately fitted yet: row819 ptn[0,0,13] predicts
303.55375K versus300.30938K, an error3.24436K. This is a fresh initialization,
not a trained prediction. The paired result confirms removal of the artificial
80K deep-temperature error without a loss-weight reduction or capacity change.
Carbon flux and stock group values are exactly unchanged in these two forwards;
the huge aggregate reduction is therefore not a carbon-accuracy improvement.

## Remaining issue and next work

After removing thermal domination, the four fuel pools, dead_leaves and cimean
account for99.981%/99.985% of the remaining state anchor. Example row819:
fuel_1hr176.0286 versus3.04983 (TRAIN daily-change scale0.027615),
dead_leaves138.2852 versus0.0406467 (scale0.0235181), in their native units.
This is both a large endpoint error and amplification by slow-change scales.
The observation alone cannot determine whether another output-coordinate
mismatch, learned fresh-state error, or state-supervision ownership is responsible.

Before committing to learning, make a bounded source/statistics check of these
remaining outputs and their daily-change representation together, rather than
repeatedly changing one field or lowering weights until a scalar looks small.
Keep the shared architecture and full capacity. Resolve this check before
deciding whether to compile v2 as-is or incorporate a justified coordinate fix.
The next numerical gate remains effective-model native reverse/JVP plus two
updates, then joint T7/T30, then a bounded learning contract. The old-transform
gradient pass does not certify the revised transform. No paid launch is ready.

## Evidence

Local JSON/log prefixes under `outputs/`:
`shared-recovery-8004a11-gpu` and `shared-thermal-6467453-forward`.
Server counterparts under `runtime/outputs/diagnostics/` with identical names.
The source of the thermal change is documented in
`shared_thermal_output_audit_20260921.md` and its plan in
`shared_thermal_output_plan_20260921.md`.
