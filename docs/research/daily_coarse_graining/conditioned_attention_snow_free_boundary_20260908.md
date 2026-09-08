# Warm snow-free boundary: bounded repair contract

Status: bounded forward consistency screen passes; science accuracy is mixed
with two material regressions. No model or training-gradient promotion.

## Mechanism and scope

The thermal-source audit found growing snow heat on two Teacher snow-free
windows. Keep the current conditioned Attention parameters and architecture.
Add a default-off process-head boundary, before the conservative updater:

- Day-start land snow mass, layer depth/liquid, and nobio snow are exactly zero;
  nobio surface fraction is zero.
- Every active current-day native Snowf record is zero (not just its sum).
- Every valid native Tair record, including the interpolation predecessor, is
  strictly above 273.15 K. This sufficient condition excludes freezing dew.
- Snow thermal state is finite and can reach the no-snow reference within the
  existing 1e9 J/m2 heat and 100 K temperature tendency bounds.

For this branch only, supply zero snow inventory transfers, snow phase and
sublimation fluxes, depth, grain and ages; density is 50 kg/m3. Thermal
tendencies reset heat to zero and temperature to 273.15 K. The existing updater
and fast-owner materializer propagate canonical endpoints and their mirrors.
No Teacher label, future state, reconstructed half-hour forcing, or subdaily
scan enters inference. Rain remains allowed. All other regimes use the
unchanged learned outputs. The selector is discrete; no derivative is claimed
across its freezing or zero-snow regime boundaries.

## Source proof

All references are under `fortran_source/ORCHIDEE/src_sechiba/`:

- `condveg.f90::condveg_frac_snow:872-890`: zero depth gives zero snow fraction.
- `diffuco.f90::diffuco_snow:1216-1275`: zero snow fraction and zero nobio
  fraction give zero sublimation beta.
- `diffuco.f90::diffuco_comb:3135-3183`: freezing condensation can override
  beta to one even without existing snow. Strictly warm Tair excludes it;
  warm dew instead sets snow beta to zero.
- `enerbil.f90::enerbil_evapveg:1760`: vevapsno contains snow beta as a factor.
- `explicitsnow.f90::explicitsnow_fall:1198-1266`: zero snowfall adds no depth.
- `explicitsnow.f90::explicitsnow_main:222-235,259-315,395-402,459-500`:
  no depth resets temperature/liquid, zero sublimation does not add mass,
  enthalpy is recomputed, and zero snow resets ages/density/grain/depth/liquid.
- `explicitsnow.f90::explicitsnow_profile:1681-1690` and
  `explicitsnow_melt_refrz:1441-1558`: zero depth/mass skips snow heating and
  phase changes. Rain does not create land snow.

The native forcing is linearly interpolated for temperature; requiring both
endpoints above freezing suffices without building the 48 records. Current-day
Snowf uses its current native interval; the predecessor is not precipitation.
This is a PFT14 explicit-snow/no-nobio result, not an all-configuration claim.

## Matched comparison and bounded gates

Reference and repair use the same accepted update-3 weights, prepared v6
artifacts and science losses; no optimizer update or checkpoint retention.
The reference is a causal comparison, not an acceptable scientific baseline.

1. Local tests: warm/rain, snowpack, snowfall, freezing/dew possibility,
   predecessor/padding, and nonzero nobio cases. Outside the branch, outputs
   must be identical. Inside it, canonical state/mirrors and budgets must
   agree, and local forward/reverse/finite-difference thermal derivatives must
   agree away from regime boundaries.
2. Free-CPU existing-label panel: prepared artifacts 251, 322 and 442, both
   7- and 30-day inputs. Count coverage and compare selected Teacher endpoints;
   disclose any absent active-snow coverage. Do not regenerate labels.
3. One free-GPU, at most 45-minute zero-training run: artifact 322 windows 5/1
   at 7 days, and artifact 442 window 0 at 30 days. Compare primary science
   losses, hard failures, snow trajectories and production/recorded scores.
   Source-selected heat/depth/liquid must stay at reference (thermal roundoff
   tolerance 1e-6 J/m2 / 1e-10 K); every physical hard failure must remain zero.
   Recorded and production score tolerance remains atol=1e-6, rtol=1e-5.

A passed forward screen permits the existing streamed/JVP/FD gradient gate
and broader 40-artifact integration; it does not by itself authorize resumed
paid training, fixed-monitor promotion, 365-day acceptance or Gate D2.

## Implementation and current evidence

Implementation commit: `e0fd6e3`; prepared-panel path correction: `c882529`.
The canonical Teacher is unchanged. `MechanisticStrongBaselineProcessHead`
has a default-off `warm_snow_free_boundary` option inherited by the current
conditioned Attention head. The correction changes 22 learned output fields,
adds no parameters, and keeps the existing complete output schema.

Local admission: 10 tests pass in 44.20 seconds, covering the regime selector,
canonical binding/materialized mirrors, water/carbon budgets, and thermal
forward/reverse/finite-difference checks. Ruff and `git diff --check` pass.
The 18 related mechanistic-output/thermal-source tests also pass (259.34 s).
The real-binding fixture needed the existing valid soil/static fixture values;
those fixture errors were fixed locally before any GPU execution.

Free `test01` label read completed at `c882529` in seconds. Artifacts 251,
322 and 442 contain 24 seven-day windows and 12 thirty-day windows: 528 day
positions (not claimed to be unique). Every position is selected. All nine
checked Teacher snow endpoint families match the source reference exactly,
with zero undefined/nonfinite values. There is **no real active-snow coverage**
in this panel; only synthetic unchanged-output evidence covers those regimes.
Artifact 1 was initially named but is absent from this prepared manifest;
the bounded panel was corrected to available artifact 251 before the GPU run.
Report: `runtime/outputs/diagnostics/conditioned-snow-free-labels-c882529.json`.

The first screen at `c882529` finished the reference 322/5 computation, then
stopped at 278.52 seconds when its report reducer received `HardCounts`
instead of a host mapping. The partial JSON preserves that case; it does not
contain repair-arm evidence. The reducer is corrected and a complete
screen/report-loop test using the real `HardCounts` type passes at `4d85256`.
The screen also records all relevant HYDROL snow endpoints and their
DIFFUCO/ENERBIL mirrors. The paired forward screen completed on free
`gln01`, GPU 1, at `4d85256` in 1347.72 seconds (22.46 minutes), within the
45-minute timeout, reusing compilation caches. Report/log stem:
`runtime/outputs/diagnostics/conditioned-snow-free-rollout-4d85256`.
No Slurm/paid job, new Teacher labels, training update, or checkpoint was made.

## Paired result and interpretation

All six reference/repair cases completed; `all_repair_forward_gates_pass=true`.
The repaired paths select all 7/7/30 days, and all 17 checked snow owners and
mirrors match their source reference exactly. Both arms have zero physical
hard failures under the existing policy. Declared dynamic-state mask changes
are recorded separately, not hidden or reclassified: repair counts are 7,
0 and 29 respectively; unexpected mask changes are zero.

The following is the unchanged, normalized science composite, **not** a
percentage error in biomass or productivity. Lower is better.

| Train window | Reference | Repair | Change | Recorded minus production, repair |
| --- | ---: | ---: | ---: | ---: |
| 322/5, 7 days | 0.160806598 | 0.156316134 | -2.79% | +5.78e-8 |
| 322/1, 7 days | 0.098416067 | 0.333321197 | +238.69% | -1.94e-8 |
| 442/0, 30 days | 0.128476724 | 0.248246001 | +93.22% | +1.79e-9 |

Every repaired production/recorded score passes the frozen comparison
tolerance. Reference 322/5 fails (difference -6.08975e-5). Reference 322/1
and 442/0 pass scoring consistency but fail the source snow-state check.

GPP normalized MSE changes from 0.988537 to 0.962295, 0.570830 to 1.926283,
and 0.760815 to 1.470874 respectively. Growth respiration and biomass errors
also rise in the two regressed windows; nonliving carbon error falls in all
three. Heterotrophic error is zero in this bounded panel and supplies no
evidence of learned improvement. Raw primary trajectories and masks remain
in the JSON for later inspection.

This is evidence that the source correction removes an unphysical recurrent
state and the observed scoring instability on these windows. It is **not**
evidence of general gradient correctness or a useful replacement model.
Existing weights do not preserve their former science scores under the
corrected transition. Compensation for erroneous states is a possible
explanation, not an attribution established by this screen. Do not reject
the source rule solely to recover a weak model's former score, and do not
claim that retraining is guaranteed to repair the accuracy regression.

## Exact next operation

Keep the accepted recovery at update 3 and this option default off. Before
any new optimizer update, bind the corrected head on these same three
windows and compare full streamed one-day-VJP BPTT with JVP and finite
differences on the existing declared parameter directions and epsilon ladder.
The local thermal-only AD test does not substitute for that network test.
Do not compile a whole-30-day no-rematerialization reverse graph again.

Rebuild **all** one-day, direct/state and horizon gradient functions with the
corrected head. This screen's `with_boundary` helper rebuilds forward steps
only; its inherited gradient helpers must not be used for corrected training.
If derivative agreement passes, run the bounded train-only integration and
discarded-step descent check with corrected forward losses as the starting
point, while reporting actual GPP/respiration/biomass errors. The scientific
goal remains the absolute long-horizon targets, not outperforming the old
incorrect-snow checkpoint. Any retained continuation requires the corrected
boundary option in its experiment identity and the existing science gates.
No new architecture, v6 regeneration, paid training, monitor evaluation or
product promotion is authorized by this forward-screen result alone.
