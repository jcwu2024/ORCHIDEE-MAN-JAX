# Warm snow-free boundary: bounded repair contract

Status: implementation and zero-training admission in progress. No model promotion.

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
2. Free-CPU existing-label panel: prepared artifacts 1, 322 and 442, both
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
