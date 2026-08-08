# Long-Horizon Biomass And Productivity Objective v1

Status: **objective, metrics, and source ledger frozen; successor architecture and loss coefficients not selected**.

Date: 2026-08-08.

## Decision

The next neural candidate must be judged by the complete daily operator, not
only by its direct process-head loss. The accepted trajectory boundary is:

```text
named process-head outputs
  -> conservative fast update
  -> exact retained daily processes
  -> daily GPP/NPP/respiration and next-day carbon stocks
  -> 1/7/30-day cumulative and terminal objectives
```

The implementation is
`daily_operator/long_horizon_objective.py`; its machine contract is
`manifests/coarse_graining/long_horizon_biomass_productivity_objective_v1.json`.
It is architecture-independent and does not reconstruct 48 forcing records or
run a 48-step state scan.

The existing equal-label one-day process loss remains a separate required
anchor. Long-horizon terms cannot replace it. Final coefficients relative to
that anchor are deliberately unset: they must be frozen before training after
one bounded train/train gradient-magnitude preflight, not chosen after seeing
rollout results.

## Frozen Objectives

At horizons `H = 1, 7, 30`:

- cumulative GPP, NPP, maintenance respiration, growth respiration, and
  heterotrophic respiration use complete-window masks;
- the cumulative residual is divided by `H * frozen_daily_scale`, so a
  30-day amount does not dominate merely because its units are 30 times larger;
- terminal AGB, BGB, total live biomass, litter, particulate soil carbon, DOC,
  and ecosystem carbon use the full operator state after day `H`;
- normalized MSE and the square of signed mean bias are separate components;
- fields are macro-averaged equally, then the three horizons are averaged
  equally;
- negative declared stocks, nonfinite defined values, mask/discrete mismatch,
  and water/carbon budget failures remain zero-tolerance hard counts.

This directly addresses the earlier mixed-horizon failure: a lower aggregate
state RMSE cannot conceal biased NPP, growth respiration, or a small but
persistent daily carbon tendency.

## Carbon Stock Definition

The frozen v6 data do not contain scalar `AGB` or `BGB` labels. They contain
the full 12-part biomass carbon state per PFT. The operational split follows
the exact above/below routing used by Fortran `littercalc_leak` and the accepted
JAX `littercalc_litter_increments`:

| Stock | Included plant parts | Area conversion |
| --- | --- | --- |
| AGB | leaf, sapwood above, heartwood above, fruit, reserve, stilt/pneumatophore AGR sapwood and heartwood | multiply each PFT by its source `pft_fraction`, then sum PFTs |
| BGB | sapwood below, heartwood below, root | same |
| Total live biomass | all 12 carbon parts | exactly `AGB + BGB` |

Reserve is reported separately as a diagnostic but remains inside operational
AGB because the source routes reserve loss to above litter. This is an
explicit source convention, not a general ecological claim about reserve
location.

`deepC_peat` is never added to ecosystem carbon: Fortran derives it from
`carbon_32l`, so adding both would double-count soil carbon.

## Source And v6 Ledger

| Quantity/process | Fortran owner | Coherent v6 coverage | Neural/retained ownership | Long-horizon use or gap |
| --- | --- | --- | --- | --- |
| GPP | `stomate.f90::stomate_main:3198-3208` | `daily_interface.gpp_daily`, PFT axis | direct fast process output | cumulative 1/7/30-day flux and signed bias |
| Maintenance respiration | `stomate_resp.f90::maint_respiration:122-389`; `stomate_npp.f90::npp_calc:314-380` | direct `resp_maint_part`; total `resp_maint` in next state | direct part-resolved fast output consumed by exact `npp_calc` | cumulative total plus direct-label one-day anchor |
| Allocation | `stomate_alloc.f90::alloc:146-834`; `stomate_npp.f90::npp_calc:393-491` | endpoints and retained outputs, no independent `bm_alloc` label | retained exact | do not infer allocation from endpoints; concurrent respiration/losses make it non-identifiable |
| Growth respiration and NPP | `stomate_npp.f90::npp_calc:455-530`; call at `stomate_lpj.f90:1118-1131` | `resp_growth` and `npp_daily` in next state | retained exact result of predicted GPP/maintenance plus source allocation | cumulative 1/7/30-day flux and signed bias |
| Mortality | `lpj_gap.f90::gap:117-364`; call at `stomate_lpj.f90:1234-1243` | effects in biomass and `bm_to_litter`; no admitted standalone neural label | retained exact | terminal biomass/litter; future learned mortality needs a new capture contract |
| Turnover | `stomate_turnover.f90::turn:169-947`; call at `stomate_lpj.f90:1260-1268` | `turnover_daily` and biomass state | retained exact | terminal AGB/BGB and litter input |
| Litter input/stocks | `stomate_litter.f90::littercalc_leak:1637-2873` | exact input derivation plus litter endpoints and typed respiration/transfers | exact derivation plus constrained transfer outputs | cumulative heterotrophic respiration; terminal litter |
| Soil POC/DOC | `stomate_soilcarbon.f90::soilcarbon_leak:641-2416` | typed decomposition/export/transport plus `carbon_32l` and DOC endpoints | constrained process outputs and exact derivations | terminal POC/DOC and ecosystem carbon |
| Carbon closure | `stomate_lpj.f90:1872`; litter/soil internal checks | accepted Gate C combined carbon balance | conservative updater and retained tail | every day hard gate; 7/30-day accumulated residual reported |

## Remaining Architecture Decision

This work does not select a successor network. The architecture discussion
must decide which source-backed transfers need learned capacity, how fast
productivity and slow carbon compartments exchange context, and the explicit
coefficients joining the one-day anchor to the four implemented long-horizon
components. No GPU run, 365-day evaluation, Gate D2, or product gate is opened
by this contract alone.
