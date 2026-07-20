# Routing Flow Sequence Ledger

Scope: source-order wiring ledger for `routing.f90::routing_flow` lines
2944-5369. This is not a numerical long-run validation result; it records which
source blocks have local JAX kernels and what remains before the full daily
routing-flow wrapper can be claimed closed.

## Source Order

| Source block | Fortran owner | JAX helper | Status |
| --- | --- | --- | --- |
| Initial area/flood sums, stream climatology partition, fast/slow/stream outflows | `routing_flow` lines 3180-3388 | `routing_reservoir_outflow_step` | Local kernel covered |
| Stream sediment/POC erosion, deposition, re-erosion, bank erosion | `routing_flow` lines 3390-3539 | `routing_stream_erosion_step` | Local kernel covered |
| Flood/pond vertical water balance and basin DOC/CO2 input distribution | `routing_flow` lines 3558-3681 | `routing_flood_pond_input_step` | Local kernel covered |
| Floodplain drainage/lateral outflow and particulate deposition | `routing_flow` lines 3686-3915 | `routing_floodplain_flux_step` | Local kernel covered |
| Pond drainage, pond inflow, POC/sediment trapping | `routing_flow` lines 3924-3993 | `routing_pond_flux_step` | Local kernel covered |
| Basin-to-basin transport accumulation | `routing_flow` lines 4003-4061 | `routing_transport_between_basins_step` | Local kernel covered |
| Swamp return and floodplain flooding | `routing_flow` lines 4074-4206 | `routing_swamp_flood_step` | Local kernel covered |
| Reservoir writeback and negative-water correction | `routing_flow` lines 4208-4372 | `routing_reservoir_update_step` | Local kernel covered |
| Stream/flood/pond fractions and river-bed transfer limit | `routing_flow` lines 4374-4500 | `routing_area_fractions_step` | Local kernel covered |
| DOC/POC/CO2 chemistry and pCO2/FCO2 diagnostics | `routing_flow` lines 4502-4929; `poc_decomposition` lines 11688-11737 | `routing_co2_chemistry_step`, `routing_poc_decomposition` | Local kernel covered |
| Returnflow and reinfiltration aggregation | `routing_flow` lines 4940-4970 | `routing_return_reinfiltration_step` | Local kernel covered |
| Irrigation withdrawal/adduction | `routing_flow` lines 4975-5205 | `routing_irrigation_step` | Local kernel covered |
| Netflow diagnostics, reservoir diagnostics, outlet aggregation | `routing_flow` lines 5219-5329 | `routing_flow_diagnostics_step` | Local kernel covered |
| Lake overflow cap and coastal redistribution | `routing_flow` lines 5331-5366 | `routing_lake_overflow_step` | Local kernel covered |

## Wiring Boundary

`routing_flow_step` now passes explicit state through the audited helpers above
in source order for the conservative disabled-surface-switch micro-case. It
also converts Fortran 1-based `route_togrid`/`route_tobasin` values, including
`nbasmax+1:nbasmax+3` outlet slots, only at the local transport boundary.
`routing_daily_boundary_step` now covers the surrounding `routing_main` daily
gate: accumulator due check, source-order `routing_flow`, `routing_lake`,
lake-return addition, accumulator reset, and daily-to-SECHIBA timestep scaling.

Do not claim full multi-landpoint/global routing-flow parity yet. Wrapper-level
micro-cases now cover the disabled-surface outlet path, active floodplain/pond
composition, lake overflow/coastal redistribution, irrigation demand using the
Fortran `precip` input rather than `flood_inp`, new-flood-scheme swamp and
floodplain routing, active stream DOC/CO2 chemistry, neighbor irrigation
adduction, river-balance diagnostics, broader combined active-state writeback
interactions through the daily boundary, and Fortran 1-based route guards. The
remaining routing blocker is coupling the wrapper to complete production
routing map assets. The local kernels and wrapper-level branch compositions
above are covered separately; the open item is real production asset wiring.
