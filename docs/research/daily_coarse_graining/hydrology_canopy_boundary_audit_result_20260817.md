# Hydrology And Canopy Boundary Audit Result

Date: 2026-08-17.

Status: completed non-test source/representation audit. The architecture
consequence below was initially a proposal for user discussion and was
subsequently approved and implemented under the v2 local acceptance record.

## Scope

This audit asks which dominant HYDROL endpoints can be evaluated once from the
daily boundary and whether the current canopy storage/debt transfer coordinate
matches the Teacher state.

The panel contains 20 accepted v6 point-years: ten landpoints, each in 1961
and 1991. Sixteen point-years are train/train and four are spatial validation.
No spatial-test or temporal-test shard was read. The audit did not regenerate
Teacher data, read v5, materialize forcing records, or execute a subdaily state
scan. It ran on the free Explore1000 `test04` CPU node.

The generated report is
`outputs/research/daily-coarse-graining/hydrology_canopy_boundary_audit_20py_20260817_v2.json`.
It covers 7,290 daily endpoints.

## Hydrology Result

The audit calls the existing source-backed HYDROL diagnostics once at the
daily endpoint. It uses final `mc/mcl`, vegetation and soil-tile state, paper
static fields, and either the true final `nroot` or day-start `nroot`.

| Reconstruction | Passing point-years | Maximum absolute error | Panel RMSE | Decision |
| --- | ---: | ---: | ---: | --- |
| final `nroot` from day-start `nroot` plus final moisture | 5/20 | 0.997979 | 0.00789318 | not exact |
| `us` from the attempted one-boundary root update | 5/20 | 0.410670 | 0.00127470 | not exact |
| `humrelv` from the attempted one-boundary root update | 5/20 | 1.0 | 0.0104932 | not exact |
| `humrel` from the attempted one-boundary root update | 5/20 | 1.0 | 0.0257029 | not exact |
| `us` from true final `nroot` and final moisture | 20/20 | 2.22e-16 | 4.36e-19 | exact derived |
| `humrelv` from true final `nroot` and final moisture | 20/20 | 2.22e-16 | 1.62e-18 | exact derived |
| `humrel` from true final `nroot` and final moisture | 20/20 | 2.22e-16 | 3.96e-18 | exact derived |
| grid `evap_bare_lim` from final `evap_bare_lim_ns` | 20/20 | 0.0 | 0.0 | exact derived |
| `temp_hydro` from final `stempdiag` | 0/20 | 9.88667 K | 0.665490 K | learned primitive |

The failed `nroot` reconstruction confirms a real history dependency. The
dynamic-root source update only operates where the current root weight is
nonzero. A layer that became zero earlier in the day cannot be recovered from
final moisture alone. A single final-moisture call therefore cannot replace
the 48 source updates.

The complementary result is stronger and directly useful: once final `nroot`
is supplied, final `us`, `humrelv`, and `humrel` are exact one-boundary source
diagnostics, including the 101 audited under-`mcr` tile events. The neural
boundary should learn final `nroot`, not all four mutually dependent fields.
It should also learn tile-level `evap_bare_lim_ns` and reconstruct the grid
aggregate exactly. `temp_hydro` remains learned because final `stempdiag` is
not the thermal state that entered the last half-hour HYDROL call.

This removes four fields and 147 scalar values per daily sample from the
independent neural output boundary: `us` (132), `humrelv` (12), `humrel` (2),
and `evap_bare_lim` (1). The complete public boundary remains unchanged.

## Canopy Result

The source allows signed `qsintveg`. Positive values are canopy water storage;
negative values are evaporation debt. The final positive storage obeys its
source capacity in all 7,290 endpoints: there are zero storage-over-capacity
and zero positive-storage-at-zero-capacity cases.

The final endpoint has three materially populated regimes:

- positive storage occurs on 3,771/7,290 endpoints (`51.73%`);
- the endpoint is zero on 493/7,290 endpoints (`6.76%`);
- final debt is positive on 3,026/7,290 endpoints (`41.51%`);
- day-start debt is positive on `41.44%` of endpoints;
- debt increases during the day on `20.10%` of endpoints;
- positive final debt has median `0.0007717` and maximum `0.0397241 kg m-2`.

The captured daily `water.vevapwet` label is a signed daily sum. It includes
dew/condensation and is nonpositive on 406 positive-debt endpoints. Debt
increase exceeds that signed daily sum in 104 endpoints. It is therefore not
a valid hard scale for either final debt or debt incoming. A gross positive
wet-canopy evaporation scale would require information that v6 does not store
and final inference is forbidden to reconstruct through a 48-step scan.

The present four-head storage/debt outgoing/incoming representation is poorly
conditioned for endpoint fitting. Multiple transfer combinations produce the
same final signed state, while the scientific target of interest is the final
`qsintveg` endpoint.

## Architecture Consequence For Discussion

The proposed revision retains the 49.7-million-parameter axis-aligned
process-Attention backbone. It changes only the hydrology/canopy output
coordinate and the exact daily-boundary composer.

1. The hydrology expert learns final `nroot`, `evap_bare_lim_ns`,
   `temp_hydro`, `ae_ns`, and the remaining genuine primitives. Final `nroot`
   uses a support-aware normalized layer representation so root weights remain
   nonnegative, sum to one for an active PFT, and cannot activate a layer that
   was zero at day start.
2. After the conservative primitive state update, one source-backed HYDROL
   boundary call reconstructs `us`, `humrelv`, and `humrel` from predicted
   final `nroot/mc/mcl`. A deterministic tile aggregation reconstructs
   `evap_bare_lim` from predicted `evap_bare_lim_ns`.
3. The canopy head predicts the final signed `qsintveg` endpoint directly:
   positive storage is represented as occupancy in the physical capacity;
   negative debt uses a separately normalized nonnegative magnitude and an
   explicit storage/zero/debt status loss. Signed daily `vevapwet` is not used
   as a hard debt bound.
4. The predicted endpoint is converted deterministically to the conservative
   updater's storage and debt transfers. For each nonnegative component, the
   converter removes the necessary part of the day-start component and adds
   only the positive remainder needed to reach the predicted endpoint. This
   removes cancellation and preserves the existing budget/state boundary.

No weak matched control is required. The revised model must first pass the
absolute local forward/loss/gradient, one-day state/budget, and one-point-year
memorization gates. Only after those pass may the 64-point-year, complete-v6,
7/30-day, 365-day, Gate D2, sealed-test, or product stages reopen.

The user subsequently reviewed and approved this output mechanism. Its
implementation and current gate are recorded in
`hydrology_canopy_output_v2_local_acceptance_20260817.md`.
