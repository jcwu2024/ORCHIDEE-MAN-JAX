# Target-Oriented GPP State Reachability Result (2026-09-01)

Status: accepted zero-training dependency evidence.

## Provenance

- Explore1000 job: `14899147`
- Git commit: `b730b99e8b8c7794b0b8c27f7a24d06734c92af4`
- Reachability contract SHA256:
  `1bf2bbfc13c2b6b4b04d323b1a95917e3b33d501dd249f7b2fd36084307c82af`
- Predecessor real-v6 admission report SHA256:
  `cf8eb1d2ba66c70a2d8580caef32647d59765f863c0043a749761a9148d7e623`
- Result report SHA256:
  `dc53765e84f14b0d08e7d9d6bafbd0527a36349f57f012dac1cf8d5e1a36d020`
- Runtime: 18 minutes 30 seconds on one V100.
- Peak live JAX device allocation: 1,269,505,280 bytes.

The report is stored outside Git at
`runtime/outputs/diagnostics/target-oriented-gpp-state-reachability-14899147.json`
on Explore1000.

## Executed Boundary

The audit used 12 target-blind calendar positions from the 1961 train/train
point-year and the accepted 1962 restart Day 1 boundary. For each case it
computed the float64 `(2386, 2)` forward-mode Jacobian from compact
bare-soil/PFT14 `daily.fast_gpp` to next-day
`slowproc_stomate_previous_step_state` after the canonical conservative update
and retained exact tail.

It performed zero optimizer updates, created no optimizer, wrote no
checkpoint, regenerated no Teacher data, used no v5 data, read no fresh or
sealed target, reconstructed no 48-record forcing, and ran no 48-step state
scan.

## Acceptance Result

The audit passed all frozen gates:

- all 13 Jacobians are finite;
- the bare-soil derivative column is exactly zero in every case;
- each PFT14 case reaches 37 to 41 coordinates and 14 to 16 fields;
- the 13-case exact-nonzero union contains 41 coordinates in 16 fields;
- 39 coordinates exceed the `1e-12` normalized reporting threshold, while two
  additional exact-nonzero coordinates remain in the authoritative union;
- all declared next-state values are finite; and
- water and carbon budget residuals are exactly zero in every case.

The maximum absolute PFT14 derivative is `10.382934536186148`, and the maximum
normalized sensitivity is `1.5311164716805354`.

## Reachable Field Inventory

| Markov field | Union coordinates | Maximum normalized sensitivity |
| --- | ---: | ---: |
| `age` | 1 | `4.18596e-7` |
| `assim_param` | 1 | `2.16429e-4` |
| `biomass` | 11 | `1.59333e-2` |
| `bm_to_litter` | 7 | `3.31542e-3` |
| `carb_mass_total` | 1 | `7.48700e-8` |
| `frac_age` | 4 | `3.06919e-2` |
| `gpp_week` | 1 | `1.27359e-1` |
| `lai` | 1 | `1.01967e-2` |
| `leaf_age` | 4 | `2.44496e-3` |
| `leaf_frac` | 4 | `3.06919e-2` |
| `maxgppweek_thisyear` | 1 | `1.17310e-1` |
| `npp_daily` | 1 | `1.53112` |
| `resp_growth` | 1 | `1.21355` |
| `tot_bare_soil` | 1 | `2.68187e-2` |
| `turnover_daily` | 1 | `1.48652e-2` |
| `veget` | 1 | `2.68187e-2` |

This is a local first-order dependency inventory, not evidence that the other
slow-state coordinates can never change after longer nonlinear rollouts. It
does establish which next-day state coordinates can carry an infinitesimal GPP
error across the accepted one-day operator at the audited boundaries.

## Decision

The GPP-to-next-day-state path is sufficiently broad to justify explicit
one-day state supervision, but the supervision must not flatten all 2,386
carbon coordinates into one undifferentiated loss. The next gate is to freeze a
field-grouped one-day state-anchor and objective-calibration contract over the
41 exact reachable coordinates. The contract must be train-only and must be
accepted locally before any optimizer update. Training, fresh-target tuning,
365-day rollout, Gate D2, sealed test, and product acceptance remain closed.
