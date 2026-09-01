# Target-Oriented GPP State Reachability V1

Date: 2026-09-01

Status: frozen zero-training dependency audit; execution pending

## Question

The coupled candidate learns only `daily.fast_gpp`. Its one-day state anchor
must therefore supervise only next-day slow-process coordinates that GPP can
actually change through the complete daily operator. Penalizing all 2,386
input-state coordinates would dilute the loss with states outside the learned
GPP boundary.

## Frozen Panel

The audit uses existing train/train v6 data from landpoint `001.0-071.0`.
Twelve ordinary 1961 cases cover Day 2 and one target-blind mid-month state
from each remaining noleap month. A thirteenth case covers the accepted 1962
restart Day 1 boundary. Selection uses only year and calendar position, never
target values.

## Differentiated Boundary

For each case, run the coupled candidate once and freeze every learned output
except `daily.fast_gpp`. Treat the compact GPP array as an independent
`(1, 2)` input: bare soil plus PFT14. Propagate it through the existing
mechanistic reconstruction, conservative daily update, post-update exact
materialization, and canonical retained exact STOMATE tail.

Compute the complete float64 forward-mode Jacobian of all 2,386 next-day
`slowproc_stomate_previous_step_state` coordinates with respect to both GPP
slots. The bare-soil column must be exactly zero. A coordinate is reachable
when the PFT14 derivative is finite and exactly nonzero in at least one of the
13 cases.

The exact nonzero union controls the inventory. A scale-normalized sensitivity
is reported using the existing train/train GPP scale and amended next-state
scale, but its `1e-12` reporting threshold does not remove a valid dependency.

## Output

The machine report must retain, for every reachable source field:

- its Markov state slice, shape, and physical axes;
- reachable flat indices within the field and global continuous-state indices;
- per-case reach counts;
- maximum absolute and normalized sensitivity.

The future one-day loss will average within each active source field and then
across fields. Exact coefficients are not authorized by this audit and remain
part of the following objective-calibration contract.

## Gates

Every Jacobian value and declared next-state value must be finite. The PFT14
column must reach at least one coordinate in every case, the union must be
nonempty, every coordinate must have one Markov owner, and all per-case water
and carbon budget residuals must remain within `2e-12`.

The audit performs zero optimizer updates, creates no optimizer, writes no
checkpoint, and does not regenerate v6, use v5, inspect fresh or sealed data,
reconstruct half-hour forcing, or execute a subdaily state scan.

The machine contract is
`manifests/coarse_graining/target_oriented_gpp_state_reachability_v1.json`,
canonical SHA256
`1bf2bbfc13c2b6b4b04d323b1a95917e3b33d501dd249f7b2fd36084307c82af`.
