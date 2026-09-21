# Shared `cimean` source audit (2026-09-21)

This is a bounded, read-only source audit at repository HEAD `8e17b48`.
The codebase-memory MCP transport was unavailable, so the evidence below is
from direct `rg`/source reads. No code, checkpoint, Teacher shard, or runtime
job was changed or started by this audit.

## Decision

`cimean` is not a generic learnable value on the bare-soil coordinate. In the
Fortran paper layout, compact slot 0 is source PFT1 (bare soil), and its exact
DIFFUCO value is `Ca` (canopy/ambient CO2 concentration). It is not a zero
value and must not be replaced with a vegetation-trained mean, a pooled
bare+vegetation statistic, or a Teacher-derived endpoint shortcut.

The compact PFT14 coordinate remains a genuine photosynthesis output and must
remain learnable. Do not freeze the whole `cimean` head or remove its PFT14
parameter gradients. The minimum rigorous route is a trait-conditioned
source-owned bare-soil identity at materialization (or an equivalent exact
source owner before normalization/state reconstruction), while retaining the
shared network and the PFT14 loss/gradient path.

## Fortran provenance and branch semantics

The source declaration calls `cimean` mean leaf/intercellular CO2, in ppm:

- `fortran_source/ORCHIDEE/src_sechiba/diffuco.f90:391-405`,
  `diffuco_main` declares `cimean(kjpindex,nvm)` as an output.
- `diffuco.f90:665-678` dispatches to `diffuco_trans_co2` when `ok_co2` is
  enabled; the non-CO2 branch also owns the output.
- `diffuco.f90:2284-2292` initializes all PFT outputs and explicitly assigns
  `cimean(:,1)=Ca(:)`. This is the source PFT1/bare-soil column.
- `diffuco.f90:2299-2328` loops only over `jv=2..nvm`, initializes each
  vegetation column with `cimean(:,jv)=Ca(:)`, and then enters the photosynthesis
  mask.
- `diffuco.f90:2341-2347` defines the mask using PFT-specific LAI threshold and
  `veget_max > min_sechiba`. Points outside that mask never overwrite the `Ca`
  initialization.
- `diffuco.f90:2884-2895` writes `cimean` only for the `nia`/assimilation points:
  the calculated mean-Ci formula is used when the conductance denominator is
  above `min_sechiba`; otherwise `gamma_star` is used. Thus `Ca` is the exact
  fallback for inactive vegetation points as well, but active vegetation is not
  an identity.

The source does not define a bare-soil stomatal/photosynthetic `cimean` target.
The PFT1 identity is therefore semantic (source PFT id/role), not an arbitrary
compact array position. The paper catalog confirms this at
`configs/pft_catalogs/orchidee_man_paper_250919.json` (PFT1 role
`bare_soil`; PFT14 role `vegetated`), and the daily contract selects source
indices `[0,13]` (`research/daily_coarse_graining/daily_markov_contract.py:1132-1135`).

## Current JAX and shared-operator behavior

The direct JAX DIFFUCO owner already preserves the Fortran branch:

- `jax_orchidee/sechiba/diffuco.py:2173-2179` initializes the PFT background
  `cimean` to `ca[:,None]`.
- `diffuco.py:2979-2989` records the same source provenance.
- `diffuco.py:3051-3058` computes active `cimean` and applies
  `jnp.where(assimilate, cimean_active, ca[:,None])`.
- `diffuco.py:3504-3525`/`3556-3563` inserts the locally computed PFT column
  into the full PFT array and carries `cimean` as a named output.
- `jax_orchidee/driver/orchestration.py:2870-2879` copies the owner result
  into `diffuco_previous_step_state` alongside `gpp`, `gsmean`, `rveget`, and
  `rstruct`.

The shared successor contract currently treats
`endpoint.fast_state.diffuco_previous_step_state.cimean` as an
`independent_neural_endpoint`, finite signed output of compact shape `1 x 2`:
`manifests/coarse_graining/successor_fast_owner_reconstruction_v1.json:563-594`.
Its decoder uses `normalized_day_end_minus_day_start_owner` and the day-start
slice from the continuous state (`.../successor_fast_owner_decoder.py:120-142`,
`.../output_transform.py:54-66,93-95`). The endpoint is therefore physically
decoded as `day_start + mean + scale * raw`; the scale is not omitted.

The current applicability is too weak for this field. In
`successor_fast_owner_training.py:67-76,96-117`, the target mask is only
`teacher_defined & inputs.pft.active_pft_mask`; it does not inspect the
`role_bare_soil` trait. The contract's compact active layout makes both PFT1
and PFT14 eligible. The current shared output path consequently allows the
bare slot to be predicted and pooled with vegetation.

The saved fresh-forward evidence is consistent with this diagnosis, not with
a missing scale: `outputs/shared-thermal-6467453-forward.json` records the
two cimean predictions `596.9668588816351` and `598.1166298041667` against
Teacher `317.27` and `318.57`, with state-delta scale
`0.07627253973421866`. The reported fast TRAIN pooled delta statistics
(mean `0.0010743508`, scale `382.810705`, copied to both compact slots) mix
the exact bare identity with variable vegetation Ci. The full TRAIN state
statistics similarly report cimean scales approximately
`0.0762725397` (bare) and `541.3760855` (PFT14); this is a roughly
`5018.98x` scale ratio. These are diagnostics only; they do not authorize
changing the accepted 75.58M architecture or replacing statistics by fixed
array-slot rules.

## Can `Ca` be obtained legally at day start and day end?

Yes, for the current paper protocol, without Teacher endpoint leakage.

Fortran input provenance is explicit:

- `fortran_source/ORCHIDEE/src_driver/dim2_driver.f90:708-719` reads
  `ATM_CO2` and initializes `for_ccanopy(:,:)=atmco2`.
- `dim2_driver.f90:1007-1010` refreshes the same annual `atmco2` value before
  each forcing call; `dim2_driver.f90:1111-1117` passes `for_ccanopy` through
  `intersurf_initialize_2d`.
- `fortran_source/ORCHIDEE/src_sechiba/intersurf.f90:113-152,298-306`
  identifies and gathers `ccanopy` as the CO2 concentration in the canopy.

The existing JAX route mirrors this source:

- `jax_orchidee/driver/domain.py:986-994` reads annual CO2 from the paper
  configuration.
- `jax_orchidee/driver/bundle.py:279-291` implements
  `ccanopy_from_co2` as a land-point vector, with the above Fortran references.
- `research/daily_coarse_graining/teacher_shards.py:109,635-641` stores the
  legal annual condition as `annual_co2_ppm`.
- `research/daily_coarse_graining/daily_markov_contract.py:1760-1783`
  reconstructs `ccanopy` from that annual condition and broadcasts it over
  the 48 half-hour steps.
- `manifests/coarse_graining/successor_process_head_candidate_v1.json:78-80`
  and `strong_baseline_axis_aligned_process_expert_v1.json:110-112` route
  `annual_co2_ppm` through the `annual_co2` source family.

Therefore the exact bare PFT physical endpoint for the current day can be
reconstructed as `cimean_bare(day_end) = Ca(current_year)` on the annual-CO2
paper route. The decoder's day-start base is the supplied input
`diffuco_previous_step_state.cimean`; it is not guaranteed to equal the
current year's `Ca` at a cross-year handoff (the previous day can carry the
prior year's value). Consequently the signed target is
`delta_bare = Ca(current_year) - cimean_bare(day_start_input)`, and must not be
silently clamped to zero. It is zero for ordinary within-year inputs whose
day-start state already carries the same annual value, but cross-year examples
must retain their nonzero delta and its correct boundary gradients. This is a
source identity, not a justification for using the observed Teacher endpoint
as an input. A future time-varying/coupled-CO2 route would need its declared
time-resolved `ccanopy` input and must not silently assume annual constancy.

## Minimal rigorous recommendation

1. Preserve the shared PFT-axis network, the full output shape, and the PFT14
   `cimean` head gradients. Keep cimean loss accounting explicit rather than
   dropping the whole field.
2. Identify bare soil from the numerical `role_bare_soil` trait channel
   (`research/daily_coarse_graining/daily_operator/training_data.py:202-220`),
   not from compact index 0. Apply the exact source owner `Ca` to only that
   coordinate before physical/state reconstruction (and apply the matching
   bare-coordinate endpoint/delta target semantics). This preserves PFT-axis
   permutation equivariance.
3. Route `Ca` from the already admitted annual condition/forcing owner. Do not
   read it from the cimean Teacher target, day-end target, or an inferred
   endpoint. Do not use the pooled vegetation statistic for the bare identity.
4. Keep the bare coordinate defined and source-supervised if it remains in the
   shared output contract; an applicability mask that merely hides it is not a
   faithful representation of `cimean=Ca`. If the implementation chooses an
   exact deterministic owner instead, remove only that coordinate from the
   neural loss/schema with an explicit contract update and retain full shape
   during materialization.
5. Retain parameter/state derivatives needed by the PFT14 active path, all
   shared trunk parameters, and the day-start boundary. The direct bare
   coordinate head should not learn an arbitrary residual after the exact
   source-owned identity is applied, but the nonzero cross-year
   `Ca - day_start_state` delta must retain its legitimate state/boundary
   gradient path. This must not zero PFT14 or the shared trunk's PFT14
   gradient. The physical `Ca`/annual-CO2 route remains an input dependency
   where that route is trainable; no gradient should be manufactured through
   Teacher data.
6. Validate only the bounded contract change: bare day-end output equals legal
   current-day `Ca`, bare delta equals `current-day Ca - supplied day-start
   cimean` (including cross-year nonzero cases), PFT14 output/gradient is
   unchanged, compact PFT permutation preserves identities, inactive slots
   remain inactive, and finite/physical gates still pass. Do not regenerate
   v6, switch to v5, or reduce shared-transformer capacity as part of this
   audit.

## Limits

This report audits the paper compact layout, the named shared successor
consumer, and the annual-CO2 input route. It does not establish all Fortran
PFT/configuration branches, all coupling modes, or all historical restart
cases. In particular, the current-day endpoint equality to annual `Ca` is
asserted only for the current annual `ATM_CO2` route; the input day-start state
may carry a prior-year value at a cross-year handoff. A coupled or subannual
CO2 forcing requires separate source evidence.
