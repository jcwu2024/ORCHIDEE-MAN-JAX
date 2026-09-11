# Bare-soil output source audit: `gsmean`, `dead_leaves`, and fuel pools

Date: 2026-09-11

## Decision

Apply an exact-zero applicability rule to the semantic bare-soil coordinate of
`gsmean` only. The Fortran producer clears the full `gsmean` array on every
call and can assign values only inside a vegetation loop beginning at PFT 2.
This is an unconditional source invariant, not a pattern inferred from sampled
Teacher data.

Do **not** apply the same correction to `dead_leaves`, `fuel_1hr`,
`fuel_10hr`, `fuel_100hr`, or `fuel_1000hr`. Those five fields are
`INTENT(inout)` pools, are read from and written to restart files, and are not
cleared by the active litter routine. A zero bare-soil value can be preserved
along the paper run when the entering value is zero, but that conditional
zero-by-history property is not an unconditional output contract.

This is a source audit for the immutable-v6 fresh-initialization route. It does
not regenerate Teacher data, change the neural architecture, validate the
concurrent `gsmean` implementation, or close all configuration branches.
Here, **bare soil** means the source PFT identity (Fortran PFT 1) and must be
resolved in the compact JAX interface by its semantic `role_bare_soil` trait;
it never means an assumed positional slot.

## Field-by-field disposition

| Field | Fortran semantics | Paper-path bare-soil behavior | Wider restart/configuration behavior | Learned-boundary decision |
|---|---|---|---|---|
| `gsmean` | Recomputed `INTENT(out)` diagnostic | Reset to exact zero every call; PFT 1 is outside the only assignment loop | Still source-defined zero at PFT 1, independent of prior/restart state | Materialize zero by semantic identity; keep defined and supervised; hard-gate named output and next state |
| `dead_leaves` | Prognostic `INTENT(inout)` pool; restart-carried | If it enters as zero, no PFT-1 addition occurs and decay preserves zero | A nonzero PFT-1 restart is retained and can decay because decomposition spans all PFTs | Retain/source-model; no unconditional hard zero |
| `fuel_1hr` | Prognostic `INTENT(inout)` pool; restart-carried | If it enters as zero, the active addition path does not add at PFT 1 | A nonzero PFT-1 restart is not cleared; inactive fire/DGVM/LCC/peat paths may also alter fuel ownership | Retain/source-model with the fuel family; no unconditional hard zero |
| `fuel_10hr` | Same fuel-family mechanism | Same conditional zero-by-history behavior | Same restart and configuration qualification | Same; no unconditional hard zero |
| `fuel_100hr` | Same fuel-family mechanism | Same conditional zero-by-history behavior | Same restart and configuration qualification | Same; no unconditional hard zero |
| `fuel_1000hr` | Same fuel-family mechanism | Same conditional behavior once a zero entry state is established | Same, plus its restart reader lacks a visible `val_exp` prefill immediately before the missing-field test | Same; do not claim a universal cold-start-zero contract |

## 1. Exact source invariant for `gsmean`

The paper definition selects the CO2-aware branch with
`STOMATE_OK_CO2 = y` in
`fortran_run_scripts/paper_250919/run.def.vn:64`. In
`fortran_source/ORCHIDEE/src_sechiba/diffuco.f90`, `diffuco_main`
dispatches to `diffuco_trans_co2` at lines 665-672; the alternative call is at
lines 673-680. The producer spans lines 2018-2982, with `gsmean` declared
`INTENT(out)` at line 2087.

The decisive sequence is:

1. `gsmean(:,:) = zero` clears the whole output at line 2289.
2. The only vegetation loop is `DO jv = 2,nvm` at line 2299, excluding PFT 1.
3. Assimilation eligibility is built at lines 2330-2384, and `gstot` is reset
   at lines 2387-2391.
4. The only final assignment is inside `IF (nia .GT. 0)`, lines 2878-2885.
5. The vegetation loop ends at line 2976.

Therefore PFT 1 is exactly zero after every call, for every forcing, prior
state, and restart content. Vegetation PFTs failing the assimilation mask also
retain their initialized zero, but this audit authorizes only the static,
semantically identifiable bare-soil rule. It does not replace the dynamic
Fortran gates with a guessed neural eligibility mask.

The non-CO2 fallback is compatible: `diffuco_trans` also starts its vegetation
loop at PFT 2 (`diffuco.f90:1835`) and clears the actual argument receiving
`gsmean` as `vbetaco2(:,:) = zero` at line 1906.

### Learned-boundary consequence

The successor owner manifest declares
`endpoint.fast_state.diffuco_previous_step_state.gsmean` as an
`independent_neural_endpoint`, compact shape `(1, 2)`, nonnegative, and
represented as normalized day-end-minus-day-start
(`manifests/coarse_graining/successor_fast_owner_reconstruction_v1.json:868-903`).
`SuccessorFastOwnerMaterializer` consumes learned named endpoints while
building the successor state
(`research/daily_coarse_graining/daily_operator/fast_owner_reconstruction.py:204-386`,
especially lines 294-321). Without an applicability rule, a neural value for
the bare-soil coordinate therefore becomes both an exported named value and
successor state.

The source-backed integration contract is:

- identify bare soil from the named PFT trait `role_bare_soil`, not from
  compact index 0;
- replace only the bare-soil coordinate of the named `gsmean` endpoint with
  exact zero before physical/successor reconstruction;
- leave the coordinate defined and in the loss mask rather than hiding it as
  inapplicable data; and
- count any nonzero bare-soil value in either the named output or reconstructed
  next state as a hard failure.

At audit time, the concurrent implementation has this shape in
`research/daily_coarse_graining/daily_operator/output_applicability.py:18-125`,
is applied opt-in at
`research/daily_coarse_graining/daily_operator/target_oriented_process_head.py:108-116`,
is exposed through
`research/daily_coarse_graining/daily_operator/fresh_start.py:27-43`, and adds
named-output/next-state hard checks in
`research/daily_coarse_graining/daily_operator/fresh_start_training.py:37-106`.
Those files are owned and validated by the implementation task, not by this
audit.

## 2. Why `dead_leaves` and fuels are different

### Active litter routine carries rather than resets them

The paper enables `OK_LEAK` in
`fortran_run_scripts/paper_250919/Job0_bio:220-223`, selecting
`littercalc_leak` in
`fortran_source/ORCHIDEE/src_stomate/stomate_litter.f90:1637-2873`.
Within that routine:

- `dead_leaves` is `INTENT(inout)` at lines 1751-1760;
- all four fuels are `INTENT(inout)` at lines 1822-1828; and
- the initialization block at lines 1907-1934 resets outputs and temporaries,
  but not any of these five carried fields.

New material is added only inside `DO j = 2,nvm` at lines 2146-2321:

- `dead_leaves` addition: lines 2196-2199;
- `fuel_1hr` addition: lines 2245-2265;
- `fuel_10hr` addition: lines 2268-2282;
- `fuel_100hr` addition: lines 2284-2298; and
- `fuel_1000hr` addition: lines 2300-2317.

This prevents new PFT-1 additions, but it does not clear incoming PFT-1
storage. Aboveground decomposition then loops over all PFTs at lines
2563-2728. It multiplies structural and metabolic `dead_leaves` by decay
factors at lines 2581 and 2665, respectively. Thus an arbitrary nonzero
bare-soil `dead_leaves` restart is semantically live and can change.

The fuel detail is even more explicit. The temporary
`fuel_nlitt_total_pft` is initialized to zero at line 1933 and populated from
the four carried fuel arrays only inside the PFT-2-and-up addition loop at
lines 2316-2317. Fuel decomposition/synchronization spans all PFTs at lines
2600-2640 and 2678-2717, but its updates require the temporary total to exceed
`min_stomate` (lines 2606 and 2683). Consequently, an arbitrary nonzero PFT-1
fuel restart is not reset by this routine: the PFT-1 temporary total remains
zero and the guarded synchronization does not erase the carried arrays.

Therefore the paper-path statement is only conditional: starting from a
known zero PFT-1 pool, no active addition creates a value and zero remains
zero. Empty LAI, absent vegetation, or three observed Teacher zeros do not turn
a stored, restart-carried pool into a source-defined zero output.

### Restart provenance

`readstart` in
`fortran_source/ORCHIDEE/src_stomate/stomate_io.f90:34-1747` declares
`dead_leaves` and all fuels at lines 273 and 280-283, reads `dead_leaves` at
lines 1056-1060, and reads the fuels at lines 1081-1098. Missing-field fallback
sets `dead_leaves`, `fuel_1hr`, `fuel_10hr`, and `fuel_100hr` to zero only when
the entire retrieved array remains the sentinel `val_exp`.

For `fuel_1000hr`, lines 1096-1098 perform the read and sentinel test without
an immediately preceding visible `fuel_1000hr = val_exp` assignment. This may
be benign in a specific initialized call chain, but these lines alone do not
prove a universal missing-field or cold-start zero. The report deliberately
leaves that initialization edge unresolved rather than converting it into a
heuristic contract.

`writerestart` writes `dead_leaves` at lines 2729-2731 and all four fuels at
lines 2742-2752. The main STOMATE call chain also transports the five fields
through restart read, `littercalc_leak`, and restart write
(`fortran_source/ORCHIDEE/src_stomate/stomate.f90:1503-1561`,
`3291-3315`, and `5440-5496`). These are persistent state variables, not
throwaway diagnostics.

The JAX reference follows the same state boundary: its explicit cold-start
helper initializes all five arrays to zero
(`jax_orchidee/stomate/reference.py:703-776`), while restart readers and entry
state assembly preserve stored values (`reference.py:1176-1193`,
`1262-1280`, and `1522-1581`). The restart writer requires and writes all five
fields (`jax_orchidee/stomate/restart_io.py:1646-1726`).

### Paper-path qualification

The active paper job imposes the PFT14-only vegetation map
(`fortran_run_scripts/paper_250919/Job0_bio:61-77`), disables land-cover
change at line 63, disables DGVM in lines 113-126, disables fire at line 134,
and runs a cold start followed by yearly restart chaining at lines 277-290 and
313-322. Under this exact route, a correctly established zero bare-soil value
for the five litter/fuel fields is preserved by induction.

That observation is narrower than a source invariant. Outside the paper
configuration, SPITFIRE receives and consumes the four fuels
(`fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90:1193-1215` and
`fortran_source/ORCHIDEE/src_stomate/lpj_spitfire.f90:163-1247`, especially
846-855), DGVM cover redistributes fuels over all PFTs
(`fortran_source/ORCHIDEE/src_stomate/lpj_cover.f90:73-387`, especially
219-286), land-cover routines recalculate fuels over all PFTs
(`fortran_source/ORCHIDEE/src_stomate/stomate_lcchange.f90:467-489`,
`957-974`, and `1591-1608`), and the peat-cover path also redistributes them
(`fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90:2569-3337`, especially
3000-3101). These branches are inactive in the paper job and were not closed
here. Their existence reinforces the decision not to hard-zero the fuel
family globally.

## 3. JAX edge case and endpoint ownership

The JAX litter kernels preserve the main distinction: additions to
`dead_leaves` and fuels are masked away from source PFT index 0
(`jax_orchidee/stomate/carbon_kernels.py:853-945` and `1035-1096`), while
aboveground decomposition handles all PFTs (`carbon_kernels.py:1403-1525`).
The composed path is `littercalc_leak_core_with_controls`, lines 1677-1873.

There is a potential arbitrary-restart edge mismatch that must remain visible.
The JAX fuel synchronization recomputes totals directly from all four fuel
arrays at lines 1099-1180, whereas the audited Fortran temporary total remains
zero for PFT 1 unless populated inside the PFT-2-and-up addition loop. A
nonzero artificial bare-soil restart can therefore take a different update
path. This report neither fixes nor waives that discrepancy; it is another
reason not to impose an unconditional zero contract on the fuels.

The owner manifest treats `dead_leaves` and each fuel as a separate
`independent_neural_endpoint`, with absolute teacher-native targets and
nonnegative constraints
(`manifests/coarse_graining/successor_fast_owner_reconstruction_v1.json:5174-5375`).
They must remain a consistent carried family at the learned boundary. The
fact that all five bare-soil coordinates happen to be zero in a small Teacher
sample is not sufficient evidence to replace their modeled state semantics.

## 4. Loss-localization motivation, not proof

The completed loss audit identified the bare-soil `gsmean` coordinate as the
initial optimization failure:

- `gsmean` state-head gradient norm: `758.4475834`;
- full gradient norm: approximately `758.4517`;
- next largest audited state heads: `fuel_1hr` `0.4634`, `dead_leaves`
  `0.3143`, `fuel_1000hr` `0.2697`, and `fuel_100hr` `0.02786`; and
- replacing only the exported bare-soil `gsmean` coordinate with zero in an
  offline accounting counterfactual reduced initial state loss from about
  `229.5` to about `0.48`.

Sources are `outputs/fresh-start-loss-balance-v1.json` and
`docs/research/daily_coarse_graining/fresh_start_loss_balance_audit_20260911.md`.
The counterfactual localized bad objective accounting; it did not demonstrate
learning or establish source semantics. The Fortran audit above makes the
`gsmean` correction admissible, while the different Fortran ownership
prevents extending it to `dead_leaves` or fuels.

## 5. Bounded recommendation and unresolved work

For the current fresh-initialization run, enable only the source-backed
semantic bare-soil `gsmean` materialization and its hard output/state gate.
Keep all parameters trainable and keep the corrected coordinate defined and
supervised. Do not remove `gsmean` from the loss and do not generalize this
rule to `rveget`, `rstruct`, litter pools, or other outputs merely because
their sampled bare-soil values are zero.

A narrower immutable-v6-only optimization for `dead_leaves` or fuels would
require a separately stated product contract and an explicit scan covering
all immutable-v6 cases across cold starts, ordinary days, and restart years.
No such full scan was performed here, and even that empirical contract would
not become a general Fortran invariant.

Unresolved and explicitly out of scope:

- universal initialization behavior for a missing `fuel_1000hr` restart
  field;
- arbitrary nonzero bare-soil fuel restart parity between the current Fortran
  temporary-total logic and JAX direct-total logic;
- complete fire, DGVM, land-cover-change, and peat branch coverage; and
- any Teacher-v6 regeneration, training rerun, broader process correction, or
  hard-zero rule for `dead_leaves` and fuels.

The final source decision is intentionally asymmetric:
**hard-materialize bare-soil `gsmean = 0`; retain and model all five
restart-carried litter/fuel fields.**
