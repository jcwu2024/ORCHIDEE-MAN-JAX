# Litter metadata to GPP input-route audit

Date: 2026-09-13

## Decision

The nine `litter_metadata` leaves do reach GPP in the actual fresh
109,464,375-parameter, update-1998 model. They reach it through several learned
context paths, including a coordinate-preserving carbon adapter. They are not
among the six explicitly named structured-GPP state inputs.

The completed leaf intervention identifies `litterpart` as the dominant leaf
on this panel. Replacing only its four coordinates rescues 95.10% of the T30
free-minus-Teacher GPP MSE gap; the reciprocal replacement restores 78.69%.
None of the other eight leaves is independently material at the fixed 10%
criterion. This is strong leaf-level mediation evidence, but it still does not
identify one neural token route or a Fortran process cause.

The corresponding ORCHIDEE-MAN variables do not have a demonstrated direct
same-day role in the fast photosynthesis calculation. In the source inspected
here they belong to litter decomposition, litter-quality routing, dead-leaf
cover, and fire/fuel bookkeeping. Thus the strong intervention result is a
diagnosis of same-day neural input mediation at fixed weights, not evidence
that litter metadata directly controls same-day GPP in Fortran and not unique
process causality.

The most concrete mechanism is `litterpart`: its PFT14 training scale is only
0.00789091 around a mean of 0.99993773, while free-running carries reach
81.35 standard deviations. Those values enter both coordinate attention and
pooled whole-field summaries. The remaining nonconstant leaves differ by only
about 1.03--1.47 normalized RMS over the same T30 intervention inputs.

## Evidence boundary

This audit follows the exact head built by
`research/daily_coarse_graining/daily_operator/fresh_start.py:22-68` and used by
`outputs/trained-gpp-state-interventions-v1.json`, whose source checkout is
`a136fda9e659d2f98952afe39a315153a0c94ddd`. The construction is:

1. distribution-aware maintenance;
2. structured GPP;
3. coupled GPP/carbon feedback;
4. conditioned potential attention.

All neural parameters were fresh-initialized by that construction. The coupled
candidate has 109,464,359 parameters; the final conditioned head adds 16
learned attention-temperature parameters
(`target_oriented_coupled_gpp_conditioned_attention.py:128-173`). The
intervention itself performs no optimizer update and propagates no hybrid
output. Each arm changes only that day's continuous carry under the same
forcing, parameters, and discrete carry.

The group result is strong and bidirectional. Across the eight T30 windows
(240 days), GPP MSE is 0.75941339 for the free carry, 0.01023296 for the
Teacher carry, 0.04533842 after replacing the nine leaves in the free carry,
and 0.59685162 after the reciprocal replacement in the Teacher carry. These
are rescue and damage fractions of 0.95314151 and 0.78301386. Rescue is
positive in 7/8 windows and damage in 8/8. T7 corroborates directionally
(rescue 0.88560353; damage 1.13015950). The separately completed leaf panel
does provide leaf-level attribution: `litterpart` alone has T30 rescue
0.95104496 and damage 0.78693458, and T7 rescue 0.87796868 and damage
1.12753540. None of the other eight leaves independently crosses the fixed
10% bidirectional-materiality criterion. Replacing `litter_metadata` and DOC
together gives T30 rescue 0.98142186 and damage 0.94250834. The artifact's
classification remains
`same_day_state_mediation_diagnosis_not_unique_process_causality`.

The compact PFT axis contains `[bare soil, PFT14]`, corresponding to native
indices `[0, 13]`
(`manifests/coarse_graining/daily_teacher_669_data_product_policy.json:33`).
The numerical audit below uses the exact intervention inputs: the archived
Teacher day-start carry and, after the common first day, the preceding neural
free endpoint (`diagnose_trained_gpp_state_interventions.py:56-78,367-370`).

## The nine leaves

All 94 coordinates use ordinary per-coordinate z-scores from the actual runtime
normalization. `_load_runtime` loads the process statistics and exposes
`process_statistics.input_normalization()`
(`scripts/hpc/run_successor_process_head_canonical_gate.py:243-254,331-354`);
the target-oriented head receives that object, and `build_fresh_head` preserves
its continuous moments while adding log-tail conditioning only for DOC
(`fresh_start.py:22-41`). The authoritative arrays are therefore
`outputs/v6-process-training-statistics.npz` keys
`input__continuous__mean` and `input__continuous__scale`.

As a cross-check, those two arrays are bitwise identical across all 3,854
coordinates to `state__mean` and `state__scale` in
`outputs/v6-training-statistics.npz`. This follows the process-statistics
builder's explicit copying of the base continuous input moments
(`training_statistics.py:584-603`); there is no hidden pooling across PFTs in
this input-normalization step. `Zero-var` below means the runtime training
variance was zero; the stored fallback scale is 1. The normalized RMS and
maximum are over the 240 T30 day-start pairs actually supplied to the
intervention.

| Leaf | Slice; shape | Zero-var | Free-Teacher z RMS | Maximum free abs z | Source-backed role |
|---|---:|---:|---:|---:|---|
| `litterpart` | `[3246:3250]`; `(1,2,2)` | 2/4 | 28.2534 | 81.3529 | Above-ground litter fraction by PFT/pool; legacy fire input |
| `lignin_struc` | `[3036:3040]`; `(1,2,2)` | 4/4 | 0 | 0 | Legacy above/below lignin ratio passed to LPJ fire |
| `lignin_struc_above` | `[3040:3042]`; `(1,2)` | 1/2 | 1.4675 | 3.6896 | Above-ground structural-litter quality and decay/routing |
| `lignin_struc_below` | `[3042:3106]`; `(1,2,32)` | 53/64 | 1.0269 | 4.5968 | Layered below-ground structural-litter quality and decay/routing |
| `dead_leaves` | `[2482:2486]`; `(1,2,2)` | 2/4 | 1.1553 | 3.0405 | Dead-leaf stock, litter decay, cover diagnostic, legacy fire |
| `fuel_1hr` | `[2992:2996]`; `(1,2,2,1)` | 2/4 | 1.2432 | 3.2148 | SPITFIRE fine-fuel load |
| `fuel_10hr` | `[2988:2992]`; `(1,2,2,1)` | 2/4 | 1.1053 | 4.9859 | SPITFIRE fuel load |
| `fuel_100hr` | `[2984:2988]`; `(1,2,2,1)` | 2/4 | 1.1224 | 4.9859 | SPITFIRE fuel load |
| `fuel_1000hr` | `[2980:2984]`; `(1,2,2,1)` | 2/4 | 1.2897 | 4.9859 | SPITFIRE coarse-fuel load |

For `litterpart`, the two PFT14 pool coordinates both have training mean
0.99993773 and scale 0.00789091. Teacher inputs remain within 0.00790 standard
deviations of the mean. The two PFT14 free-minus-Teacher RMS differences are
39.83 and 40.08 standard deviations and their extrema reach 81.35. The two
bare-soil coordinates have zero training variance, mean 0, fallback scale 1,
and much smaller difference RMS values of 0.272 and 0.279. The combined raw
RMS difference is only 0.29619; the extreme learned input is caused by its
normalization relative to a nearly constant PFT14 training distribution.

The endpoint's learned-output and recurrent-input scales are materially
different. The four fast-output coordinates use pooled output mean/scale
`0.5/0.5`, process-loss scale 1, and next-state scale 1, whereas the next GPP
call standardizes the PFT14 coordinates by 0.00789091. On T30 Teacher-reset
rollouts, the learned daily endpoints average
`[0.35522, 0.36047, 0.66768, 0.67572]` rather than the Teacher
`[0, 0, 1, 1]`; the two PFT14 endpoint RMSE values, 0.34351 and 0.33745, become
43.53 and 42.77 recurrent-input standard deviations. Thus the present pooled
endpoint objective can regard a residual as moderate while the following
day's GPP input sees it as extreme. These normalization and endpoint summaries
come from `outputs/trained-litterpart-feedback-summary-v1.json`; they diagnose
a supervision-scale mismatch, not a defect in the input normalization.

`lignin_struc` is identically zero in both carries and cannot explain the
observed contrast by value replacement in this panel. The other leaves can
still contribute or interact, but their normalized displacement alone is not
comparable to `litterpart`. The leaf panel identifies `litterpart` at leaf
level; it does not identify which of its coordinates or internal token routes
mediates the response, exclude interactions, or establish a unique process
cause.

## Actual learned routes to GPP

The broad access is intentional in the frozen architecture contract:
`slowproc_stomate_previous_step_state` is assigned to the `state_carbon`
family, and the `canopy_gpp` expert admits that family
(`strong_baseline_axis_aligned_process_expert_v1.json`, `source_members` and
`experts.canopy_gpp.admitted_sources`). Every one of the nine leaves therefore
has all of the following routes.

### 1. Shared whole-field token

`assemble_strong_baseline_context` standardizes the complete continuous carry
(`strong_baseline_architecture.py:1041-1047`). It reduces each field, including
each of these leaves, to mean, standard deviation, minimum, and maximum over
its defined coordinates (`strong_baseline_architecture.py:899-915,1053-1061`).
The resulting field tokens are part of the shared source bank
(`strong_baseline_architecture.py:1233-1269`). Consequently one extreme
`litterpart` coordinate changes several pooled statistics, not just one local
token.

### 2. Canopy-GPP axial context

The layout assigns every slow-process state field to `state_carbon` and builds
axis pools for each expert that admits the family
(`strong_baseline_architecture.py:339-421`). Runtime axial summaries are
formed from the same standardized values and mixed along each physical axis
(`strong_baseline_architecture.py:973-1027,1094-1118`). This gives the canopy
expert PFT, litter-pool, element, and deep-layer summaries where applicable.

### 3. Parent GPP hidden and structured shared bank

The target-oriented domain attention consumes both shared source tokens and
the canopy expert's axial tokens, then mixes science contexts
(`target_oriented_attention.py:328-433`). The GPP field decoder retains its
hidden representation (`target_oriented_attention.py:500-603`). Structured
GPP projects that hidden representation into a per-PFT `parent_token`
(`target_oriented_structured_gpp.py:576-609`).

Separately, structured GPP places the admitted parent source tokens and canopy
axis tokens directly in the bank read by all three GPP branches
(`target_oriented_structured_gpp.py:654-727`). Thus pooled litter context can
affect potential, limitation, and correction even apart from the parent-hidden
path.

### 4. Coordinate-preserving carbon adapter

The coupled adapter selects all 107 continuous slow-process fields, 2,386
coordinates in total, including all nine audited leaves
(`target_oriented_coupled_gpp_carbon_feedback.py:150-225`). It embeds each
normalized coordinate together with a defined flag, field identity, and
non-PFT axis identities/positions
(`target_oriented_coupled_gpp_carbon_feedback.py:299-338`). The three
potential/limitation/correction queries attend over those coordinate tokens and
receive separate residual injections
(`target_oriented_coupled_gpp_carbon_feedback.py:341-382`).

The adapter mask permits global coordinates plus coordinates owned by the same
compact PFT. It therefore blocks a PFT14 query from directly attending to a
bare-soil-owned `litterpart` coordinate inside this adapter. This protection is
not global: the whole-field token pools all defined coordinates, and non-PFT
axial summaries can also aggregate across the active compact PFT slots.

### 5. Final output

The final structured response is

`normalization_scale * softplus(potential) * sigmoid(limitation) * correction`

for active vegetated PFTs (`target_oriented_structured_gpp.py:728-747`). The
six explicitly named dynamic state owners used to build the local structured
query are `assim_param`, `temp_growth`, `humrel`, `qsintveg`,
`control_salinity`, and `control_inudate`
(`target_oriented_structured_gpp.py:219-244,401-447`). None of the nine litter
metadata leaves is one of those direct owners.

In compact form, the learned routes are:

```text
nine normalized litter leaves
  +--> whole-field summary -------------------+
  +--> canopy axial summaries ----------------+--> domain/science GPP hidden --+
  |                                           |                                 |
  |                                           +--> structured shared bank ------+--> potential
  +--> per-coordinate carbon tokens --> three query residuals -----------------+--> limitation
                                                                                +--> correction
                                                                                     |
                                                                                     +--> GPP
```

## Fortran comparison

The ordering rules out interpreting these paths as a recovered direct
same-day source dependency. In `sechiba`, `diffuco_main` computes `gpp` during
the fast canopy step (`fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:988-1004`).
Only later does `sechiba` call `slowproc_main` and pass that already-computed
`gpp` along with `deadleaf_cover` and the carbon states
(`sechiba.f90:1131-1217`). `slowproc_main` similarly passes GPP and the carbon
states separately into `stomate_main`, then computes NPP as GPP minus growth
and maintenance respiration
(`fortran_source/ORCHIDEE/src_sechiba/slowproc.f90:964-1013,1033-1061`).

The audited variables have these source-backed slow roles:

- `lignin_struc_above` and `lignin_struc_below` are updated as structural
  litter-quality ratios (`stomate_litter.f90:2352-2392`). Above-ground lignin
  controls structural-litter decay and partitioning to heterotrophic
  respiration and carbon inputs (`stomate_litter.f90:2556-2600`); layered
  below-ground lignin controls the corresponding decay and routing
  (`stomate_litter.f90:1957-1970,2727-2765`).
- `dead_leaves` is decremented with litter decomposition and is used to derive
  `deadleaf_cover` (`stomate_litter.f90:1178-1203,1314-1364`). It is also a
  modifiable legacy fire state (`lpj_fire.f90:120-176`).
- `litterpart` is recomputed from above-ground litter contributions
  (`stomate_litter.f90:2394-2410`). In the legacy `fire` interface its own
  source comment calls it dispensable and probably redundant
  (`lpj_fire.f90:120-176`).
- the four fuel pools are maintained with litter decay and read by SPITFIRE to
  construct dead/net fuel load and fire characteristics
  (`stomate_litter.f90:2602-2660`;
  `lpj_spitfire.f90:568-625`).

The paper configuration enables STOMATE and fire
(`fortran_run_scripts/paper_250919/run.def.vn:63,109`), so these are not merely
dead declarations. They can affect later states through litter/carbon/fire and
ultimately later biomass, LAI, cover, or other fast-process inputs. That is a
physically plausible delayed recurrent route. It does not provide a direct
same-day edge from these metadata values to the already-computed Fortran GPP.

## Exact reconstruction feasibility is conditional

The repository already contains the exact source formula for one
`littercalc_leak` call. For a litter pool with pre-call stock `A`, PFT
numerator `p*A`, total increment `dA`, and PFT increment `dA_pft`, the update
is

```text
A_added = A + dA
p_added = where(A_added > min_stomate,
                (p * A + dA_pft) / A_added,
                0)
```

Fortran constructs `dA_pft` from `bm_to_litter` and `turnover` and applies the
ratio at `stomate_litter.f90:2106-2172,2322-2325,2393-2406`.
`littercalc_litter_increments` and `littercalc_apply_pool_increments` are the
source-equivalent JAX implementations
(`jax_orchidee/stomate/carbon_kernels.py:853-945,970-1031`). The needed daily
`bm_to_litter` and `turnover_daily` arrays are already exposed in
`DayStartProcessState` (`input_assembly.py:179-184`), and
`derive_litter_input` applies the exact source partition to their daily totals
(`process_derivations.py:227-288`). These facts make a daily deterministic
owner worth testing, but do not yet prove one.

In the actual `OK_LEAK` route, STOMATE scales those daily arrays by
`dt_sechiba/one_day` and calls `littercalc_leak` every half hour
(`stomate.f90:3277-3315`). Each call computes `p_added` and only afterward
decays `litter_above` (`stomate_litter.f90:2563-2581,2652-2665`), leaving the
ratio to be carried into the next call. Consequently, substituting day-start
stock plus a single summed daily increment is not generally source-exact:
earlier and later additions receive different products of the intervening
temperature-, moisture-, lignin-, soil-water-, and flood-dependent decay
factors. The `min_stomate` branch is also evaluated at each call. The existing
one-call kernel and daily litter-input sum do not establish that those 48
mixed addition/decay steps commute or collapse.

The current daily-candidate contract also forbids reconstructing the 48-step
scan. A deterministic replacement therefore requires either (a) a proved
source-equivalent recurrence elimination using quantities available to the
daily operator, or (b) a source-backed retained daily calculation producing the
terminal fast value without a subdaily scan. Teacher readback is validation
evidence only, never an inference input. The replacement must match the typed
v6 fast `litterpart` endpoint over
relevant Teacher rows, including cold start, bare-soil/PFT masks, zero-stock
and threshold branches. Constantly forcing `[0, 0, 1, 1]` is not acceptable:
that is the value on this intervention panel, not the general source rule, and
the training population includes PFT14 zero cases. Until that proof and
validation exist, `successor_fast_owner_reconstruction_v1.json:5135-5172`
correctly remains `independent_neural_endpoint`.

## Interpretation and bounded next evidence

The model's broad `state_carbon -> canopy_gpp` access was an explicit design
choice, and a learned use of correlated slow state is not automatically a code
defect. Here, however, the combination of no source-backed same-day edge and an
81-standard-deviation `litterpart` input, together with the leaf-specific
bidirectional intervention, makes `litterpart` proxy/context dependence and
off-manifold amplification the leading explanation. This is a learned
mediation result, not proof of a direct Fortran cause or of one internal neural
route.

The immediate bounded repair keeps the learned endpoint and directly supervises
its next-state `litterpart` residual using actual recurrent input scales,
especially0.00789091 for the two PFT14 coordinates, with existing defined and
active-PFT masks. The shared one-day integration and training-only gradient
calibration are specified in `feedback_and_canopy_loss_amendment_20260913.md`.
The source review above found no already-proved daily shortcut; proving a
new exact owner is not a prerequisite for testing this supported loss change.
If a daily deterministic owner is established later, it still needs relevant
ordinary/cold-start/mask/threshold typed-target parity before adoption.
This audit does not imply changing the input normalization, GPP architecture,
or access routes, and it does not justify removing `litterpart` from the GPP
input.
