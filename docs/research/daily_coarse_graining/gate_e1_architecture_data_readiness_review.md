# Gate E1 Architecture And Data-Readiness Review

Status: **accepted**.

Date: 2026-08-05.

This review covers the uncommitted Gate E1 implementation at base commit
`23a05b5`. It does not start Gate E2, select a network, train a model, submit a
compute job, regenerate Teacher data, or change the canonical Teacher.

## Decision

Gate E1 is accepted. The two finite review blockers are closed:

1. **Unique canonical input assembly: closed.** Public `DailyOperatorInput`
   contains only canonical continuous/discrete day-start state, native forcing,
   the frozen PFT interface, named static/annual registries, and calendar
   controls. `CanonicalDailyOperatorInputAssembler` alone reconstructs grouped
   water/carbon/thermal state, litter/lignin day-start views, masks, and exact
   process-static views. Thirty E1 tests include fail-closed mutations of all
   three state domains, litter/turnover, both lignin views, and every retained
   process-static field.
2. **Reusable differentiable canonical retained tail: closed.** The package
   exports `CanonicalRetainedTailAdapter`, its explicit static owner, and
   `build_canonical_retained_tail_adapter`. The verification script supplies
   only the true-label Oracle head and imports the formal adapter. The adapter
   consumes `RetainedTailInput`, packs the historical target only inside the
   accepted retained-owner boundary, and exposes neither a Teacher endpoint
   nor an anonymous fast target. Real-owner eager/JIT agreement is within
   `5.69e-14`; forward JVP and reverse VJP are finite. The staged graph has no
   callback primitive and no length-48 scan.

The cold-continuation, ordinary-day, and restart-year cases all execute the
same `daily_operator_transition` composition and reproduce the complete
canonical next state. Restart composition and the accepted file roundtrip
evidence pass. No identity tail is used for these real-owner checks.

## Inference Contract Audit

The intended inference categories are correct: canonical day-start continuous
and discrete state; native forcing values/times/durations/predecessor and
record masks; PFT state, named parameters, traits, fractions and active mask;
named static/annual conditions; and calendar/discrete controls. Landpoint/PFT
identity, Teacher endpoints, day-end state, rain/snow labels, litter labels,
dependent POC/DOC terms, and Teacher daily temperature labels are absent.
Rain/snow are integrated from `Rainf`/`Snowf`; litter input and dependent
decomposition terms use frozen exact formulas; daily temperature context is a
predicted output. The source/JAXPR audit sees one length-5 forcing scan and no
length-48 scan.

The inference contract is accepted. Named process views are legal only as
assembler-owned derived values; callers cannot supply them independently.
Persisted/cached assembled views can be admitted only through the host-side
fail-closed consistency gate before staging.

## Frozen 47-Label Matrix

The machine authority is
`manifests/coarse_graining/daily_flux_label_inventory_v1.json`; source owners,
units, roles and dependencies below were checked against it. `C` means the
existing v5 contract/shards contain the label source, `D` means exact in-graph
derivation, and `S` means only the bounded supplemental capture has the typed
field and full 669-point coverage must still be generated. Shapes use source
axes; `P` is the active PFT axis and is 14 in the Gate C capture, `L=32`,
`T=6`, `Z=11`, `Q=7`, `K=3`, and `E=1`. Masks are: `state` for the frozen
defined-status mask, `PFT` for active/process-capability masks, `finite` for a
capture finite mask, and `formula` for the conjunction of dependency masks.

| Label | Ready | Prediction/exact owner | Shape; unit; mask |
| --- | --- | --- | --- |
| water.canopy_inventory_endpoints | C | predicted conservative inventory; hydrol canopy endpoint owner | `[npts,P]` signed storage/debt; kg m-2; state+PFT |
| water.soil_inventory_endpoints | C | predicted conservative inventory; hydrol soil endpoint owner | `[npts,Z,T]` for `mc/mcl`; source state geometry; state |
| water.snow_inventory_endpoints | C | predicted conservative inventory; explicit-snow owner | `snow[npts]`, `snow_nobio[npts,*]`, `snowliq[npts,K]`; kg m-2; state |
| water.flood_inventory_endpoints | C | predicted conservative inventory; hydrol flood owner | `[npts]`; kg m-2; state |
| water.rain_input | C | exact native `Rainf*duration` aggregation | `[npts]`; kg m-2 day-1; forcing record mask |
| water.snowfall_input | C | exact native `Snowf*duration` aggregation | `[npts]`; kg m-2 day-1; forcing record mask |
| water.canopy_precipitation_partition | S | predicted; complete-day SECHIBA capture | `precip2canopy/ground [npts,P]`; kg m-2 day-1; PFT+finite |
| water.canopy_to_ground | S | predicted; complete-day SECHIBA capture | `[npts,P]`; kg m-2 day-1; PFT+finite |
| water.wet_canopy_evaporation | S | predicted; complete-day SECHIBA capture | `[npts,P]`; kg m-2 day-1; PFT+finite |
| water.transpiration | S | predicted; complete-day SECHIBA capture | `[npts,P]` plus rootsink `[npts,Z,T]`; kg m-2 day-1; PFT+finite |
| water.bare_soil_evaporation | S | predicted; complete-day SECHIBA capture | `[npts]` plus `[npts,T]`; kg m-2 day-1; soil-tile+finite |
| water.snow_sublimation | S | predicted; complete-day SECHIBA capture | `[npts]` plus `[npts,nnobio]`; kg m-2 day-1; finite |
| water.flood_evaporation | S | predicted; complete-day SECHIBA capture | `[npts]`; kg m-2 day-1; finite |
| water.snow_phase_and_melt_transfer | S | predicted; complete-day SECHIBA capture | two `[npts]` plus three `[npts,K]`; kg m-2 day-1; snow+finite |
| water.soil_infiltration | S | predicted; complete-day SECHIBA capture | two `[npts,T]`; kg m-2 day-1; soil-tile+finite |
| water.soil_vertical_transfer | S | predicted; complete-day SECHIBA capture | `[npts,Z,T]`; kg m-2 day-1; interface/tile+finite |
| water.runoff | S | predicted; complete-day SECHIBA capture | `[npts]` plus two `[npts,T]`; kg m-2 day-1; tile+finite |
| water.drainage | S | predicted; complete-day SECHIBA capture | `[npts]` plus `[npts,T]`; kg m-2 day-1; tile+finite |
| water.routing_exchange | S | predicted; complete-day SECHIBA capture | four `[npts,1]`/`[npts]` roles; kg m-2 day-1; finite |
| carbon.fast_gpp | C | predicted; daily DIFFUCO fold owner | `[npts,P]`; gC m-2 day-1; PFT |
| carbon.fast_maintenance_respiration | C | predicted; daily maintenance fold owner | `[npts,P,12]`; gC m-2 day-1; PFT+part |
| carbon.ok_leak_inventory_endpoints | C | predicted conservative inventory; OK_LEAK owner | litter/POC/DOC/canopy source axes; gC m-2; state+PFT |
| carbon.litter_input | D | exact litter increment partition from day-start state | above `[npts,2,P,E]`, below `[npts,2,P,L,E]`; gC m-2 day-1; formula+PFT |
| carbon.litter_respiration | S | predicted; compiled OK_LEAK capture | `[npts,P,2]` plus `[npts,P]`; gC m-2 day-1; PFT+finite |
| carbon.litter_to_doc | S | predicted; compiled OK_LEAK capture | `[npts,P,L,Q,E]` plus `[npts,P,Q,E]`; gC m-2 day-1; PFT+finite |
| carbon.poc_gross_decomposition | S | predicted; compiled OK_LEAK capture | ordinary/flood `[npts,3,E,P,L]`; gC m-2 day-1; PFT+finite |
| carbon.poc_respiration | D | exact `(1-CUE)*gross POC` owner | `[npts,P]`; gC m-2 day-1; formula+PFT |
| carbon.poc_to_doc | D | exact CUE/destination-mask owner | `[npts,P,L,Q,E]`; gC m-2 day-1; formula+PFT |
| carbon.doc_gross_decomposition | S | predicted; compiled OK_LEAK capture | ordinary/flood `[npts,P,L,Q]`; gC m-2 day-1; PFT+finite |
| carbon.doc_to_poc | D | exact CUE/fraction/lignin owner | `[npts,3,P,L]`; gC m-2 day-1; formula+PFT |
| carbon.doc_respiration | D | exact `(1-CUE)*gross DOC` owner | `[npts,P]`; gC m-2 day-1; formula+PFT |
| carbon.doc_external_input | S | predicted; compiled OK_LEAK capture | two `[npts,T]` plus three `[npts,P,E]`; gC m-2 day-1; PFT+finite |
| carbon.doc_export | S | predicted; compiled OK_LEAK capture | three `[npts,P,Q,E]` plus `[npts,P,Q,L,E]`; gC m-2 day-1; PFT+finite |
| carbon.doc_free_adsorbed_equilibration | S | predicted; compiled OK_LEAK capture | `[npts,P,L,Q,E]`; gC m-2 day-1; PFT+finite |
| carbon.doc_vertical_water_transport | S | predicted; compiled OK_LEAK capture | `[npts,P,L,Q,E]`; gC m-2 day-1; interface+PFT+finite |
| carbon.doc_vertical_diffusion | S | predicted; compiled OK_LEAK capture | `[npts,P,L,Q,E]`; gC m-2 day-1; interface+PFT+finite |
| carbon.cryoturbation_redistribution | S | predicted; compiled OK_LEAK capture | carbon `[npts,3,P,L]`, DOC `[npts,P,L,2,Q,E]`, litter `[npts,2,P,L,E]`; gC m-2 day-1; PFT+finite |
| carbon.perma_peat_redistribution | S | predicted; compiled OK_LEAK capture | `[npts,3,P,L]`; gC m-2 day-1; peat-capability+PFT+finite |
| energy.surface_temperature_tendency | D | exact endpoint difference | `[npts]`; K day-1; formula+state |
| energy.soil_temperature_tendency | D | exact endpoint difference | `ptn[npts,L,P]`, `stempdiag[npts,Z]`; K day-1; formula+state+PFT |
| energy.snow_thermal_tendency | D | exact endpoint difference | `snowheat/snowtemp [npts,K]`; source thermal units/day; formula+state |
| energy.daily_temperature_context | C | predicted compact output; daily fold supervision | `tsurf[npts]`, `tsoil[npts,Z]`; K; state/defined |
| energy.net_radiation_integral | S | predicted signed components; energy capture | `[npts]` plus `[npts,P]`; J m-2 day-1; explicit finite mask (`netrad_pft` may be NaN) |
| energy.sensible_heat_integral | S | predicted; energy capture | `[npts]` plus `[npts,P]`; J m-2 day-1; PFT+finite |
| energy.latent_heat_integral | S | predicted; energy capture | four `[npts]`; J m-2 day-1; finite |
| energy.ground_heat_integral | S | predicted signed components; energy capture | two `[npts]` plus `[npts,P]`; J m-2 day-1; PFT+finite |
| energy.phase_change_integral | S | predicted signed components; energy capture | two `[npts]` plus `[npts,K]`; J m-2 day-1; snow+finite |

Counts are `10 C + 8 D + 29 S = 47`. `PROCESS_LABEL_BINDINGS`
binds all 47 once, but that audit currently checks IDs rather than the complete
shape/unit/mask schema. Before E2, the typed sidecar manifest must make those
shape/unit/mask declarations machine-readable and the binding audit must
validate them, not rely only on this review table.

## Supplemental Data Size

The one-day Gate C capture report supplies the measured float64 shapes. The
following is the uncompressed payload for the union of fields used by each
family; totals use `669 * 50 * 365 = 12,209,250` point-days and do not assume a
compression ratio.

| Family | Labels / stored fields | Bytes per point-day | 669x50y order |
| --- | ---: | ---: | ---: |
| `water_transfer_daily_v1` | 13 / 28 | 2,072 | 25.30 GB (23.56 GiB) |
| `ok_leak_transfer_daily_v1` | 11 / 24 | 279,872 | 3.417 TB (3.108 TiB) |
| `energy_flux_daily_v1` | 5 / 14 plus 14-byte `netrad_pft` mask | 454 | 5.54 GB (5.16 GiB) |
| **Total** | 29 / 66 unique required fields | **282,398** | **3.448 TB (3.136 TiB)** |

The 96-day capture/Oracle assets prove source capture and replay only. They
are not full-coverage training data. In particular, the OK_LEAK tensor family
dominates storage; E2 data preparation must first define a lossless typed
representation and measure compression on a bounded sample before any 669
generation request. It may not collapse scientifically required axes merely
to reduce size.

## Reader, Split, And Statistics Readiness

The accepted spatial/temporal split metadata can be reused unchanged and the
existing 669 parent shards must remain immutable. The current reader is not
ready for typed E1 labels: `markov_dataset.py` loads a fixed set containing
`state_trajectory`, anonymous historical `fast_day_target`, native forcing,
conditions, diagnostics and discrete state; collation and statistics are
hard-coded to those names. It has no hash-joined supplemental sidecar reader,
typed-label masks, or train/train-only statistics for the new families.

Minimum E2 data work after Gate E1 acceptance:

1. freeze a versioned supplemental manifest keyed by parent dataset ID,
   contract hash, landpoint, year and day index, including field owner, axes,
   units, dtype and defined-mask policy;
2. add a hash-verifying sidecar reader that joins without modifying parent
   shards and preserves existing spatial/temporal splits;
3. collate typed `ProcessPrediction` targets rather than expose the anonymous
   historical fast target to the candidate;
4. fit finite-only statistics from the train/train split for predicted labels
   and reuse exact derivations after denormalization; never fit on validation
   or test;
5. run missing-day, duplicate-key, shape/unit/mask drift and parent-hash
   fail-closed tests;
6. measure a bounded OK_LEAK compression/layout sample before proposing the
   full generation resources and cost.

These data tasks belong to E2 preparation. The 29 supplemental labels and
typed sidecar work are not Gate E1 blockers.

## Predeclared E2 Entry Contract

No specific network is selected. E2 starts from
`bind_daily_operator(input_assembler=canonical_assembler,
process_head=candidate_head,
retained_tail=canonical_retained_tail_adapter)`, where
`candidate_head(parameters, context, assembled_input)` returns the frozen typed
`ProcessPrediction`. The input assembler, conservative updater, exact
derivations, and canonical retained tail are fixed infrastructure, not
trainable architecture choices. The candidate must not consume IDs, endpoints,
day-end state, typed Teacher process labels, or the historical anonymous fast
target at inference.

Before any larger or paid run, predeclare and pass: one real assembled sample
through eager/JIT and forward/reverse gradients with the real tail; typed
sidecar read/collate/statistics on train/train only; finite one-day loss and
budget closure on a tiny local batch; bounded tiny-fit with parent behavior
reported; cold/ordinary/restart one-day checks; and a free 7-day rollout smoke.
These are candidate admission gates, not evidence that E2 has started here.

## Accepted Handoff

Gate E2 may begin with the data-preparation items above and the stable API in
this review. It must not treat the 96-day Oracle/capture evidence as full
training coverage. Network selection, loss weighting, optimization, tuning,
paid computation, and full supplemental generation are E2 decisions and were
not performed by Gate E1.
