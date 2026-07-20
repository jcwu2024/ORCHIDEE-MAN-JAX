# STOMATE Daily Trace Contract

Scope: paper-case PFT14 mangrove active branch. This contract defines trace
columns needed before wiring the Phase 1B kernels through the Phase 1D daily
scheduling helpers. It is not a full STOMATE loop specification.

## Audited Daily Scheduling Spans

Time-step constraints and units:

- `src_stomate/stomate.f90`, `stomate_main`, lines 1818-1852:
  `dt_days` must be an integer number of days, no more than `max_dt_days`, and
  `dt_sechiba <= dt_days * one_day`.
- `src_stomate/stomate.f90`, `stomate_main`, lines 3183-3185:
  variables named `*_daily` are actually STOMATE-step means/totals if
  `dt_days /= 1`.

Daily accumulation:

- `src_stomate/stomate.f90`, `stomate_main`, lines 3198-3208:
  calls `stomate_accu(do_slow, ..., *_daily)`, including
  `stomate_accu(do_slow, gpp_d, gpp_daily)`.
- `src_stomate/stomate.f90`, `stomate_accu_r1d`, lines 9341-9363:
  `field_out(:) = field_out(:) + field_in(:) * dt_sechiba`; if `ldmean`, then
  `field_out(:) = field_out(:) / dt_stomate`.
- `src_stomate/stomate.f90`, `stomate_accu_r2d`, lines 9365-9387:
  same formula for 2-D fields.
- `src_stomate/stomate.f90`, `stomate_accu_r3d`, lines 9389-9411:
  same formula for 3-D fields.

Entry normalization before daily accumulation:

- `src_stomate/stomate.f90`, `stomate_main`, lines 2921-2931:
  computes current-year `veget_cov` and `veget_cov_max` by dividing `veget`
  and `veget_max` by `1 - totfrac_nobio`, with zero output when the
  biological fraction is too small.
- `src_stomate/stomate.f90`, `stomate_main`, lines 2980-2990:
  on `date == 1`, normalizes `vegetnew_firstday` with
  `totfrac_nobio_new`.
- `src_stomate/stomate.f90`, `stomate_main`, lines 2993-3016:
  normalizes `veget_max_new` into `veget_cov_max_new` with
  `totfrac_nobio_new` for land-cover-change and dynamic-peat branches.
- `src_sechiba/slowproc.f90`, `slowproc_main`, lines 937-985:
  sums `frac_nobio_lastyear` into `totfrac_nobio_lastyear` and passes it as
  `stomate_main`'s current `totfrac_nobio`.
- `src_sechiba/slowproc.f90`, `slowproc_main`, lines 1116-1121:
  computes `tot_bare_soil` separately from `veget_max` and `veget`; it is not
  a substitute for either `totfrac_nobio` or `totfrac_nobio_new`.

Maintenance respiration scheduling:

- `src_stomate/stomate.f90`, `stomate_main`, lines 3244-3252:
  calls `maint_respiration(kjpindex,lai,t2m,t2m_longterm,stempdiag,height,
  veget_cov_max,rprof,biomass,resp_maint_part_radia,sla_calc)`.
- `src_stomate/stomate.f90`, `stomate_main`, lines 3254-3263:
  zeros `resp_maint_radia`/`flood_root_radia`, sums
  `resp_maint_part_radia(:,j,k)` over parts for PFTs `j=2,nvm`, and computes
  flood-root diagnostic.
- `src_stomate/stomate.f90`, `stomate_main`, lines 3265-3267:
  accumulates `resp_maint_part = resp_maint_part + resp_maint_part_radia`.

Daily active-process block:

- `src_stomate/stomate.f90`, `stomate_main`, lines 4200-4218:
  updates `lai` from `biomass * sla_calc` for non-crop PFTs or calls `setlai`.
- `src_stomate/stomate.f90`, `stomate_main`, lines 4221-4251:
  calls `season`.
- `src_stomate/stomate.f90`, `stomate_main`, lines 4297-4338:
  calls `StomateLpj`, passing `gpp_daily`, `resp_maint_part`, `PFTpresent`,
  `biomass`, `lai`, `rprof`, `npp_daily`, `resp_maint_d`, and
  `resp_growth_d`.
- `src_stomate/stomate_lpj.f90`, `StomateLpj`, lines 1070-1082:
  active path calls `phenology`.
- `src_stomate/stomate_lpj.f90`, `StomateLpj`, lines 1093-1102:
  active path calls `alloc`, producing `f_alloc`.
- `src_stomate/stomate_lpj.f90`, `StomateLpj`, lines 1111-1113:
  recalculates `slai` with `setlai` after allocation.
- `src_stomate/stomate_lpj.f90`, `StomateLpj`, lines 1115-1131:
  calls `npp_calc` with `gpp_daily`, `f_alloc`, `bm_alloc`,
  `resp_maint_part`, `biomass`, `resp_maint`, `resp_growth`, and `npp_daily`.

Daily reset:

- `src_stomate/stomate.f90`, `stomate_main`, lines 4950-5000:
  resets accumulated daily fields including `gpp_daily(:,:)=zero`,
  `resp_maint_part(:,:,:)=zero`, and many forcing/litter/carbon accumulators.
- `src_stomate/stomate.f90`, `stomate_main`, lines 5034-5036:
  maps STOMATE respiration outputs to SECHIBA-facing outputs after daily work:
  `resp_maint = resp_maint_radia * veget_cov_max`,
  `resp_growth = resp_growth_d * veget_cov_max * dt_sechiba / one_day`.

History writes relevant to validation:

- `src_stomate/stomate_lpj.f90`, lines 1675-1707:
  XIOS sends `LAI`, `GPP`, pool masses, `MAINT_RESP`, `GROWTH_RESP`, and AGR
  maintenance diagnostics.
- `src_stomate/stomate_npp.f90`, lines 681-720:
  writes `BM_ALLOC_*`, `SLA_CALC`, `NPP_ABOVE`, and `NPP_BELOW`.

## Required Trace Columns

### 1. Before/after `stomate_accu` for GPP

Trace at `stomate.f90` lines 3198-3208 and `stomate_accu_r2d` lines 9365-9387.

Required columns:

- `itime`, `date`, `day`, `sec`, `do_slow`
- `dt_sechiba`, `dt_stomate`, `dt_days`
- `gpp_d_before_accu[npts,nvm]`
- `gpp_daily_before_accu[npts,nvm]`
- `gpp_daily_after_accu[npts,nvm]`
- PFT14 active slice: `gpp_d[:,14]`, `gpp_daily[:,14]`

Purpose: validate `field_out += field_in * dt_sechiba` and slow-step division
by `dt_stomate`. This is scheduling parity only, not carbon-process parity.

### 2. After `maint_respiration` and accumulation

Trace at `stomate.f90` lines 3244-3267 plus `stomate_resp.f90` lines
122-170, 203-232, 234-303, 319-376.

Required columns:

- `biomass_before_maint[npts,nvm,nparts,nelements]`
- `lai_before_maint[npts,nvm]` and `lai_after_maint[npts,nvm]`
- `t2m[npts]`, `t2m_longterm[npts]`, `stempdiag[npts,nslm]`
- `rprof[npts,nvm]`, `sla_calc[npts,nvm]`
- `resp_maint_part_radia_after[npts,nvm,nparts]`
- `resp_maint_radia_after_sum[npts,nvm]`
- `resp_maint_part_before_accum[npts,nvm,nparts]`
- `resp_maint_part_after_accum[npts,nvm,nparts]`
- PFT14 AGR slices:
  `resp_maint_part_radia[:,14,iagrsapst/iagrsappn/iagrhrtst/iagrhrtpn]`

Purpose: validate maintenance kernel and STOMATE accumulation semantics. Do not
infer missing coefficients from these outputs.

### 3. Before/after `alloc`

Trace at `stomate_lpj.f90` lines 1093-1102 and `stomate_alloc.f90` lines
146-152, 187-201, 335-759, 760-834.

Required columns:

- `biomass_before_alloc[npts,nvm,nparts,nelements]`
- `biomass_after_alloc[npts,nvm,nparts,nelements]`
- `lai_before_alloc[npts,nvm]`
- `senescence[npts,nvm]`
- `moiavail_week[npts,nvm]`, `tsoil_month[npts,nslm]`,
  `soilhum_month[npts,nslm]`
- `age[npts,nvm]`, `leaf_age[npts,nvm,nleafages]`,
  `leaf_frac[npts,nvm,nleafages]`
- `when_growthinit[npts,nvm]`, `rprof[npts,nvm]`, `sla_calc[npts,nvm]`
- internal allocation ratios:
  `LtoLSR[npts]`, `StoLSR[npts]`, `RtoLSR[npts]`, `alloc_sap_above[npts]`
- output `f_alloc[npts,nvm,nparts]`
- PFT14 AGR fractions:
  `f_alloc[:,14,isapabove/iagrsapst/iagrsappn/isapbelow/iroot/icarbres]`

Purpose: unlock full allocation parity. Until this trace exists, `f_alloc`
must remain an explicit input and must not be fabricated.

### 4. Before/after `npp_calc`

Trace at `stomate_lpj.f90` lines 1115-1131 and `stomate_npp.f90` lines
116-180, 230-247, 280-295, 299-386, 447-531.

Required columns:

- `biomass_before_npp[npts,nvm,nparts,nelements]`
- `biomass_after_npp[npts,nvm,nparts,nelements]`
- `gpp_daily[npts,nvm]`
- `f_alloc[npts,nvm,nparts]`
- `resp_maint_part[npts,nvm,nparts]`
- `PFTpresent[npts,nvm]`
- `bm_alloc_after[npts,nvm,nparts,nelements]`
- `resp_maint_after[npts,nvm]`
- `resp_growth_after[npts,nvm]`
- `npp_daily_after[npts,nvm]`
- `leaf_age_before_after`, `leaf_frac_before_after`, `age_before_after`
- AGR pools before/after:
  `isapabove`, `iheartabove`, `iagrsapst`, `iagrsappn`, `iagrhrtst`,
  `iagrhrtpn`

Purpose: validate closed `npp_calc` algebra and identify where the current
Phase 1B kernel must stop before leaf-age/SLA bookkeeping.

## Current JAX Phase 1D Boundary

Implemented helpers:

- `stomate_accumulate_daily`: exact `stomate_accu` algebra.
- `stomate_normalize_by_bio_fraction`, `stomate_veget_cover_fractions`,
  `stomate_vegetnew_firstday`, and `stomate_veget_cov_max_new`: exact local
  entry normalization algebra for the current-year and next-year non-bio
  denominators. Current trace coverage remains partial unless audited
  `totfrac_nobio`/`totfrac_nobio_new` are supplied explicitly.
- `accumulate_resp_maint_part`: exact `resp_maint_part` accumulation.
- `sum_resp_maint_radia`: exact post-`maint_respiration` part summation,
  including bare-soil zeroing and optional flood-root diagnostic.
- `reset_daily_on_slow`: exact explicit-field reset behavior for named fields.
- `prepare_daily_carbon_inputs`: trace-ready boundary packer.
- `require_explicit_f_alloc`: guard that prevents placeholder allocation.

Blocked by missing trace:

- Full `alloc`. The 2026-06-23 server trace exposes PFT14 `f_alloc` by part
  in `orchjax_stomate_lpj_trace.txt:after_alloc`, but still lacks the complete
  allocation input state needed to validate `alloc` itself.
- Full `bm_alloc` and process parity for `npp_calc`.
- Full `StomateLpj` or full STOMATE loop.

## 2026-06-23 Server Trace Slice

Local package:
`outputs/server_1961_trace_full_20260623/MANIFEST.txt`.

Record formats inspected:

- `orchjax_stomate_daily_trace.txt`
  - Tags: `gpp_before_accu`, `gpp_after_accu`.
  - Fields: `itime`, `ik`, `pft`, `do_slow`, `dt_sechiba`, `dt_stomate`,
    `dt_days`, `gpp_d`, `gpp_daily`.
  - Provenance: `src_stomate/stomate.f90`, `stomate_main` lines 3198-3208;
    `stomate_accu_r2d` lines 9365-9387.
- `orchjax_stomate_maint_trace.txt`
  - Tag: `maint_after`, one record per `itime/ik/pft/part`.
  - Fields: `itime`, `ik`, `pft`, `part`, `do_slow`, `dt_sechiba`,
    `dt_stomate`, `dt_days`, `lai`, `t2m`, `t2m_longterm`, `height`,
    `sla_calc`, `biomass`, `resp_maint_part_radia`,
    `resp_maint_part_after_accum`, `resp_maint_radia_after_sum`.
  - Provenance: `src_stomate/stomate.f90`, `stomate_main` lines 3244-3267;
    `src_stomate/stomate_resp.f90`, `maint_respiration` lines 122-170,
    203-232, 234-303, 319-376.
- `orchjax_stomate_lpj_trace.txt`
  - Tag: `after_alloc`, one record per `itime/pft/part`.
  - Fields currently named in JAX: `itime`, `pft`, `part`, `veget_cov_max`,
    `f_alloc`, `biomass_after_alloc`, `unmapped_after_alloc_01`,
    `unmapped_after_alloc_02`, `senescence`, `height`, `sla_calc`.
  - This is sufficient to treat `f_alloc` as explicit trace truth for a
    downstream boundary, but not sufficient to close full allocation.
- `orchjax_stomate_npp_trace.txt`
  - Tag: `after_npp`, one record per `itime/pft/part`.
  - Observed fields include after-NPP part state and PFT-level diagnostics, but
    the trace does not expose the full pre-`npp_calc` state or full `bm_alloc`
    by element needed to validate `npp_closed_update` end to end.

Validated lightweight parity:

- `tests/parity/test_stomate_server_1961_trace_slice.py` verifies the first
  PFT14 `gpp_before_accu`/`gpp_after_accu` record with exact
  `stomate_accumulate_daily` algebra.
- The same test reads the first nonzero PFT14 `maint_after` 12-part group and
  validates `sum_resp_maint_radia` and `accumulate_resp_maint_part` exactly.
- The first PFT14 `after_alloc` 12-part group sums `f_alloc` to 1.0 and can
  clear the `prepare_daily_carbon_inputs` `f_alloc` blocker, while
  `biomass_before_alloc`, `bm_alloc`, and full process parity remain blocked.
