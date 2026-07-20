# Server 1961 Fixed-Format Trace Contract

Scope: local truth base for the fixed-format trace package under
`outputs/server_1961_trace_full_20260623`. These traces are validation truth
only. They do not replace Fortran source truth, and the JAX reader must not
fabricate missing process state.

The STOMATE traces in this package are process/kernel truth for the server run
that produced them. They are not a whole-driver parity sentinel for a local run
initialized from `reference/case_001_071/stomate_start.nc`: that local restart
has PFT14 leaf biomass `236.51477764695636`, while the first server
`stomate_lpj.after_alloc` PFT14 leaf record has `biomass_after_alloc`
`50.9244200281159`. Driver-level comparisons must use a trace generated from
the same initial restart set, or initialize the JAX driver from the server run's
matching restart payload.

## Package Provenance

Local manifest:

- `outputs/server_1961_trace_full_20260623/MANIFEST.txt`
- Created UTC: `2026-06-23T14:55:24Z`
- Source root:
  `/public/share/qhcess/qhcess2/User/jcwu/orchidee_man_jax_trace_clean_20260619_012142`
- Run output:
  `/public/share/qhcess/qhcess2/User/jcwu/orchidee_man_jax_trace_clean_20260619_012142/outputs/run_1961_trace_full`
- Run script:
  `/public/share/qhcess/qhcess2/User/jcwu/orchidee_man_jax_trace_clean_20260619_012142/scripts/run_trace_full_1961.ksh`
- Patched Fortran build tree used by the run script:
  `/public/share/qhcess/qhcess2/User/jcwu/orchidee_man_jax_trace_clean_20260619_012142/modipsl_250919_dev`

Local run evidence included in the package:

- `logs/run_trace_full_1961*.log`
- `logs/out_orchidee_1961.txt`
- `run/run.def`, `run/run.def.1961`, `run/used_run.def`
- restart/history NetCDF files under `netcdf/`

The patched Fortran files that emitted `orchjax_*` traces are not included as
patch diffs in this local package. Therefore this contract cites the trace
package and the relevant Fortran module/subroutine path where known, but does
not claim exact patch insertion line numbers unless they were already audited
elsewhere.

## Fixed-Format Reader Contract

Reader implementation:

- `jax_orchidee.trace.fixed_format.stream_records`
- `jax_orchidee.trace.server_1961`
- `jax_orchidee.trace.server_1961.find_server_record` for streaming first
  match by tag plus key indices such as `itime`, `ji`, `jv`, and `isl`/`jsl`.

Format:

- A record starts with a textual tag token, for example `pre`, `main`,
  `post_layer`, or `gpp_before_accu`.
- The record continues over subsequent numeric/logical lines until the next
  textual tag.
- Values are parsed as Fortran-style integers, floats, or `T`/`F` booleans.
- The reader supports `tags=...` filtering and `limit=...` streaming so large
  files such as `orchjax_hydrol_main_trace.txt` are not materialized.

## Minimal Supported Schemas

The local schema registry supports:

- `driver_forcing`: `orchjax_driver_forcing_trace.txt`, tag `first_forcing`.
  Relevant Fortran source truth: `dim2_driver.f90` and `readdim2.f90` forcing
  selection path.
- `grid`: `orchjax_grid_trace.txt`, tag `grid_scatter`. Relevant Fortran
  source truth: `grid.f90` and `haversine.f90` geometry/scatter path.
- `intersurf_boundary`: `orchjax_intersurf_boundary_trace.txt`, tag
  `initialize`. Relevant Fortran source truth: `intersurf.f90` initialization
  boundary path.
- `intersurf_main`: `orchjax_intersurf_main_trace.txt`, tag `main`.
  Relevant Fortran source truth: `intersurf.f90` main boundary path.
- `slowproc_soilt`: `orchjax_slowproc_soilt_trace.txt`, tag `soil_overlap`.
  Relevant Fortran source truth: `slowproc.f90`, `slowproc_soilt`, plus
  `interpol_help.f90` overlap aggregation.
- `slowproc_read_data`: `orchjax_slowproc_read_data_trace.txt`, tag `tide`.
  Relevant Fortran source truth: `slowproc.f90`, `slowproc_read_data`.
- `slowproc_read_annual`: `orchjax_slowproc_read_annual_trace.txt`, tag
  `salinity`. Relevant Fortran source truth: `slowproc.f90`,
  `slowproc_read_annual`.
- `hydrol_main`: `orchjax_hydrol_main_trace.txt`, tag `pre`.
  Relevant Fortran source truth: `hydrol.f90`, `hydrol_soil` pre-solve/RHS
  path; prior audited spans are summarized in
  `docs/source_audits/hydrol_full_solve_trace_contract.md`.
- `hydrol_post`: `orchjax_hydrol_post_trace.txt`, tags `post_tile` and
  `post_layer`. Relevant Fortran source truth: `hydrol.f90`, `hydrol_soil`
  post-solve/drainage path.
- `hydrol_update`: `orchjax_hydrol_update_trace.txt`, tag `mc_after_update`.
  Relevant Fortran source truth: `hydrol.f90`, `hydrol_soil` layer update path.
- `hydrol_alt`: `orchjax_hydrol_alt_trace.txt`, tag `alt_first_solve`.
  Relevant Fortran source truth: `hydrol.f90`, alternate first solve path.
- `hydrol_alt_residual`: `orchjax_hydrol_alt_residual_trace.txt`, tag
  `alt_residual`. Relevant Fortran source truth: `hydrol.f90`, alternate
  residual-boundary path.
- `stomate_daily`: `orchjax_stomate_daily_trace.txt`, tags
  `gpp_before_accu` and `gpp_after_accu`. Fortran provenance:
  `src_stomate/stomate.f90`, `stomate_main` lines 3198-3208 and
  `stomate_accu_r2d` lines 9365-9387.
- `stomate_maint`: `orchjax_stomate_maint_trace.txt`, tag `maint_after`.
  Relevant Fortran source truth: `stomate.f90` maintenance scheduling and
  `stomate_resp.f90`, `maint_respiration`.
- `stomate_npp`: `orchjax_stomate_npp_trace.txt`, tag `after_npp`. Relevant
  Fortran source truth: `stomate_lpj.f90`, `StomateLpj`, and
  `stomate_npp.f90`, `npp_calc`.
- `stomate_lpj`: `orchjax_stomate_lpj_trace.txt`, tag `after_alloc`.
  Relevant Fortran source truth: `stomate_lpj.f90`, `StomateLpj`, and
  `stomate_alloc.f90`, `alloc`.

## Still Unnamed Or Partially Audited

Some fixed-format fields are intentionally preserved under `unmapped_*` names
in `jax_orchidee.trace.server_1961`:

- `hydrol_main.pre`: `unmapped_pre_01` through `unmapped_pre_05`
- `hydrol_post.post_tile`: `unmapped_post_tile_01`
- `grid.grid_scatter`: `unmapped_grid_01`, a sentinel-like `1.0e20` field
  whose emitted variable name is not in the local package.
- `slowproc_soilt.soil_overlap`: `unmapped_soilt_01` through
  `unmapped_soilt_04`; these are likely intermediate aggregation diagnostics
  and require the server patch for exact names.
- `hydrol_alt_residual.alt_residual`: `unmapped_alt_residual_01`
- `stomate_lpj.after_alloc`: `unmapped_lpj_01`
- `stomate_npp.after_npp`: `unmapped_npp_01` and `unmapped_npp_02`

Those values are parsed and testable as trace truth, but their exact Fortran
variable names should be audited against the server patch before they are used
as process contracts.

Not yet registered in this file after the second pass:

- No known package traces remain unregistered. Some registered tags are still
  only minimally named as listed above.
