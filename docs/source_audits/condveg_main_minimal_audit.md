# CONDVEG Main Minimal Audit

Scope: source-equivalent CONDVEG module after ENERBIL for the paper-case
branch. This audit covers only `fortran_source/ORCHIDEE` and does not borrow
logic from external JAX ports.

## Position After ENERBIL

`condveg_main` is the SECHIBA process after ENERBIL/HYDROL and before
THERMOSOIL in the paper-case order audited elsewhere for ENERBIL. Same-step
CONDVEG outputs therefore update later state/diagnostics; they are not used
to manufacture current ENERBIL inputs.

## Source Anchors

- `fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_main` lines
  332-338: interface and output fields `emis`, `albedo`, `frac_snow_veg`,
  `frac_snow_nobio`, `z0m`, `z0h`, `roughheight`, and `roughheight_pft`.
- `condveg_main` lines 398-400: calls `condveg_frac_snow`.
- `condveg_main` lines 402-403: assigns `emis(:)=emis_scal`.
- `condveg_main` lines 407-428: roughness branch. `impaze=true` writes
  prescribed `z0_scal`/`roughheight_scal`; otherwise `rough_dyn=true` calls
  `condveg_z0cdrag_dyn`, and `rough_dyn=false` calls `condveg_z0cdrag`.
- `condveg_main` lines 430-435: calls `condveg_albedo`.
- `condveg_main` lines 447-455: computes mean snow albedo diagnostic as
  `(albedo_snow(:,1)+albedo_snow(:,2))/2` only where `snow>0`, otherwise `0`.
- `condveg_frac_snow` lines 858-898: explicit and default snow-cover
  formulas.
- `condveg_z0cdrag_dyn` lines 1519-1720: dynamic roughness formulas used by
  active `IMPOSE_AZE=false` and `ROUGH_DYN=true`.
- `condveg_z0cdrag` lines 1288-1474: static roughness formulas used when
  `IMPOSE_AZE=false` and `ROUGH_DYN=false`.
- `condveg_albedo` lines 612-839: complete albedo helper, audited as a
  dependency but not run by the minimal `condveg_main_minimal` kernel.

## Active Paper-Case Branches

The local expanded run configuration records:

- `outputs/server_1961_trace_full_20260623/run/used_run.def` line 2448:
  `IMPOSE_AZE = FALSE`.
- `outputs/server_1961_trace_full_20260623/run/used_run.def` line 2456:
  `ROUGH_DYN = TRUE`.
- `outputs/server_1961_trace_full_20260623/run/used_run.def` line 179:
  `OK_EXPLICITSNOW = TRUE`.

Therefore the active minimal branch is:

1. `condveg_frac_snow` explicit-snow formula from lines 879-887 plus
   non-vegetated snow fraction from lines 892-894.
2. `emis(:)=emis_scal` from `condveg_main` line 403. The caller must pass the
   already-audited module value; this minimal after-ENERBIL helper does not
   infer initialization.
3. Dynamic roughness through `condveg_z0cdrag_dyn`, lines 1578-1718.
4. Albedo remains delegated until its constants/state are explicit.

## Implemented JAX Coverage

`jax_orchidee/sechiba/condveg.py` implements:

- `condveg_frac_snow`: exact formulas from `condveg_frac_snow` lines 878-894.
- `condveg_main_emissivity`: direct fill from `condveg_main` lines 402-403.
- `condveg_prescribed_roughness`: branch assignments from `condveg_main`
  lines 407-415, requiring explicit `z0_scal` and `roughheight_scal`.
- `condveg_z0cdrag_dyn`: dynamic roughness algebra from
  `condveg_z0cdrag_dyn` lines 1578-1718 for the source configuration
  `nnobio=1`, `iice=1`.
- `condveg_z0cdrag`: static roughness algebra from `condveg_z0cdrag` lines
  1325-1474 for the source configuration `nnobio=1`, `iice=1`. The helper
  requires explicit PFT module parameters (`is_tree`, `z0_over_height`, and
  `ratio_z0m_z0h`) and `tot_bare_soil`.
- `condveg_albedo_snow_mean`: diagnostic loop from `condveg_main` lines
  447-455, provided only when a caller already has exact `albedo_snow`.
- `condveg_main_minimal`: source-order composition for snow fraction,
  emissivity, and the active dynamic roughness branch.
- `condveg_main_minimal_coverage`: audit helper that reports albedo and
  unported/static roughness gaps rather than filling placeholders.

Relevant constant anchors:

- `constantes_var.f90` lines 371-392: `pb_std`, `ZeroCelsius`, `ct_karman`.
- `constantes_var.f90` line 460: `min_wind`.
- `constantes_var.f90` lines 688-723: `height_displacement`, `z0_bare`,
  `z0_ice`, `Cdrag_foliage`, `Prandtl`, `Ct`, `c1`, `c2`, `c3`.
- `constantes_var.f90` lines 185-188: `ivis`, `inir`, `nnobio=1`,
  `iice=1`.
- `constantes_soil_var.f90` lines 97-99: `sn_dens`.

## Deliberate Gaps

`albedo`, `albedo_snow`, `alb_bare`, and `alb_veget` are not emitted by
`condveg_main_minimal`. The `condveg_albedo` helper depends on runtime/static
state such as `alb_bg_modis`, `alb_bare_model`, soil albedo arrays, leaf
albedo arrays, snow albedo arrays, `fixed_snow_albedo`, and ice albedo. Those
inputs are explicit in lines 670-835, but not all are supplied by the current
minimal after-ENERBIL interface. Filling them with defaults would be a hidden
approximation.

The `ROUGH_DYN=false` branch calls `condveg_z0cdrag` at `condveg_main` lines
423-425 and is not active for the paper case. It is now source-backed when
the caller supplies explicit PFT roughness parameters and `tot_bare_soil`;
missing inputs remain an explicit coverage failure rather than a placeholder.

The `IMPOSE_AZE=true` branch is implemented only when the caller supplies the
exact prescribed scalars. Missing `z0_scal` or `roughheight_scal` is treated
as missing input, matching the no-placeholder rule.
