"""Bounded, source-extracted Fortran thermal initialization/consumer oracle.

No model driver, restart file factory, remote command or historical producer is
executed. The reference compiles the original thermal procedures, initialization
statement spans, and HYDROL moisture/temperature spans. Synthetic raw state and
explicit frozen boundary inputs are shared with current production JAX calls.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

from fortran_oracle_common import (  # noqa: E402
    COMPILE_FLAGS, DEFAULT_COMPILER, compiler_environment, float_comparison,
    write_point_comparisons,
)

THERMAL = ROOT / "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90"
HYDROL = ROOT / "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90"
SECHIBA = ROOT / "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"
PARAMETERS = ROOT / "fortran_source/ORCHIDEE/src_parameters/constantes_soil_var.f90"
FROZEN = ROOT / "outputs/reference_mode/micro_oracles"
COEFFICIENTS = {
    "soilcap": (1,), "soilcap_pft": (1, 14), "soilflx": (1,),
    "soilflx_pft": (1, 14), "cgrnd": (1, 31, 14),
    "dgrnd": (1, 31, 14), "cgrnd_snow": (1, 3),
    "dgrnd_snow": (1, 3), "lambda_snow": (1,),
}
MOISTURE_FIELDS = (
    "shumdiag_perma", "mc_layh", "mcl_layh", "tmc_layh",
    "mc_layh_pft", "mcl_layh_pft", "tmc_layh_pft",
)
HUMLEV_FIELDS = (
    "mc_layt", "mcl_layt", "tmc_layt", "mc_layt_pft",
    "mcl_layt_pft", "tmc_layt_pft", "shum_ngrnd_perma",
)
SPANS: list[dict[str, object]] = []


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, record: object) -> None:
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def span(path: Path, first: int, last: int, name: str) -> str:
    data = b"".join(path.read_bytes().splitlines(keepends=True)[first - 1:last])
    SPANS.append({"name": name, "path": path.relative_to(ROOT).as_posix(),
                  "first": first, "last": last, "sha256": hashlib.sha256(data).hexdigest()})
    return data.decode("utf-8")


def routine(path: Path, name: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    first = next(i for i, line in enumerate(lines, 1)
                 if re.match(rf"\s*SUBROUTINE\s+{name}\s*\(", line, re.I))
    last = next(i for i, line in enumerate(lines[first - 1:], first)
                if re.match(rf"\s*END\s+SUBROUTINE\s+{name}\b", line, re.I))
    return span(path, first, last, name)


def source_table(name: str) -> np.ndarray:
    text = PARAMETERS.read_text(encoding="utf-8")
    body = re.search(rf"::\s*{name}\s*=\s*&[^\n]*\n(.*?)/\)", text, re.I | re.S)
    if body is None:
        raise ValueError(f"source table not found: {name}")
    literals = re.findall(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?)_r_std", body[1])
    values = np.asarray([float(x.replace("D", "e").replace("d", "e")) for x in literals])
    if values.shape != (12,):
        raise ValueError(f"unexpected source table dimensions: {name}: {values.shape}")
    return values


def make_base() -> tuple[dict[str, np.ndarray], dict[str, object]]:
    from jax_orchidee.parameters.vertical_soil import vertical_soil_init
    from jax_orchidee.sechiba.thermosoil import QZ_USDA, SMCMAX_USDA, SO_CAPA_DRY_NS_USDA
    geometry = vertical_soil_init()
    if (geometry.nslm, geometry.ngrnd) != (11, 32):
        raise ValueError("This bounded oracle declares nslm=11, ngrnd=32")
    cold_path = FROZEN / "thermosoil_cold_coef_actual/production_inputs.npz"
    transition_path = FROZEN / "thermosoil_first_transition_actual/production_inputs.npz"
    with np.load(cold_path) as cold, np.load(transition_path) as transition:
        base = {name: cold[name].copy() for name in (
            "QZ", "SMCMAX", "mcs", "so_capa_dry_ns", "zlt", "dlt",
            "dz1", "refSOC", "organic_layer_thick", "soilc_total", "scalars",
        )}
        for name in ("cgrnd", "dgrnd", "cgrnd_snow", "dgrnd_snow"):
            base[f"rest_{name}"] = transition[f"{name}_previous"].copy()
        restored_capacity = transition["soilcap_initial"].copy()
    # These are explicit boundary arrays, not claimed as historical Fortran results.
    base.update({
        "znt": geometry.znt, "znh": geometry.znh,
        "hydro_dz": geometry.dnh * 1000., "hydro_dh": geometry.dlh * 1000.,
        # control_initialize:595 uses THERMAL NODES, not layer boundaries.
        "diaglev": geometry.znt[:11],
        "mc": .22 + .006 * np.arange(11)[None, :, None] + .015 * np.arange(3)[None, None, :],
        "resdist": np.asarray([[.2, .35, .45]]), "vegtot_old": np.asarray([.8]),
        "pref_soil_veg": (np.arange(14) % 3 + 1).astype(np.int32),
        "njsc": np.asarray([7], np.int32),
        # getdiff:2805 rejects PFT-dependent frozen fractions. Vary depth while
        # keeping the phase identical across PFTs, as required by the source.
        "ptn": np.broadcast_to(np.linspace(271., 277., 32)[None, :, None], (1, 32, 14)).copy(),
        "shum_ngrnd_permalong": np.broadcast_to(np.linspace(.4, .7, 32)[None, :, None], (1, 32, 14)).copy(),
        "veget_max": np.zeros((1, 14)), "snow": np.asarray([0.]),
        "temp_sol_new": np.asarray([279.25]),
        "temp_sol_new_pft": 279.25 + .01 * np.arange(14)[None, :],
        "post_temp_sol": np.asarray([281.75]),
        "snowdz": np.zeros((1, 3)), "snowrho": np.asarray([[200., 250., 300.]]),
        "snowtemp": np.asarray([[268., 269., 270.]]), "pb": np.asarray([950.]),
        "frac_snow_veg": np.asarray([.375]), "frac_snow_nobio": np.asarray([[.125]]),
        "totfrac_nobio": np.asarray([.125]), "ok_lai_i": np.zeros((14,), np.int32),
        "rest_soilcap": restored_capacity * 1.125,
        "rest_soilcap_pft": np.broadcast_to(restored_capacity[:, None], (1, 14)).copy(),
        "rest_soilflx": np.asarray([11.25]), "rest_soilflx_pft": np.full((1, 14), 11.25),
        "rest_lambda_snow": np.asarray([.125]),
    })
    base["mcl"] = .65 * base["mc"]
    base["veget_max"][0, [0, 1, 13]] = [.1, .25, .5]
    base["ok_lai_i"][-1] = 1
    for field, expected in (("QZ", QZ_USDA), ("SMCMAX", SMCMAX_USDA),
                            ("so_capa_dry_ns", SO_CAPA_DRY_NS_USDA)):
        original = source_table({"QZ": "QZ_usda", "SMCMAX": "SMCMAX_usda",
                                 "so_capa_dry_ns": "so_capa_dry_ns_usda"}[field])
        np.testing.assert_array_equal(base[field], original)
        np.testing.assert_array_equal(np.asarray(expected), original)
    np.testing.assert_array_equal(base["mcs"], source_table("mcs_usda"))
    np.testing.assert_array_equal(base["zlt"], geometry.zlt)
    np.testing.assert_array_equal(base["dlt"], geometry.dlt)
    np.testing.assert_array_equal(base["dz1"], 1. / np.diff(geometry.znt))
    return base, {
        "frozen_boundaries": [{"path": str(p), "sha256": sha(p)} for p in (cold_path, transition_path)],
        "input_classes": {
            "frozen_geometry_tables": ["zlt", "dlt", "dz1", "QZ", "SMCMAX", "mcs", "so_capa_dry_ns", "scalars"],
            "frozen_data_boundaries": ["refSOC", "organic_layer_thick", "soilc_total"],
            "JAX_derived_frozen_restart_boundaries": ["rest_cgrnd", "rest_dgrnd", "rest_cgrnd_snow", "rest_dgrnd_snow", "rest_soilcap (scaled)", "rest_soilcap_pft"],
            "synthetic_source_admissible_state": "raw mc != mcl, resdist != soiltile, vegtot_old, ptn, long humidity, vegetation, snow, pressure, surface temperatures and other restored coefficients",
            "original_Fortran_outputs": "only outputs.bin generated in this package's case directory",
            "JAX_outputs": "only jax.npz; never fed to the Fortran reference",
        },
        "dimensions": {"npts": 1, "ngrd": 32, "nslm": 11, "nvm": 14, "nstm": 3, "nsnow": 3},
        "geometry_authority": "vertical_soil_init default geometry, checked against unchanged frozen geometry; not a new geometry oracle",
    }


def make_cases(base: dict[str, np.ndarray], case_set: str = "standard") -> list[tuple[str, dict[str, np.ndarray], dict[str, object]]]:
    definitions = [("complete_no_snow", 0., None), ("complete_thin_partial", .0005, None),
        ("complete_thin_whole", .005, None), ("complete_thin_upper", .01, None),
        ("complete_above_thin", .010001, None), ("complete_thick", .04, None),
        ("cold_all_coefficients_missing", .005, "all"),
        *[(f"missing_{name}", .005, name) for name in COEFFICIENTS],
        ("complete_cooling", .005, None)]
    if case_set == "sparse-vegetation":
        definitions = [("complete_sparse_vegetation", 0., None)]
    result = []
    for name, depth, missing in definitions:
        values = {k: np.array(v, copy=True) for k, v in base.items()}
        values["snowdz"][:] = depth / 3.
        values["snow"][:] = np.sum(values["snowdz"] * values["snowrho"], axis=1)
        if name == "complete_cooling":
            values["post_temp_sol"][:] = 276.75
        if name == "complete_sparse_vegetation":
            values["veget_max"][0, [0, 1, 13]] = [.05, .125, .25]
        if missing == "all":
            values["ptn"][:] = 280.
            values["shum_ngrnd_permalong"][:] = 1.
        for coefficient in COEFFICIENTS:
            if missing in ("all", coefficient):
                values[f"rest_{coefficient}"][:] = 999999.
        result.append((name, values, {"missing_coefficient": missing, "snow_depth_m": depth,
            "source_phase": "final coef overwrites thin-snow" if missing else "restored coefficients retained; final var_init thin-snow retained",
            "cold_default_boundary": missing == "all"}))
    return result


def production_energy_kwargs(operands: dict[str, object], profile: object,
                             caller: str = "thermosoil_explicit_step") -> dict[str, object]:
    """Evaluate the actual production energy call operands without a full step."""
    from jax_orchidee.sechiba import thermosoil as th
    tree = ast.parse(Path(th.__file__).read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == caller)
    call = next(node for node in ast.walk(function) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "thermosoil_energy_diagnostics")
    namespace = {**operands, "profile": profile, "energy_pcapa_en": operands["pcapa_en_previous"],
        "SN_CAPA": th.SN_CAPA, "thermosoil_energy_diagnostics": lambda **kwargs: kwargs,
        "getdiff": SimpleNamespace(pcapa_en=operands["pcapa_en_previous"]),
        "thermosoil_veget_max_bg": th.thermosoil_veget_max_bg}
    return eval(compile(ast.Expression(call), str(th.__file__), "eval"), namespace)


def compile_unit(inputs: dict[str, np.ndarray]) -> tuple[str, list[tuple[str, tuple[int, ...]]]]:
    header = (ROOT / "scripts/dev/oracle_thermosoil_cold_coef_actual.f90.template").read_text().split("contains", 1)[0]
    header = header.replace("thermosoil_cold_coef_actual_oracle", "thermal_init_oracle")
    header = header.replace("real(r_std), parameter :: so_capa_ice=", "real(r_std) :: so_capa_ice=")
    declarations = """
  real(r_std), parameter :: so_cond_wet=1.89_r_std, val_exp=999999._r_std
  real(r_std) :: znt(ngrnd), dz5(ngrnd-1), ptn_beg(kjpindex,ngrnd,nvm),temp_sol_beg(kjpindex)
  real(r_std) :: shum_ngrnd_perma(kjpindex,ngrnd,nvm), surfheat_incr(kjpindex),coldcont_incr(kjpindex)
  real(r_std) :: cgrnd(kjpindex,ngrnd-1,nvm),dgrnd(kjpindex,ngrnd-1,nvm)
  real(r_std) :: pcappa_supp(kjpindex,ngrnd,nvm)
  logical :: satsoil=.false.
  interface restget_p
    module procedure rest_rank1,rest_rank2,rest_rank3
  end interface
"""
    for name, shape in COEFFICIENTS.items():
        declarations += f"  real(r_std) :: rest_{name}({','.join(map(str, shape))})\n"
    stubs = """contains
  subroutine ipslerr_p(level,a,b,c,d)
    integer,intent(in)::level
    character(*),intent(in)::a,b,c,d
    if(level>=3) then
      write(*,*)a,b,c,d
      error stop 'unexpected ipslerr_p'
    endif
  end subroutine
  subroutine ioconf_setatt_p(a,b)
    character(*),intent(in)::a,b
  end subroutine
  subroutine thermosoil_getdiff_old_thermix_with_snow(kjpindex_in,snow,njsc)
    integer(i_std),intent(in)::kjpindex_in,njsc(kjpindex_in)
    real(r_std),intent(in)::snow(kjpindex_in)
    error stop 'outside explicit snow domain'
  end subroutine
"""
    for rank in (1, 2, 3):
        colons = ",".join(":" for _ in range(rank))
        stubs += f"""
  subroutine rest_rank{rank}(rest_id,name,nbp,n1,n2,kjit,flag,values,gather,nbpg,index_g)
    integer,intent(in)::rest_id,nbp,n1,n2,kjit,nbpg,index_g(:)
    logical,intent(in)::flag
    character(*),intent(in)::name,gather
    real(r_std),intent(out)::values({colons})
    select case(name)
"""
        for name, shape in COEFFICIENTS.items():
            if len(shape) == rank:
                stubs += f"    case('{name}'); values=rest_{name}\n"
        stubs += "    case default; error stop 'unknown injected restart field'\n    end select\n  end subroutine\n"
    thermal_routines = "\n".join(routine(THERMAL, name) for name in (
        "thermosoil_cond_pft", "thermosoil_cond_nopft", "thermosoil_cond",
        "thermosoil_getdiff", "thermosoil_coef", "thermosoil_humlev",
        "thermosoil_getdiff_thinsnow", "thermosoil_diaglev", "thermosoil_var_init",
        "thermosoil_profile", "thermosoil_energy"))
    hydro = """
module hydro_oracle
  use defprec
  implicit none
  integer,parameter::kjpindex=1,nslm=11,nstm=3,nsnow=3
  real(r_std),parameter::zero=0._r_std,un=1._r_std,trois=3._r_std,huit=8._r_std,min_sechiba=1.e-8_r_std,sn_dens=330._r_std
  logical::peat_hydro=.false.,ok_explicitsnow=.true.
  real(r_std)::soilmoist(1,11),shumdiag_perma(1,11),mc_layh(1,11),mcl_layh(1,11)
  real(r_std)::mc_layh_s(1,11,3),mcl_layh_s(1,11,3),mcs(1),mcs_mineral(12)
  real(r_std)::temp_hydro(1,11),zz(11),diaglev(11)
contains
  subroutine moisture_init(mc,mcl,resdist,vegtot_old,njsc,dz,dh)
    real(r_std),intent(in)::mc(1,11,3),mcl(1,11,3),resdist(1,3),vegtot_old(1),dz(11),dh(11)
    integer,intent(in)::njsc(1)
    integer::jst,ji,jsl
"""
    hydro += span(HYDROL, 4073, 4073, "hydrol_default_mcs")
    hydro += span(HYDROL, 4518, 4535, "hydrol_initial_soilmoist")
    hydro += span(HYDROL, 4603, 4613, "hydrol_initial_shumdiag")
    hydro += span(HYDROL, 4632, 4646, "hydrol_initial_volumetric_moisture")
    hydro += "\n  end subroutine\n" + routine(HYDROL, "hydrol_calculate_temp_hydro") + "\nend module\n"
    program = """
program thermal_initialization_driver
  use defprec
  use thermal_init_oracle
  use hydro_oracle,only: moisture_init,hydrol_calculate_temp_hydro, &
       hsoilmoist=>soilmoist,hshum=>shumdiag_perma,hmc=>mc_layh,hmcl=>mcl_layh, &
       mc_layh_s,mcl_layh_s,hydro_mcs=>mcs_mineral,temp_hydro,hzz=>zz,hdiaglev=>diaglev
  implicit none
  integer::iu,ou,jv,jg,m,rest_id=0,nbp_glo=1,kjit=0,index_g(1)=[1]
  logical::calculate_coef,is_crop_soil(3)
  real(r_std)::shumdiag_perma(1,11),mc_layh(1,11),mcl_layh(1,11),tmc_layh(1,11),soilmoist(1,11)
  real(r_std)::mc_layh_pft(1,11,14),mcl_layh_pft(1,11,14),soilmoist_pft(1,11,14),tmc_layh_pft(1,11,14),thawed_humidity(1)
  real(r_std)::stempdiag(1,11),profil_froz(1,32,14),veget_max_bg(1,14)
  real(r_std)::soilcap(1),soilcap_pft(1,14),soilflx(1),soilflx_pft(1,14)
  real(r_std)::lambda_snow(1),cgrnd_snow(1,3),dgrnd_snow(1,3)
  character(1024)::input_path,output_path
"""
    module_names = {"QZ", "SMCMAX", "mcs", "so_capa_dry_ns", "zlt", "dlt", "dz1", "refSOC", "znt", "ptn", "shum_ngrnd_permalong", *[f"rest_{x}" for x in COEFFICIENTS]}
    for name, values in inputs.items():
        if name in module_names:
            continue
        kind = "integer(i_std)" if values.dtype.kind in "iu" else "real(r_std)"
        program += f"  {kind} :: {name}({','.join(map(str, values.shape))})\n"
    program += """
  call get_command_argument(1,input_path)
  call get_command_argument(2,output_path)
  open(newunit=iu,file=trim(input_path),access='stream',form='unformatted',status='old',action='read')
"""
    for name in inputs:
        program += f"  read(iu) {name}\n"
    program += """
  close(iu)
  dt_sechiba=scalars(1);lambda=scalars(2);phigeoth=scalars(3);psnowdzmin=scalars(4)
  use_toporganiclayer_tempdiff=.false.;use_soilc_tempdiff=.true.;use_refSOC=.true.
  ok_freeze_thermix=.true.;ok_explicitsnow=.true.;ok_LAIdev=ok_lai_i/=0;veget_mask_2d=.true.
  ptn_pftmean=0.;thawed_humidity=1.;cgrnd_snow=0.;dgrnd_snow=0.;lambda_snow=0.
  hydro_mcs=mcs;hzz=znh*1000.;hdiaglev=diaglev
  call moisture_init(mc,mcl,resdist,vegtot_old,njsc,hydro_dz,hydro_dh)
  shumdiag_perma=hshum;mc_layh=hmc;mcl_layh=hmcl;tmc_layh=hsoilmoist;soilmoist=hsoilmoist
  is_crop_soil=.false.
"""
    program += span(SECHIBA, 704, 714, "sechiba_initial_soil_to_pft_mapping")
    program += "  tmc_layh_pft=soilmoist_pft\n"
    program += "\n  open(newunit=ou,file=trim(output_path),access='stream',form='unformatted',status='replace',action='write')\n"
    layout: list[tuple[str, tuple[int, ...]]] = []

    def dump(prefix: str, fields: list[tuple[str, str, tuple[int, ...]]]) -> None:
        nonlocal program
        for name, expression, shape in fields:
            program += f"  write(ou) {expression}\n"
            layout.append((f"{prefix}.{name}", shape))

    dump("moisture", [(n, "soilmoist_pft" if n == "tmc_layh_pft" else n, (1, 11, 14) if n.endswith("_pft") else (1, 11)) for n in MOISTURE_FIELDS])
    program += span(THERMAL, 570, 573, "thermal_begin_memory")
    program += span(THERMAL, 602, 610, "thermal_initial_pftmean")
    # Exact early phase, then source restart restoration and final unconditional var_init.
    program += span(THERMAL, 624, 640, "thermal_early_var_init_and_coef")
    program += span(THERMAL, 665, 723, "thermal_coefficient_reads_and_final_var_init")
    dump("transient", [(n, n, (1, 32, 14)) for n in ("pcapa", "pcapa_en", "pkappa")])
    program += span(THERMAL, 725, 742, "thermal_conditional_final_coef")
    dump("humlev", [(n, n, (1, 32, 14) if n.endswith("_pft") or n == "shum_ngrnd_perma" else (1, 32)) for n in HUMLEV_FIELDS])
    dump("initial", [(n, n, s) for n, s in COEFFICIENTS.items()])
    dump("initial", [(n, n, (1, 32, 14)) for n in ("pcapa", "pcapa_en", "pkappa", "profil_froz", "pcappa_supp", "ptn_beg")])
    dump("initial", [("temp_sol_beg", "temp_sol_beg", (1,)), ("stempdiag", "stempdiag", (1, 11)), ("veget_max_bg", "veget_max_bg", (1, 14))])
    program += "  call hydrol_calculate_temp_hydro(1,stempdiag,snow,snowdz)\n"
    dump("hydrol_entry", [("stempdiag", "stempdiag", (1, 11)), ("temp_hydro", "temp_hydro", (1, 11))])
    program += """
  call thermosoil_profile(1,post_temp_sol,temp_sol_new_pft,ptn,stempdiag, &
      snowtemp,veget_max,frac_snow_veg,frac_snow_nobio,totfrac_nobio,cgrnd_snow,dgrnd_snow)
  call thermosoil_energy(1,post_temp_sol,soilcap,veget_max_bg)
"""
    dump("consumer", [("ptn", "ptn", (1, 32, 14)), ("stempdiag", "stempdiag", (1, 11)),
        ("surfheat_incr", "surfheat_incr", (1,)), ("coldcont_incr", "coldcont_incr", (1,)),
        ("ptn_beg", "ptn_beg", (1, 32, 14)), ("temp_sol_beg", "temp_sol_beg", (1,))])
    program += "  close(ou)\nend program\n"
    return header + declarations + stubs + thermal_routines + "\nend module\n" + hydro + program, layout


def jax_results(values: dict[str, np.ndarray], case: dict[str, object]) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    import jax
    from jax_orchidee.driver.restart_state import restart_thermosoil_initial_fields
    from jax_orchidee.sechiba import hydrol as hy
    from jax_orchidee.sechiba import thermosoil as th

    v = values
    grid = SimpleNamespace(znh=v["znh"], dnh=v["hydro_dz"] / 1000., dlh=v["hydro_dh"] / 1000.,
        znt=v["znt"], zlt=v["zlt"], dlt=v["dlt"], dz1=v["dz1"], dz5=(v["zlt"][:-1] - v["znt"][:-1]) * v["dz1"])
    source = {"veget": v["veget_max"], "veget_max": v["veget_max"], "njsc": v["njsc"],
        "mc": v["mc"], "mcl": v["mcl"], "water2infilt": np.asarray([[.1, .2, .3]]),
        "resdist": v["resdist"], "vegtot_old": v["vegtot_old"], "qsintveg": np.zeros((1, 14)),
        "drysoil_frac": np.asarray([.2]), "ptn": v["ptn"], "refsoc": v["refSOC"],
        "temp_sol": v["temp_sol_new"], "snowdz": v["snowdz"], "snowrho": v["snowrho"],
        "snowtemp": v["snowtemp"], "shum_ngrnd_permalong": v["shum_ngrnd_permalong"]}
    context = SimpleNamespace(run_def_values={"OK_EXPLICITSNOW": "TRUE", "USE_SOILC_TEMPDIFF": "TRUE",
        "use_refSOC": "TRUE", "OK_FREEZE_THERMIX": "TRUE"},
        hydrol_ok_pc=True, hydrol_ok_leak=False, hydrol_soil_peat_hydro=False,
        hydrol_tides=False, hydrol_agri_peat=False, ok_freeze_cwrr=True, hydrol_depth_max_h=2.,
        first_step_bundle=SimpleNamespace(vegetation=SimpleNamespace(soiltile=np.asarray([[.5, .3, .2]]))),
        cwrr_grid=grid, hydrol_humcste=np.full((14,), 1.), run_scalars=SimpleNamespace(pref_soil_veg=v["pref_soil_veg"]),
        hydrol_dz_mm=v["hydro_dz"], hydrol_dh_mm=v["hydro_dh"], thermosoil_satsoil=False,
        nvm=14, ok_laidev=v["ok_lai_i"].astype(bool))
    hydro = hy.hydrol_var_init(veget=source["veget"], veget_max=v["veget_max"],
        soiltile=context.first_step_bundle.vegetation.soiltile, njsc=v["njsc"],
        znh_m=grid.znh, dnh_m=grid.dnh, dlh_m=grid.dlh, humcste=context.hydrol_humcste,
        altmax=np.full((1, 14), 2.), mc=v["mc"], mcl=v["mcl"], water2infilt=source["water2infilt"],
        resdist=v["resdist"], vegtot_old=v["vegtot_old"], qsintveg=source["qsintveg"],
        drysoil_frac=source["drysoil_frac"], ok_pc=True, ok_freeze_cwrr=True)
    moisture = hy.hydrol_cold_start_thermosoil_moisture_inputs(hydro, pref_soil_veg=v["pref_soil_veg"],
        dz_mm=v["hydro_dz"], dh_mm=v["hydro_dh"], njsc=v["njsc"], mcs=hydro.mcs)
    coefficient_inputs = dict(moisture=moisture, temp_sol_new_pft=v["temp_sol_new_pft"],
        snowdz=v["snowdz"], snowrho=v["snowrho"], snowtemp=v["snowtemp"], njsc=v["njsc"], pb=v["pb"],
        dlt=v["dlt"], dz1=v["dz1"], zlt=v["zlt"], znt=v["znt"], dz5=grid.dz5,
        dt_sechiba=float(v["scalars"][0]), frac_snow_veg=v["frac_snow_veg"],
        frac_snow_nobio=v["frac_snow_nobio"], totfrac_nobio=v["totfrac_nobio"], ok_laidev=context.ok_laidev)
    closure = th.thermosoil_cold_start_coef_closure(ptn=v["ptn"], temp_sol_new=v["temp_sol_new"],
        veget_max=v["veget_max"], refsoc=v["refSOC"], use_refSOC=True, use_soilc_tempdiff=True,
        shum_ngrnd_permalong=v["shum_ngrnd_permalong"], **coefficient_inputs)
    restart = {name: v[f"rest_{name}"] for name in COEFFICIENTS}
    restart.update(ptn=v["ptn"], refSOC=v["refSOC"], shum_ngrnd_prmlng=v["shum_ngrnd_permalong"])
    if case["cold_default_boundary"]:
        restart.pop("ptn")
        restart.pop("shum_ngrnd_prmlng")
    initialized = th.thermosoil_initialize(ngrnd=32, nscm=12, temp_sol_new=v["temp_sol_new"],
        veget_max=v["veget_max"], coefficient_inputs=coefficient_inputs, restart_fields=restart)
    thin = th.thermosoil_getdiff_thinsnow(ptn=v["ptn"], shum_ngrnd_permalong=v["shum_ngrnd_permalong"],
        snowdz=v["snowdz"], pcapa=closure.getdiff.pcapa, pcapa_en=closure.getdiff.pcapa_en,
        pkappa=closure.getdiff.pkappa, profil_froz=closure.getdiff.profil_froz,
        veget_mask_2d=np.ones((1, 14), bool), zlt=v["zlt"])
    final_material = closure.getdiff if initialized.calculate_coef else thin
    # Current public initialize result does not carry pcapa/en/kappa; verify the
    # actual admitted C34 helper and the cold closure used by production owners.
    helper = None
    if not initialized.calculate_coef:
        helper = restart_thermosoil_initial_fields(source, context=context, payload=SimpleNamespace(pb=v["pb"]), altmax=np.full((1, 14), 2.))
    state = SimpleNamespace(ptn=initialized.ptn, refsoc=initialized.refsoc,
        **{name: getattr(initialized, name) for name in COEFFICIENTS},
        gtemp=np.asarray([123.]), temp_sol_beg=initialized.temp_sol_beg,
        pcapa_en=helper["pcapa_en"] if helper else final_material.pcapa_en,
        veget_mask_2d=initialized.veget_mask_2d, shum_ngrnd_permalong=initialized.shum_ngrnd_permalong)
    captured: dict[str, object] = {}

    class BoundaryCaptured(Exception):
        pass

    def capture(**kwargs):
        captured.update(kwargs)
        raise BoundaryCaptured

    # Execute the real first-owner argument assembler, stop at its component
    # delegate, and then execute only profile + energy with those exact operands.
    with patch.object(th, "thermosoil_explicit_step", capture):
        try:
            th.run_thermosoil_first_step_module(moisture=moisture, thermosoil_restart=state,
                enerbil_payload={"temp_sol_new": v["post_temp_sol"], "temp_sol_new_pft": v["temp_sol_new_pft"], "soilcap": initialized.soilcap},
                condveg_result=SimpleNamespace(frac_snow_veg=v["frac_snow_veg"], frac_snow_nobio=v["frac_snow_nobio"]),
                snowrho=v["snowrho"], snowtemp=v["snowtemp"], snowdz=v["snowdz"], njsc=v["njsc"],
                veget_max=v["veget_max"], pb=v["pb"], totfrac_nobio=v["totfrac_nobio"],
                dlt=v["dlt"], dz1=v["dz1"], zlt=v["zlt"], znt=v["znt"], dz5=grid.dz5,
                dt_sechiba=float(v["scalars"][0]), ok_laidev=context.ok_laidev)
        except BoundaryCaptured:
            pass
    if not captured:
        raise AssertionError("real owner did not reach its thermal component boundary")
    profile = th.thermosoil_profile_explicit_snow(**{k: captured[k] for k in (
        "ptn", "cgrnd", "dgrnd", "cgrnd_snow", "dgrnd_snow", "temp_sol_new", "snowtemp", "veget_max",
        "frac_snow_veg", "frac_snow_nobio", "totfrac_nobio", "veget_mask_2d")}, nslm=11)
    energy_kwargs = production_energy_kwargs(captured, profile)
    energy = th.thermosoil_energy_diagnostics(**energy_kwargs)
    # One narrow shared JIT graph: the first energy consumer, never the model.
    if not hasattr(jax_results, "energy_jit"):
        jax_results.energy_jit = jax.jit(th.thermosoil_energy_diagnostics)
    energy_jit = jax_results.energy_jit(**energy_kwargs)
    _, hydro_diag, temp_hydro = hy.hydrol_first_step_profil_froz_from_arrays(
        ptn=initialized.ptn, veget_max=v["veget_max"], mc=v["mc"], njsc=v["njsc"],
        zz_mm=v["znh"] * 1000., dh_mm=v["hydro_dh"], diaglev_m=v["diaglev"],
        snow=v["snow"], snowdz=v["snowdz"])
    result = {f"moisture.{name}": getattr(moisture, name) for name in MOISTURE_FIELDS}
    result.update({f"humlev.{name}": getattr(closure.humlev, name) for name in HUMLEV_FIELDS})
    result.update({f"transient.{name}": getattr(thin, name) for name in ("pcapa", "pcapa_en", "pkappa")})
    result.update({f"initial.{name}": getattr(initialized, name) for name in COEFFICIENTS})
    result.update({f"initial.{name}": getattr(final_material, name) for name in ("pcapa", "pcapa_en", "pkappa", "profil_froz")})
    for name in ("pcappa_supp", "ptn_beg", "temp_sol_beg", "stempdiag", "veget_max_bg"):
        result[f"initial.{name}"] = getattr(initialized, name)
    result.update({"hydrol_entry.stempdiag": hydro_diag, "hydrol_entry.temp_hydro": temp_hydro,
                   "consumer.ptn": profile.ptn, "consumer.stempdiag": profile.stempdiag})
    result.update({f"consumer.{name}": getattr(energy, name) for name in ("surfheat_incr", "coldcont_incr", "ptn_beg", "temp_sol_beg")})
    extra = {f"jit_energy.{name}": getattr(energy_jit, name) for name in ("surfheat_incr", "coldcont_incr", "ptn_beg", "temp_sol_beg")}
    if helper:
        extra.update({f"c34_helper.{name}": helper[name] for name in ("pcapa_en", "temp_sol_beg", "stempdiag")})
    extra.update({"owner.pcapa_en_previous": captured["pcapa_en_previous"], "owner.temp_sol_beg": captured["temp_sol_beg"], "owner.soilcap_initial": captured["soilcap_initial"]})
    result.update(extra)
    return {k: np.asarray(x, dtype=np.float64) for k, x in result.items()}, {
        "calculate_coef": initialized.calculate_coef, "ptn_source": initialized.ptn_source,
        "cold_branch": bool(np.sum(np.asarray(final_material.pcapa_en)[:, 0, :] * np.asarray(initialized.veget_max_bg)) <= th.SN_CAPA),
        "C34_actual_helper_executed": helper is not None,
        "real_owner_assembler_executed": True, "full_thermal_step_executed": False,
        "first_hydrol_wrapper_executed": True, "narrow_energy_JIT_executed": True,
    }


def run(output: Path, case_set: str = "standard", reuse_reference: Path | None = None) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"Use a new attempt directory, never overwrite evidence: {output}")
    output.mkdir(parents=True)
    (output / "runner.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "invocation.json", {"argv": sys.argv, "cwd": str(Path.cwd()),
        "runner_sha256": sha(Path(__file__)), "python": sys.executable})
    started = time.perf_counter()
    base, metadata = make_base()
    cases = make_cases(base, case_set)
    text, layout = compile_unit(base)
    source = output / "oracle.f90"
    source.write_text(text, encoding="utf-8", newline="\n")
    write_json(output / "source-spans.json", SPANS)
    write_json(output / "output-layout.json", layout)
    write_json(output / "input-origin.json", metadata)
    executable = output / "oracle.exe"
    command = [str(DEFAULT_COMPILER), *COMPILE_FLAGS, str(source.resolve()), "-o", str(executable.resolve())]
    if reuse_reference is not None:
        if sha(reuse_reference / "oracle.f90") != sha(source):
            raise ValueError("Original Fortran compile unit changed; cannot reuse executable")
        shutil.copyfile(reuse_reference / "oracle.exe", executable)
        write_json(output / "compile.json", {"reused_from": str(reuse_reference.resolve()),
            "new_compilation": False, "source_sha256": sha(source), "executable_sha256": sha(executable),
            "original_compile": json.loads((reuse_reference / "compile.json").read_text())})
    else:
        compile_start = time.perf_counter()
        compiled = subprocess.run(command, cwd=output, env=compiler_environment(DEFAULT_COMPILER), text=True, capture_output=True, timeout=60)
        (output / "compile.stdout").write_text(compiled.stdout, encoding="utf-8")
        (output / "compile.stderr").write_text(compiled.stderr, encoding="utf-8")
        write_json(output / "compile.json", {"command": command, "cwd": str(output.resolve()), "returncode": compiled.returncode,
            "elapsed_seconds": time.perf_counter()-compile_start, "source_sha256": sha(source), "compiler_sha256": sha(DEFAULT_COMPILER),
            "compiler_version": subprocess.check_output([str(DEFAULT_COMPILER), "--version"], text=True).splitlines()[0]})
        compiled.check_returncode()
    summaries = []
    import jax
    for name, values, definition in cases:
        case_dir = output / name
        case_dir.mkdir()
        np.savez(case_dir / "inputs.npz", **values)
        binary = case_dir / "inputs.bin"
        with binary.open("wb") as handle:
            for value in values.values():
                handle.write(np.asarray(value, dtype="<i4" if value.dtype.kind in "iu" else "<f8").ravel(order="F").tobytes())
        input_layout = [{"field": k, "shape": list(v.shape), "dtype": str(v.dtype), "array_sha256": hashlib.sha256(v.tobytes()).hexdigest()} for k, v in values.items()]
        write_json(case_dir / "inputs.json", {**definition, "fields": input_layout, "binary_sha256": sha(binary)})
        out_binary = case_dir / "outputs.bin"
        remote_free_command = [str(executable.resolve()), str(binary.resolve()), str(out_binary.resolve())]
        run_start = time.perf_counter()
        completed = subprocess.run(remote_free_command, cwd=case_dir, env=compiler_environment(DEFAULT_COMPILER), text=True, capture_output=True, timeout=20)
        (case_dir / "fortran.stdout").write_text(completed.stdout, encoding="utf-8")
        (case_dir / "fortran.stderr").write_text(completed.stderr, encoding="utf-8")
        write_json(case_dir / "execution.json", {"command": remote_free_command, "cwd": str(case_dir.resolve()),
            "returncode": completed.returncode, "elapsed_seconds": time.perf_counter()-run_start, "executable_sha256": sha(executable)})
        completed.check_returncode()
        raw = np.fromfile(out_binary, dtype="<f8")
        fortran, offset = {}, 0
        for field, shape in layout:
            size = int(np.prod(shape))
            fortran[field] = raw[offset:offset+size].reshape(shape, order="F")
            offset += size
        if offset != raw.size:
            raise ValueError(f"Fortran output size: expected {offset}, found {raw.size}")
        jax_start = time.perf_counter()
        actual, observed = jax_results(values, definition)
        for field in actual.keys() - fortran.keys():
            reference = field.replace("jit_energy.", "consumer.").replace("c34_helper.", "initial.")
            reference = {"owner.pcapa_en_previous": "initial.pcapa_en", "owner.temp_sol_beg": "initial.temp_sol_beg", "owner.soilcap_initial": "initial.soilcap"}.get(reference, reference)
            fortran[field] = fortran[reference]
        if fortran.keys() != actual.keys():
            raise ValueError("Oracle field sets differ")
        np.savez(case_dir / "fortran.npz", **fortran)
        np.savez(case_dir / "jax.npz", **actual)
        comparisons = [float_comparison(k, fortran[k], actual[k], rtol=1.e-12, atol=1.e-12) for k in fortran]
        write_point_comparisons(case_dir / "points.csv", fortran, actual, rtol=1.e-12, atol=1.e-12)
        expected_missing = definition["missing_coefficient"] is not None
        if observed["calculate_coef"] != expected_missing:
            raise AssertionError("Unexpected coefficient source branch")
        summary = {"name": name, **definition, **observed, "status": "passed" if all(c["passed"] for c in comparisons) else "failed",
            "jax_seconds": time.perf_counter()-jax_start, "comparisons": comparisons,
            "fortran_output_sha256": sha(out_binary)}
        write_json(case_dir / "comparison.json", summary)
        summaries.append(summary)
        print(json.dumps({"case": name, "status": summary["status"], "cold_branch": observed["cold_branch"], "failed": [c["name"] for c in comparisons if not c["passed"]]}), flush=True)
    result = {"schema": "thermal-initialization-oracle-v1", "status": "passed" if all(c["status"] == "passed" for c in summaries) else "failed",
        "rtol": 1.e-12, "atol": 1.e-12, "case_count": len(summaries), "field_comparisons": sum(len(s["comparisons"]) for s in summaries),
        "scalar_comparisons": sum(sum(c["count"] for c in s["comparisons"]) for s in summaries),
        "elapsed_seconds": time.perf_counter()-started, "jax_version": jax.__version__, "jax_x64": bool(jax.config.jax_enable_x64),
        "devices": [str(d) for d in jax.devices()], "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (THERMAL, HYDROL, SECHIBA, PARAMETERS)},
        "cases": [{k: v for k, v in s.items() if k != "comparisons"} for s in summaries],
        "limits": ["synthetic same-input component oracle, not exact historical 001/319 replay", "frozen restart coefficients are declared boundary values, not regenerated Fortran truth", "restget_p is a supplied-array seam, no IOIPSL/NetCDF reader compiled", "thermal main delegate is intercepted; actual assembler, profile and energy are executed separately", "default USDA / explicit snow / refSOC / nonpeat HYDROL domain only", "public thermosoil_initialize result lacks material arrays; production C34 helper and cold closure are checked separately", "no full timestep/day/year or whole-model numerical acceptance"],
        "production_patches": [], "verified_ledger_entries": [], "numerical_acceptance": False}
    write_json(output / "comparison.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-set", choices=("standard", "sparse-vegetation"), default="standard")
    parser.add_argument("--reuse-reference", type=Path)
    args = parser.parse_args()
    attempt = args.output.resolve()
    try:
        report = run(attempt, args.case_set, args.reuse_reference)
    except FileExistsError:
        # A rejected existing destination belongs to earlier evidence.
        raise
    except Exception as exc:
        if attempt.is_dir():
            import traceback
            write_json(attempt / "failure.json", {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise
    raise SystemExit(0 if report["status"] == "passed" else 1)
