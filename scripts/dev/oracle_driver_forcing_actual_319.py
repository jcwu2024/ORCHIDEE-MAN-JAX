from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.domain import (  # noqa: E402
    PaperPointForcingCache,
    _forcing_model_step_from_cached_point,
    load_case_config,
    read_domain_grid,
)
from jax_orchidee.driver.forcing_interpolation import (  # noqa: E402
    ForcingInterpolationRecord,
    ForcingInterpolationState,
    forcing_read_interpol_transition,
)
from scripts.dev.fortran_oracle_common import (  # noqa: E402
    DEFAULT_COMPILER,
    compiler_environment,
)


FAMILY = "driver_forcing_actual_319"
LANDPOINT_ID = "319.0-057.0"
CONFIG = ROOT / "configs/orchidee_man_250919.yaml"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_driver/readdim2.f90"
SOLAR_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/solar.f90"
STUBS = ROOT / "scripts/dev/oracle_lane_driver_forcing_owner_stubs.f90"
TEMPLATE = ROOT / "scripts/dev/oracle_driver_forcing_actual_319.f90.template"
OUTPUT_DIR = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
RUN_DEF = (
    ROOT
    / "outputs/paper_250919_materialized_run_defs/arg2_1.0/319.0-057.0"
    / "I5/S26_69.834_0.0019_0.3980_97.918/used_run.def"
)

YEAR = 1961
NT = 1460
SPLIT = 12
NB_SPREAD = 6
NSTEPS = 17520
DT_FORCE = 21600.0
RTOL = 1.0e-12
ATOL = 1.0e-12
SOLAR_ROUNDOFF_LEDGER_ATOL = 2.3e-7

RAW_FIELDS = (
    "Tair",
    "PSurf",
    "Qair",
    "Wind_N",
    "Wind_E",
    "Rainf",
    "Snowf",
    "LWdown",
    "SWdown",
)
REAL_FIELDS = (
    "Tair",
    "Qair",
    "PSurf",
    "Wind_N",
    "Wind_E",
    "Wind",
    "Rainf",
    "Snowf",
    "LWdown",
    "SWdown",
    "state.mean_coszang",
    "state.julian_for",
    "state.julian0",
    "state.previous.Tair",
    "state.current.Tair",
    "state.previous.Qair",
    "state.current.Qair",
    "state.previous.PSurf",
    "state.current.PSurf",
    "state.previous.Wind_N",
    "state.current.Wind_N",
    "state.previous.Wind_E",
    "state.current.Wind_E",
    "state.previous.LWdown",
    "state.current.LWdown",
    "state.previous.SWdown",
    "state.current.SWdown",
    "state.current.Rainf",
    "state.current.Snowf",
)
INTEGER_FIELDS = (
    "state.last_read",
    "state.itau_read_nm1",
    "state.itau_read_n",
)
PUBLIC_FIELDS = REAL_FIELDS[:10]


def _read_run_def_limits() -> dict[str, float]:
    values: dict[str, float] = {}
    wanted = {
        "LIMIT_WEST": "west",
        "LIMIT_EAST": "east",
        "LIMIT_SOUTH": "south",
        "LIMIT_NORTH": "north",
    }
    for raw in RUN_DEF.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw:
            continue
        key, value = (part.strip() for part in raw.split("=", 1))
        if key in wanted:
            values[wanted[key]] = float(value.replace("D", "E").replace("d", "e"))
    if set(values) != set(wanted.values()):
        raise ValueError(f"incomplete point-319 domain limits in {RUN_DEF}: {values}")
    return values


def _actual_inputs() -> tuple[PaperPointForcingCache, object, dict[str, object]]:
    limits = _read_run_def_limits()
    domain = read_domain_grid(CONFIG, YEAR, domain_override=limits)
    if domain.iim != 1 or domain.jjm != 1 or domain.nbindex != 1:
        raise ValueError(
            f"expected point-319 forcing zoom to be 1x1 land, got "
            f"{domain.iim}x{domain.jjm}, nbindex={domain.nbindex}"
        )
    i0, j0 = (int(value) for value in domain.forcing_indices_zero_based[0])
    config = load_case_config(CONFIG)
    forcing_path = Path(
        config["drivers"]["atmospheric_forcing"]["file_pattern"].format(year=YEAR)
    )
    source_dtypes: dict[str, str] = {}
    with xr.open_dataset(forcing_path, decode_times=False) as dataset:
        records: dict[str, np.ndarray] = {}
        for name in RAW_FIELDS:
            variable = dataset[name]
            source_dtypes[name] = str(variable.dtype)
            values = np.asarray(
                variable.isel(latitude=j0, longitude=i0).values,
                dtype=np.float64,
            )
            if values.shape != (NT,):
                raise ValueError(f"{name}: expected {(NT,)}, got {values.shape}")
            records[name] = values.reshape((NT, 1, 1))
        areas = np.asarray(dataset["Areas"].isel(latitude=j0, longitude=i0).values, dtype=np.float64).reshape((1, 1))
        contfrac = np.asarray(dataset["contfrac"].isel(latitude=j0, longitude=i0).values, dtype=np.float64).reshape((1, 1))
        height_lev1 = float(np.asarray(dataset["Height_Lev1"].values))
        height_levuv = float(np.asarray(dataset["Height_Levuv"].values))
    cache = PaperPointForcingCache(
        **records,
        Areas=areas,
        contfrac=contfrac,
        Height_Lev1=height_lev1,
        Height_Levuv=height_levuv,
    )
    metadata = {
        "landpoint_id": LANDPOINT_ID,
        "year": YEAR,
        "forcing_file": str(forcing_path),
        "forcing_file_size": forcing_path.stat().st_size,
        "forcing_indices_zero_based": [i0, j0],
        "zoom_shape": [domain.iim, domain.jjm],
        "kindex": np.asarray(domain.kindex, dtype=np.int32).tolist(),
        "longitude": float(domain.lon[0, 0]),
        "latitude": float(domain.lat[0, 0]),
        "contfrac": float(contfrac[0, 0]),
        "height_lev1": height_lev1,
        "height_levuv": height_levuv,
        "raw_record_count": NT,
        "model_step_count": NSTEPS,
        "dt_force_seconds": DT_FORCE,
        "dt_model_seconds": DT_FORCE / SPLIT,
        "split": SPLIT,
        "nb_spread": NB_SPREAD,
        "source_dtypes": source_dtypes,
        "domain_limits": limits,
    }
    return cache, domain, metadata


def _write_input(path: Path, cache: PaperPointForcingCache, domain: object) -> str:
    ordered = (
        np.asarray([domain.lon[0, 0]], dtype=np.float64),
        np.asarray([domain.lat[0, 0]], dtype=np.float64),
        np.asarray(cache.contfrac, dtype=np.float64),
        np.asarray([cache.Height_Lev1], dtype=np.float64),
        np.asarray([cache.Height_Levuv], dtype=np.float64),
        *(np.asarray(getattr(cache, name), dtype=np.float64)[:, 0, 0] for name in RAW_FIELDS),
    )
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for values in ordered:
            payload = np.ascontiguousarray(values, dtype=np.float64).tobytes()
            handle.write(payload)
            digest.update(payload)
    return digest.hexdigest()


def _span(path: Path, start: int, end: int) -> bytes:
    return b"".join(path.read_bytes().splitlines(keepends=True)[start - 1 : end])


def _generated_solar_source() -> bytes:
    return (
        b"module solar\nuse defprec\nuse constantes\nuse ioipsl_para\n"
        b"implicit none\ncontains\n"
        + _span(SOLAR_SOURCE, 55, 268)
        + b"\nend module solar\n"
    )


def _generated_readdim2_source() -> tuple[bytes, bytes]:
    owner = _span(SOURCE, 660, 1687)
    landind = _span(SOURCE, 1899, 1968)
    zoom = _span(SOURCE, 2034, 2051)
    terminator = b"END SUBROUTINE forcing_read_interpol"
    if owner.count(terminator) != 1:
        raise ValueError("forcing_read_interpol terminator drift")
    observation = b"""
   ! Oracle observation hook: copies local SAVE state without changing formulas.
   IF (.NOT. ((itau_read == 0).AND.(itau_split == 0))) THEN
      oracle_last_read=last_read
      oracle_itau_read_nm1=itau_read_nm1
      oracle_itau_read_n=itau_read_n
      oracle_mean_coszang=mean_coszang(1,1)
      oracle_julian_for=julian_for
      oracle_julian0=julian0
      oracle_tair_nm1=tair_nm1(1,1); oracle_tair_n=tair_n(1,1)
      oracle_qair_nm1=qair_nm1(1,1); oracle_qair_n=qair_n(1,1)
      oracle_pb_nm1=pb_nm1(1,1); oracle_pb_n=pb_n(1,1)
      oracle_u_nm1=u_nm1(1,1); oracle_u_n=u_n(1,1)
      oracle_v_nm1=v_nm1(1,1); oracle_v_n=v_n(1,1)
      oracle_lwdown_nm1=lwdown_nm1(1,1); oracle_lwdown_n=lwdown_n(1,1)
      oracle_swdown_nm1=swdown_nm1(1,1); oracle_swdown_n=swdown_n(1,1)
      oracle_rainf_n=rainf_n(1,1); oracle_snowf_n=snowf_n(1,1)
   ENDIF
"""
    owner_instrumented = owner.replace(terminator, observation + terminator)
    declarations = b"""module readdim2
  use ioipsl_para
  use weather
  use constantes
  use solar
  use grid
  use mod_orchidee_para
  implicit none
  integer,save :: iim_full=1,jjm_full=1,llm_full=1,ttm_full=1460
  integer,save :: iim_zoom=1,jjm_zoom=1,i_test=1,j_test=1
  integer,save,allocatable :: i_index(:),j_index(:)
  real,save,allocatable :: data_full(:,:)
  logical,save :: interpol=.true.,daily_interpol=.false.,is_watchout=.false.
  real,save :: oracle_lon=0.,oracle_lat=0.,oracle_contfrac=1.
  real,save :: oracle_height_lev1=2.,oracle_height_levuv=10.
  real,save,allocatable :: oracle_tair(:),oracle_psurf(:),oracle_qair(:)
  real,save,allocatable :: oracle_wind_n(:),oracle_wind_e(:)
  real,save,allocatable :: oracle_rainf(:),oracle_snowf(:)
  real,save,allocatable :: oracle_lwdown(:),oracle_swdown(:)
  integer,save :: oracle_last_read=0,oracle_itau_read_nm1=0,oracle_itau_read_n=0
  real,save :: oracle_mean_coszang=0.,oracle_julian_for=0.,oracle_julian0=0.
  real,save :: oracle_tair_nm1=0.,oracle_tair_n=0.,oracle_qair_nm1=0.,oracle_qair_n=0.
  real,save :: oracle_pb_nm1=0.,oracle_pb_n=0.,oracle_u_nm1=0.,oracle_u_n=0.
  real,save :: oracle_v_nm1=0.,oracle_v_n=0.,oracle_lwdown_nm1=0.,oracle_lwdown_n=0.
  real,save :: oracle_swdown_nm1=0.,oracle_swdown_n=0.,oracle_rainf_n=0.,oracle_snowf_n=0.
contains
"""
    backend = b"""
  subroutine oracle_load(path)
    character(len=*),intent(in)::path
    integer::unit
    allocate(oracle_tair(ttm_full),oracle_psurf(ttm_full),oracle_qair(ttm_full))
    allocate(oracle_wind_n(ttm_full),oracle_wind_e(ttm_full))
    allocate(oracle_rainf(ttm_full),oracle_snowf(ttm_full))
    allocate(oracle_lwdown(ttm_full),oracle_swdown(ttm_full))
    open(newunit=unit,file=path,access='stream',form='unformatted',status='old',action='read')
    read(unit) oracle_lon,oracle_lat,oracle_contfrac,oracle_height_lev1,oracle_height_levuv
    read(unit) oracle_tair,oracle_psurf,oracle_qair,oracle_wind_n,oracle_wind_e
    read(unit) oracle_rainf,oracle_snowf,oracle_lwdown,oracle_swdown
    close(unit)
  end subroutine oracle_load

  subroutine flinquery_var(fid,name,exists)
    integer,intent(in)::fid
    character(len=*),intent(in)::name
    logical,intent(out)::exists
    select case(trim(name))
    case('Wind_N','Wind_E','contfrac'); exists=.true.
    case default; exists=.false.
    end select
  end subroutine flinquery_var

  subroutine forcing_just_read(iim,jjm,zlev,zlevuv,ttm,itb,ite,swdown,rainf,snowf,tair,u,v,qair,pb,lwdown,swnet,eair,peta,peqa,petb,peqb,cdrag,ccanopy,force_id,wind_n_exists,check)
    integer,intent(in)::iim,jjm,ttm,itb,ite,force_id
    logical,intent(in)::wind_n_exists,check
    real,intent(out)::zlev(iim,jjm),zlevuv(iim,jjm),swdown(iim,jjm)
    real,intent(out)::rainf(iim,jjm),snowf(iim,jjm),tair(iim,jjm)
    real,intent(out)::u(iim,jjm),v(iim,jjm),qair(iim,jjm),pb(iim,jjm)
    real,intent(out)::lwdown(iim,jjm),swnet(iim,jjm),eair(iim,jjm)
    real,intent(out)::peta(iim,jjm),peqa(iim,jjm),petb(iim,jjm),peqb(iim,jjm)
    real,intent(out)::cdrag(iim,jjm),ccanopy(iim,jjm)
    if(itb/=ite .or. itb<1 .or. itb>ttm_full) error stop 'invalid forcing record index'
    zlev=oracle_height_lev1; zlevuv=oracle_height_levuv
    tair=oracle_tair(itb); pb=oracle_psurf(itb); qair=oracle_qair(itb)
    u=oracle_wind_n(itb); v=oracle_wind_e(itb)
    rainf=oracle_rainf(itb); snowf=oracle_snowf(itb)
    lwdown=oracle_lwdown(itb); swdown=oracle_swdown(itb)
    swnet=0.; eair=0.; peta=0.; peqa=0.; petb=0.; peqb=0.; cdrag=0.; ccanopy=0.
  end subroutine forcing_just_read

  subroutine forcing_just_read_tmax(iim,jjm,ttm,itb,ite,tmax,force_id)
    integer,intent(in)::iim,jjm,ttm,itb,ite,force_id
    real,intent(out)::tmax(iim,jjm)
    tmax=oracle_tair(itb)
  end subroutine forcing_just_read_tmax

  subroutine flinget_buffer(fid,name,ii,jj,ll,tt,itb,ite,data)
    integer,intent(in)::fid,ii,jj,ll,tt,itb,ite
    character(len=*),intent(in)::name
    real,intent(out)::data(:,:)
    if(trim(name)=='contfrac') then
      data=oracle_contfrac
    else
      data=0.
    endif
  end subroutine flinget_buffer

end module readdim2
"""
    return declarations + owner_instrumented + b"\n" + landind + b"\n" + zoom + backend, owner


def _compile_and_run(work: Path, input_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    generated_solar = work / "solar_actual_319.f90"
    generated_readdim2 = work / "readdim2_actual_319.f90"
    harness = work / "oracle_driver_forcing_actual_319.f90"
    executable = work / "oracle_driver_forcing_actual_319.exe"
    real_output = work / "fortran_real_outputs.bin"
    integer_output = work / "fortran_integer_outputs.bin"
    solar_bytes = _generated_solar_source()
    readdim2_bytes, owner_bytes = _generated_readdim2_source()
    generated_solar.write_bytes(solar_bytes)
    generated_readdim2.write_bytes(readdim2_bytes)
    harness.write_bytes(TEMPLATE.read_bytes())
    command = [
        str(DEFAULT_COMPILER),
        "-cpp",
        "-fdefault-real-8",
        "-ffree-line-length-none",
        "-O0",
        "-fcheck=all",
        "-ffpe-trap=invalid,zero,overflow",
        str(STUBS),
        str(generated_solar),
        str(generated_readdim2),
        str(harness),
        "-o",
        str(executable),
    ]
    environment = compiler_environment(DEFAULT_COMPILER)
    built = subprocess.run(
        command,
        cwd=work,
        env=environment,
        text=True,
        capture_output=True,
    )
    if built.returncode:
        raise RuntimeError(
            "Fortran Oracle compilation failed:\n"
            + built.stdout
            + "\n"
            + built.stderr
        )
    run = subprocess.run(
        [str(executable), str(input_path), str(real_output), str(integer_output)],
        cwd=work,
        env=environment,
        text=True,
        capture_output=True,
    )
    if run.returncode:
        raise RuntimeError(
            "Fortran Oracle execution failed:\n" + run.stdout + "\n" + run.stderr
        )
    real_values = np.fromfile(real_output, dtype=np.float64)
    integer_values = np.fromfile(integer_output, dtype=np.int32)
    expected_real = len(REAL_FIELDS) * NSTEPS
    expected_integer = len(INTEGER_FIELDS) * NSTEPS
    if real_values.size != expected_real or integer_values.size != expected_integer:
        raise ValueError(
            f"unexpected Fortran output sizes: real={real_values.size}/{expected_real}, "
            f"integer={integer_values.size}/{expected_integer}"
        )
    version = subprocess.run(
        [str(DEFAULT_COMPILER), "--version"],
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()[0]
    metadata = {
        "compiler": str(DEFAULT_COMPILER),
        "compiler_version": version,
        "compile_command": command,
        "compile_stdout": built.stdout,
        "compile_stderr": built.stderr,
        "run_stdout": run.stdout,
        "run_stderr": run.stderr,
        "source": str(SOURCE),
        "source_span": "readdim2.f90::forcing_read_interpol lines 660-1687",
        "source_span_sha256": hashlib.sha256(owner_bytes).hexdigest(),
        "solar_source": str(SOLAR_SOURCE),
        "solar_span": "solar.f90::solarang lines 55-181; time_zone lines 201-268",
        "solar_compile_span_sha256": hashlib.sha256(_span(SOLAR_SOURCE, 55, 268)).hexdigest(),
        "generated_readdim2_sha256": hashlib.sha256(readdim2_bytes).hexdigest(),
        "generated_solar_sha256": hashlib.sha256(solar_bytes).hexdigest(),
        "template_sha256": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(),
        "observation_hook": "read-only copy of forcing_read_interpol local SAVE state before return",
    }
    return (
        real_values.reshape((len(REAL_FIELDS), NSTEPS), order="F"),
        integer_values.reshape((len(INTEGER_FIELDS), NSTEPS), order="F"),
        metadata,
    )


def _jax_outputs(cache: PaperPointForcingCache, domain: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    records = tuple(
        ForcingInterpolationRecord.from_mapping(
            {
                "tair": cache.Tair[index],
                "pb": cache.PSurf[index],
                "qair": cache.Qair[index],
                "u": cache.Wind_N[index],
                "v": cache.Wind_E[index],
                "rainf": cache.Rainf[index],
                "snowf": cache.Snowf[index],
                "swdown": cache.SWdown[index],
                "lwdown": cache.LWdown[index],
                "zlev": np.full((1, 1), cache.Height_Lev1),
                "zlevuv": np.full((1, 1), cache.Height_Levuv),
            }
        )
        for index in range(NT)
    )
    state = forcing_read_interpol_transition(
        records,
        ForcingInterpolationState(),
        itauin=0,
        itau_split=0,
        split=SPLIT,
        nb_spread=NB_SPREAD,
        dt_force=DT_FORCE,
        lon=domain.lon,
        lat=domain.lat,
        daily_interpol=False,
        is_watchout=False,
    ).state
    owner = np.empty((len(REAL_FIELDS), NSTEPS), dtype=np.float64)
    owner_integer = np.empty((len(INTEGER_FIELDS), NSTEPS), dtype=np.int32)
    production = np.empty((len(PUBLIC_FIELDS), NSTEPS), dtype=np.float64)
    for model_step in range(NSTEPS):
        transition = forcing_read_interpol_transition(
            records,
            state,
            itauin=model_step // SPLIT + 1,
            itau_split=model_step % SPLIT + 1,
            split=SPLIT,
            nb_spread=NB_SPREAD,
            dt_force=DT_FORCE,
            lon=domain.lon,
            lat=domain.lat,
            daily_interpol=False,
            is_watchout=False,
        )
        state = transition.state
        values = transition.values
        previous = state.previous
        current = state.current
        if previous is None or current is None or state.mean_coszang is None:
            raise RuntimeError(f"incomplete JAX interpolation state at step {model_step + 1}")
        owner[:, model_step] = (
            float(values.tair[0, 0]),
            float(values.qair[0, 0]),
            float(values.pb[0, 0]),
            float(values.u[0, 0]),
            float(values.v[0, 0]),
            float(np.hypot(values.u[0, 0], values.v[0, 0])),
            float(values.rainf[0, 0]),
            float(values.snowf[0, 0]),
            float(values.lwdown[0, 0]),
            float(values.swdown[0, 0]),
            float(state.mean_coszang[0, 0]),
            float(state.julian_for),
            float(state.julian0),
            float(previous.tair[0, 0]),
            float(current.tair[0, 0]),
            float(previous.qair[0, 0]),
            float(current.qair[0, 0]),
            float(previous.pb[0, 0]),
            float(current.pb[0, 0]),
            float(previous.u[0, 0]),
            float(current.u[0, 0]),
            float(previous.v[0, 0]),
            float(current.v[0, 0]),
            float(previous.lwdown[0, 0]),
            float(current.lwdown[0, 0]),
            float(previous.swdown[0, 0]),
            float(current.swdown[0, 0]),
            float(current.rainf[0, 0]),
            float(current.snowf[0, 0]),
        )
        owner_integer[:, model_step] = (
            state.last_read,
            state.itau_read_nm1,
            state.itau_read_n,
        )
        direct = _forcing_model_step_from_cached_point(
            cache,
            model_tstep=model_step,
            split=SPLIT,
            nb_spread=NB_SPREAD,
            lon=domain.lon,
            lat=domain.lat,
            land_indices_key=((0, 0),),
            dt_force=DT_FORCE,
        )
        production[:, model_step] = (
            float(direct.Tair[0, 0]),
            float(direct.Qair[0, 0]),
            float(direct.PSurf[0, 0]),
            float(direct.Wind_N[0, 0]),
            float(direct.Wind_E[0, 0]),
            float(np.hypot(direct.Wind_N[0, 0], direct.Wind_E[0, 0])),
            float(direct.Rainf[0, 0]),
            float(direct.Snowf[0, 0]),
            float(direct.LWdown[0, 0]),
            float(direct.SWdown[0, 0]),
        )
    return owner, owner_integer, production


def _comparison(name: str, fortran: np.ndarray, jax: np.ndarray, *, exact: bool = False) -> dict[str, object]:
    fortran = np.asarray(fortran)
    jax = np.asarray(jax)
    if exact:
        passed_mask = fortran == jax
        absolute = np.abs(fortran.astype(np.int64) - jax.astype(np.int64))
        result: dict[str, object] = {
            "name": name,
            "comparison": "exact",
            "count": int(fortran.size),
            "max_abs_error": int(absolute.max(initial=0)),
            "passed": bool(np.all(passed_mask)),
        }
    else:
        absolute = np.abs(fortran - jax)
        relative = absolute / np.maximum(np.abs(fortran), np.finfo(np.float64).tiny)
        passed_mask = np.isclose(fortran, jax, rtol=RTOL, atol=ATOL)
        result = {
            "name": name,
            "comparison": "float64",
            "count": int(fortran.size),
            "rtol": RTOL,
            "atol": ATOL,
            "max_abs_error": float(absolute.max(initial=0.0)),
            "max_rel_error": float(relative.max(initial=0.0)),
            "passed": bool(np.all(passed_mask)),
        }
    if not result["passed"]:
        first = int(np.flatnonzero(~passed_mask)[0])
        result.update(
            {
                "first_mismatch_step": first + 1,
                "first_fortran": float(fortran[first]),
                "first_jax": float(jax[first]),
                "first_abs_error": float(absolute[first]),
            }
        )
    return result


def _write_public_evidence(
    output_dir: Path,
    fortran: np.ndarray,
    jax: np.ndarray,
) -> None:
    with (output_dir / "fortran_outputs.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("step", "field", "value"))
        for step in range(NSTEPS):
            for field_index, field in enumerate(PUBLIC_FIELDS):
                writer.writerow((step + 1, field, f"{fortran[field_index, step]:.17e}"))
    with (output_dir / "point_comparisons.csv").open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("step", "field", "fortran", "jax", "abs_error", "rel_error", "strict_passed"))
        for step in range(NSTEPS):
            for field_index, field in enumerate(PUBLIC_FIELDS):
                expected = fortran[field_index, step]
                actual = jax[field_index, step]
                absolute = abs(expected - actual)
                relative = absolute / max(abs(expected), np.finfo(np.float64).tiny)
                writer.writerow(
                    (
                        step + 1,
                        field,
                        f"{expected:.17e}",
                        f"{actual:.17e}",
                        f"{absolute:.17e}",
                        f"{relative:.17e}",
                        str(bool(np.isclose(expected, actual, rtol=RTOL, atol=ATOL))).lower(),
                    )
                )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache, domain, input_metadata = _actual_inputs()
    with tempfile.TemporaryDirectory(prefix="orchidee_driver_forcing_actual_319_") as raw:
        work = Path(raw)
        input_path = work / "actual_forcing_319.bin"
        input_metadata["selected_input_sha256"] = _write_input(input_path, cache, domain)
        fortran_real, fortran_integer, build = _compile_and_run(work, input_path)
    jax_real, jax_integer, production = _jax_outputs(cache, domain)

    comparisons = [
        _comparison(f"owner.{field}", fortran_real[index], jax_real[index])
        for index, field in enumerate(REAL_FIELDS)
    ]
    comparisons.extend(
        _comparison(
            f"owner.{field}",
            fortran_integer[index],
            jax_integer[index],
            exact=True,
        )
        for index, field in enumerate(INTEGER_FIELDS)
    )
    comparisons.extend(
        _comparison(
            f"production.{field}",
            fortran_real[index],
            production[index],
        )
        for index, field in enumerate(PUBLIC_FIELDS)
    )

    sw_index = PUBLIC_FIELDS.index("SWdown")
    owner_sw_absolute = np.abs(fortran_real[sw_index] - jax_real[sw_index])
    production_sw_absolute = np.abs(fortran_real[sw_index] - production[sw_index])
    solar_roundoff = {
        "ledger_atol": SOLAR_ROUNDOFF_LEDGER_ATOL,
        "owner_max_abs_error": float(owner_sw_absolute.max(initial=0.0)),
        "owner_within_ledger": bool(np.all(owner_sw_absolute <= SOLAR_ROUNDOFF_LEDGER_ATOL)),
        "production_max_abs_error": float(production_sw_absolute.max(initial=0.0)),
        "production_within_ledger": bool(np.all(production_sw_absolute <= SOLAR_ROUNDOFF_LEDGER_ATOL)),
        "policy": "reported separately; strict 1e-12 comparison remains the family gate",
    }
    failed = [item for item in comparisons if not item["passed"]]
    first_mismatch = None
    if failed:
        candidates = [item for item in failed if "first_mismatch_step" in item]
        if candidates:
            earliest = min(candidates, key=lambda item: (int(item["first_mismatch_step"]), str(item["name"])))
            first_mismatch = {
                key: earliest[key]
                for key in (
                    "name",
                    "first_mismatch_step",
                    "first_fortran",
                    "first_jax",
                    "first_abs_error",
                )
            }

    _write_public_evidence(OUTPUT_DIR, fortran_real, jax_real)
    (OUTPUT_DIR / "inputs.json").write_text(
        json.dumps(input_metadata, indent=2) + "\n", encoding="ascii"
    )
    result = {
        "schema_version": 2,
        "family": FAMILY,
        "status": "passed" if not failed else "failed",
        "landpoint_id": LANDPOINT_ID,
        "year": YEAR,
        "comparisons": comparisons,
        "solar_roundoff_ledger": solar_roundoff,
        "first_mismatch": first_mismatch,
        "build": build,
    }
    (OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="ascii"
    )
    summary = {
        "family": FAMILY,
        "status": result["status"],
        "model_steps": NSTEPS,
        "strict_comparisons": len(comparisons),
        "failed_comparisons": len(failed),
        "first_mismatch": first_mismatch,
        "solar_roundoff_ledger": solar_roundoff,
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
