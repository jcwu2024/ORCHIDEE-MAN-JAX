from __future__ import annotations

import json
import csv
import subprocess
import tempfile
from pathlib import Path

from fortran_oracle_common import compiler_environment

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_driver/readdim2.f90"
SOLAR_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_global/solar.f90"
COMPILER = Path(r"C:\msys64\ucrt64\bin\gfortran.exe")

_BASE_FIELDS_FOR_ORACLE = (
    "tair", "qair", "swdown", "rainf", "snowf", "pb", "u", "v",
    "lwdown", "zlev", "zlevuv",
)
_WATCHOUT_FIELDS_FOR_ORACLE = (
    "swnet", "eair", "petacoef", "peqacoef", "petbcoef", "peqbcoef",
    "cdrag", "ccanopy",
)


ORACLE_BACKEND = r"""
  subroutine flinquery_var(fid,name,exists)
    integer,intent(in)::fid; character(len=*),intent(in)::name; logical,intent(out)::exists
    select case(trim(name))
    case('Wind_N'); exists=oracle_wind_n
    case('Wind_E'); exists=.true.
    case('contfrac'); exists=oracle_contfrac
    case('levels'); exists=oracle_watchout
    case('neighboursNN'); exists=oracle_neighbours
    case default; exists=.false.
    end select
  end subroutine
  subroutine oracle_flinget(name,itb,data)
    integer,intent(in)::itb; character(len=*),intent(in)::name; real,intent(out)::data(:,:)
    integer::i,j
    do j=1,size(data,2); do i=1,size(data,1)
      select case(trim(name))
      case('contfrac'); data(i,j)=merge(1.,0.5,i+j/=4)
      case('Tair'); data(i,j)=270.+5.*itb+i+2*j
      case('Tmin'); data(i,j)=265.+3.*itb+i+j
      case('Tmax'); data(i,j)=280.+4.*itb+i+j
      case('Rainf'); data(i,j)=1.e-5*(itb+i)
      case('Snowf'); data(i,j)=1.e-6*(itb+j)
      case('precip'); data(i,j)=1.e-5*(itb+i+j)
      case('SWdown'); data(i,j)=merge(10000.,100.*itb+10.*i+j,oracle_high_solar)
      case('LWdown'); data(i,j)=200.+10.*itb+i+j
      case('levels'); data(i,j)=2.+.25*itb+.1*i
      case('SWnet'); data(i,j)=50.*itb+i+j
      case('Eair'); data(i,j)=1000.*itb+10*i+j
      case('petAcoef'); data(i,j)=.1*itb+.01*i
      case('peqAcoef'); data(i,j)=.2*itb+.01*j
      case('petBcoef'); data(i,j)=.3*itb+.01*i
      case('peqBcoef'); data(i,j)=.4*itb+.01*j
      case('cdrag'); data(i,j)=.005*itb+.001*i
      case('ccanopy'); data(i,j)=350.+itb+i+j
      case('resolutionX'); data(i,j)=1000.*i
      case('resolutionY'); data(i,j)=2000.*j
      case('neighboursNN','neighboursNE','neighboursEE','neighboursSE','neighboursSS','neighboursSW','neighboursWW','neighboursNW'); data(i,j)=merge(1.e30,10.*i+j,i==1.and.j==1)
      case('PSurf'); data(i,j)=90000.+1000.*itb+10.*i+j
      case('Qair'); data(i,j)=.002+.001*itb+.0001*i
      case('Wind_N'); data(i,j)=itb+i
      case('Wind_E'); data(i,j)=-itb-j
      case default; data(i,j)=0.
      end select
    enddo; enddo
  end subroutine
  subroutine forcing_just_read(iim,jjm,zlev,zlevuv,ttm,itb,ite,swdown,rainf,snowf,tair,u,v,qair,pb,lwdown,swnet,eair,peta,peqa,petb,peqb,cdrag,ccanopy,force_id,wind_n_exists,check)
    integer,intent(in)::iim,jjm,ttm,itb,ite,force_id; logical,intent(in)::wind_n_exists,check
    real,intent(out)::zlev(iim,jjm),zlevuv(iim,jjm),swdown(iim,jjm),rainf(iim,jjm),snowf(iim,jjm),tair(iim,jjm),u(iim,jjm),v(iim,jjm),qair(iim,jjm),pb(iim,jjm),lwdown(iim,jjm),swnet(iim,jjm),eair(iim,jjm),peta(iim,jjm),peqa(iim,jjm),petb(iim,jjm),peqb(iim,jjm),cdrag(iim,jjm),ccanopy(iim,jjm)
    call oracle_flinget(merge('Tmin','Tair',daily_interpol),itb,tair)
    call oracle_flinget('SWdown',itb,swdown); call oracle_flinget('LWdown',itb,lwdown)
    call oracle_flinget('PSurf',itb,pb); call oracle_flinget('Qair',itb,qair)
    call oracle_flinget('Wind_N',itb,u); call oracle_flinget('Wind_E',itb,v)
    if(daily_interpol) then; call oracle_flinget('precip',itb,rainf); snowf=0.; else; call oracle_flinget('Rainf',itb,rainf); call oracle_flinget('Snowf',itb,snowf); endif
    if(oracle_watchout) then
      call oracle_flinget('levels',itb,zlev); zlevuv=zlev
      call oracle_flinget('SWnet',itb,swnet); call oracle_flinget('Eair',itb,eair)
      call oracle_flinget('petAcoef',itb,peta); call oracle_flinget('peqAcoef',itb,peqa)
      call oracle_flinget('petBcoef',itb,petb); call oracle_flinget('peqBcoef',itb,peqb)
      call oracle_flinget('cdrag',itb,cdrag); call oracle_flinget('ccanopy',itb,ccanopy)
    else
      zlev=2.;zlevuv=10.;swnet=0.;eair=0.;peta=0.;peqa=0.;petb=0.;peqb=0.;cdrag=0.;ccanopy=0.
    endif
  end subroutine
  subroutine forcing_just_read_tmax(iim,jjm,ttm,itb,ite,tmax,force_id)
    integer,intent(in)::iim,jjm,ttm,itb,ite,force_id; real,intent(out)::tmax(iim,jjm); call oracle_flinget('Tmax',itb,tmax)
  end subroutine
  subroutine flinget_buffer(fid,name,ii,jj,ll,tt,itb,ite,data)
    integer,intent(in)::fid,ii,jj,ll,tt,itb,ite; character(len=*),intent(in)::name; real,intent(out)::data(:,:)
    call oracle_flinget(name,itb,data)
  end subroutine
"""


def _generated_source() -> str:
    lines = SOURCE.read_text(encoding="utf-8", errors="replace").splitlines(True)
    owner = "".join(lines[659:1687])
    landind = "".join(lines[1898:1968])
    zoom = "".join(lines[2033:2051])
    declarations = r"""module readdim2
  use ioipsl_para
  use weather
  use constantes
  use solar
  use grid
  use mod_orchidee_para
  implicit none
  integer,save :: iim_full=2,jjm_full=2,llm_full=1,ttm_full=4,iim_zoom=2,jjm_zoom=2,i_test=1,j_test=1
  integer,save,allocatable :: i_index(:),j_index(:)
  real,save,allocatable :: data_full(:,:)
  logical,save :: interpol=.true.,daily_interpol=.false.,is_watchout=.false.
  logical,save :: oracle_watchout=.false.,oracle_contfrac=.true.,oracle_neighbours=.true.
  logical,save :: oracle_wind_n=.true.,oracle_high_solar=.false.
contains
"""
    return (
        declarations
        + owner
        + "\n"
        + landind
        + "\n"
        + zoom
        + "\n"
        + ORACLE_BACKEND
        + "\nend module readdim2\n"
    )


def _generated_solar_source() -> str:
    lines = SOLAR_SOURCE.read_text(encoding="utf-8", errors="replace").splitlines(True)
    return (
        "module solar\nuse defprec\nuse constantes\nuse ioipsl_para\nimplicit none\ncontains\n"
        + "".join(lines[54:268])
        + "\nend module solar\n"
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="orchidee_forcing_owner_") as raw:
        work = Path(raw)
        generated = work / "readdim2_oracle.f90"
        generated.write_text(_generated_source(), encoding="utf-8")
        generated_solar = work / "solar_oracle.f90"
        generated_solar.write_text(_generated_solar_source(), encoding="utf-8")
        exe = work / "oracle.exe"
        command = [
            str(COMPILER),
            "-cpp",
            "-fdefault-real-8",
            "-ffree-line-length-none",
            "-ffunction-sections",
            "-Wl,--gc-sections",
            str(ROOT / "scripts/dev/oracle_lane_driver_forcing_owner_stubs.f90"),
            str(generated_solar),
            str(generated),
            str(ROOT / "scripts/dev/oracle_lane_driver_forcing_owner_harness.f90"),
            "-o",
            str(exe),
        ]
        environment = compiler_environment(COMPILER)
        built = subprocess.run(
            command, cwd=work, env=environment, text=True, capture_output=True
        )
        if built.returncode:
            print(built.stdout)
            print(built.stderr)
            return built.returncode
        scenario_outputs = {}
        for scenario in (
            "standard", "watchout", "daily", "netrad", "hourly",
            "scalar_wind", "no_contfrac", "daily_watchout", "daily_wrap",
            "wrap", "high_daily", "high_standard", "daily_split1",
        ):
            output = work / f"fortran_outputs_{scenario}.csv"
            run = subprocess.run(
                [str(exe), str(output), scenario],
                cwd=work,
                env=environment,
                text=True,
                capture_output=True,
            )
            if run.returncode:
                print(run.stdout)
                print(run.stderr)
                return run.returncode
            scenario_outputs[scenario] = output
        target = ROOT / "outputs/reference_mode/micro_oracles/driver_forcing_owner"
        target.mkdir(parents=True, exist_ok=True)
        for scenario, output in scenario_outputs.items():
            (target / f"fortran_outputs_{scenario}.csv").write_bytes(output.read_bytes())
        output = scenario_outputs["standard"]
        rows: dict[tuple[str, int], list[float]] = {}
        with output.open(newline="", encoding="ascii") as handle:
            for name, step, value in csv.reader(handle):
                rows.setdefault((name, int(step)), []).append(float(value))
        import sys

        sys.path.insert(0, str(ROOT))
        import numpy as np
        from jax_orchidee.driver.domain import (
            PaperPointForcingCache,
            _forcing_model_step_from_cached_point,
        )
        from jax_orchidee.driver.forcing_interpolation import (
            ForcingInterpolationRecord,
            ForcingInterpolationState,
            forcing_grid_metadata,
            forcing_read_interpol_transition,
        )

        shape = (24, 2, 2)

        def series(name: str) -> np.ndarray:
            result = np.empty(shape)
            for t in range(1, 25):
                for j in range(2):
                    for i in range(2):
                        if name == "Tair":
                            value = 270.0 + 5.0 * t + i + 1 + 2 * (j + 1)
                        elif name == "PSurf":
                            value = 90000.0 + 1000.0 * t + 10 * (i + 1) + j + 1
                        elif name == "Qair":
                            value = 0.002 + 0.001 * t + 0.0001 * (i + 1)
                        elif name == "Wind_N":
                            value = t + i + 1
                        elif name == "Wind_E":
                            value = -t - (j + 1)
                        elif name == "Rainf":
                            value = 1.0e-5 * (t + i + 1)
                        elif name == "Snowf":
                            value = 1.0e-6 * (t + j + 1)
                        elif name == "SWdown":
                            value = 100.0 * t + 10.0 * (i + 1) + j + 1
                        elif name == "LWdown":
                            value = 200.0 + 10.0 * t + i + 1 + j + 1
                        elif name == "Tmin":
                            value = 265.0 + 3.0 * t + i + 1 + j + 1
                        elif name == "Tmax":
                            value = 280.0 + 4.0 * t + i + 1 + j + 1
                        elif name == "precip":
                            value = 1.0e-5 * (t + i + 1 + j + 1)
                        elif name == "levels":
                            value = 2.0 + 0.25 * t + 0.1 * (i + 1)
                        elif name == "swnet":
                            value = 50.0 * t + i + 1 + j + 1
                        elif name == "eair":
                            value = 1000.0 * t + 10.0 * (i + 1) + j + 1
                        elif name == "petacoef":
                            value = 0.1 * t + 0.01 * (i + 1)
                        elif name == "peqacoef":
                            value = 0.2 * t + 0.01 * (j + 1)
                        elif name == "petbcoef":
                            value = 0.3 * t + 0.01 * (i + 1)
                        elif name == "peqbcoef":
                            value = 0.4 * t + 0.01 * (j + 1)
                        elif name == "cdrag":
                            value = 0.005 * t + 0.001 * (i + 1)
                        elif name == "ccanopy":
                            value = 350.0 + t + i + 1 + j + 1
                        result[t - 1, i, j] = value
            return result

        cache = PaperPointForcingCache(
            **{
                name: series(name)
                for name in (
                    "Tair",
                    "PSurf",
                    "Qair",
                    "Wind_E",
                    "Wind_N",
                    "Rainf",
                    "Snowf",
                    "SWdown",
                    "LWdown",
                )
            },
            Areas=np.ones((2, 2)),
            contfrac=np.ones((2, 2)),
            Height_Lev1=2.0,
            Height_Levuv=10.0,
        )
        lon = np.asarray([[-30.0, 30.0], [0.0, 60.0]])
        lat = np.asarray([[-20.0, 30.0], [10.0, 60.0]])
        comparisons = []
        records = tuple(
            ForcingInterpolationRecord.from_mapping(
                {
                    "tair": series("Tair")[index],
                    "pb": series("PSurf")[index],
                    "qair": series("Qair")[index],
                    "u": series("Wind_N")[index],
                    "v": series("Wind_E")[index],
                    "rainf": series("Rainf")[index],
                    "snowf": series("Snowf")[index],
                    "swdown": series("SWdown")[index],
                    "lwdown": series("LWdown")[index],
                    "zlev": np.full((2, 2), 2.0),
                    "zlevuv": np.full((2, 2), 10.0),
                }
            )
            for index in range(4)
        )
        complete_state = forcing_read_interpol_transition(
            records,
            ForcingInterpolationState(),
            itauin=0,
            itau_split=0,
            split=4,
            nb_spread=2,
            dt_force=21600.0,
            lon=lon,
            lat=lat,
            daily_interpol=False,
            is_watchout=False,
        ).state
        for model_step in range(8):
            result = _forcing_model_step_from_cached_point(
                cache,
                model_tstep=model_step,
                split=4,
                nb_spread=2,
                lon=lon,
                lat=lat,
                dt_force=21600.0,
            )
            complete = forcing_read_interpol_transition(
                records,
                complete_state,
                itauin=model_step // 4 + 1,
                itau_split=model_step % 4 + 1,
                split=4,
                nb_spread=2,
                dt_force=21600.0,
                lon=lon,
                lat=lat,
                daily_interpol=False,
                is_watchout=False,
            )
            complete_state = complete.state
            for output_name, attr in (
                ("tair", "Tair"),
                ("qair", "Qair"),
                ("swdown", "SWdown"),
                ("rainf", "Rainf"),
                ("snowf", "Snowf"),
            ):
                expected = np.asarray(getattr(result, attr)).ravel(order="F")
                actual = np.asarray(rows[(output_name, model_step + 1)])
                error = np.max(np.abs(actual - expected))
                passed = bool(np.allclose(actual, expected, rtol=1e-12, atol=1e-14))
                comparisons.append(
                    {
                        "name": f"{output_name}.step{model_step + 1}",
                        "max_abs_error": float(error),
                        "passed": passed,
                    }
                )
                complete_expected = np.asarray(
                    getattr(complete.values, output_name)
                ).ravel(order="F")
                complete_error = np.max(np.abs(actual - complete_expected))
                comparisons.append(
                    {
                        "name": f"complete_owner.{output_name}.step{model_step + 1}",
                        "max_abs_error": float(complete_error),
                        "passed": bool(
                            np.allclose(
                                actual, complete_expected, rtol=1e-12, atol=1e-14
                            )
                        ),
                    }
                )
        def read_rows(path):
            parsed = {}
            with path.open(newline="", encoding="ascii") as handle:
                for name, step, value in csv.reader(handle):
                    parsed.setdefault((name, int(step)), []).append(float(value))
            return parsed

        for scenario, daily, watchout, conserve, scenario_split, scenario_spread, nsteps, itau_base, high_solar in (
            ("watchout", False, True, False, 4, 2, 8, 1, False),
            ("daily", True, False, False, 24, 6, 48, 1, False),
            ("netrad", False, False, True, 4, 2, 8, 1, False),
            ("hourly", False, False, False, 1, 1, 2, 1, False),
            ("scalar_wind", False, False, False, 4, 2, 8, 1, False),
            ("no_contfrac", False, False, False, 4, 2, 8, 1, False),
            ("daily_watchout", True, True, False, 24, 6, 48, 1, False),
            ("daily_wrap", True, False, False, 24, 6, 589, 1, False),
            ("wrap", False, False, False, 4, 2, 5, 24, False),
            ("high_daily", True, False, False, 24, 6, 48, 1, True),
            ("high_standard", False, False, False, 4, 2, 8, 1, True),
            ("daily_split1", True, False, False, 1, 1, 2, 1, False),
        ):
            scenario_rows = read_rows(scenario_outputs[scenario])
            raw_contfrac = None if scenario == "no_contfrac" else np.asarray(
                [[1.0, 1.0], [1.0, 0.5]], dtype=np.float64
            )
            raw_neighbours = None
            raw_resolution = None
            if watchout:
                raw_neighbours = np.empty((2, 2, 8), dtype=np.int32)
                for direction in range(8):
                    raw_neighbours[:, :, direction] = np.asarray(
                        [[999999999, 12], [21, 22]], dtype=np.int32
                    )
                raw_resolution = np.empty((2, 2, 2), dtype=np.float64)
                raw_resolution[:, :, 0] = np.asarray([[1000.0, 1000.0], [2000.0, 2000.0]])
                raw_resolution[:, :, 1] = np.asarray([[2000.0, 4000.0], [2000.0, 4000.0]])
            grid = forcing_grid_metadata(
                series("Tmin" if daily else "Tair")[0],
                contfrac=raw_contfrac,
                watchout=watchout,
                neighbours=raw_neighbours,
                resolution=raw_resolution,
                ii_begin=1,
                ii_end=1,
            )
            actual_contfrac = np.asarray(scenario_rows[("contfrac", 0)])
            expected_contfrac = grid.contfrac.ravel(order="F")
            comparisons.append({
                "name": f"{scenario}.contfrac.initialization",
                "max_abs_error": float(np.max(np.abs(actual_contfrac - expected_contfrac))),
                "passed": bool(np.array_equal(actual_contfrac, expected_contfrac)),
            })
            if watchout:
                for direction in range(8):
                    actual_neighbour = np.asarray(
                        scenario_rows[(f"neighbours_{direction + 1}", 0)], dtype=np.int32
                    )
                    expected_neighbour = grid.neighbours[:, :, direction].ravel(order="F")
                    comparisons.append({
                        "name": f"{scenario}.neighbours{direction + 1}.initialization",
                        "max_abs_error": float(np.max(np.abs(actual_neighbour - expected_neighbour))),
                        "passed": bool(np.array_equal(actual_neighbour, expected_neighbour)),
                    })
            scenario_records = tuple(
                ForcingInterpolationRecord.from_mapping(
                    {
                        "tair": series("Tmin" if daily else "Tair")[index],
                        "tmax": series("Tmax")[index] if daily else None,
                        "pb": series("PSurf")[index], "qair": series("Qair")[index],
                        "u": series("Wind_N")[index], "v": series("Wind_E")[index],
                        "rainf": series("precip" if daily else "Rainf")[index],
                        "snowf": np.zeros((2, 2)) if daily else series("Snowf")[index],
                        "swdown": np.full((2, 2), 10000.0) if high_solar else series("SWdown")[index],
                        "lwdown": series("LWdown")[index],
                        "zlev": series("levels")[index] if watchout else np.full((2, 2), 2.0),
                        "zlevuv": series("levels")[index] if watchout else np.full((2, 2), 10.0),
                        **({name: series(name)[index] for name in (
                            "swnet", "eair", "petacoef", "peqacoef", "petbcoef",
                            "peqbcoef", "cdrag", "ccanopy",
                        )} if watchout else {}),
                    }
                ) for index in range(24 if scenario in {"hourly", "daily_wrap", "wrap"} else 4)
            )
            scenario_state = forcing_read_interpol_transition(
                scenario_records, ForcingInterpolationState(), itauin=0, itau_split=0,
                split=scenario_split, nb_spread=scenario_spread,
                dt_force=86400.0 if daily else 3600.0 if scenario == "hourly" else 21600.0,
                lon=lon, lat=lat,
                daily_interpol=daily, is_watchout=watchout, netrad_cons=conserve,
            ).state
            output_names = list(_BASE_FIELDS_FOR_ORACLE)
            if watchout:
                output_names.extend(_WATCHOUT_FIELDS_FOR_ORACLE)
            for model_step in range(nsteps):
                complete = forcing_read_interpol_transition(
                    scenario_records, scenario_state,
                    itauin=model_step // scenario_split + itau_base,
                    itau_split=model_step % scenario_split + 1,
                    split=scenario_split, nb_spread=scenario_spread,
                    dt_force=86400.0 if daily else 3600.0 if scenario == "hourly" else 21600.0,
                    lon=lon, lat=lat, daily_interpol=daily, is_watchout=watchout,
                    netrad_cons=conserve,
                )
                scenario_state = complete.state
                for name in output_names:
                    actual = np.asarray(scenario_rows[(name, model_step + 1)])
                    expected = np.asarray(getattr(complete.values, name)).ravel(order="F")
                    error = np.max(np.abs(actual - expected))
                    comparisons.append({
                        "name": f"{scenario}.{name}.step{model_step + 1}",
                        "max_abs_error": float(error),
                        "passed": bool(np.allclose(actual, expected, rtol=1e-12, atol=1e-14)),
                    })
        status = "passed" if all(item["passed"] for item in comparisons) else "failed"
        metadata = {"command": command, "status": "passed"}
        (target / "build.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="ascii"
        )
        (target / "comparison.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "family": "driver_forcing_owner",
                    "status": status,
                    "comparisons": comparisons,
                    "build": metadata,
                },
                indent=2,
            )
            + "\n",
            encoding="ascii",
        )
        if status != "passed":
            print(
                json.dumps(
                    [item for item in comparisons if not item["passed"]], indent=2
                )
            )
            return 1
    return 0


def run_oracle(output_dir: Path, compiler: Path = COMPILER) -> dict[str, object]:
    """Adapter for the unified family runner."""

    del output_dir, compiler
    status = main()
    result_path = (
        ROOT / "outputs/reference_mode/micro_oracles/driver_forcing_owner/comparison.json"
    )
    result = json.loads(result_path.read_text(encoding="ascii"))
    if status != 0:
        result["status"] = "failed"
    return result


if __name__ == "__main__":
    raise SystemExit(main())
