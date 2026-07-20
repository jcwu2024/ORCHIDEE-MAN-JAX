from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import textwrap

import numpy as np
import pytest

from jax_orchidee.driver.dim2 import resolve_dim2_time_control
from jax_orchidee.driver.driver_forcing_completion import (
    DownwardSolarFluxState,
    NetCDFStatusError,
    downward_solar_flux,
    driver_output_status,
    iter_driver_step_dispatch_statuses,
    nccheck,
    require_fixed_paper_weather_gate,
    resolve_driver_weather_timestep,
    solarang,
    weathgen_qsat_2d,
)


ROOT = Path(__file__).resolve().parents[2]
SOLAR_SOURCE = ROOT / "fortran_source" / "ORCHIDEE" / "src_global" / "solar.f90"
WEATHER_SOURCE = ROOT / "fortran_source" / "ORCHIDEE" / "src_driver" / "weather.f90"
GFORTRAN = Path(r"C:\msys64\ucrt64\bin\gfortran.exe")
if not GFORTRAN.is_file():
    discovered = shutil.which("gfortran")
    GFORTRAN = Path(discovered) if discovered else GFORTRAN


def test_fixed_weather_gate_and_driver_dispatch_output_states():
    """dim2_driver.f90::driver lines 285-311, 802-837, 1051, 1403, 1425, 1441."""

    paper = require_fixed_paper_weather_gate(dt_force=21600.0, dt_sechiba=1800.0)
    assert (paper.dt, paper.split, paper.weathergen) == (1800.0, 12, False)
    generated = resolve_driver_weather_timestep(
        dt_force=21600.0, dt_sechiba=1800.0, weathergen=True
    )
    assert (generated.dt, generated.split, generated.weathergen) == (21600.0, 1, True)
    with pytest.raises(NotImplementedError, match="ALLOW_WEATHERGEN=false"):
        require_fixed_paper_weather_gate(
            dt_force=21600.0, dt_sechiba=1800.0, weathergen=True
        )

    time = resolve_dim2_time_control(
        dt_force=3600.0,
        dt_sechiba=1800.0,
        forcing_length=3,
        date0=0.0,
        time_length=3,
    )
    statuses = iter_driver_step_dispatch_statuses(time)
    assert len(statuses) == 6
    assert [item.initialize_intersurf for item in statuses] == [True, False, False, False, False, False]
    assert [item.final_forcing_step for item in statuses] == [False, False, False, False, False, True]
    assert [item.prepare_load_balance for item in statuses] == [False, False, True, True, False, False]

    standard_root = driver_output_status(is_watchout=False, is_root_prc=True)
    assert standard_root.write_standard_restart_fields
    assert standard_root.write_pbl_restart_fields
    assert standard_root.close_forcing_history
    assert standard_root.close_restart_and_dump_config
    watchout_nonroot = driver_output_status(is_watchout=True, is_root_prc=False)
    assert not watchout_nonroot.write_pbl_restart_fields
    assert not watchout_nonroot.close_restart_and_dump_config


def test_formula_shapes_thresholds_state_and_netcdf_guard():
    lon = np.asarray([[-179.0, -179.0], [179.0, 179.0], [0.0, 0.0]])
    lat = np.asarray([[-45.0, 45.0], [-45.0, 45.0], [-45.0, 45.0]])
    angle = solarang(100.99, 1.0, lon, lat, one_year=365.0)
    assert angle.shape == lon.shape
    assert np.all((angle >= 0.0) & (angle <= 1.0))

    first = downward_solar_flux(
        np.asarray([-60.0, 21.0]),
        calendar_str="gregorian",
        jday=1.5,
        rtime=12.0,
        cloud=np.asarray([0.0, 0.5]),
        nband=2,
        one_year=365.0,
    )
    assert first.state.initialized and first.state.step == 1.0
    np.testing.assert_allclose(first.solad[:, 1] / first.solad[:, 0], 0.54 / 0.46)
    retained = downward_solar_flux(
        np.asarray([-60.0, 21.0]),
        calendar_str="360d",
        jday=80.25,
        rtime=6.0,
        cloud=np.asarray([0.0, 0.5]),
        nband=3,
        one_year=360.0,
        state=first.state,
    )
    assert retained.state.step == 1.0
    np.testing.assert_allclose(retained.solad[:, 0], retained.solad[:, 2])

    temperature = np.asarray([[213.15, 273.15, 300.0], [373.15, 200.0, 280.0]])
    pressure = np.asarray([[100000.0, 100000.0, 90000.0], [80000.0, 70000.0, 200.0]])
    qsat = weathgen_qsat_2d(temperature, pressure)
    assert qsat.shape == temperature.shape
    assert qsat[1, 2] == 1.0  # Source MAX denominator guard.

    assert nccheck(0) is None
    with pytest.raises(NetCDFStatusError, match="invalid coordinate") as caught:
        nccheck(-40, strerror=lambda _: "invalid coordinate")
    assert caught.value.status == -40


def _extract_lines(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start - 1 : end])


def _oracle_source() -> str:
    # Procedure bodies are byte-for-byte source slices; only their enclosing
    # module dependencies and the deterministic calls below are supplied here.
    solarang_source = _extract_lines(SOLAR_SOURCE, 55, 181)
    timezone_source = _extract_lines(SOLAR_SOURCE, 201, 268)
    downward_source = _extract_lines(SOLAR_SOURCE, 290, 597)
    qsat_source = _extract_lines(WEATHER_SOURCE, 3623, 3668)
    return textwrap.dedent(
        f"""
        module calendar
        end module calendar

        module extracted_oracle
          implicit none
          real, parameter :: pi=3.1415926535897932384626433832795, zero=0.0
          real, parameter :: one_year=365.0, zero_t=273.15
          integer, parameter :: numout=6
        contains
          subroutine getin_p(name, value)
            character(len=*), intent(in) :: name
            real, intent(inout) :: value
          end subroutine getin_p
        {solarang_source}
        {timezone_source}
        {downward_source}
        {qsat_source}
        end module extracted_oracle

        program driver_forcing_oracle
          use extracted_oracle
          implicit none
          real :: lon(3,2), lat(3,2), csang(3,2)
          real :: latitude(2), cloud(2), solad2(2,2), solai2(2,2)
          real :: solad3(2,3), solai3(2,3), t(2,3), p(2,3), qsat(2,3)
          character(len=20) :: gregorian_calendar, other_calendar
          integer :: j
          lon(:,1)=(/-179.0,179.0,0.0/); lon(:,2)=lon(:,1)
          lat(:,1)=-45.0; lat(:,2)=45.0
          call solarang(100.99,1.0,3,2,lon,lat,csang)
          write(*,'(*(ES25.16,1X))') csang
          call solarang(200.01,1.0,3,2,lon,lat,csang)
          write(*,'(*(ES25.16,1X))') csang

          latitude=(/-60.0,21.0/); cloud=(/0.0,0.5/)
          gregorian_calendar='gregorian'; other_calendar='360d'
          call downward_solar_flux(2,latitude,gregorian_calendar,1.5,12.0,cloud,2,solad2,solai2)
          write(*,'(*(ES25.16,1X))') solad2
          write(*,'(*(ES25.16,1X))') solai2
          call downward_solar_flux(2,latitude,other_calendar,80.25,6.0,cloud,3,solad3,solai3)
          write(*,'(*(ES25.16,1X))') solad3
          write(*,'(*(ES25.16,1X))') solai3

          t(:,1)=(/213.15,373.15/); t(:,2)=(/273.15,200.0/); t(:,3)=(/300.0,280.0/)
          p(:,1)=(/100000.0,80000.0/); p(:,2)=(/100000.0,70000.0/); p(:,3)=(/90000.0,200.0/)
          call weathgen_qsat_2d(2,3,t,p,qsat)
          write(*,'(*(ES25.16,1X))') qsat
        end program driver_forcing_oracle
        """
    )


@pytest.mark.skipif(not GFORTRAN.is_file(), reason="gfortran is unavailable")
def test_source_extracted_gfortran_oracle_matches_solar_and_qsat(tmp_path):
    source_path = tmp_path / "driver_forcing_oracle.f90"
    executable = tmp_path / "driver_forcing_oracle.exe"
    source_path.write_text(_oracle_source(), encoding="utf-8")
    env = os.environ.copy()
    env["PATH"] = f"{GFORTRAN.parent}{os.pathsep}{env.get('PATH', '')}"
    compiled = subprocess.run(
        [
            str(GFORTRAN),
            "-fdefault-real-8",
            "-ffree-line-length-none",
            str(source_path),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run([str(executable)], capture_output=True, text=True, env=env)
    assert executed.returncode == 0, executed.stderr
    rows = []
    for line in executed.stdout.splitlines():
        try:
            values = [float(token) for token in line.split()]
        except ValueError:
            continue
        if values:
            rows.append(np.asarray(values, dtype=np.float64))
    # Three initialization WRITE lines from the extracted downward procedure.
    numeric = [row for row in rows if row.size in {4, 6}]
    assert len(numeric) == 7

    lon = np.asarray([[-179.0, -179.0], [179.0, 179.0], [0.0, 0.0]])
    lat = np.asarray([[-45.0, 45.0], [-45.0, 45.0], [-45.0, 45.0]])
    np.testing.assert_allclose(
        solarang(100.99, 1.0, lon, lat, one_year=365.0).ravel(order="F"),
        numeric[0],
        rtol=2e-13,
        atol=2e-13,
    )
    np.testing.assert_allclose(
        solarang(200.01, 1.0, lon, lat, one_year=365.0).ravel(order="F"),
        numeric[1],
        rtol=2e-13,
        atol=2e-13,
    )

    first = downward_solar_flux(
        np.asarray([-60.0, 21.0]), calendar_str="gregorian", jday=1.5, rtime=12.0,
        cloud=np.asarray([0.0, 0.5]), nband=2, one_year=365.0,
    )
    np.testing.assert_allclose(first.solad.ravel(order="F"), numeric[2], rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(first.solai.ravel(order="F"), numeric[3], rtol=3e-13, atol=3e-13)
    second = downward_solar_flux(
        np.asarray([-60.0, 21.0]), calendar_str="360d", jday=80.25, rtime=6.0,
        cloud=np.asarray([0.0, 0.5]), nband=3, one_year=360.0, state=first.state,
    )
    np.testing.assert_allclose(second.solad.ravel(order="F"), numeric[4], rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(second.solai.ravel(order="F"), numeric[5], rtol=3e-13, atol=3e-13)

    temperature = np.asarray([[213.15, 273.15, 300.0], [373.15, 200.0, 280.0]])
    pressure = np.asarray([[100000.0, 100000.0, 90000.0], [80000.0, 70000.0, 200.0]])
    np.testing.assert_allclose(
        weathgen_qsat_2d(temperature, pressure).ravel(order="F"),
        numeric[6],
        rtol=3e-13,
        atol=3e-13,
    )


def test_initialized_downward_state_requires_saved_step():
    with pytest.raises(ValueError, match="requires step"):
        downward_solar_flux(
            np.asarray([0.0]),
            calendar_str="gregorian",
            jday=1.0,
            rtime=12.0,
            cloud=np.asarray([0.0]),
            nband=1,
            one_year=365.0,
            state=DownwardSolarFluxState(initialized=True),
        )
