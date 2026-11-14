from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jax_orchidee.driver.dim2 import (
    Dim2DriverRestart,
    Dim2IntersurfOutput,
    Dim2Switches,
    resolve_dim2_restart_fields,
    resolve_dim2_time_control,
    run_dim2_driver,
    validate_dim2_switches,
)
from jax_orchidee.driver.restart_file_io import (
    DRIVER_RESTART_VARIABLES,
    read_dim2_driver_restart,
    write_dim2_driver_restart,
)
from jax_orchidee.driver.forcing_readers import VerticalForcingOptions, forcing_just_read
from jax_orchidee.stomate.restart_io import StomateRestartPhysicalState


ROOT = Path(__file__).resolve().parents[2]


def _output(boundary, state):
    npts = boundary.payload.nbindex
    return Dim2IntersurfOutput(
        vevapp=np.full(npts, 7.25),
        fluxsens=np.full(npts, 8.5),
        fluxlat=np.full(npts, 9.75),
        coastalflow=np.zeros(npts),
        riverflow=np.zeros(npts),
        tsol_rad=np.full(npts, 281.0),
        temp_sol_new=np.full(npts, 281.0),
        qsurf=np.full(npts, 0.02),
        albedo=np.full((npts, 2), 0.2),
        emis=np.full(npts, 0.98),
        z0=np.full(npts, 0.1),
        component_state=state,
    )


def test_peqAcoef_reader_driver_and_restart_codec_keep_selected_path_explicit(tmp_path):
    # The non-WATCHOUT forcing reader must not treat unrelated PBL coefficient
    # values as active; DIM2's nonrelaxed branch sets A to zero and B to Qair.
    source = {
        "Tair": np.array([[[283.0]]]),
        "Snowf": np.array([[[0.0]]]),
        "Rainf": np.array([[[0.0002]]]),
        "Tmin": np.array([[[282.0]]]),
        "precip": np.array([[[0.0003]]]),
        "SWdown": np.array([[[112.0]]]),
        "LWdown": np.array([[[301.0]]]),
        "PSurf": np.array([[[98765.0]]]),
        "Qair": np.array([[[0.0067]]]),
        "Wind_N": np.array([[[2.1]]]),
        "Wind_E": np.array([[[3.2]]]),
        "Wind": np.array([[[4.3]]]),
        "Levels": np.array([[[4.0]]]),
        "Levels_uv": np.array([[[8.0]]]),
        "levels": np.array([[[5.0]]]),
        "SWnet": np.array([[[111.0]]]),
        "Eair": np.array([[[284123.0]]]),
        "petAcoef": np.array([[[13.25]]]),
        "peqAcoef": np.array([[[17.5]]]),
        "petBcoef": np.array([[[19.75]]]),
        "peqBcoef": np.array([[[23.125]]]),
        "cdrag": np.array([[[29.5]]]),
        "ccanopy": np.array([[[411.0]]]),
    }
    forcing = forcing_just_read(
        source,
        VerticalForcingOptions(zheight=True, zlev_fixed=2.0, zlevuv_fixed=10.0),
        ttm=1,
        is_watchout=False,
        wind_n_exists=True,
    )
    assert forcing.peqAcoef is None
    np.testing.assert_array_equal(forcing.qair, [[0.0067]])

    vector = lambda value: np.asarray(value, dtype=np.float64).reshape(-1)
    bundle = SimpleNamespace(
        domain=SimpleNamespace(
            nbindex=1,
            kindex=np.array([1], dtype=np.int32),
            lalo=np.array([[0.0, 0.0]]),
            lon=np.array([[0.0]]),
            lat=np.array([[0.0]]),
            contfrac_land=np.ones(1),
            resolution=None,
            neighbours=None,
            corners=None,
            seglength=None,
        ),
        forcing=SimpleNamespace(
            zlev=vector(forcing.zlev),
            zlevuv=vector(forcing.zlev_uv),
            u=vector(forcing.u),
            v=vector(forcing.v),
            qair=vector(forcing.qair),
            temp_air=vector(forcing.tair),
            pb=vector(forcing.pb),
            precip_rain=vector(forcing.rainf),
            precip_snow=vector(forcing.snowf),
            lwdown=vector(forcing.lwdown),
            swdown=vector(forcing.swdown),
        ),
        ccanopy=np.array([411.0]),
        vegetation=SimpleNamespace(
            veget_max=np.ones((1, 1)), veget=np.ones((1, 1)), soiltile=np.ones((1, 1))
        ),
        static_trace_fields=SimpleNamespace(
            njsc=None, clay_frac=None, sand_frac=None, silt_frac=None, bulk_dens=None,
            soil_ph=None, poor_soils=None, soilclass_sub_index=None, soilclass_sub_area=None,
            soilclass=None, salinity=None, tide_height=None,
        ),
        restart_anchors=None,
        missing_geometry_fields=("resolution", "neighbours", "corners", "seglength"),
        missing_static_fields=(
            "soilclass", "njsc", "clay_frac", "sand_frac", "silt_frac", "bulk_dens",
            "soil_ph", "poor_soils", "soilclass_sub_index", "soilclass_sub_area",
            "salinity", "tide_height",
        ),
    )
    present_restart = Dim2DriverRestart(peqAcoef=np.array([17.5]))
    resolved = resolve_dim2_restart_fields(present_restart, bundle)
    np.testing.assert_array_equal(resolved.peqAcoef, [17.5])
    missing = resolve_dim2_restart_fields(Dim2DriverRestart(), bundle)
    np.testing.assert_array_equal(missing.peqAcoef, [0.0])

    with pytest.raises(NotImplementedError, match="RELAXATION"):
        validate_dim2_switches(Dim2Switches(relaxation=True))
    with pytest.raises(NotImplementedError, match="WATCHOUT"):
        validate_dim2_switches(Dim2Switches(watchout=True))

    time = resolve_dim2_time_control(
        dt_force=3600.0, dt_sechiba=1800.0, forcing_length=1, date0=0.0, time_length=1
    )
    initialize_boundaries = []
    main_boundaries = []

    def initialize_spy(boundary, state):
        initialize_boundaries.append(boundary)
        return _output(boundary, state)

    def main_spy(boundary, state):
        main_boundaries.append(boundary)
        return _output(boundary, state)

    result = run_dim2_driver(
        time=time,
        forcing_reader=lambda _step: bundle,
        intersurf_initialize=initialize_spy,
        intersurf_main=main_spy,
        restart=present_restart,
    )
    assert result.initial_restart.peqAcoef[0] == 17.5
    assert len(initialize_boundaries) == 1
    assert len(main_boundaries) == len(result.steps)
    for boundary in (*initialize_boundaries, *main_boundaries):
        np.testing.assert_array_equal(boundary.peqAcoef, [0.0])
        np.testing.assert_array_equal(boundary.peqBcoef, [0.0067])
    np.testing.assert_array_equal(result.restart.peqAcoef, [0.0])

    # Exercise the production codec with a nonzero, distinguishable persisted
    # coefficient. This verifies IO preservation, not that the relaxed units
    # or WATCHOUT coupling convention are scientifically established.
    persisted = {
        field: np.array([100.0 + i], dtype=np.float64)
        for i, field in enumerate(DRIVER_RESTART_VARIABLES)
    }
    persisted["peqAcoef"] = np.array([17.5], dtype=np.float64)
    state = Dim2DriverRestart(**persisted)
    physical = StomateRestartPhysicalState(
        nav_lon=np.array([[0.0]]),
        nav_lat=np.array([[0.0]]),
        nav_lev=np.array([0.0]),
        time=np.array([0.0]),
        time_steps=np.array([1800.0]),
    )
    path = tmp_path / "driver_restart.nc"
    write_dim2_driver_restart(path, state=state, physical_state=physical)
    restored = read_dim2_driver_restart(path)
    for field in DRIVER_RESTART_VARIABLES:
        np.testing.assert_array_equal(getattr(restored, field), getattr(state, field))
