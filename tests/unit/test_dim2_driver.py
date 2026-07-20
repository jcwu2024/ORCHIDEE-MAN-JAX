from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jax_orchidee.driver.dim2 import (
    CP_AIR,
    GRAVITY,
    Dim2DriverRestart,
    Dim2IntersurfOutput,
    Dim2RestartTime,
    Dim2Switches,
    resolve_dim2_forcing_controls,
    dim2_switches_from_run_def,
    iter_dim2_time_indices,
    resolve_dim2_time_control,
    resolve_dim2_restart_fields,
    run_dim2_driver,
    time_length_to_forcing_steps,
    validate_dim2_switches,
)


def test_restart_time_conversion_skip_and_source_order_indices():
    """Fortran dim2_driver.f90 lines 379-401, 464-529, 802-837."""

    time = resolve_dim2_time_control(
        dt_force=21600.0,
        dt_sechiba=1800.0,
        forcing_length=10,
        date0=1000.0,
        restart=Dim2RestartTime(itau_dep_rest=4, date0_rest=1000.25, dt_rest=10800.0),
        time_skip=1,
        time_length=2,
    )
    assert (time.split, time.itau_dep, time.itau_fin, time.for_offset) == (12, 3, 5, 1)
    assert (time.istp, time.istp_old, time.split_start) == (37, 4, 1)
    indices = tuple(iter_dim2_time_indices(time))
    assert len(indices) == 24
    assert (indices[0].it, indices[0].it_force, indices[0].isplit, indices[0].istp) == (4, 5, 1, 37)
    assert indices[0].forcing_model_tstep == 48
    assert indices[-1].lstep_last
    assert indices[-1].istp == 60


def test_reset_time_uses_forcing_origin_and_time_length_parser():
    time = resolve_dim2_time_control(
        dt_force=21600.0,
        dt_sechiba=1800.0,
        forcing_length=1460,
        date0=42.0,
        restart=Dim2RestartTime(itau_dep_rest=99, date0_rest=1.0, dt_rest=1800.0, driver_reset_time=True),
        time_length="1D",
    )
    assert time.itau_dep == 0
    assert time.date0_rest == 42.0
    assert time.itau_len == 4
    assert time_length_to_forcing_steps("1Y", dt_force=21600.0) == 1460
    assert time_length_to_forcing_steps("1Y6M", dt_force=21600.0) == 2190


def test_non_integral_split_uses_fortran_int_and_unsupported_switches_are_explicit():
    # dim2_driver.f90 line 296 uses INT, not a divisibility guard.
    time = resolve_dim2_time_control(dt_force=20000, dt_sechiba=1800, forcing_length=1, date0=0)
    assert time.split == 11
    for switches in (
        Dim2Switches(weathergen=True),
        Dim2Switches(relaxation=True),
        Dim2Switches(watchout=True),
        Dim2Switches(driver_routing=True),
        Dim2Switches(river_routing=False),
    ):
        with pytest.raises(NotImplementedError):
            validate_dim2_switches(switches)
    parsed = dim2_switches_from_run_def({"WEATHERGEN": "n", "RIVER_ROUTING": "y"})
    assert not parsed.weathergen and parsed.river_routing


def test_default_time_length_shrinks_after_time_skip():
    """Fortran dim2_driver.f90 lines 519 and 533-558."""

    time = resolve_dim2_time_control(
        dt_force=21600.0,
        dt_sechiba=1800.0,
        forcing_length=10,
        date0=0.0,
        time_skip=3,
    )
    assert time.itau_dep == 3
    assert time.itau_len == 7
    assert time.itau_fin == 10


def test_restart_fallbacks_match_driver_source_defaults():
    """Fortran dim2_driver.f90 lines 722-787."""

    bundle = _bundle(0, npts=2)
    resolved = resolve_dim2_restart_fields(Dim2DriverRestart(), bundle)
    np.testing.assert_array_equal(resolved.fluxsens, 0.0)
    np.testing.assert_array_equal(resolved.vevapp, 0.0)
    np.testing.assert_allclose(resolved.old_zlev, 2.0)
    np.testing.assert_allclose(resolved.old_qair, 0.01)
    np.testing.assert_allclose(resolved.old_eair, CP_AIR * 280.0 + GRAVITY * 2.0)
    np.testing.assert_allclose(resolved.rau_old, 100000.0 / (287.05 * 280.0))
    np.testing.assert_array_equal(resolved.petAcoef, 0.0)
    np.testing.assert_allclose(resolved.petBcoef, resolved.old_eair)
    np.testing.assert_array_equal(resolved.peqAcoef, 0.0)
    np.testing.assert_allclose(resolved.peqBcoef, resolved.old_qair)

    explicit = resolve_dim2_restart_fields(
        Dim2DriverRestart(old_qair=np.asarray([0.1, 0.2])),
        bundle,
    )
    np.testing.assert_allclose(explicit.old_qair, [0.1, 0.2])


def _bundle(model_tstep: int, npts: int = 2):
    def vec(value):
        return np.full((npts, 1), value, dtype=np.float64)

    forcing = SimpleNamespace(
        zlev=vec(2.0),
        zlevuv=vec(10.0),
        u=vec(4.0 + model_tstep),
        v=vec(3.0 + model_tstep),
        qair=vec(0.01 + model_tstep * 0.001),
        temp_air=vec(280.0 + model_tstep),
        pb=vec(100000.0),
        precip_rain=vec(0.001),
        precip_snow=vec(0.0),
        lwdown=vec(300.0),
        swdown=vec(500.0),
    )
    domain = SimpleNamespace(
        nbindex=npts,
        kindex=np.arange(1, npts + 1, dtype=np.int32),
        lalo=np.column_stack((np.arange(npts), np.arange(npts))),
        lon=np.arange(npts, dtype=np.float64).reshape(npts, 1),
        lat=np.arange(npts, dtype=np.float64).reshape(npts, 1),
        contfrac_land=np.ones(npts),
        resolution=None,
        neighbours=None,
        corners=None,
        seglength=None,
    )
    vegetation = SimpleNamespace(
        veget_max=np.ones((npts, 1)), veget=np.ones((npts, 1)), soiltile=np.ones((npts, 1))
    )
    return SimpleNamespace(
        domain=domain,
        forcing=forcing,
        ccanopy=np.full(npts, 400.0),
        vegetation=vegetation,
        restart_anchors=None,
        static_trace_fields=SimpleNamespace(
            njsc=None,
            clay_frac=None,
            sand_frac=None,
            silt_frac=None,
            bulk_dens=None,
            soil_ph=None,
            poor_soils=None,
            soilclass_sub_index=None,
            soilclass_sub_area=None,
            soilclass=None,
            salinity=None,
            tide_height=None,
        ),
        missing_geometry_fields=("resolution", "neighbours", "corners", "seglength"),
        missing_static_fields=(
            "soilclass", "njsc", "clay_frac", "sand_frac", "silt_frac", "bulk_dens", "soil_ph",
            "poor_soils", "soilclass_sub_index", "soilclass_sub_area", "salinity", "tide_height",
        ),
    )


def _output(boundary, state, *, initialize=False):
    npts = boundary.payload.nbindex
    base = 281.0 if initialize else 282.0 + boundary.index.forcing_model_tstep
    return Dim2IntersurfOutput(
        vevapp=np.full(npts, 1.0),
        fluxsens=np.full(npts, 2.0),
        fluxlat=np.full(npts, 3.0),
        coastalflow=np.full(npts, 4.0),
        riverflow=np.full(npts, 5.0),
        tsol_rad=np.full(npts, base),
        temp_sol_new=np.full(npts, base),
        qsurf=np.full(npts, 0.02),
        albedo=np.full((npts, 2), 0.2 if initialize else 0.25),
        emis=np.full(npts, 0.98),
        z0=np.full(npts, 0.5 if initialize else 0.6),
        component_state=(state or 0) + 1,
    )


def test_driver_first_step_double_call_conversion_and_dynamic_landpoints():
    """Fortran lines 851-853, 914-1020, 1051-1180 and 1293-1408."""

    time = resolve_dim2_time_control(
        dt_force=3600.0, dt_sechiba=1800.0, forcing_length=1, date0=0.0, time_length=1
    )
    reads = []
    initialize_calls = []
    main_calls = []

    def reader(model_tstep):
        reads.append(model_tstep)
        return _bundle(model_tstep, npts=3)

    def initialize(boundary, state):
        initialize_calls.append(boundary)
        return _output(boundary, state, initialize=True)

    def main(boundary, state):
        main_calls.append(boundary)
        return _output(boundary, state)

    result = run_dim2_driver(
        time=time,
        forcing_reader=reader,
        intersurf_initialize=initialize,
        intersurf_main=main,
    )
    assert reads == [0, 1]
    assert len(initialize_calls) == 1
    assert len(main_calls) == 2
    assert result.component_state == 3
    assert result.istp_old == 2
    assert main_calls[0].payload.nbindex == 3
    np.testing.assert_allclose(main_calls[0].payload.pb, 1000.0)
    np.testing.assert_allclose(main_calls[0].payload.precip_rain, 1.8)
    np.testing.assert_allclose(main_calls[0].eair, CP_AIR * 280.0 + GRAVITY * 2.0)
    np.testing.assert_allclose(main_calls[0].swnet, 400.0)
    np.testing.assert_allclose(initialize_calls[0].swnet, 435.0)
    assert not np.allclose(initialize_calls[0].payload.u, main_calls[0].payload.u)
    np.testing.assert_allclose(result.restart.old_qair, 0.011)
    np.testing.assert_allclose(result.restart.old_zlev, 2.0)
    np.testing.assert_allclose(result.restart.albedo_vis, 0.25)
    np.testing.assert_allclose(result.steps[0].dtdt, (282.0 - 281.0) / 1800.0)


def test_restart_albedo_overrides_but_source_z0_condition_preserves_initialize_output():
    time = resolve_dim2_time_control(
        dt_force=1800.0, dt_sechiba=1800.0, forcing_length=1, date0=0.0, time_length=1
    )
    seen = []
    initialized = []
    restart = Dim2DriverRestart(
        albedo_vis=np.asarray([0.1, 0.2]),
        albedo_nir=np.asarray([0.3, 0.4]),
        z0=np.asarray([0.7, 0.8]),
    )

    def main(boundary, state):
        seen.append(boundary)
        return _output(boundary, state)

    def initialize(boundary, state):
        initialized.append(boundary)
        return _output(boundary, state, initialize=True)

    run_dim2_driver(
        time=time,
        forcing_reader=lambda _: _bundle(0),
        intersurf_initialize=initialize,
        intersurf_main=main,
        restart=restart,
    )
    # Restart fields are read only after intersurf_initialize_2d.
    # The initialize callback therefore receives the hard-coded z0=0.1.
    expected_initialize_u = 4.0 * np.log(2.0 / 0.1) / np.log(10.0 / 0.1)
    np.testing.assert_allclose(initialized[0].payload.u, expected_initialize_u)
    np.testing.assert_allclose(initialized[0].swnet, 435.0)
    np.testing.assert_allclose(seen[0].swnet, [400.0, 350.0])
    # dim2_driver.f90 line 1157 assigns z0 only when tmp_z0 is all val_exp;
    # a normal present restart z0 therefore leaves initialized z0=0.5 in use.
    expected_u = 4.0 * np.log(2.0 / 0.5) / np.log(10.0 / 0.5)
    np.testing.assert_allclose(seen[0].payload.u, expected_u)


def test_nonrelaxation_overwrites_restart_implicit_coefficients_each_step():
    time = resolve_dim2_time_control(
        dt_force=1800.0, dt_sechiba=1800.0, forcing_length=1, date0=0.0, time_length=1
    )
    seen = []
    restart = Dim2DriverRestart(
        petAcoef=np.full(2, 9.0),
        petBcoef=np.full(2, 9.0),
        peqAcoef=np.full(2, 9.0),
        peqBcoef=np.full(2, 9.0),
    )

    def main(boundary, state):
        seen.append(boundary)
        return _output(boundary, state)

    result = run_dim2_driver(
        time=time,
        forcing_reader=lambda _: _bundle(0),
        intersurf_initialize=lambda boundary, state: _output(boundary, state, initialize=True),
        intersurf_main=main,
        restart=restart,
    )
    np.testing.assert_allclose(seen[0].petAcoef, 0.0)
    np.testing.assert_allclose(seen[0].peqAcoef, 0.0)
    np.testing.assert_allclose(seen[0].petBcoef, CP_AIR * 280.0 + GRAVITY * 2.0)
    np.testing.assert_allclose(seen[0].peqBcoef, 0.01)
    np.testing.assert_allclose(result.restart.petBcoef, seen[0].petBcoef)


def test_landpoint_count_cannot_change_mid_transition():
    time = resolve_dim2_time_control(
        dt_force=3600.0, dt_sechiba=1800.0, forcing_length=1, date0=0.0, time_length=1
    )
    with pytest.raises(ValueError, match="landpoint count changed"):
        run_dim2_driver(
            time=time,
            forcing_reader=lambda step: _bundle(step, npts=2 if step == 0 else 3),
            intersurf_initialize=lambda boundary, state: _output(boundary, state, initialize=True),
            intersurf_main=lambda boundary, state: _output(boundary, state),
        )


def test_forcing_controls_follow_source_threshold_truncation_and_overrides():
    long_step = resolve_dim2_forcing_controls(dt_force=21600.0, dt=1800.0, split=12)
    assert long_step.no_inter and not long_step.inter_lin
    assert long_step.nb_spread == 6
    assert not long_step.netrad_cons

    short_step = resolve_dim2_forcing_controls(
        dt_force=7199.0,
        dt=1800.0,
        split=4,
        inter_lin=True,
        spread_override=9,
        netrad_cons_override=False,
    )
    assert not short_step.no_inter and short_step.inter_lin
    assert short_step.nb_spread == 4
    assert not short_step.netrad_cons


def test_forcing_controls_split_one_disables_interpolation_and_forces_spread_one():
    result = resolve_dim2_forcing_controls(
        dt_force=1800.0,
        dt=1800.0,
        split=1,
        inter_lin=True,
        spread_override=0,
        netrad_cons_override=True,
    )
    assert result.no_inter and not result.inter_lin
    assert result.nb_spread == 1
    assert not result.netrad_cons
