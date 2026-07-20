from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import (  # noqa: E402
    explicitsnow_compactn_step,
    explicitsnow_age_ice_step,
    explicitsnow_fall_step,
    explicitsnow_gone_step,
    explicitsnow_grain_step,
    explicitsnow_initialize_zero_state,
    explicitsnow_levels_step,
    explicitsnow_liquid_heat_excess_step,
    explicitsnow_main_step,
    explicitsnow_maxmass_step,
    explicitsnow_melt_refrz_step,
    explicitsnow_profile_step,
    explicitsnow_sublimation_step,
    explicitsnow_transf_step,
    explicitsnow_final_cleanup_step,
    hydrol_bucket_snow_step,
    hydrol_explicit_snow_step,
    hydrol_explicit_snow_zero_state,
    hydrol_cold_start_thermosoil_moisture_inputs,
    hydrol_cold_start_state,
    hydrol_layer_moisture_content,
    snow3lgrain_0d_explicit,
    snow3lheat_explicit,
    snow3lhold_explicit,
    snow3lliq_explicit,
    snow3ltemp_explicit,
)


def _manual_hydrol_bucket_point(
    *,
    precip_rain,
    precip_snow,
    temp_sol_new,
    soilcap,
    frac_nobio,
    totfrac_nobio,
    vevapsno,
    snow,
    snow_age,
    snow_nobio,
    snow_nobio_age,
    dt_sechiba=1800.0,
):
    chalfu0 = 0.3336e6
    tp_00 = 273.15
    snowcri = 1.5
    sneige = 1.5e-3
    maxmass_snow = 3000.0
    max_snow_age = 50.0
    snow_trans = 0.2
    min_sechiba = 1.0e-8
    snow += (1.0 - totfrac_nobio) * precip_snow
    if snow > snowcri:
        subsnownobio = frac_nobio * vevapsno
        subsnowveg = vevapsno - subsnownobio
    elif frac_nobio > min_sechiba:
        subsnownobio = vevapsno
        subsnowveg = 0.0
    else:
        subsnownobio = 0.0
        subsnowveg = vevapsno
    subsinksoil = 0.0
    if subsnowveg > snow:
        if (1.0 - totfrac_nobio) > min_sechiba:
            subsinksoil = (subsnowveg - snow) / (1.0 - totfrac_nobio)
        subsnowveg = snow
        snow = 0.0
        vevapsno = subsnowveg + subsnownobio
    else:
        snow -= subsnowveg

    snowmelt = 0.0
    if temp_sol_new > tp_00:
        if snow > sneige:
            snowmelt = (1.0 - frac_nobio) * (temp_sol_new - tp_00) * soilcap / chalfu0
            if snowmelt < snow:
                snow -= snowmelt
            else:
                snowmelt = snow
                snow = 0.0
        elif snow >= 0.0:
            snowmelt = snow
            snow = 0.0
        else:
            snow = 0.0
            snowmelt = 0.0
    if snow > maxmass_snow:
        snow_d1k = soilcap / chalfu0
        snowmelt += min(snow - maxmass_snow, snow_d1k)
        snow -= snowmelt

    snow_nobio += frac_nobio * precip_snow + frac_nobio * precip_rain
    snow_nobio -= subsnownobio
    if temp_sol_new > tp_00:
        snowmelt_tmp = frac_nobio * (temp_sol_new - tp_00) * soilcap / chalfu0
        if snowmelt_tmp > snow_nobio:
            snowmelt_tmp = max(0.0, snow_nobio)
        snowmelt += snowmelt_tmp
        snow_nobio -= snowmelt_tmp
    icemelt = 0.0
    if snow_nobio > maxmass_snow:
        snow_d1k = soilcap / chalfu0
        icemelt = min(snow_nobio - maxmass_snow, snow_d1k)
        snow_nobio -= icemelt

    if snow <= 0.0:
        snow_age = 0.0
    else:
        snow_age = (
            snow_age + (1.0 - snow_age / max_snow_age) * dt_sechiba / 86400.0
        ) * np.exp(-precip_snow / snow_trans)
    if snow_nobio <= 0.0:
        snow_nobio_age = 0.0
    else:
        d_age = (
            snow_nobio_age
            + (1.0 - snow_nobio_age / max_snow_age) * dt_sechiba / 86400.0
        ) * np.exp(-precip_snow / snow_trans) - snow_nobio_age
        if d_age > min_sechiba:
            xx = max(tp_00 - temp_sol_new, 0.0)
            xx = (xx / 7.0) ** 4.0
            d_age /= 1.0 + xx
        snow_nobio_age = max(snow_nobio_age + d_age, 0.0)

    return {
        "snow": snow,
        "snow_age": snow_age,
        "snow_nobio": snow_nobio,
        "snow_nobio_age": snow_nobio_age,
        "vevapsno": vevapsno,
        "tot_melt": icemelt + snowmelt,
        "snowmelt": snowmelt,
        "snowdepth": snow / 330.0,
        "subsnownobio": subsnownobio,
        "subsnowveg": subsnowveg,
        "subsinksoil": subsinksoil,
        "icemelt": icemelt,
    }


def test_hydrol_bucket_snow_vegetation_and_ice_source_order():
    """Fortran: hydrol.f90::hydrol_snow lines 4693-4966."""

    inputs = {
        "precip_rain": np.asarray([2.0, 0.0], dtype=np.float64),
        "precip_snow": np.asarray([1.0, 0.0], dtype=np.float64),
        "temp_sol_new": np.asarray([274.15, 274.15], dtype=np.float64),
        "soilcap": np.asarray([0.3336e6, 0.3336e6], dtype=np.float64),
        "frac_nobio": np.asarray([[0.2], [0.0]], dtype=np.float64),
        "totfrac_nobio": np.asarray([0.2, 0.0], dtype=np.float64),
        "vevapsno": np.asarray([0.5, 0.5], dtype=np.float64),
        "snow": np.asarray([2.0, 0.2], dtype=np.float64),
        "snow_age": np.asarray([4.0, 3.0], dtype=np.float64),
        "snow_nobio": np.asarray([[1.0], [0.0]], dtype=np.float64),
        "snow_nobio_age": np.asarray([[2.0], [0.0]], dtype=np.float64),
    }
    result = hydrol_bucket_snow_step(**inputs)
    expected = [
        _manual_hydrol_bucket_point(
            precip_rain=inputs["precip_rain"][idx],
            precip_snow=inputs["precip_snow"][idx],
            temp_sol_new=inputs["temp_sol_new"][idx],
            soilcap=inputs["soilcap"][idx],
            frac_nobio=inputs["frac_nobio"][idx, 0],
            totfrac_nobio=inputs["totfrac_nobio"][idx],
            vevapsno=inputs["vevapsno"][idx],
            snow=inputs["snow"][idx],
            snow_age=inputs["snow_age"][idx],
            snow_nobio=inputs["snow_nobio"][idx, 0],
            snow_nobio_age=inputs["snow_nobio_age"][idx, 0],
        )
        for idx in range(2)
    ]

    for name in ("snow", "snow_age", "vevapsno", "tot_melt", "snowmelt", "snowdepth", "subsnowveg", "subsinksoil", "icemelt"):
        np.testing.assert_allclose(np.asarray(getattr(result, name)), [item[name] for item in expected])
    for name in ("snow_nobio", "snow_nobio_age", "subsnownobio"):
        np.testing.assert_allclose(np.asarray(getattr(result, name))[:, 0], [item[name] for item in expected])
    assert any("hydrol_snow lines 4693-4966" in item for item in result.provenance)


def test_hydrol_bucket_snow_refuses_unsupported_extra_nonbio_surfaces():
    with pytest.raises(ValueError, match="nnobio > 1"):
        hydrol_bucket_snow_step(
            precip_rain=np.asarray([0.0], dtype=np.float64),
            precip_snow=np.asarray([0.0], dtype=np.float64),
            temp_sol_new=np.asarray([273.15], dtype=np.float64),
            soilcap=np.asarray([0.3336e6], dtype=np.float64),
            frac_nobio=np.asarray([[0.1, 0.2]], dtype=np.float64),
            totfrac_nobio=np.asarray([0.3], dtype=np.float64),
            vevapsno=np.asarray([0.0], dtype=np.float64),
            snow=np.asarray([0.0], dtype=np.float64),
            snow_age=np.asarray([0.0], dtype=np.float64),
            snow_nobio=np.zeros((1, 2), dtype=np.float64),
            snow_nobio_age=np.zeros((1, 2), dtype=np.float64),
        )


def test_explicitsnow_initialize_zero_state_matches_no_restart_defaults():
    state = explicitsnow_initialize_zero_state(kjpindex=2, nsnow=3)

    np.testing.assert_allclose(np.asarray(state.snowrho), 50.0)
    np.testing.assert_allclose(np.asarray(state.snowtemp), 273.15)
    np.testing.assert_allclose(np.asarray(state.snowdz), 0.0)
    np.testing.assert_allclose(np.asarray(state.snowheat), 0.0)
    np.testing.assert_allclose(np.asarray(state.snowgrain), 0.0)
    assert np.asarray(state.snowrho).shape == (2, 3)
    assert any("explicitsnow_initialize lines 56-88" in item for item in state.provenance)

    with pytest.raises(ValueError, match="positive"):
        explicitsnow_initialize_zero_state(kjpindex=0)


def test_hydrol_explicit_snow_zero_state_refuses_nonzero_snowfall_path():
    with pytest.raises(ValueError, match="precip_snow"):
        hydrol_explicit_snow_zero_state(
            precip_snow=np.asarray([1.0], dtype=np.float64),
            precip_rain=np.asarray([0.0], dtype=np.float64),
            vevapsno=np.asarray([0.0], dtype=np.float64),
            snow=np.asarray([0.0], dtype=np.float64),
            snowdz=np.zeros((1, 3), dtype=np.float64),
            snowrho=np.full((1, 3), 50.0, dtype=np.float64),
            snowtemp=np.full((1, 3), 273.15, dtype=np.float64),
            snow_age=np.asarray([0.0], dtype=np.float64),
            snow_nobio=np.zeros((1, 1), dtype=np.float64),
            snow_nobio_age=np.zeros((1, 1), dtype=np.float64),
            frac_nobio=np.zeros((1, 1), dtype=np.float64),
            totfrac_nobio=np.asarray([0.0], dtype=np.float64),
        )


def test_explicitsnow_fall_initializes_empty_snowpack_layers_from_new_snow():
    """Fortran: explicitsnow.f90::explicitsnow_fall lines 1196-1274."""

    precip_snow = np.asarray([6.0], dtype=np.float64)
    temp_air = np.asarray([268.0], dtype=np.float64)
    u = np.asarray([0.0], dtype=np.float64)
    v = np.asarray([0.0], dtype=np.float64)
    totfrac_nobio = np.asarray([0.0], dtype=np.float64)
    snowtemp = np.full((1, 3), 273.15, dtype=np.float64)

    result = explicitsnow_fall_step(
        precip_snow=precip_snow,
        temp_air=temp_air,
        u=u,
        v=v,
        totfrac_nobio=totfrac_nobio,
        snowrho=np.full((1, 3), 50.0, dtype=np.float64),
        snowdz=np.zeros((1, 3), dtype=np.float64),
        snowheat=np.zeros((1, 3), dtype=np.float64),
        snowgrain=np.zeros((1, 3), dtype=np.float64),
        snowtemp=snowtemp,
    )

    speed = 0.1
    rhosnew = max(50.0, 109.0 + 6.0 * (268.0 - 273.15) + 26.0 * np.sqrt(speed))
    dsnowfall = precip_snow[0] / rhosnew
    psnowhmass = precip_snow[0] * (2.106e3 * (snowtemp[0, 0] - 273.15) - 0.3336e6)
    newgrain = min(2.0e-4, np.asarray(snow3lgrain_0d_explicit(rhosnew)))

    np.testing.assert_allclose(np.asarray(result.rhosnew), [rhosnew])
    np.testing.assert_allclose(np.asarray(result.dsnowfall), [dsnowfall])
    np.testing.assert_allclose(np.asarray(result.psnowhmass), [psnowhmass])
    np.testing.assert_allclose(np.asarray(result.snowdz), np.full((1, 3), dsnowfall / 3.0))
    np.testing.assert_allclose(np.asarray(result.snowheat), np.full((1, 3), psnowhmass / 3.0))
    np.testing.assert_allclose(np.asarray(result.snowrho), np.full((1, 3), rhosnew))
    np.testing.assert_allclose(np.asarray(result.snowgrain), np.full((1, 3), newgrain))
    assert any("explicitsnow_fall lines 1139-1274" in item for item in result.provenance)


def test_explicitsnow_fall_adds_new_snow_to_existing_top_layer_only():
    """Fortran: explicitsnow.f90::explicitsnow_fall lines 1209-1243."""

    snowdz = np.asarray([[0.10, 0.20, 0.30]], dtype=np.float64)
    snowrho = np.asarray([[100.0, 150.0, 200.0]], dtype=np.float64)
    snowgrain = np.asarray([[1.0e-4, 1.5e-4, 2.0e-4]], dtype=np.float64)
    snowheat = np.asarray([[-10.0, -20.0, -30.0]], dtype=np.float64)
    snowtemp = np.full((1, 3), 270.0, dtype=np.float64)
    precip_snow = np.asarray([3.0], dtype=np.float64)
    temp_air = np.asarray([271.0], dtype=np.float64)
    u = np.asarray([2.0], dtype=np.float64)
    v = np.asarray([1.0], dtype=np.float64)

    result = explicitsnow_fall_step(
        precip_snow=precip_snow,
        temp_air=temp_air,
        u=u,
        v=v,
        totfrac_nobio=np.asarray([0.0], dtype=np.float64),
        snowrho=snowrho,
        snowdz=snowdz,
        snowheat=snowheat,
        snowgrain=snowgrain,
        snowtemp=snowtemp,
    )

    speed = np.sqrt(5.0)
    rhosnew = max(50.0, 109.0 + 6.0 * (271.0 - 273.15) + 26.0 * np.sqrt(speed))
    dsnowfall = precip_snow[0] / rhosnew
    newgrain = min(2.0e-4, np.asarray(snow3lgrain_0d_explicit(rhosnew)))
    top_dz = snowdz[0, 0] + dsnowfall
    top_rho = (snowdz[0, 0] * snowrho[0, 0] + dsnowfall * rhosnew) / top_dz
    top_grain = (snowdz[0, 0] * snowgrain[0, 0] + dsnowfall * newgrain) / top_dz
    psnowhmass = precip_snow[0] * (2.106e3 * (snowtemp[0, 0] - 273.15) - 0.3336e6)

    expected_dz = snowdz.copy()
    expected_dz[0, 0] = top_dz
    expected_rho = snowrho.copy()
    expected_rho[0, 0] = top_rho
    expected_grain = snowgrain.copy()
    expected_grain[0, 0] = top_grain
    expected_heat = snowheat.copy()
    expected_heat[0, 0] += psnowhmass

    np.testing.assert_allclose(np.asarray(result.snowdz), expected_dz)
    np.testing.assert_allclose(np.asarray(result.snowrho), expected_rho)
    np.testing.assert_allclose(np.asarray(result.snowgrain), expected_grain)
    np.testing.assert_allclose(np.asarray(result.snowheat), expected_heat)


def test_explicitsnow_levels_step_matches_three_source_depth_branches():
    """Fortran: explicitsnow.f90::explicitsnow_levels lines 1568-1630."""

    result = explicitsnow_levels_step(np.asarray([0.03, 0.12, 0.40], dtype=np.float64))

    expected = np.asarray(
        [
            [0.01, 0.01, 0.01],
            [0.03, 0.06, 0.03],
            [0.05, min(0.5, (0.40 - 0.05) * 0.34 + 0.05), 0.40 - min(0.5, (0.40 - 0.05) * 0.34 + 0.05) - 0.05],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(np.asarray(result.snowdz), expected)
    np.testing.assert_allclose(np.asarray(result.snowdz).sum(axis=1), [0.03, 0.12, 0.40])
    assert any("explicitsnow_levels lines 1568-1630" in item for item in result.provenance)

    with pytest.raises(ValueError, match="nsnow=3"):
        explicitsnow_levels_step(np.asarray([0.1], dtype=np.float64), nsnow=4)


def test_snow3l_thermodynamic_conversions_round_trip_dry_and_wet_layers():
    """Fortran: qsat_moisture.f90::snow3lheat/temp/liq 1D helper formulas."""

    snowrho = np.asarray([120.0, 250.0], dtype=np.float64)
    snowdz = np.asarray([0.10, 0.30], dtype=np.float64)
    snowtemp = np.asarray([268.15, 273.15], dtype=np.float64)
    snowliq = np.asarray([0.0, 0.003], dtype=np.float64)

    heat = snow3lheat_explicit(snowliq, snowrho, snowdz, snowtemp)
    diagnosed_temp = snow3ltemp_explicit(heat, snowrho, snowdz)
    diagnosed_liq = snow3lliq_explicit(heat, snowrho, snowdz, diagnosed_temp)
    rebuilt_heat = snow3lheat_explicit(diagnosed_liq, snowrho, snowdz, diagnosed_temp)

    np.testing.assert_allclose(np.asarray(diagnosed_temp), snowtemp)
    np.testing.assert_allclose(np.asarray(diagnosed_liq), snowliq)
    np.testing.assert_allclose(np.asarray(rebuilt_heat), np.asarray(heat))


def test_snow3lhold_explicit_matches_density_limited_source_formula():
    """Fortran: qsat_moisture.f90::snow3lhold_1d lines 659-686."""

    snowrho = np.asarray([100.0, 250.0, 750.0, 900.0], dtype=np.float64)
    snowdz = np.asarray([0.2, 0.2, 0.2, 0.2], dtype=np.float64)

    result = snow3lhold_explicit(snowrho, snowdz)

    zrho = np.minimum(750.0, snowrho)
    hold_ratio = 0.03 + (0.10 - 0.03) * np.maximum(0.0, 200.0 - zrho) / 200.0
    expected = hold_ratio * snowdz * zrho / 1000.0
    expected[zrho >= 750.0] = 0.0
    np.testing.assert_allclose(np.asarray(result), expected)


def test_snow3ltemp_one_dimensional_guard_matches_source_bad_temperature_reset():
    """Fortran: qsat_moisture.f90::snow3ltemp_1d lines 969-970."""

    snowrho = np.asarray([100.0], dtype=np.float64)
    snowdz = np.asarray([0.1], dtype=np.float64)
    very_cold_heat = np.asarray([-1.0e8], dtype=np.float64)

    guarded = snow3ltemp_explicit(very_cold_heat, snowrho, snowdz)
    unguarded = snow3ltemp_explicit(
        very_cold_heat,
        snowrho,
        snowdz,
        one_dimensional_guard=False,
    )

    np.testing.assert_allclose(np.asarray(guarded), [273.15])
    assert np.asarray(unguarded)[0] <= 100.0


def test_explicitsnow_grain_dry_snow_without_temperature_gradient_is_unchanged():
    """Fortran: explicitsnow.f90::explicitsnow_grain dry branch lines 731-740."""

    snowgrain = np.asarray([[2.0e-4, 2.2e-4, 2.4e-4]], dtype=np.float64)
    result = explicitsnow_grain_step(
        snowliq=np.zeros((1, 3), dtype=np.float64),
        snowdz=np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        gtemp=np.asarray([268.0], dtype=np.float64),
        snowtemp=np.asarray([[268.0, 268.0, 268.0]], dtype=np.float64),
        pb=np.asarray([1013.0], dtype=np.float64),
        snowgrain=snowgrain,
    )

    np.testing.assert_allclose(np.asarray(result.snowgrain), snowgrain)
    np.testing.assert_allclose(np.asarray(result.ztheta), 0.0)
    assert any("explicitsnow_grain lines 609-752" in item for item in result.provenance)


def test_explicitsnow_grain_wet_snow_uses_source_growth_limiter():
    """Fortran: explicitsnow.f90::explicitsnow_grain wet branch lines 743-748."""

    snowdz = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64)
    snowliq = np.asarray([[0.002, 0.004, 0.006]], dtype=np.float64)
    snowgrain = np.asarray([[2.0e-4, 2.5e-4, 3.0e-4]], dtype=np.float64)
    result = explicitsnow_grain_step(
        snowliq=snowliq,
        snowdz=snowdz,
        gtemp=np.asarray([273.15], dtype=np.float64),
        snowtemp=np.asarray([[273.15, 273.15, 273.15]], dtype=np.float64),
        pb=np.asarray([1013.0], dtype=np.float64),
        snowgrain=snowgrain,
        dt_sechiba=1800.0,
    )

    theta = snowliq / snowdz
    expected = snowgrain + (1800.0 * 4.0e-12 / snowgrain) * np.minimum(0.14, theta + 0.05)
    np.testing.assert_allclose(np.asarray(result.ztheta), theta)
    np.testing.assert_allclose(np.asarray(result.snowgrain), expected)


def test_explicitsnow_compactn_increases_density_and_conserves_layer_mass():
    """Fortran: explicitsnow.f90::explicitsnow_compactn lines 819-860."""

    snowtemp = np.asarray([[268.15, 271.15, 273.15]], dtype=np.float64)
    snowrho = np.asarray([[100.0, 200.0, 300.0]], dtype=np.float64)
    snowdz = np.asarray([[0.10, 0.20, 0.30]], dtype=np.float64)

    result = explicitsnow_compactn_step(
        snowtemp=snowtemp,
        snowrho=snowrho,
        snowdz=snowdz,
        dt_sechiba=1800.0,
    )

    zsmass = np.cumsum(snowdz * snowrho, axis=1)
    zsettle = 2.8e-6 * np.exp(
        -0.04 * (273.15 - np.minimum(273.15, snowtemp))
        - 460.0 * np.maximum(0.0, snowrho - 150.0)
    )
    zviscocity = 3.7e7 * np.exp(
        0.081 * (273.15 - np.minimum(273.15, snowtemp)) + 0.018 * snowrho
    )
    expected_rho = snowrho + snowrho * 1800.0 * ((9.80665 * zsmass / zviscocity) + zsettle)
    expected_dz = snowdz * (snowrho / expected_rho)

    np.testing.assert_allclose(np.asarray(result.zsmass), zsmass)
    np.testing.assert_allclose(np.asarray(result.zsettle), zsettle)
    np.testing.assert_allclose(np.asarray(result.zviscocity), zviscocity)
    np.testing.assert_allclose(np.asarray(result.snowrho), expected_rho)
    np.testing.assert_allclose(np.asarray(result.snowdz), expected_dz)
    np.testing.assert_allclose(np.asarray(result.snowrho) * np.asarray(result.snowdz), snowrho * snowdz)
    assert any("explicitsnow_compactn lines 760-869" in item for item in result.provenance)


def test_explicitsnow_compactn_leaves_dense_layers_and_snow_free_points_unchanged():
    """Fortran: compaction is skipped when no snow exists or rho >= xrhosmax."""

    snowtemp = np.asarray(
        [
            [268.15, 268.15, 268.15],
            [268.15, 268.15, 268.15],
        ],
        dtype=np.float64,
    )
    snowrho = np.asarray(
        [
            [760.0, 800.0, 900.0],
            [100.0, 200.0, 300.0],
        ],
        dtype=np.float64,
    )
    snowdz = np.asarray(
        [
            [0.10, 0.20, 0.30],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    result = explicitsnow_compactn_step(
        snowtemp=snowtemp,
        snowrho=snowrho,
        snowdz=snowdz,
    )

    np.testing.assert_allclose(np.asarray(result.snowrho), snowrho)
    np.testing.assert_allclose(np.asarray(result.snowdz), snowdz)
    np.testing.assert_allclose(np.asarray(result.zsettle), 2.8e-6)
    np.testing.assert_allclose(np.asarray(result.zviscocity), 3.7e7)


def test_explicitsnow_gone_removes_whole_snowpack_when_energy_covers_heat_deficit():
    """Fortran: explicitsnow.f90::explicitsnow_gone lines 1337-1356."""

    snowheat = np.asarray([[-1800.0, -900.0, -900.0]], dtype=np.float64)
    snowtemp = np.asarray([[268.0, 269.0, 270.0]], dtype=np.float64)
    snowdz = np.asarray([[0.10, 0.20, 0.30]], dtype=np.float64)
    snowrho = np.asarray([[100.0, 150.0, 200.0]], dtype=np.float64)
    snowliq = np.asarray([[0.001, 0.002, 0.003]], dtype=np.float64)

    result = explicitsnow_gone_step(
        pgflux=np.asarray([2.0], dtype=np.float64),
        snowheat=snowheat,
        snowtemp=snowtemp,
        snowdz=snowdz,
        snowrho=snowrho,
        snowliq=snowliq,
        grndflux=np.asarray([10.0], dtype=np.float64),
        snowmelt=np.asarray([99.0], dtype=np.float64),
        soilflxresid=np.asarray([0.5], dtype=np.float64),
        dt_sechiba=1800.0,
    )

    totsnowheat = snowheat.sum(axis=1)
    expected_mass = (snowrho * snowdz).sum(axis=1)
    expected_grndflux = 2.0 + totsnowheat / 1800.0 + 0.5

    np.testing.assert_allclose(np.asarray(result.totsnowheat), totsnowheat)
    np.testing.assert_allclose(np.asarray(result.grndflux), expected_grndflux)
    np.testing.assert_allclose(np.asarray(result.snowmelt), expected_mass)
    np.testing.assert_allclose(np.asarray(result.snowdz), 0.0)
    np.testing.assert_allclose(np.asarray(result.snowliq), 0.0)
    np.testing.assert_allclose(np.asarray(result.snowtemp), 273.15)
    np.testing.assert_allclose(np.asarray(result.snowrho), snowrho)
    assert any("explicitsnow_gone lines 1300-1366" in item for item in result.provenance)


def test_explicitsnow_gone_resets_snowmelt_but_preserves_unmelted_and_snow_free_state():
    """Fortran: gone branch keeps existing snow when energy is insufficient."""

    snowheat = np.asarray(
        [
            [-1800.0, -900.0, -900.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    snowtemp = np.asarray(
        [
            [268.0, 269.0, 270.0],
            [266.0, 267.0, 268.0],
        ],
        dtype=np.float64,
    )
    snowdz = np.asarray(
        [
            [0.10, 0.20, 0.30],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    snowrho = np.asarray(
        [
            [100.0, 150.0, 200.0],
            [50.0, 50.0, 50.0],
        ],
        dtype=np.float64,
    )
    snowliq = np.asarray(
        [
            [0.001, 0.002, 0.003],
            [0.004, 0.005, 0.006],
        ],
        dtype=np.float64,
    )

    result = explicitsnow_gone_step(
        pgflux=np.asarray([-10.0, 5.0], dtype=np.float64),
        snowheat=snowheat,
        snowtemp=snowtemp,
        snowdz=snowdz,
        snowrho=snowrho,
        snowliq=snowliq,
        grndflux=np.asarray([7.0, 8.0], dtype=np.float64),
        snowmelt=np.asarray([99.0, 88.0], dtype=np.float64),
        soilflxresid=np.asarray([0.0, 0.0], dtype=np.float64),
        dt_sechiba=1800.0,
    )

    np.testing.assert_allclose(np.asarray(result.snowmelt), 0.0)
    np.testing.assert_allclose(np.asarray(result.grndflux), [7.0, 8.0])
    np.testing.assert_allclose(np.asarray(result.snowdz), snowdz)
    np.testing.assert_allclose(np.asarray(result.snowtemp), snowtemp)
    expected_liq = snowliq.copy()
    expected_liq[1, :] = 0.0
    np.testing.assert_allclose(np.asarray(result.snowliq), expected_liq)


def test_explicitsnow_profile_updates_snow_temperature_and_clears_surface_addition():
    """Fortran: explicitsnow.f90::explicitsnow_profile lines 1679-1686."""

    cgrnd_snow = np.asarray([[250.0, 245.0, 999.0]], dtype=np.float64)
    dgrnd_snow = np.asarray([[0.20, 0.30, 999.0]], dtype=np.float64)
    lambda_snow = np.asarray([0.5], dtype=np.float64)
    temp_sol_new = np.asarray([270.0], dtype=np.float64)
    temp_sol_add = np.asarray([2.0], dtype=np.float64)
    snowtemp = np.asarray([[260.0, 261.0, 262.0]], dtype=np.float64)
    snowdz = np.asarray([[0.10, 0.20, 0.30]], dtype=np.float64)

    result = explicitsnow_profile_step(
        cgrnd_snow=cgrnd_snow,
        dgrnd_snow=dgrnd_snow,
        lambda_snow=lambda_snow,
        temp_sol_new=temp_sol_new,
        snowtemp=snowtemp,
        snowdz=snowdz,
        temp_sol_add=temp_sol_add,
    )

    first = (0.5 * 250.0 + (270.0 + 2.0)) / (0.5 * (1.0 - 0.20) + 1.0)
    second = 250.0 + 0.20 * first
    third = 245.0 + 0.30 * second
    np.testing.assert_allclose(np.asarray(result.snowtemp), [[first, second, third]])
    np.testing.assert_allclose(np.asarray(result.temp_sol_add), 0.0)
    assert any("explicitsnow_profile lines 1655-1694" in item for item in result.provenance)


def test_explicitsnow_profile_leaves_snow_free_points_unchanged():
    """Fortran: profile loop updates only grid points with positive snow depth."""

    snowtemp = np.asarray([[260.0, 261.0, 262.0]], dtype=np.float64)
    temp_sol_add = np.asarray([2.0], dtype=np.float64)

    result = explicitsnow_profile_step(
        cgrnd_snow=np.asarray([[250.0, 245.0, 999.0]], dtype=np.float64),
        dgrnd_snow=np.asarray([[0.20, 0.30, 999.0]], dtype=np.float64),
        lambda_snow=np.asarray([0.5], dtype=np.float64),
        temp_sol_new=np.asarray([270.0], dtype=np.float64),
        snowtemp=snowtemp,
        snowdz=np.zeros((1, 3), dtype=np.float64),
        temp_sol_add=temp_sol_add,
    )

    np.testing.assert_allclose(np.asarray(result.snowtemp), snowtemp)
    np.testing.assert_allclose(np.asarray(result.temp_sol_add), temp_sol_add)


def test_explicitsnow_transf_stable_three_layer_remap_matches_source_equations():
    """Fortran: explicitsnow.f90::explicitsnow_transf lines 940-1094."""

    snowdz_old = np.asarray([[0.10, 0.20, 0.30]], dtype=np.float64)
    snowdz = np.asarray([[0.05, 0.25, 0.30]], dtype=np.float64)
    snowrho = np.asarray([[100.0, 200.0, 300.0]], dtype=np.float64)
    snowheat = np.asarray([[-10.0, -20.0, -30.0]], dtype=np.float64)
    snowgrain = np.asarray([[1.0e-4, 2.0e-4, 3.0e-4]], dtype=np.float64)

    result = explicitsnow_transf_step(
        snowdz_old=snowdz_old,
        snowdz=snowdz,
        snowrho=snowrho,
        snowheat=snowheat,
        snowgrain=snowgrain,
    )

    zsnowzo = np.cumsum(snowdz_old, axis=1)
    zsnowzn = np.cumsum(snowdz, axis=1)
    zsnowddz = zsnowzn - zsnowzo
    zdelta = np.where(zsnowddz > 0.0, 1.0, 0.0)
    expected_rho = np.zeros_like(snowrho)
    expected_heat = np.zeros_like(snowheat)
    expected_grain = np.zeros_like(snowgrain)
    expected_rho[0, 0] = (
        snowdz_old[0, 0] * snowrho[0, 0]
        + zsnowddz[0, 0] * (zdelta[0, 0] * snowrho[0, 1] + (1.0 - zdelta[0, 0]) * snowrho[0, 0])
    ) / snowdz[0, 0]
    expected_heat[0, 0] = snowheat[0, 0] + zsnowddz[0, 0] * (
        zdelta[0, 0] * snowheat[0, 1] / snowdz_old[0, 1]
        + (1.0 - zdelta[0, 0]) * snowheat[0, 0] / snowdz_old[0, 0]
    )
    expected_grain[0, 0] = (
        snowdz_old[0, 0] * snowgrain[0, 0]
        + zsnowddz[0, 0] * (zdelta[0, 0] * snowgrain[0, 1] + (1.0 - zdelta[0, 0]) * snowgrain[0, 0])
    ) / snowdz[0, 0]
    expected_rho[0, 1] = (
        snowrho[0, 0] * (zsnowzo[0, 0] - zsnowzn[0, 0])
        + snowrho[0, 1] * (zsnowzn[0, 1] - zsnowzo[0, 0])
    ) / snowdz[0, 1]
    expected_heat[0, 1] = (
        snowheat[0, 0] * (zsnowzo[0, 0] - zsnowzn[0, 0]) / snowdz_old[0, 0]
        + snowheat[0, 1] * (zsnowzn[0, 1] - zsnowzo[0, 0]) / snowdz_old[0, 1]
    )
    expected_grain[0, 1] = (
        snowgrain[0, 0] * (zsnowzo[0, 0] - zsnowzn[0, 0])
        + snowgrain[0, 1] * (zsnowzn[0, 1] - zsnowzo[0, 0])
    ) / snowdz[0, 1]
    expected_rho[0, 2] = (
        snowdz_old[0, 2] * snowrho[0, 2]
        - zsnowddz[0, 1] * (zdelta[0, 1] * snowrho[0, 2] + (1.0 - zdelta[0, 1]) * snowrho[0, 1])
    ) / snowdz[0, 2]
    expected_heat[0, 2] = snowheat[0, 2] - zsnowddz[0, 1] * (
        zdelta[0, 1] * snowheat[0, 2] / snowdz_old[0, 2]
        + (1.0 - zdelta[0, 1]) * snowheat[0, 1] / snowdz_old[0, 1]
    )
    expected_grain[0, 2] = (
        snowdz_old[0, 2] * snowgrain[0, 2]
        - zsnowddz[0, 1] * (zdelta[0, 1] * snowgrain[0, 2] + (1.0 - zdelta[0, 1]) * snowgrain[0, 1])
    ) / snowdz[0, 2]

    np.testing.assert_allclose(np.asarray(result.snowrho), expected_rho)
    np.testing.assert_allclose(np.asarray(result.snowheat), expected_heat)
    np.testing.assert_allclose(np.asarray(result.snowgrain), expected_grain)
    np.testing.assert_allclose(np.asarray(result.snowdz), snowdz)
    assert any("explicitsnow_transf lines 891-1132" in item for item in result.provenance)


def test_explicitsnow_transf_thin_or_new_snow_branch_preserves_source_old_dz_weights():
    """Fortran: thin/new snowpack mix branch lines 1108-1126."""

    snowdz_old = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.01, 0.005]], dtype=np.float64)
    snowdz = np.asarray([[0.03, 0.03, 0.03], [0.006, 0.012, 0.006]], dtype=np.float64)
    snowrho = np.asarray([[100.0, 200.0, 300.0], [120.0, 240.0, 360.0]], dtype=np.float64)
    snowheat = np.asarray([[-9.0, -6.0, -3.0], [-3.0, -6.0, -9.0]], dtype=np.float64)
    snowgrain = np.asarray([[1.0e-4, 2.0e-4, 3.0e-4], [2.0e-4, 4.0e-4, 6.0e-4]], dtype=np.float64)

    result = explicitsnow_transf_step(
        snowdz_old=snowdz_old,
        snowdz=snowdz,
        snowrho=snowrho,
        snowheat=snowheat,
        snowgrain=snowgrain,
    )

    psnow = snowdz.sum(axis=1)
    expected_dz = np.repeat((psnow / 3.0)[:, None], 3, axis=1)
    expected_rho = np.repeat(((snowrho * snowdz_old).sum(axis=1) / psnow)[:, None], 3, axis=1)
    expected_heat = np.repeat((snowheat.sum(axis=1) / 3.0)[:, None], 3, axis=1)
    expected_grain = np.repeat(((snowgrain * snowdz_old).sum(axis=1) / psnow)[:, None], 3, axis=1)

    np.testing.assert_allclose(np.asarray(result.snowdz), expected_dz)
    np.testing.assert_allclose(np.asarray(result.snowrho), expected_rho)
    np.testing.assert_allclose(np.asarray(result.snowheat), expected_heat)
    np.testing.assert_allclose(np.asarray(result.snowgrain), expected_grain)


def _reference_melt_refrz_one_point(snowtemp, snowdz, snowrho, snowliq, snowmelt, grndflux):
    ph2o = 1000.0
    xci = 2.106e3
    chalfu0 = 0.3336e6
    tp_00 = 273.15
    dt_sechiba = 1800.0
    xsnowdmin = 1.0e-6
    nsnow = 3

    zsnowlwe = snowrho * snowdz / ph2o
    pcapa_snow = snowrho * xci
    zphase = np.minimum(
        pcapa_snow * np.maximum(0.0, snowtemp - tp_00) * snowdz,
        np.maximum(0.0, zsnowlwe - snowliq) * chalfu0 * ph2o,
    )
    zsnowmelt = zphase / (chalfu0 * ph2o)
    zsnowtemp = snowtemp - zphase / (pcapa_snow * snowdz)
    snowtemp = np.minimum(tp_00, zsnowtemp)
    zmeltxs = (zsnowtemp - snowtemp) * pcapa_snow * snowdz
    zwholdmax = np.asarray(snow3lhold_explicit(snowrho, snowdz))
    zcmprsfact = (zsnowlwe - np.minimum(snowliq + zsnowmelt, zwholdmax)) / (
        zsnowlwe - np.minimum(snowliq, zwholdmax)
    )
    snowdz = snowdz * zcmprsfact
    snowrho = zsnowlwe * ph2o / snowdz
    snowliq = snowliq + zsnowmelt

    zscap = snowrho * xci
    zphase2 = np.minimum(
        zscap * np.maximum(0.0, tp_00 - snowtemp) * snowdz,
        snowliq * chalfu0 * ph2o,
    )
    zsnowdz = np.maximum(xsnowdmin / nsnow, snowdz)
    snowtemp_old = snowtemp.copy()
    snowtemp = snowtemp + zphase2 / (zscap * zsnowdz)
    snowliq = snowliq - ((snowtemp - snowtemp_old) * zscap * zsnowdz / (chalfu0 * ph2o))
    snowliq = np.maximum(snowliq, 0.0)

    zwholdmax = np.asarray(snow3lhold_explicit(snowrho, snowdz))
    flowliq = np.maximum(0.0, snowliq - zwholdmax)
    snowliq = snowliq - flowliq
    snowdz = np.maximum(0.0, snowdz - flowliq * ph2o / snowrho)

    zflowliqt = np.zeros(4, dtype=np.float64)
    for jj in range(nsnow):
        zflowliqt[jj + 1] = flowliq[jj]
    flowliq = np.zeros(3, dtype=np.float64)
    zsnowliq = snowliq.copy()
    for jj in range(nsnow):
        snowliq[jj] = snowliq[jj] + zflowliqt[jj]
        flowliq[jj] = max(0.0, snowliq[jj] - zwholdmax[jj])
        snowliq[jj] = snowliq[jj] - flowliq[jj]
        snowrho[jj] = snowrho[jj] + (snowliq[jj] - zsnowliq[jj]) * ph2o / max(xsnowdmin / nsnow, snowdz[jj])
        zflowliqt[jj + 1] = zflowliqt[jj + 1] + flowliq[jj]

    snowmelt = snowmelt + zflowliqt[nsnow] * ph2o
    meltxs = np.sum(zmeltxs) / dt_sechiba
    grndflux = grndflux + meltxs
    return snowtemp, snowdz, snowrho, snowliq, snowmelt, grndflux, meltxs


def test_explicitsnow_melt_refrz_matches_source_reference_for_melt_refreeze_and_flow():
    """Fortran: explicitsnow.f90::explicitsnow_melt_refrz lines 1446-1553."""

    snowtemp = np.asarray([[274.15, 272.15, 273.15]], dtype=np.float64)
    snowdz = np.asarray([[0.10, 0.20, 0.30]], dtype=np.float64)
    snowrho = np.asarray([[100.0, 200.0, 300.0]], dtype=np.float64)
    snowliq = np.asarray([[0.0, 0.001, 0.04]], dtype=np.float64)
    snowmelt = np.asarray([2.0], dtype=np.float64)
    grndflux = np.asarray([1.0], dtype=np.float64)

    result = explicitsnow_melt_refrz_step(
        precip_rain=np.asarray([5.0], dtype=np.float64),
        pgflux=np.asarray([0.0], dtype=np.float64),
        soilcap=np.asarray([2.0e6], dtype=np.float64),
        snowtemp=snowtemp,
        snowdz=snowdz,
        snowrho=snowrho,
        snowliq=snowliq,
        snowmelt=snowmelt,
        grndflux=grndflux,
        temp_air=np.asarray([275.0], dtype=np.float64),
        soilflxresid=np.asarray([0.0], dtype=np.float64),
    )
    ref = _reference_melt_refrz_one_point(
        snowtemp[0].copy(),
        snowdz[0].copy(),
        snowrho[0].copy(),
        snowliq[0].copy(),
        snowmelt[0],
        grndflux[0],
    )

    np.testing.assert_allclose(np.asarray(result.snowtemp)[0], ref[0])
    np.testing.assert_allclose(np.asarray(result.snowdz)[0], ref[1])
    np.testing.assert_allclose(np.asarray(result.snowrho)[0], ref[2])
    np.testing.assert_allclose(np.asarray(result.snowliq)[0], ref[3])
    np.testing.assert_allclose(np.asarray(result.snowmelt), [ref[4]])
    np.testing.assert_allclose(np.asarray(result.grndflux), [ref[5]])
    np.testing.assert_allclose(np.asarray(result.meltxs), [ref[6]])
    assert any("explicitsnow_melt_refrz lines 1385-1563" in item for item in result.provenance)


def test_explicitsnow_melt_refrz_snow_free_branch_clears_depth_and_liquid_only():
    """Fortran: melt/refrz else branch lines 1556-1559."""

    snowtemp = np.asarray([[270.0, 271.0, 272.0]], dtype=np.float64)
    snowrho = np.asarray([[50.0, 50.0, 50.0]], dtype=np.float64)

    result = explicitsnow_melt_refrz_step(
        precip_rain=np.asarray([1.0], dtype=np.float64),
        pgflux=np.asarray([0.0], dtype=np.float64),
        soilcap=np.asarray([2.0e6], dtype=np.float64),
        snowtemp=snowtemp,
        snowdz=np.zeros((1, 3), dtype=np.float64),
        snowrho=snowrho,
        snowliq=np.asarray([[0.001, 0.002, 0.003]], dtype=np.float64),
        snowmelt=np.asarray([7.0], dtype=np.float64),
        grndflux=np.asarray([8.0], dtype=np.float64),
        temp_air=np.asarray([275.0], dtype=np.float64),
        soilflxresid=np.asarray([0.0], dtype=np.float64),
    )

    np.testing.assert_allclose(np.asarray(result.snowdz), 0.0)
    np.testing.assert_allclose(np.asarray(result.snowliq), 0.0)
    np.testing.assert_allclose(np.asarray(result.snowrho), snowrho)
    np.testing.assert_allclose(np.asarray(result.snowtemp), snowtemp)
    np.testing.assert_allclose(np.asarray(result.snowmelt), [7.0])
    np.testing.assert_allclose(np.asarray(result.grndflux), [8.0])


def test_explicitsnow_sublimation_depletes_vegetated_snow_and_limits_vevapsno():
    """Fortran: explicitsnow_main lines 261-285 depletion branch."""

    snowrho = np.asarray([[100.0, 100.0, 100.0]], dtype=np.float64)
    snowdz = np.asarray([[0.002, 0.002, 0.001]], dtype=np.float64)
    snowliq = np.asarray([[0.001, 0.002, 0.003]], dtype=np.float64)
    snowtemp = np.asarray([[268.0, 269.0, 270.0]], dtype=np.float64)

    result = explicitsnow_sublimation_step(
        vevapsno=np.asarray([1.0], dtype=np.float64),
        frac_nobio=np.asarray([[0.0]], dtype=np.float64),
        totfrac_nobio=np.asarray([0.25], dtype=np.float64),
        snowrho=snowrho,
        snowdz=snowdz,
        snowliq=snowliq,
        snowtemp=snowtemp,
    )

    np.testing.assert_allclose(np.asarray(result.snow), [0.0])
    np.testing.assert_allclose(np.asarray(result.snowdz), 0.0)
    np.testing.assert_allclose(np.asarray(result.snowliq), 0.0)
    np.testing.assert_allclose(np.asarray(result.snowtemp), 273.15)
    np.testing.assert_allclose(np.asarray(result.subsnowveg), [0.5])
    np.testing.assert_allclose(np.asarray(result.subsinksoil), [(1.0 - 0.5) / 0.75])
    np.testing.assert_allclose(np.asarray(result.vevapsno), [0.5])
    assert any("explicitsnow_main lines 255-323" in item for item in result.provenance)


def test_explicitsnow_sublimation_partially_removes_layers_in_source_order():
    """Fortran: partial sublimation removes from upper layers first."""

    snowrho = np.asarray([[100.0, 200.0, 300.0]], dtype=np.float64)
    snowdz = np.asarray([[0.01, 0.02, 0.03]], dtype=np.float64)
    snowliq = np.asarray([[0.001, 0.002, 0.003]], dtype=np.float64)
    snowtemp = np.asarray([[268.0, 269.0, 270.0]], dtype=np.float64)

    result = explicitsnow_sublimation_step(
        vevapsno=np.asarray([2.0], dtype=np.float64),
        frac_nobio=np.asarray([[0.0]], dtype=np.float64),
        totfrac_nobio=np.asarray([0.0], dtype=np.float64),
        snowrho=snowrho,
        snowdz=snowdz,
        snowliq=snowliq,
        snowtemp=snowtemp,
    )

    expected_dz = snowdz.copy()
    expected_dz[0, 0] = 0.0
    expected_dz[0, 1] = 0.02 - (2.0 - 1.0) / 200.0
    expected_snow = (snowrho * expected_dz).sum(axis=1)
    np.testing.assert_allclose(np.asarray(result.snowdz), expected_dz)
    np.testing.assert_allclose(np.asarray(result.snow), expected_snow)
    np.testing.assert_allclose(np.asarray(result.snowliq), snowliq)
    np.testing.assert_allclose(np.asarray(result.snowtemp), snowtemp)
    np.testing.assert_allclose(np.asarray(result.subsnowveg), [2.0])
    np.testing.assert_allclose(np.asarray(result.subsnownobio), 0.0)


def test_explicitsnow_maxmass_removes_excess_from_bottom_layer_first():
    """Fortran: explicitsnow_main lines 329-360 bottom-layer maxmass removal."""

    snowrho = np.asarray([[500.0, 500.0, 500.0]], dtype=np.float64)
    snowdz = np.asarray([[2.0, 3.0, 4.0]], dtype=np.float64)
    chalfu0 = 0.3336e6

    result = explicitsnow_maxmass_step(
        soilcap=np.asarray([chalfu0], dtype=np.float64),
        snowrho=snowrho,
        snowdz=snowdz,
        chalfu0=chalfu0,
    )

    expected_melt = 3.0
    expected_dz = snowdz.copy()
    expected_dz[0, 2] -= expected_melt / 500.0
    np.testing.assert_allclose(np.asarray(result.snowmelt_from_maxmass), [expected_melt])
    np.testing.assert_allclose(np.asarray(result.snowdz), expected_dz)
    np.testing.assert_allclose(np.asarray(result.snow), (snowrho * expected_dz).sum(axis=1))
    assert any("explicitsnow_main lines 325-376" in item for item in result.provenance)


def test_explicitsnow_maxmass_can_remove_lower_layers_and_part_of_top_layer():
    """Fortran: maxmass branch lines 361-370 for locjj < nsnow."""

    snowrho = np.asarray([[500.0, 500.0, 500.0]], dtype=np.float64)
    snowdz = np.asarray([[2.0, 3.0, 4.0]], dtype=np.float64)

    result = explicitsnow_maxmass_step(
        soilcap=np.asarray([1.0e12], dtype=np.float64),
        snowrho=snowrho,
        snowdz=snowdz,
        maxmass_snow=500.0,
    )

    expected_melt = 4000.0
    expected_dz = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(np.asarray(result.snowmelt_from_maxmass), [expected_melt])
    np.testing.assert_allclose(np.asarray(result.snowdz), expected_dz)
    np.testing.assert_allclose(np.asarray(result.snow), [500.0])


def test_explicitsnow_age_ice_updates_land_age_and_ice_snow_state():
    """Fortran: explicitsnow_main lines 408-488."""

    chalfu0 = 0.3336e6
    result = explicitsnow_age_ice_step(
        precip_snow=np.asarray([0.1], dtype=np.float64),
        precip_rain=np.asarray([0.2], dtype=np.float64),
        temp_sol_new_old=np.asarray([274.15], dtype=np.float64),
        temp_sol_new=np.asarray([270.15], dtype=np.float64),
        soilcap=np.asarray([chalfu0 * 0.2], dtype=np.float64),
        frac_nobio=np.asarray([[0.5]], dtype=np.float64),
        subsnownobio=np.asarray([[0.1]], dtype=np.float64),
        snow=np.asarray([10.0], dtype=np.float64),
        snow_age=np.asarray([5.0], dtype=np.float64),
        snow_nobio=np.asarray([[1.0]], dtype=np.float64),
        snow_nobio_age=np.asarray([[4.0]], dtype=np.float64),
    )

    dt_days = 1800.0 / 86400.0
    expected_snow_age = (5.0 + (1.0 - 5.0 / 50.0) * dt_days) * np.exp(-0.1 / 0.2)
    ice_before_melt = 1.0 + 0.5 * (0.1 + 0.2) - 0.1
    expected_snowmelt_ice = 0.5 * (274.15 - 273.15) * (chalfu0 * 0.2) / chalfu0
    expected_ice_snow = ice_before_melt - expected_snowmelt_ice
    d_age = (4.0 + (1.0 - 4.0 / 50.0) * dt_days) * np.exp(-0.1 / 0.2) - 4.0
    expected_ice_age = max(4.0 + d_age, 0.0)

    np.testing.assert_allclose(np.asarray(result.snow_age), [expected_snow_age])
    np.testing.assert_allclose(np.asarray(result.snowmelt_ice), [expected_snowmelt_ice])
    np.testing.assert_allclose(np.asarray(result.icemelt), [0.0])
    np.testing.assert_allclose(np.asarray(result.snow_nobio), [[expected_ice_snow]])
    np.testing.assert_allclose(np.asarray(result.snow_nobio_age), [[expected_ice_age]])
    assert any("explicitsnow_main lines 408-499" in item for item in result.provenance)


def test_explicitsnow_age_ice_caps_land_ice_snow_at_maxmass():
    """Fortran: explicitsnow_main lines 436-440."""

    result = explicitsnow_age_ice_step(
        precip_snow=np.asarray([0.0], dtype=np.float64),
        precip_rain=np.asarray([0.0], dtype=np.float64),
        temp_sol_new_old=np.asarray([270.0], dtype=np.float64),
        temp_sol_new=np.asarray([270.0], dtype=np.float64),
        soilcap=np.asarray([1.0], dtype=np.float64),
        frac_nobio=np.asarray([[1.0]], dtype=np.float64),
        subsnownobio=np.asarray([[0.0]], dtype=np.float64),
        snow=np.asarray([0.0], dtype=np.float64),
        snow_age=np.asarray([5.0], dtype=np.float64),
        snow_nobio=np.asarray([[3500.0]], dtype=np.float64),
        snow_nobio_age=np.asarray([[4.0]], dtype=np.float64),
    )

    np.testing.assert_allclose(np.asarray(result.snow_age), [0.0])
    np.testing.assert_allclose(np.asarray(result.snow_nobio), [[3000.0]])
    np.testing.assert_allclose(np.asarray(result.icemelt), [500.0])


def test_explicitsnow_liquid_heat_excess_limits_liquid_and_adds_ground_flux():
    """Fortran: explicitsnow_main lines 386-402."""

    snowliq = np.asarray([[0.020, 0.005, 0.100]], dtype=np.float64)
    snowdz = np.asarray([[0.10, 0.20, 0.30]], dtype=np.float64)
    snowrho = np.asarray([[100.0, 200.0, 300.0]], dtype=np.float64)
    grndflux = np.asarray([1.0], dtype=np.float64)

    result = explicitsnow_liquid_heat_excess_step(
        snowliq=snowliq,
        snowdz=snowdz,
        snowrho=snowrho,
        grndflux=grndflux,
    )

    excess_mass = np.maximum(0.0, snowliq * 1000.0 - 0.10 * snowdz * snowrho)
    expected_flux = excess_mass * 0.3336e6 / 1800.0
    expected_liq = np.maximum(0.0, snowliq - expected_flux * 1800.0 / (1000.0 * 0.3336e6))
    np.testing.assert_allclose(np.asarray(result.zliqheatxs), expected_flux)
    np.testing.assert_allclose(np.asarray(result.snowliq), expected_liq)
    np.testing.assert_allclose(np.asarray(result.grndflux), grndflux + expected_flux.sum(axis=1))
    assert any("explicitsnow_main lines 386-402" in item for item in result.provenance)


def test_explicitsnow_final_cleanup_resets_empty_snow_and_sums_total_melt():
    """Fortran: explicitsnow_main lines 493-515."""

    snowrho = np.asarray([[100.0, 200.0, 300.0], [110.0, 210.0, 310.0]], dtype=np.float64)
    snowgrain = np.asarray([[1.0e-4, 2.0e-4, 3.0e-4], [4.0e-4, 5.0e-4, 6.0e-4]], dtype=np.float64)
    snowdz = np.asarray([[0.10, 0.20, 0.30], [0.11, 0.21, 0.31]], dtype=np.float64)
    snowliq = np.asarray([[0.001, 0.002, 0.003], [0.004, 0.005, 0.006]], dtype=np.float64)

    result = explicitsnow_final_cleanup_step(
        snow=np.asarray([0.0, 10.0], dtype=np.float64),
        snowrho=snowrho,
        snowgrain=snowgrain,
        snowdz=snowdz,
        snowliq=snowliq,
        icemelt=np.asarray([1.0, 2.0], dtype=np.float64),
        snowmelt=np.asarray([3.0, 4.0], dtype=np.float64),
        snowmelt_ice=np.asarray([5.0, 6.0], dtype=np.float64),
        snowmelt_from_maxmass=np.asarray([7.0, 8.0], dtype=np.float64),
    )

    expected_rho = snowrho.copy()
    expected_rho[0, :] = 50.0
    np.testing.assert_allclose(np.asarray(result.snowrho), expected_rho)
    np.testing.assert_allclose(np.asarray(result.snowgrain)[0], 0.0)
    np.testing.assert_allclose(np.asarray(result.snowdz)[0], 0.0)
    np.testing.assert_allclose(np.asarray(result.snowliq)[0], 0.0)
    np.testing.assert_allclose(np.asarray(result.snowgrain)[1], snowgrain[1])
    np.testing.assert_allclose(np.asarray(result.tot_melt), [16.0, 20.0])
    assert any("explicitsnow_main lines 493-515" in item for item in result.provenance)


def test_explicitsnow_main_step_preserves_source_ordered_zero_snow_path():
    """Fortran: explicitsnow_main lines 198-515 with no incoming snow."""

    result = explicitsnow_main_step(
        precip_rain=np.asarray([0.0], dtype=np.float64),
        precip_snow=np.asarray([0.0], dtype=np.float64),
        temp_air=np.asarray([280.0], dtype=np.float64),
        pb=np.asarray([1013.0], dtype=np.float64),
        u=np.asarray([0.0], dtype=np.float64),
        v=np.asarray([0.0], dtype=np.float64),
        temp_sol_new=np.asarray([280.0], dtype=np.float64),
        soilcap=np.asarray([2.0e6], dtype=np.float64),
        pgflux=np.asarray([0.0], dtype=np.float64),
        frac_nobio=np.zeros((1, 1), dtype=np.float64),
        totfrac_nobio=np.asarray([0.0], dtype=np.float64),
        gtemp=np.asarray([280.0], dtype=np.float64),
        lambda_snow=np.asarray([0.0], dtype=np.float64),
        cgrnd_snow=np.zeros((1, 3), dtype=np.float64),
        dgrnd_snow=np.zeros((1, 3), dtype=np.float64),
        vevapsno=np.asarray([0.0], dtype=np.float64),
        snow_age=np.asarray([5.0], dtype=np.float64),
        snow_nobio_age=np.asarray([[2.0]], dtype=np.float64),
        snow_nobio=np.asarray([[0.0]], dtype=np.float64),
        snowrho=np.full((1, 3), 50.0, dtype=np.float64),
        snowgrain=np.zeros((1, 3), dtype=np.float64),
        snowdz=np.zeros((1, 3), dtype=np.float64),
        snowtemp=np.full((1, 3), 273.15, dtype=np.float64),
        snowheat=np.zeros((1, 3), dtype=np.float64),
        snow=np.asarray([0.0], dtype=np.float64),
        temp_sol_add=np.asarray([1.5], dtype=np.float64),
        snowliq=np.zeros((1, 3), dtype=np.float64),
        subsnownobio=np.zeros((1, 1), dtype=np.float64),
        grndflux=np.asarray([99.0], dtype=np.float64),
        snowmelt=np.asarray([42.0], dtype=np.float64),
        soilflxresid=np.asarray([0.0], dtype=np.float64),
    )

    np.testing.assert_allclose(np.asarray(result.snow), [0.0])
    np.testing.assert_allclose(np.asarray(result.snowdz), 0.0)
    np.testing.assert_allclose(np.asarray(result.snowrho), 50.0)
    np.testing.assert_allclose(np.asarray(result.snowliq), 0.0)
    np.testing.assert_allclose(np.asarray(result.snowgrain), 0.0)
    np.testing.assert_allclose(np.asarray(result.snow_age), [0.0])
    np.testing.assert_allclose(np.asarray(result.tot_melt), [0.0])
    np.testing.assert_allclose(np.asarray(result.snowmelt), [0.0])
    np.testing.assert_allclose(np.asarray(result.temp_sol_add), [1.5])
    np.testing.assert_allclose(np.asarray(result.grndflux), [0.0])
    assert any("explicitsnow_main lines 108-553" in item for item in result.provenance)


def test_explicitsnow_main_step_runs_nonzero_snowfall_sequence_without_silent_inputs():
    """Fortran: explicitsnow_main assembled from source-backed sub-kernels."""

    result = explicitsnow_main_step(
        precip_rain=np.asarray([0.0], dtype=np.float64),
        precip_snow=np.asarray([6.0], dtype=np.float64),
        temp_air=np.asarray([268.0], dtype=np.float64),
        pb=np.asarray([1013.0], dtype=np.float64),
        u=np.asarray([0.0], dtype=np.float64),
        v=np.asarray([0.0], dtype=np.float64),
        temp_sol_new=np.asarray([268.0], dtype=np.float64),
        soilcap=np.asarray([2.0e6], dtype=np.float64),
        pgflux=np.asarray([-5.0], dtype=np.float64),
        frac_nobio=np.zeros((1, 1), dtype=np.float64),
        totfrac_nobio=np.asarray([0.0], dtype=np.float64),
        gtemp=np.asarray([268.0], dtype=np.float64),
        lambda_snow=np.asarray([0.5], dtype=np.float64),
        cgrnd_snow=np.asarray([[260.0, 258.0, 0.0]], dtype=np.float64),
        dgrnd_snow=np.asarray([[0.10, 0.20, 0.0]], dtype=np.float64),
        vevapsno=np.asarray([0.0], dtype=np.float64),
        snow_age=np.asarray([0.0], dtype=np.float64),
        snow_nobio_age=np.zeros((1, 1), dtype=np.float64),
        snow_nobio=np.zeros((1, 1), dtype=np.float64),
        snowrho=np.full((1, 3), 50.0, dtype=np.float64),
        snowgrain=np.zeros((1, 3), dtype=np.float64),
        snowdz=np.zeros((1, 3), dtype=np.float64),
        snowtemp=np.full((1, 3), 273.15, dtype=np.float64),
        snowheat=np.zeros((1, 3), dtype=np.float64),
        snow=np.asarray([0.0], dtype=np.float64),
        temp_sol_add=np.asarray([0.0], dtype=np.float64),
        snowliq=np.zeros((1, 3), dtype=np.float64),
        subsnownobio=np.zeros((1, 1), dtype=np.float64),
        grndflux=np.asarray([0.0], dtype=np.float64),
        snowmelt=np.asarray([0.0], dtype=np.float64),
        soilflxresid=np.asarray([0.0], dtype=np.float64),
    )

    assert np.asarray(result.snowdz).shape == (1, 3)
    assert np.asarray(result.snowrho).shape == (1, 3)
    assert np.asarray(result.snowheat).shape == (1, 3)
    assert np.all(np.asarray(result.snowdz) >= 0.0)
    assert np.all(np.asarray(result.snowrho) >= 0.0)
    assert np.all(np.asarray(result.snowliq) >= 0.0)
    np.testing.assert_allclose(np.asarray(result.snow), np.asarray(result.snowrho * result.snowdz).sum(axis=1))
    np.testing.assert_allclose(np.asarray(result.subsnownobio), 0.0)
    np.testing.assert_allclose(np.asarray(result.subsinksoil), 0.0)
    assert result.notes[0] == "Full adapter is source-ordered for nsnow=3 and nnobio=1."


def test_hydrol_explicit_snow_step_returns_hydrol_state_and_inout_boundaries():
    """Fortran: hydrol.f90::hydrol_main lines 1177-1197."""

    result = hydrol_explicit_snow_step(
        precip_rain=np.asarray([0.0], dtype=np.float64),
        precip_snow=np.asarray([6.0], dtype=np.float64),
        temp_air=np.asarray([268.0], dtype=np.float64),
        pb=np.asarray([1013.0], dtype=np.float64),
        u=np.asarray([0.0], dtype=np.float64),
        v=np.asarray([0.0], dtype=np.float64),
        temp_sol_new=np.asarray([268.0], dtype=np.float64),
        soilcap=np.asarray([2.0e6], dtype=np.float64),
        pgflux=np.asarray([-5.0], dtype=np.float64),
        frac_nobio=np.zeros((1, 1), dtype=np.float64),
        totfrac_nobio=np.asarray([0.0], dtype=np.float64),
        gtemp=np.asarray([268.0], dtype=np.float64),
        lambda_snow=np.asarray([0.5], dtype=np.float64),
        cgrnd_snow=np.asarray([[260.0, 258.0, 0.0]], dtype=np.float64),
        dgrnd_snow=np.asarray([[0.10, 0.20, 0.0]], dtype=np.float64),
        vevapsno=np.asarray([0.0], dtype=np.float64),
        snow_age=np.asarray([0.0], dtype=np.float64),
        snow_nobio_age=np.zeros((1, 1), dtype=np.float64),
        snow_nobio=np.zeros((1, 1), dtype=np.float64),
        snowrho=np.full((1, 3), 50.0, dtype=np.float64),
        snowgrain=np.zeros((1, 3), dtype=np.float64),
        snowdz=np.zeros((1, 3), dtype=np.float64),
        snowtemp=np.full((1, 3), 273.15, dtype=np.float64),
        snowheat=np.zeros((1, 3), dtype=np.float64),
        snow=np.asarray([0.0], dtype=np.float64),
        temp_sol_add=np.asarray([0.0], dtype=np.float64),
        snowliq=np.zeros((1, 3), dtype=np.float64),
        subsnownobio=np.zeros((1, 1), dtype=np.float64),
        grndflux=np.asarray([0.0], dtype=np.float64),
        snowmelt=np.asarray([0.0], dtype=np.float64),
        soilflxresid=np.asarray([0.0], dtype=np.float64),
    )

    assert np.asarray(result.snow_state.snowdz).shape == (1, 3)
    assert np.asarray(result.snowliq).shape == (1, 3)
    np.testing.assert_allclose(
        np.asarray(result.snow_state.snow),
        np.asarray(result.snow_state.snowrho * result.snow_state.snowdz).sum(axis=1),
    )
    np.testing.assert_allclose(np.asarray(result.subsnownobio), 0.0)
    np.testing.assert_allclose(np.asarray(result.subsinksoil), 0.0)
    assert any("hydrol_main lines 1177-1197" in item for item in result.provenance)
    assert any("explicitsnow_main lines 108-553" in item for item in result.snow_state.provenance)


def test_hydrol_cold_start_state_matches_hydrol_init_missing_restart_defaults():
    veget_max = np.asarray([[0.0, 0.25, 0.75]], dtype=np.float64)
    soiltile = np.asarray([[0.0, 0.25, 0.75, 0.0, 0.0, 0.0]], dtype=np.float64)
    pref_soil_veg = np.asarray([1, 2, 3], dtype=np.int32)

    state = hydrol_cold_start_state(
        veget_max=veget_max,
        soiltile=soiltile,
        pref_soil_veg=pref_soil_veg,
        nslm=4,
        nsnow=3,
        peat_hydro=True,
        peat_nodr=True,
        tides=True,
    )

    np.testing.assert_allclose(np.asarray(state.mc), 0.3)
    np.testing.assert_allclose(np.asarray(state.mcl), np.asarray(state.mc))
    np.testing.assert_allclose(np.asarray(state.us), 0.0)
    np.testing.assert_allclose(np.asarray(state.humrelv), 0.0)
    np.testing.assert_allclose(np.asarray(state.vegstressv), 0.0)
    np.testing.assert_allclose(np.asarray(state.humrel), 0.0)
    np.testing.assert_allclose(np.asarray(state.water2infilt), 0.0)
    np.testing.assert_allclose(np.asarray(state.ae_ns), 0.0)
    np.testing.assert_allclose(np.asarray(state.evap_bare_lim_ns), 0.0)
    np.testing.assert_allclose(np.asarray(state.evap_bare_lim), 0.0)
    np.testing.assert_allclose(np.asarray(state.zwt_force), 1.0e20)
    assert state.zforce is False
    np.testing.assert_allclose(np.asarray(state.free_drain_coef), [[1.0, 1.0, 1.0, 0.0, 1.0, 0.0]])
    np.testing.assert_allclose(np.asarray(state.snow), 0.0)
    np.testing.assert_allclose(np.asarray(state.snow_age), 0.0)
    np.testing.assert_allclose(np.asarray(state.snow_nobio), 0.0)
    np.testing.assert_allclose(np.asarray(state.qsintveg), 0.0)
    np.testing.assert_allclose(np.asarray(state.profil_froz_hydro), 0.0)
    np.testing.assert_allclose(np.asarray(state.profil_froz_hydro_ns), 0.0)
    np.testing.assert_allclose(np.asarray(state.kk), 276.48)
    np.testing.assert_allclose(np.asarray(state.kk_moy), 276.48)
    np.testing.assert_allclose(np.asarray(state.temp_hydro), 280.0)
    np.testing.assert_allclose(np.asarray(state.run2peat), 0.0)
    np.testing.assert_allclose(np.asarray(state.wt_ab_tide), 0.0)
    np.testing.assert_allclose(np.asarray(state.resdist), soiltile)
    np.testing.assert_allclose(np.asarray(state.vegtot), [1.0])
    np.testing.assert_array_equal(np.asarray(state.mask_veget), [[0, 1, 1]])
    np.testing.assert_array_equal(np.asarray(state.mask_soiltile), [[0, 1, 1, 0, 0, 0]])
    assert state.explicit_snow is not None
    np.testing.assert_allclose(np.asarray(state.explicit_snow.snowrho), 50.0)
    assert any("hydrol_init lines 2813-2843" in item for item in state.provenance)


def test_hydrol_cold_start_thermosoil_moisture_inputs_follow_var_init_mapping():
    veget_max = np.asarray([[0.0, 0.25, 0.75]], dtype=np.float64)
    soiltile = np.asarray([[0.0, 0.25, 0.75]], dtype=np.float64)
    pref_soil_veg = np.asarray([1, 2, 3], dtype=np.int32)
    dz_mm = np.asarray([1.0, 3.0, 6.0, 8.0], dtype=np.float64)
    dh_mm = np.asarray([1.0, 3.0, 6.0, 8.0], dtype=np.float64)
    state = hydrol_cold_start_state(
        veget_max=veget_max,
        soiltile=soiltile,
        pref_soil_veg=pref_soil_veg,
        nslm=4,
        ok_freeze_cwrr=False,
    )

    payload = hydrol_cold_start_thermosoil_moisture_inputs(
        state,
        pref_soil_veg=pref_soil_veg,
        dz_mm=dz_mm,
        dh_mm=dh_mm,
        njsc=np.asarray([2], dtype=np.int32),
        peat_hydro=False,
    )

    np.testing.assert_allclose(np.asarray(payload.mc_layh), 0.3)
    np.testing.assert_allclose(np.asarray(payload.mcl_layh), 0.3)
    for jv, jst_fortran in enumerate(pref_soil_veg):
        jst = int(jst_fortran) - 1
        np.testing.assert_allclose(np.asarray(payload.mc_layh_pft)[:, :, jv], np.asarray(state.mc)[:, :, jst])
        np.testing.assert_allclose(np.asarray(payload.mcl_layh_pft)[:, :, jv], np.asarray(state.mcl)[:, :, jst])
    expected_soilmoist = np.zeros((1, 4), dtype=np.float64)
    for jst in range(soiltile.shape[1]):
        expected_soilmoist += soiltile[:, jst, None] * np.asarray(
            hydrol_layer_moisture_content(np.asarray(state.mc)[:, :, jst], dz_mm)
        )
    expected_soilmoist *= np.asarray(state.vegtot_old)[:, None]
    np.testing.assert_allclose(np.asarray(payload.tmc_layh), expected_soilmoist)
    expected_shumdiag_perma = np.clip(expected_soilmoist / dh_mm[None, :], 0.0, 1.0)
    np.testing.assert_allclose(np.asarray(payload.shumdiag_perma), expected_shumdiag_perma)
    assert any("hydrol_var_init lines 4512-4644" in item for item in payload.provenance)
