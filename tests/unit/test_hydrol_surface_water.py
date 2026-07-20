from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import (  # noqa: E402
    hydrol_irrigation_demand_ratio,
    hydrol_routing_soil_water_split,
    hydrol_soil_surface_water_setup,
)


def test_hydrol_irrigation_demand_ratio_matches_fortran_drip_normalization():
    veget = np.asarray([[0.0, 0.30, 0.20, 0.10]], dtype=np.float64)
    veget_max = np.asarray([[0.0, 0.40, 0.20, 0.10]], dtype=np.float64)
    transpot = np.asarray([[0.0, 0.80, 0.40, 0.50]], dtype=np.float64)
    evapot = np.asarray([0.20], dtype=np.float64)
    precip_rain = np.asarray([0.10], dtype=np.float64)
    vegstress_old = np.asarray([[1.0, 0.20, 0.30, 0.90]], dtype=np.float64)
    soil_deficit = np.zeros_like(veget)
    ok_laidev = np.asarray([False, True, True, True])
    threshold = np.asarray([0.0, 0.50, 0.50, 0.50], dtype=np.float64)
    fulfill = np.asarray([0.0, 2.0, 4.0, 1.0], dtype=np.float64)

    result = hydrol_irrigation_demand_ratio(
        veget=veget,
        veget_max=veget_max,
        transpot=transpot,
        evapot=evapot,
        precip_rain=precip_rain,
        vegstress_old=vegstress_old,
        soil_deficit=soil_deficit,
        ok_laidev=ok_laidev,
        irrig_threshold=threshold,
        irrig_fulfill=fulfill,
        irrig_drip=True,
        irrig_dosmax=1.0,
    )

    tempfrac_pft2 = veget[0, 1] / veget_max[0, 1]
    demand_pft2 = transpot[0, 1] * tempfrac_pft2 + evapot[0] * (1.0 - tempfrac_pft2) - precip_rain[0]
    raw_pft2 = min(1.0, fulfill[1] * demand_pft2) * veget_max[0, 1]
    tempfrac_pft3 = veget[0, 2] / veget_max[0, 2]
    demand_pft3 = transpot[0, 2] * tempfrac_pft3 + evapot[0] * (1.0 - tempfrac_pft3) - precip_rain[0]
    raw_pft3 = min(1.0, fulfill[2] * demand_pft3) * veget_max[0, 2]
    expected = np.asarray([[0.0, raw_pft2, raw_pft3, 0.0]], dtype=np.float64)
    expected = expected / expected.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(np.asarray(result.irrig_demand_ratio), expected)


def test_hydrol_irrigation_demand_ratio_matches_fortran_flooding_branch():
    veget = np.asarray([[0.0, 0.30, 0.20]], dtype=np.float64)
    veget_max = np.asarray([[0.0, 0.40, 0.25]], dtype=np.float64)
    transpot = np.zeros_like(veget)
    vegstress_old = np.asarray([[1.0, 0.20, 0.20]], dtype=np.float64)
    soil_deficit = np.asarray([[0.0, 0.70, -0.50]], dtype=np.float64)
    ok_laidev = np.asarray([False, True, True])
    threshold = np.asarray([0.0, 0.50, 0.50], dtype=np.float64)
    fulfill = np.ones(3, dtype=np.float64)

    result = hydrol_irrigation_demand_ratio(
        veget=veget,
        veget_max=veget_max,
        transpot=transpot,
        evapot=np.asarray([0.0], dtype=np.float64),
        precip_rain=np.asarray([0.0], dtype=np.float64),
        vegstress_old=vegstress_old,
        soil_deficit=soil_deficit,
        ok_laidev=ok_laidev,
        irrig_threshold=threshold,
        irrig_fulfill=fulfill,
        irrig_drip=False,
        irrig_dosmax=0.50,
    )

    np.testing.assert_allclose(np.asarray(result.irrig_demand_ratio), [[0.0, 1.0, 0.0]])


def test_hydrol_routing_soil_water_split_distributes_irrigation_to_crop_tiles():
    vegtot = np.asarray([0.80], dtype=np.float64)
    returnflow = np.asarray([0.06], dtype=np.float64)
    reinfiltration = np.asarray([0.02], dtype=np.float64)
    irrigation = np.asarray([0.40], dtype=np.float64)
    ratio = np.asarray([[0.0, 0.25, 0.75]], dtype=np.float64)
    veget_max = np.asarray([[0.0, 0.20, 0.30]], dtype=np.float64)
    soiltile = np.asarray([[0.10, 0.20, 0.30, 0.40]], dtype=np.float64)
    pref_soil_veg = np.asarray([1, 4, 4], dtype=np.int32)
    is_crop_soil = np.asarray([False, False, False, True])
    ok_laidev = np.asarray([False, True, True])

    result = hydrol_routing_soil_water_split(
        vegtot=vegtot,
        returnflow=returnflow,
        reinfiltration=reinfiltration,
        irrigation=irrigation,
        irrig_demand_ratio=ratio,
        veget_max=veget_max,
        pref_soil_veg=pref_soil_veg,
        soiltile=soiltile,
        is_crop_soil=is_crop_soil,
        ok_laidev=ok_laidev,
    )

    np.testing.assert_allclose(np.asarray(result.returnflow_soil), [0.0])
    np.testing.assert_allclose(np.asarray(result.reinfiltration_soil), [(0.06 + 0.02) / 0.80])
    np.testing.assert_allclose(np.asarray(result.irrig_fin), [[0.0, 0.40 * 0.25 / 0.20, 0.40 * 0.75 / 0.30]])
    np.testing.assert_allclose(np.asarray(result.irrigation_soil), [[0.0, 0.0, 0.0, 0.40 / 0.40]])


def test_hydrol_routing_soil_water_split_refuses_active_non_crop_irrigation():
    with pytest.raises(ValueError, match="crop soil tile"):
        hydrol_routing_soil_water_split(
            vegtot=np.asarray([1.0], dtype=np.float64),
            returnflow=np.asarray([0.0], dtype=np.float64),
            reinfiltration=np.asarray([0.0], dtype=np.float64),
            irrigation=np.asarray([0.10], dtype=np.float64),
            irrig_demand_ratio=np.asarray([[0.0, 1.0]], dtype=np.float64),
            veget_max=np.asarray([[0.0, 0.50]], dtype=np.float64),
            pref_soil_veg=np.asarray([1, 2], dtype=np.int32),
            soiltile=np.asarray([[0.50, 0.50]], dtype=np.float64),
            is_crop_soil=np.asarray([False, False]),
            ok_laidev=np.asarray([False, True]),
        )


def test_hydrol_soil_surface_water_setup_matches_fortran_non_crop_branch():
    water2infilt = np.asarray([0.20, 0.05], dtype=np.float64)
    ae_ns = np.asarray([0.12, -0.04], dtype=np.float64)
    subsinksoil = np.asarray([0.03, -0.01], dtype=np.float64)
    precisol_ns = np.asarray([0.40, 0.10], dtype=np.float64)
    reinfiltration = np.asarray([0.02, 0.03], dtype=np.float64)

    result = hydrol_soil_surface_water_setup(
        water2infilt=water2infilt,
        ae_ns_tile=ae_ns,
        subsinksoil=subsinksoil,
        precisol_ns_tile=precisol_ns,
        reinfiltration_soil=reinfiltration,
        is_crop_soil=False,
    )

    water_to_infiltrate = (
        water2infilt
        + reinfiltration
        - np.minimum(ae_ns, 0.0)
        - np.minimum(subsinksoil, 0.0)
        + precisol_ns
    )
    water_to_extract = np.maximum(ae_ns, 0.0) + np.maximum(subsinksoil, 0.0)
    temp = np.minimum(water_to_infiltrate, water_to_extract)
    expected_water2infilt = water_to_infiltrate - temp
    expected_water2extract = water_to_extract - temp

    np.testing.assert_allclose(np.asarray(result.temp), temp)
    np.testing.assert_allclose(np.asarray(result.water2infilt), expected_water2infilt)
    np.testing.assert_allclose(np.asarray(result.water2extract), expected_water2extract)
    np.testing.assert_allclose(np.asarray(result.flux_top), expected_water2extract)
    np.testing.assert_allclose(np.asarray(result.flux_infilt), expected_water2infilt)
    np.testing.assert_allclose(np.asarray(result.ae_ns), ae_ns + subsinksoil)


def test_hydrol_soil_surface_water_setup_matches_fortran_crop_irrigation_branch():
    water2infilt = np.asarray([0.02], dtype=np.float64)
    ae_ns = np.asarray([0.50], dtype=np.float64)
    subsinksoil = np.asarray([0.10], dtype=np.float64)
    precisol_ns = np.asarray([0.05], dtype=np.float64)
    reinfiltration = np.asarray([0.01], dtype=np.float64)
    irrigation = np.asarray([0.30], dtype=np.float64)

    crop = hydrol_soil_surface_water_setup(
        water2infilt=water2infilt,
        ae_ns_tile=ae_ns,
        subsinksoil=subsinksoil,
        precisol_ns_tile=precisol_ns,
        reinfiltration_soil=reinfiltration,
        irrigation_soil_tile=irrigation,
        is_crop_soil=True,
    )
    non_crop = hydrol_soil_surface_water_setup(
        water2infilt=water2infilt,
        ae_ns_tile=ae_ns,
        subsinksoil=subsinksoil,
        precisol_ns_tile=precisol_ns,
        reinfiltration_soil=reinfiltration,
        irrigation_soil_tile=irrigation,
        is_crop_soil=False,
    )

    crop_infiltrate = water2infilt + irrigation + reinfiltration + precisol_ns
    crop_extract = ae_ns + subsinksoil
    crop_temp = np.minimum(crop_infiltrate, crop_extract)
    np.testing.assert_allclose(np.asarray(crop.water2infilt), crop_infiltrate - crop_temp)
    np.testing.assert_allclose(np.asarray(crop.water2extract), crop_extract - crop_temp)
    assert np.asarray(crop.water2extract)[0] < np.asarray(non_crop.water2extract)[0]
