from __future__ import annotations

import numpy as np
import pytest

from jax_orchidee.driver.interpolation_core12 import (
    AggregatePacket,
    InterpolationTarget,
)
from jax_orchidee.sechiba.hydrol_thermosoil_completion import (
    RefSocFileData,
    hydro_subgrid_main,
    hydrol_finalize,
    hydrol_rotation_update,
    read_refsocfile,
)


def test_hydro_subgrid_preserves_minloc_indices_masks_and_freeze_scaling():
    width = 1000
    wtop = np.vstack(
        [
            np.linspace(1.0, 0.0, width),
            np.linspace(1.0, 0.0, width),
            np.ones(width),
        ]
    )
    fsat_table = np.vstack(
        [np.linspace(0.0, 1.0, width), np.zeros(width), np.ones(width)]
    )
    fwet_table = np.tile(np.linspace(0.0, 1.0, width), (3, 1))
    result = hydro_subgrid_main(
        tab_fsat=fsat_table,
        tab_wtop=wtop,
        humtot=np.array([500.0, 500.0, 500.0]),
        profil_froz_hydro=np.array([[0.5] * 9, [0.0] * 9, [0.0] * 9]),
        tab_fwet=fwet_table,
        tab_wtop_wet=wtop,
        pd_top=1.0,
        ruu_ch=np.full(3, 1000.0),
        ti_min=np.zeros(3),
        ti_max=np.array([1.0, -99.99, 1.0]),
        pas=np.ones(3),
        dz=np.ones(9),
        wtd_bornes=(2.5, 5.0, 7.5, 10.0),
    )

    assert result.fsat_index_fortran[0] == 500
    assert result.fwet_index_fortran[0] == 500
    np.testing.assert_allclose(result.fsat[0], fsat_table[0, 499] * 0.5)
    expected_cumulative = fwet_table[0, np.array([489, 479, 469, 459])]
    expected_bands = np.diff(np.r_[fwet_table[0, 499], expected_cumulative]) * 0.5
    np.testing.assert_allclose(
        np.array([result.fwt1[0], result.fwt2[0], result.fwt3[0], result.fwt4[0]]),
        expected_bands,
    )
    np.testing.assert_array_equal(result.fsat[1:], 0.0)
    np.testing.assert_array_equal(result.fwet[2:], 0.0)


def test_hydro_subgrid_requires_source_declared_table_width():
    tables = np.ones((1, 900))
    with pytest.raises(ValueError, match="exactly 1000 columns"):
        hydro_subgrid_main(
            tab_fsat=tables,
            tab_wtop=tables,
            humtot=[1.0],
            profil_froz_hydro=np.zeros((1, 9)),
            tab_fwet=tables,
            tab_wtop_wet=tables,
            pd_top=1.0,
            ruu_ch=[0.0],
            ti_min=[0.0],
            ti_max=[-99.99],
            pas=[0.0],
            dz=np.ones(9),
        )


def test_read_refsocfile_routes_structured_reader_retry_and_one_based_indices():
    source = RefSocFileData(
        longitude=np.array([10.0, 20.0]),
        latitude=np.array([40.0, 50.0]),
        mask=np.array([[1.0, -1.0], [2.0, 0.0]]),
        soil_organic_carbon=np.array(
            [[[1.0, 10.0], [2.0, 20.0]], [[3.0, 30.0], [4.0, 40.0]]]
        ),
    )
    target = InterpolationTarget(
        lalo=np.array([[45.0, 15.0], [46.0, 16.0]]),
        resolution=np.ones((2, 2)),
        neighbours=np.zeros((2, 4), dtype=np.int64),
        contfrac=np.ones(2),
    )
    seen = []

    def aggregate(request):
        seen.append(request)
        width = request.nbvmax
        area = np.zeros((2, width))
        index = np.zeros((2, width, 2), dtype=np.int64)
        if width == 16:
            return AggregatePacket(index, area, False)
        area[0, :2] = [1.0, 3.0]
        index[0, :2] = [[1, 1], [2, 1]]
        return AggregatePacket(index, area, True)

    calls = []
    result = read_refsocfile(
        target=target,
        aggregate=aggregate,
        path="provided-by-callback.nc",
        reader=lambda path: calls.append(path) or source,
    )

    assert calls == ["provided-by-callback.nc"]
    assert result.nbvmax_attempts == (16, 32)
    np.testing.assert_allclose(result.refsoc, [[2.5, 25.0], [0.0, 0.0]])
    np.testing.assert_array_equal(seen[-1].mask, [[1, 0], [1, 0]])
    np.testing.assert_array_equal(seen[-1].longitude, [[10.0, 10.0], [20.0, 20.0]])
    np.testing.assert_array_equal(seen[-1].latitude, [[40.0, 50.0], [40.0, 50.0]])


def test_read_refsocfile_requires_real_data_boundary():
    target = InterpolationTarget(
        np.zeros((1, 2)), np.ones((1, 2)), np.zeros((1, 4)), np.ones(1)
    )
    with pytest.raises(ValueError, match="structured source data or an actual path"):
        read_refsocfile(target=target, aggregate=lambda request: None)


def test_hydrol_finalize_routes_optional_writes_and_explicit_snow_in_source_order():
    base_names = {
        "mc",
        "mcl",
        "us",
        "free_drain_coef",
        "zwt_force",
        "water2infilt",
        "ae_ns",
        "vegstress",
        "snow",
        "snow_age",
        "snow_nobio",
        "snow_nobio_age",
        "qsintveg",
        "evap_bare_lim_ns",
        "evap_bare_lim",
        "resdist",
        "vegtot_old",
        "drysoil_frac",
        "humrel",
        "refSOC_1d",
        "fwet_out",
        "run2peat",
        "wt_ab",
        "wtp",
        "fwet_new",
        "liqwt_ratio",
        "wt_ab_tide",
        "run2man",
        "tot_water_end",
        "tot_watveg_beg",
        "tot_watsoil_beg",
        "snow_beg",
        "snowrho",
        "snowtemp",
        "snowdz",
        "snowheat",
        "snowgrain",
    }
    values = {
        name: np.array([index], dtype=np.float64)
        for index, name in enumerate(sorted(base_names))
    }
    writes = []
    snow_calls = []
    result = hydrol_finalize(
        kjit=9,
        rest_id=4,
        state=values,
        inputs=values,
        restart_writer=lambda request: writes.append(request) or request.name,
        use_refsoc_hydrol=True,
        check_waterbal=True,
        ok_explicitsnow=True,
        explicitsnow_finalizer=lambda payload: (
            snow_calls.append(payload) or "snow-finalized"
        ),
    )

    names = [request.name for request in writes]
    assert names[:4] == ["moistc", "moistcl", "us", "free_drain_coef"]
    assert names[19:22] == ["refSOC_1d", "fwet_out", "run2peat"]
    assert names[-4:] == [
        "tot_water_beg",
        "tot_watveg_beg",
        "tot_watsoil_beg",
        "snow_beg",
    ]
    np.testing.assert_array_equal(writes[-4].value, values["tot_water_end"])
    assert result.write_results[-1] == "snow_beg"
    assert result.explicit_snow_result == "snow-finalized"
    assert snow_calls[0]["kjit"] == 9 and snow_calls[0]["rest_id"] == 4


def test_hydrol_rotation_updates_target_store_diagnostics_and_all_point_resdist():
    result = hydrol_rotation_update(
        ip_fortran=1,
        rot_matrix=np.array([[0.0, 0.2], [0.0, 0.0]]),
        old_veget_max=np.array([0.5, 0.5]),
        veget_max=np.array([[0.4, 0.6], [0.5, 0.5]]),
        soiltile=np.array([[0.4, 0.6], [0.3, 0.7]]),
        qsintveg=np.zeros((2, 2)),
        pref_soil_veg=np.array([1, 2]),
        mc=np.array([[[1.0, 2.0], [1.0, 2.0]], [[3.0, 4.0], [3.0, 4.0]]]),
        water2infilt=np.zeros((2, 2)),
        tmc=np.array([[2.0, 4.0], [6.0, 8.0]]),
        humtot=np.array([3.0, 7.4]),
        resdist=np.array([[0.5, 0.5], [0.5, 0.5]]),
        dz=np.array([1.0, 2.0]),
        check_cwrr=True,
        allowed_err=1.0e-12,
    )

    np.testing.assert_allclose(result.rot_matrix_tile, [[0.0, 0.2], [0.0, 0.0]])
    np.testing.assert_allclose(result.mc[0, :, 1], 11.0 / 6.0)
    np.testing.assert_allclose(result.tmc[0], [2.0, 11.0 / 3.0])
    np.testing.assert_allclose(result.humtot[0], 3.0)
    np.testing.assert_array_equal(result.resdist, [[0.4, 0.6], [0.3, 0.7]])


def test_hydrol_rotation_exposes_source_undefined_zero_rotation_state():
    with pytest.raises(RuntimeError, match="soil_upd is undefined"):
        hydrol_rotation_update(
            ip_fortran=1,
            rot_matrix=np.zeros((1, 1)),
            old_veget_max=[1.0],
            veget_max=[[1.0]],
            soiltile=[[1.0]],
            qsintveg=[[0.0]],
            pref_soil_veg=[1],
            mc=np.ones((1, 2, 1)),
            water2infilt=np.zeros((1, 1)),
            tmc=np.ones((1, 1)),
            humtot=np.ones(1),
            resdist=np.ones((1, 1)),
            dz=np.ones(2),
        )
