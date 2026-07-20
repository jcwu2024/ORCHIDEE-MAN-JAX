from __future__ import annotations

import numpy as np
import pytest

import jax_orchidee.sechiba.hydrol as hydrol_module
from jax_orchidee.sechiba.hydrol import (
    hydrol_diag_soil_source_owner,
    hydrol_initialize_pft14_completion,
    hydrol_main_scientific_output_packet,
    hydrol_main_step,
    hydrol_soil_tail_diagnostics,
)
from tests.unit.test_hydrol_main import _bucket_main_inputs


def test_hydrol_initialize_fixed_dispatch_initializes_alma_and_itopmax():
    """Fortran: hydrol.f90::hydrol_initialize lines 625-688 and 833-904."""

    val_exp = 1.0e20
    result = hydrol_initialize_pft14_completion(
        qsintveg=np.asarray([[1.0, 2.0], [0.5, 0.25]]),
        humtot=np.asarray([10.0, 20.0]),
        snow=np.asarray([3.0, 4.0]),
        snow_nobio=np.asarray([[1.0], [2.0]]),
        tot_watveg_beg=np.full(2, val_exp),
        tot_watsoil_beg=np.full(2, val_exp),
        snow_beg=np.full(2, val_exp),
        mx_eau_var=np.asarray([100.0, 200.0]),
        znh_m=np.asarray([0.01, 0.05, 0.1, 0.101, 0.2]),
        soilmoist=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        val_exp=val_exp,
    )

    assert result.peat_nodr and result.ok_ru2peat and result.ok_wt_ab
    assert result.max_wt_ab == 100.0
    assert result.alma is not None
    np.testing.assert_allclose(result.alma.tot_watveg_beg, [3.0, 0.75])
    np.testing.assert_allclose(result.alma.tot_watsoil_beg, [10.0, 20.0])
    np.testing.assert_allclose(result.alma.snow_beg, [4.0, 6.0])
    assert result.itopmax == 3
    np.testing.assert_allclose(result.soilmoist_out, [[1.0, 2.0], [3.0, 4.0]])
    for value in result.topmodel_state.values():
        np.testing.assert_allclose(value, 0.0)


def test_hydrol_initialize_preserves_external_file_boundaries():
    kwargs = dict(
        qsintveg=np.zeros((1, 2)),
        humtot=np.zeros(1),
        snow=np.zeros(1),
        snow_nobio=np.zeros((1, 1)),
        tot_watveg_beg=np.zeros(1),
        tot_watsoil_beg=np.zeros(1),
        snow_beg=np.zeros(1),
        mx_eau_var=np.ones(1),
        znh_m=np.asarray([0.2]),
        soilmoist=np.zeros((1, 1)),
    )
    with pytest.raises(NotImplementedError, match="refSOC"):
        hydrol_initialize_pft14_completion(**kwargs, use_refSOC_hydrol=True)
    with pytest.raises(NotImplementedError, match="TOPMODEL parameter IO"):
        hydrol_initialize_pft14_completion(**kwargs, topm_calcul=True)
    with pytest.raises(NotImplementedError, match="new-TOPMODEL"):
        hydrol_initialize_pft14_completion(**kwargs, topmodel_new=True)


def _tail_inputs():
    tile = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    return {
        "soiltile": np.asarray([[0.25, 0.75], [0.4, 0.6]]),
        "vegtot": np.asarray([0.8, 0.0]),
        "wtd_ns": tile,
        "ru_corr_ns": tile,
        "ru_corr2_ns": tile + 1.0,
        "dr_corr_ns": tile + 2.0,
        "dr_corrnum_ns": tile + 3.0,
        "dr_force_ns": tile + 4.0,
        "ru_infilt_ns": tile + 5.0,
        "qinfilt_ns": tile + 6.0,
        "evap_bare_lim_ns": np.asarray([[5.0e-4, 5.0e-4], [2.0, 3.0]]),
        "r_soil_ns": tile + 7.0,
        "mc": np.arange(12.0).reshape(2, 3, 2),
        "dr_ns": tile + 8.0,
        "ru_ns": tile + 9.0,
        "qflux": np.arange(12.0).reshape(2, 3, 2) + 10.0,
    }


def test_hydrol_soil_tail_uses_source_weights_threshold_and_checks():
    kwargs = _tail_inputs()
    check = np.asarray([[1.0, 3.0], [5.0, 7.0]])
    result = hydrol_soil_tail_diagnostics(
        **kwargs,
        check_infilt_ns=check,
        check_tr_ns=check + 1.0,
        check_over_ns=check + 2.0,
        check_under_ns=check + 3.0,
        check_cwrr2=True,
        tides=True,
        wtp_tide=0.4,
        dwtp_tide=-0.1,
    )

    np.testing.assert_allclose(result.wtd, [1.75, 3.6])
    np.testing.assert_allclose(result.ru_corr, [1.4, 0.0])
    np.testing.assert_allclose(result.dr_force, [-4.6, 0.0])
    np.testing.assert_allclose(result.check_infilt, [2.0, 0.0])
    np.testing.assert_allclose(result.evap_bare_lim_ns[0], 0.0)
    np.testing.assert_allclose(result.evap_bare_lim, [0.0, 0.0])
    np.testing.assert_allclose(result.soil_mc, kwargs["mc"])
    np.testing.assert_allclose(result.wat_flux, kwargs["qflux"])


def test_hydrol_soil_tail_keeps_tidal_io_and_fatal_error_explicit():
    kwargs = _tail_inputs()
    with pytest.raises(NotImplementedError, match="tidal forcing"):
        hydrol_soil_tail_diagnostics(**kwargs, tides=True)
    with pytest.raises(RuntimeError, match="fatal errors"):
        hydrol_soil_tail_diagnostics(**kwargs, error=True)


def test_hydrol_diag_soil_completion_delegates_to_existing_exact_owner(monkeypatch):
    marker = object()
    captured = {}

    def fake_owner(result, **kwargs):
        captured["result"] = result
        captured["kwargs"] = kwargs
        return marker

    monkeypatch.setattr(hydrol_module, "hydrol_module_diagnostics", fake_owner)
    module_result = object()
    assert hydrol_diag_soil_source_owner(module_result, nroot="root") is marker
    assert captured == {"result": module_result, "kwargs": {"nroot": "root"}}


def test_hydrol_main_output_packet_owns_scaling_and_fixed_history_selection():
    module, diagnostic, snow, irrigation, routing = _bucket_main_inputs()
    result = hydrol_main_step(
        module_inputs=module,
        diagnostic_inputs=diagnostic,
        snow_inputs=snow,
        irrigation_inputs=irrigation,
        routing_inputs=routing,
        humcste=np.ones(module["veget_max"].shape[1]),
        dlh=np.asarray([1.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0]),
        tmcs=np.full(module["soiltile"].shape, 100.0),
        itopmax=2,
        zmaxh_m=2.0,
        ok_explicitsnow=False,
        alma_inputs={
            "tot_watveg_beg": np.asarray([0.1]),
            "tot_watsoil_beg": np.asarray([10.0]),
            "snow_beg": np.asarray([3.0]),
            "mx_eau_var": np.asarray([100.0]),
        },
    )
    packet = hydrol_main_scientific_output_packet(
        result,
        precip_rain=module["precip_rain"],
        precip_snow=snow["precip_snow"],
        qsintmax=module["qsintmax"],
        ok_explicitsnow=False,
        ok_freeze_cwrr=False,
        peat_hydro=False,
        tides=False,
        almaoutput=False,
        hist2_id=-1,
    )
    fields = dict(packet.xios_fields)

    np.testing.assert_allclose(fields["runoff"], result.runoff / 1800.0)
    np.testing.assert_allclose(fields["irrig_fin"], result.routing.irrig_fin * 48.0)
    np.testing.assert_allclose(fields["snowdz"], result.state_writeback["snowdepth"])
    assert packet.primary_history_enabled
    assert not packet.secondary_history_enabled
    assert "serializer" in packet.history_serializer_boundary
    with pytest.raises(ValueError, match="HydrolWaterBalanceResult"):
        hydrol_main_scientific_output_packet(
            result,
            precip_rain=module["precip_rain"],
            precip_snow=snow["precip_snow"],
            qsintmax=module["qsintmax"],
            ok_explicitsnow=False,
            ok_freeze_cwrr=False,
            peat_hydro=False,
            tides=False,
            check_waterbal=True,
        )
