"""Selected real HYDROL/canopy handoff and a portable writer regression.

Actual module results use declared component fixtures. Model/calendar admission,
unrelated thermal readiness and complete scientific restart generation are
outside this bounded state-writer test. HYDROL feeds the NEXT DIFFUCO call.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import jax
import numpy as np
import pytest

from jax_orchidee.driver import orchestration as driver
from jax_orchidee.sechiba.hydrol import hydrol_peat_water_table_diagnostics
from scripts.dev import check_hydrol_diffuco_state_handoff as handoff

ROOT = Path(__file__).resolve().parents[2]


def _portable_diagnostic_shell(enabled):
    """Real narrow diagnostic, synthetic unrelated writer fields only.

    This regression protects state selection. It is not counted as complete
    HYDROL execution or an independent numerical agreement result.
    """
    p = json.loads((ROOT / "tests/fixtures/hydrol_fwet_carry_reference.json").read_text())
    mc = np.full((1, 7, 1), p["mc"], dtype=np.float64)
    point, tile = np.zeros(1), np.zeros((1, 1))
    diag = hydrol_peat_water_table_diagnostics(
        mc=mc, mcl=mc, soiltile=np.ones((1, 1)), dz_mm=p["dz_mm"], dh_mm=p["dh_mm"],
        wt_ab=point, topmodel_new=enabled, numlayers=7,
        temp_hydro=np.full((1, 7), p["temp_hydro"]), mcs=p["mcs"],
        **{n: p[n] for n in ("param_vp", "param_kp", "param_qp", "param_fmax")})
    soil = NS(**{n: tile for n in ("mc", "mcl", "water2infilt", "run2peat", "run2man", "wt_ab", "wt_ab_tide", "profil_froz_hydro_ns")})
    result = NS(ok=True, module=NS(soil=soil, canop=NS(qsintveg=tile), flood=NS(flood_res=point)),
                diagnostics=NS(ae_ns=tile, mc_layh_s=tile, fwet_new=diag.fwet_new),
                outputs=NS(**{n: tile for n in ("drainage_per_soil", "runoff_per_soil", "wat_flux", "runoff2peat", "canopy2ground", "precip2ground", "precip2canopy")}),
                module_inputs={}, snow_state=None, snow_step=None)
    return result, diag, np.asarray(p["prior_fwet_new"])


@pytest.mark.parametrize("route", ("ordinary", "runtime", "compiled_fast"))
@pytest.mark.parametrize("enabled", (False, True))
def test_portable_fwet_diagnostic_survives_actual_carry(route, enabled):
    hydrol, diag, prior = _portable_diagnostic_shell(enabled)
    packet = handoff.carry(route, hydrol)
    actual = packet.fields_by_component["hydrol_previous_step_state"]["fwet_new"]
    if enabled:
        assert np.all(np.abs(np.asarray(diag.fwet_new) - prior) > 0.1)
        np.testing.assert_array_equal(actual, diag.fwet_new)
    else:
        assert diag.fwet_new is None
        np.testing.assert_array_equal(actual, prior)


@pytest.fixture(scope="module")
def real_chain():
    assert jax.config.jax_enable_x64
    assert all(d.platform == "cpu" for d in jax.devices())
    return handoff.real_components()


@pytest.mark.parametrize("route", handoff.ROUTES)
def test_real_selected_state_reaches_all_builders(real_chain, route):
    _, enabled_hydrol, canopy = real_chain
    packet = handoff.carry(route, enabled_hydrol, canopy)
    assert len(handoff.check_selected(packet, enabled_hydrol, canopy)) >= 35


@pytest.mark.parametrize("route", ("ordinary", "runtime", "compiled_fast"))
def test_real_default_topmodel_carries_prior_when_no_diagnostic(real_chain, route):
    hydrol, _, canopy = real_chain
    assert hydrol.diagnostics.fwet_new is None
    packet = handoff.carry(route, hydrol, canopy)
    np.testing.assert_array_equal(packet.fields_by_component["hydrol_previous_step_state"]["fwet_new"], [0.123])


def test_real_selected_fast_layout_writer_jit(real_chain):
    _, hydrol, canopy = real_chain
    values = handoff.compact_selected_jit(hydrol, canopy)
    expected = (hydrol.diagnostics.fwet_new, hydrol.diagnostics.ae_ns,
                hydrol.diagnostics.humrel,
                canopy.enerbil.payload["vevapp"] / canopy.enerbil.payload["dt_sechiba"])
    for actual, reference in zip(values[:-1], expected[:-1], strict=True):
        np.testing.assert_array_equal(actual, reference)
    # Arithmetic unit conversion uses the retained scientific tolerance.
    np.testing.assert_allclose(values[-1], expected[-1], rtol=1e-12, atol=1e-12)


def test_real_nonzero_canopy_storage_survives_both_mirrors_and_next_mapper():
    hydrol = handoff.real_hydrol_canopy_storage()
    actual = np.asarray(hydrol.module.canop.qsintveg)
    assert actual[0, 13] > 0.01
    for route in handoff.ROUTES:
        packet = handoff.carry(route, hydrol)
        fields = packet.fields_by_component
        for component in ("hydrol_previous_step_state", "diffuco_previous_step_state"):
            np.testing.assert_array_equal(fields[component]["qsintveg"], actual)
        mapped, _, _ = driver._diffuco_restart_payload_from_previous_state(packet)
        np.testing.assert_array_equal(mapped["qsintveg"], actual)


def test_real_refreshed_nonzero_frozen_profile_is_carried_from_soil_result():
    from scripts.dev.oracle_hydrol_component_owner import run_owner_case
    hydrol = run_owner_case(18, use_jit=False)
    actual = np.asarray(hydrol.module.soil.profil_froz_hydro_ns)
    assert np.any(actual > 0.0)
    for route in handoff.ROUTES:
        packet = handoff.carry(route, hydrol)
        np.testing.assert_array_equal(packet.fields_by_component["hydrol_previous_step_state"]["profil_froz_hydro_ns"], actual)


def test_actual_diffuco_outputs_enter_current_enerbil_and_next_hydrol_inputs_enter_next_diffuco(real_chain):
    _, hydrol, canopy = real_chain
    for name in ("vbeta", "vbeta1", "vbeta2", "vbeta3", "vbeta4", "vbeta5", "q_cdrag", "q_cdrag_pft"):
        np.testing.assert_array_equal(canopy.enerbil_precall.payload[name], canopy.diffuco.result.payload.payload[name])
    packet = handoff.carry("runtime", hydrol, canopy)
    next_call = handoff.next_diffuco_consumer(packet, canopy)
    assert next_call.enerbil is not None
    for name in ("humrel", "qsintveg", "evap_bare_lim", "drysoil_frac"):
        np.testing.assert_array_equal(next_call.fixture.diffuco_precall.payload[name],
                                      packet.fields_by_component["diffuco_previous_step_state"][name])
    assert np.isfinite(next_call.diffuco.result.payload.payload["gpp"]).all()


def test_saved_controls_are_not_recomputed_by_shared_writer(real_chain):
    _, _, canopy = real_chain
    fields = driver._empty_previous_step_state_fields()
    driver._add_diffuco_saved_controls_to_previous_fields(fields, precall=canopy.fixture.diffuco_precall)
    before = dict(fields["diffuco_previous_step_state"])
    assert set(before) >= {"control_salinity", "control_inudate"}
    driver._add_diffuco_saved_controls_to_previous_fields(
        fields, precall={"control_salinity": np.asarray([0.991]), "control_inudate": np.asarray([0.992])})
    for name in before:
        np.testing.assert_array_equal(fields["diffuco_previous_step_state"][name], before[name])


@pytest.mark.skipif(not (handoff.PAPER / "sechiba_start.nc").is_file(), reason="declared real single-point restart fixture absent")
def test_actual_selected_values_survive_real_singlepoint_files_and_reader_phases(real_chain, tmp_path):
    _, hydrol, canopy = real_chain
    packet = handoff.carry("runtime", hydrol, canopy)
    result = handoff.single_point_roundtrip(packet, canopy, tmp_path)
    assert result["complete_file_fields"] == 93
    assert {"fwet_new", "humrel", "qsintveg", "evapora"} <= set(result["overlaid_real_fields"])
    assert "us" in result["shape_exclusions"]  # Seven-layer component vs eleven-layer saved schema.
