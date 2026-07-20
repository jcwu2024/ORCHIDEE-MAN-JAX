from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jax_orchidee.sechiba.main import (
    SechibaMainPFT14FixedSwitches,
    build_sechiba_main_output_packet_pft14,
    sechiba_get_cmd,
    sechiba_main_crop_rotation_transition,
    sechiba_main_pft14_fixed,
    sechiba_main_tail_pft14_fixed,
    sechiba_main_full_irrigation_transition,
)


_IS_TREE = np.asarray([False, True, True, False] + [False] * 10)
_NATURAL = np.asarray([True, True, False, True] + [False] * 10)


def _tail_state() -> dict[str, object]:
    npts, nvm, nflow, nnobio, nstm, nsnow = 2, 14, 3, 2, 3, 3
    land = np.asarray([1.0, 2.0])
    pft = np.arange(1, npts * nvm + 1, dtype=np.float64).reshape(npts, nvm)
    return {
        "riverflow": np.full((npts, nflow), 1.0),
        "coastalflow": np.full((npts, nflow), 2.0),
        "returnflow": np.full((npts, nflow), 3.0),
        "reinfiltration": np.full((npts, nflow), 4.0),
        "irrigation": np.full((npts, nflow), 5.0),
        "sed_deposition": np.full((npts, nflow), 6.0),
        "poc_deposition": np.full((npts, nflow), 7.0),
        "flood_frac": np.full(npts, 8.0),
        "stream_frac": np.full(npts, 9.0),
        "streamfl_frac": np.full(npts, 10.0),
        "flood_res": np.full(npts, 11.0),
        "fastr": np.full(npts, 12.0),
        "co2_flux": np.arange(npts * nvm, dtype=np.float64).reshape(npts, nvm),
        "veget_max": np.full((npts, nvm), 0.05),
        "temp_sol_new": np.asarray([280.0, 281.0]),
        "temp_sol_new_pft": np.full((npts, nvm), 279.0),
        "temp_sol_pft": np.full((npts, nvm), 278.0),
        "z0m": np.asarray([0.1, 0.2]),
        "z0h": np.asarray([0.01, 0.02]),
        "emis": np.asarray([0.96, 0.97]),
        "qsurf": np.asarray([0.005, 0.006]),
        "fluxsens": land + 20.0,
        "fluxlat": land + 30.0,
        "vevapnu": land + 0.1,
        "snow": land + 0.2,
        "snow_age": land + 0.3,
        "snow_nobio": np.full((npts, nnobio), 0.4),
        "snow_nobio_age": np.full((npts, nnobio), 0.5),
        "frac_snow_nobio": np.asarray([[0.1, 0.2], [0.2, 0.3]]),
        "totfrac_nobio": np.asarray([0.2, 0.3]),
        "frac_snow_veg": np.asarray([0.4, 0.5]),
        "reinf_slope": land + 0.6,
        "njsc": np.asarray([2, 3]),
        "veget": pft / 100.0,
        "frac_nobio": np.asarray([[0.02, 0.03], [0.04, 0.05]]),
        "soiltile": np.full((npts, nstm), 1.0 / nstm),
        "rstruct": pft + 1.0,
        "gpp": pft + 2.0,
        "drysoil_frac": land + 0.7,
        "vevapflo": land + 0.8,
        "k_litt": land + 0.9,
        "vbeta": land + 1.0,
        "vbeta_pft": pft + 3.0,
        "vbeta1": land + 1.1,
        "vbeta2": pft + 4.0,
        "vbeta3": pft + 5.0,
        "vbeta4": land + 1.2,
        "vbeta4_pft": pft + 6.0,
        "vbeta5": land + 1.3,
        "gsmean": pft + 7.0,
        "cimean": pft + 8.0,
        "rveget": pft + 9.0,
        "rsol": land + 1.4,
        "vevapwet": pft / 10.0,
        "vevapsno": land + 1.5,
        "transpir": pft / 20.0,
        "contfrac": np.asarray([0.8, 0.9]),
        "tsol_rad": np.asarray([280.0, 282.0]),
        "roughheight": land + 1.6,
        "roughheight_pft": pft + 10.0,
        "lai": pft / 30.0,
        "vevapp": land + 1.7,
        "fusion": land + 1.8,
        "tq_cdrag": land + 1.9,
        "tq_cdrag_pft": pft + 11.0,
        "soilflx": land + 2.0,
        "soilcap": land + 2.1,
        "soilflx_pft": pft + 12.0,
        "soilcap_pft": pft + 13.0,
        "vevapnu_pft": pft + 14.0,
        "grndflux": land + 2.2,
        "snowtemp": np.full((npts, nsnow), 270.0),
        "snowliq": np.full((npts, nsnow), 0.01),
        "snowdz": np.full((npts, nsnow), 0.02),
        "snowrho": np.full((npts, nsnow), 200.0),
        "snowgrain": np.full((npts, nsnow), 0.03),
        "snowheat": np.full((npts, nsnow), 0.04),
        "pgflux": land + 2.3,
    }


def _rotation_state() -> tuple[dict[str, object], dict[str, object]]:
    state = _tail_state()
    npts, nvm, nstm = 2, 14, 3
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    veget_max[:, 0] = 0.2
    veget_max[0, 12:14] = [0.3, 0.3]
    veget_max[1, 13] = 0.6
    state["veget_max"] = veget_max
    state["veget"] = veget_max * 0.75
    state["veget"][:, 0] = veget_max[:, 0]
    state["totfrac_nobio"] = np.full(npts, 0.2)
    state["frac_nobio"] = np.asarray([[0.2, 0.0], [0.2, 0.0]])
    state["soiltile"] = np.asarray([[0.25, 0.375, 0.375], [0.25, 0.0, 0.75]])
    state["f_rot_sech"] = np.asarray([True, False])
    state["rot_cmd"] = np.asarray([[501413, 0], [0, 0]], dtype=np.int32)
    state["qsintveg"] = np.zeros((npts, nvm), dtype=np.float64)
    state["qsintveg"][0, [0, 12, 13]] = [0.001, 0.013, 0.014]
    state["qsintveg"][1, [0, 13]] = [0.001, 0.014]
    state["mc"] = np.arange(npts * 2 * nstm, dtype=np.float64).reshape(npts, 2, nstm) / 10.0 + 1.0
    state["water2infilt"] = np.arange(npts * nstm, dtype=np.float64).reshape(npts, nstm) / 20.0
    state["tmc"] = np.ones((npts, nstm), dtype=np.float64)
    state["humtot"] = np.ones(npts, dtype=np.float64)
    state["resdist"] = np.asarray(state["soiltile"]).copy()
    state["ptn"] = np.arange(npts * 2 * nvm, dtype=np.float64).reshape(npts, 2, nvm) + 260.0
    state["cgrnd"] = np.arange(npts * nvm, dtype=np.float64).reshape(npts, 1, nvm) + 1.0
    state["dgrnd"] = np.arange(npts * nvm, dtype=np.float64).reshape(npts, 1, nvm) + 2.0
    state["temp_sol_new_pft"] = np.arange(npts * nvm, dtype=np.float64).reshape(npts, nvm) + 270.0
    state["soilcap_pft"] = np.arange(npts * nvm, dtype=np.float64).reshape(npts, nvm) + 10.0
    state["soilflx_pft"] = np.arange(npts * nvm, dtype=np.float64).reshape(npts, nvm) + 20.0
    parameters = {
        "ok_laidev": np.asarray([False] + [True] * 13),
        "pref_soil_veg": np.asarray([1] * 12 + [2, 3]),
        "ext_coeff": np.full(nvm, 0.5),
        "dz": np.asarray([0.5, 1.0]),
    }
    return state, parameters


def test_fixed_tail_matches_single_point_routing_else_and_state_writeback():
    source = _tail_state()
    result = sechiba_main_tail_pft14_fixed(
        state=source,
        dt_sechiba=1800.0,
        is_tree=_IS_TREE,
        natural=_NATURAL,
    )

    for name in (
        "riverflow",
        "coastalflow",
        "returnflow",
        "reinfiltration",
        "irrigation",
        "sed_deposition",
        "poc_deposition",
        "flood_frac",
        "stream_frac",
        "streamfl_frac",
        "flood_res",
        "fastr",
        "DOC_to_topsoil",
        "DOC_to_subsoil",
        "sed_deposition_d",
        "poc_deposition_d",
    ):
        np.testing.assert_array_equal(np.asarray(result.state[name]), 0.0)

    expected_co2 = np.sum(source["co2_flux"][:, 1:] * source["veget_max"][:, 1:], axis=1)
    np.testing.assert_allclose(np.asarray(result.state["netco2flux"]), expected_co2)
    np.testing.assert_array_equal(np.asarray(result.state["temp_sol"]), source["temp_sol_new"])
    np.testing.assert_array_equal(np.asarray(result.state["temp_sol_pft"]), source["temp_sol_new_pft"])
    for source_name, output_name in (
        ("z0m", "z0m_out"),
        ("z0h", "z0h_out"),
        ("emis", "emis_out"),
        ("qsurf", "qsurf_out"),
    ):
        np.testing.assert_array_equal(np.asarray(result.state[output_name]), source[source_name])
    assert result.state["done_stomate_lcchange"] is False
    assert result.process_order.index("routing_main[single-point-no-call]") < result.process_order.index("sechiba_end")


def test_full_irrigation_transition_matches_source_drip_and_flooding_paths():
    common = dict(
        veget_max=np.asarray([[0.0, 0.5, 0.4]]),
        veget=np.asarray([[0.0, 0.25, 0.4]]),
        vegstress=np.asarray([[1.0, 0.2, 0.8]]),
        transpot=np.asarray([[0.0, 4.0, 3.0]]),
        evapot=np.asarray([1.0]), precip_rain=np.asarray([0.5]),
        soil_deficit=np.asarray([[0.0, 0.7, 0.2]]), irrig_frac=np.asarray([0.5]),
        ok_laidev=np.asarray([False, True, True]), irrig_threshold=np.asarray([0.0, 0.5, 0.5]),
        irrig_fulfill=np.asarray([0.0, 0.8, 1.0]), irrig_dosmax=1.0, nflow=2,
    )
    drip = sechiba_main_full_irrigation_transition(**common, irrig_drip=True)
    flood = sechiba_main_full_irrigation_transition(**common, irrig_drip=False)
    np.testing.assert_allclose(np.asarray(drip), [[0.5 * min(1.0, 0.8 * max(0.0, 4.0 * 0.5 + 1.0 * 0.5 - 0.5)) * 0.5, 0.0]])
    np.testing.assert_allclose(np.asarray(flood), [[0.5 * min(1.0, 0.7) * 0.5, 0.0]])


def test_fixed_tail_routes_full_irrigation_only_from_explicit_source_packet():
    source = _tail_state()
    packet = dict(
        veget_max=source["veget_max"], veget=np.maximum(source["veget_max"] * 0.5, 0.0),
        vegstress=np.zeros_like(source["veget_max"]), transpot=np.ones_like(source["veget_max"]),
        evapot=np.ones(source["veget_max"].shape[0]), precip_rain=np.zeros(source["veget_max"].shape[0]),
        soil_deficit=np.ones_like(source["veget_max"]), irrig_frac=np.ones(source["veget_max"].shape[0]),
        ok_laidev=np.ones(source["veget_max"].shape[1], dtype=bool),
        irrig_threshold=np.ones(source["veget_max"].shape[1]), irrig_fulfill=np.ones(source["veget_max"].shape[1]),
        irrig_dosmax=1.0, irrig_drip=False,
    )
    switches = SechibaMainPFT14FixedSwitches(do_fullirr=True)
    result = sechiba_main_tail_pft14_fixed(state=source, dt_sechiba=1800.0, is_tree=_IS_TREE, natural=_NATURAL, switches=switches, full_irrigation_inputs=packet)
    assert result.process_order.index("full_irrigation[source-routed]") < result.process_order.index("netco2flux")
    expected = sechiba_main_full_irrigation_transition(
        **{
            name: value
            for name, value in packet.items()
            if name not in {"veget_max", "veget"}
        },
        veget_max=source["veget_max"],
        veget=source["veget"],
        nflow=source["irrigation"].shape[1],
    )
    np.testing.assert_allclose(result.state["irrigation"], expected)


def test_rotation_command_decode_and_integrated_state_writebacks_match_source_contract():
    assert sechiba_get_cmd(501413, ok_laidev=[False] + [True] * 13) == (14, 13, 0.5)
    assert sechiba_get_cmd(0, ok_laidev=[False] + [True] * 13) == (0, 0, 0.0)
    with pytest.raises(RuntimeError, match="percent"):
        sechiba_get_cmd(1010000, ok_laidev=[False] + [True] * 13)
    with pytest.raises(RuntimeError, match="target"):
        sechiba_get_cmd(501415, ok_laidev=[False] + [True] * 13)

    state, parameters = _rotation_state()
    result = sechiba_main_crop_rotation_transition(state=state, **parameters)
    rotated = result.state
    np.testing.assert_allclose(np.asarray(rotated["veget_max"])[0, 12:14], [0.45, 0.15])
    np.testing.assert_array_equal(np.asarray(rotated["f_rot_sech"]), [False, False])
    np.testing.assert_array_equal(np.asarray(rotated["rot_cmd"]), 0)
    np.testing.assert_allclose(np.sum(np.asarray(rotated["soiltile"]), axis=1), 1.0)
    np.testing.assert_allclose(rotated["resdist"], rotated["soiltile"])
    assert not np.array_equal(np.asarray(rotated["mc"])[0], np.asarray(state["mc"])[0])
    assert not np.array_equal(np.asarray(rotated["ptn"])[0], np.asarray(state["ptn"])[0])
    assert not np.array_equal(
        np.asarray(rotated["temp_sol_new_pft"])[0], np.asarray(state["temp_sol_new_pft"])[0]
    )
    for name in (
        "veget_max", "veget", "soiltile", "qsintveg", "mc", "water2infilt",
        "tmc", "humtot", "resdist", "ptn", "cgrnd", "dgrnd",
        "temp_sol_new_pft", "soilcap_pft", "soilflx_pft",
    ):
        assert np.all(np.isfinite(np.asarray(rotated[name])))


def test_fixed_tail_runs_rotation_before_outputs_then_lcc_and_restart_dispatch():
    state, rotation_inputs = _rotation_state()
    carbon_fields = {
        "biomass": np.arange(4.0).reshape(2, 2),
        "litter_above": np.arange(6.0).reshape(2, 3),
        "litter_below": np.arange(8.0).reshape(2, 4),
        "carbon_32l": np.arange(10.0).reshape(2, 5),
        "DOC": np.arange(12.0).reshape(2, 6),
    }
    state.update(carbon_fields)
    veget_max_new = np.zeros((2, 14), dtype=np.float64)
    veget_max_new[:, 0] = 0.3
    veget_max_new[:, 13] = 0.5
    lcc_inputs = {
        "agri_peat": False,
        "veget_max_new": veget_max_new,
        "frac_nobio_new": np.asarray([[0.2, 0.0], [0.2, 0.0]]),
        "pref_soil_veg": np.ones(14, dtype=np.int32),
        "ext_coeff_vegetfrac": np.full(14, 0.5),
        "nstm": 3,
    }
    finalized: list[np.ndarray] = []

    def finalize_owner(current):
        finalized.append(np.asarray(current["veget_max"]).copy())
        return SimpleNamespace(state=dict(current) | {"restart_marker": np.asarray([17])})

    result = sechiba_main_tail_pft14_fixed(
        state=state,
        dt_sechiba=1800.0,
        is_tree=_IS_TREE,
        natural=_NATURAL,
        switches=SechibaMainPFT14FixedSwitches(
            ok_rotate=True,
            done_stomate_lcchange=True,
            ldrestart_write=True,
        ),
        rotation_inputs=rotation_inputs,
        lcc_inputs=lcc_inputs,
        finalize_owner=finalize_owner,
    )
    assert result.process_order.index("crop_rotation[source-routed]") < result.process_order.index("sechiba_end")
    assert result.process_order.index("modelout_packet") < result.process_order.index("slowproc_change_frac[source-routed]")
    assert result.process_order.index("slowproc_change_frac[source-routed]") < result.process_order.index("sechiba_finalize[source-routed]")
    np.testing.assert_allclose(finalized[0], veget_max_new)
    np.testing.assert_array_equal(result.state["restart_marker"], [17])
    assert result.state["done_stomate_lcchange"] is False
    for name, before in carbon_fields.items():
        np.testing.assert_array_equal(np.asarray(result.state[name]), before)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("erosion_module", True),
        ("river_routing", False),
        ("nbp_glo", 2),
        ("ok_explicitsnow", False),
    ],
)
def test_fixed_tail_explicitly_rejects_unsupported_switches(field, value):
    values = vars(SechibaMainPFT14FixedSwitches()) | {field: value}
    with pytest.raises(NotImplementedError, match=field):
        sechiba_main_tail_pft14_fixed(
            state=_tail_state(),
            dt_sechiba=1800.0,
            is_tree=_IS_TREE,
            natural=_NATURAL,
            switches=SechibaMainPFT14FixedSwitches(**values),
        )


def test_full_entry_reuses_step_owner_writes_owner_outputs_then_calls_io_boundaries():
    calls: list[str] = []
    state = _tail_state()

    def record(name):
        def boundary(fields):
            calls.append(name)
            assert isinstance(fields, tuple)
            assert fields
            assert all(field.source_subroutine == "sechiba_main" for field in fields)
            return name

        return boundary

    def step_owner(**kwargs):
        calls.append("step")
        assert kwargs == {"marker": 17}
        payload = {
            "temp_sol_new": np.asarray([290.0, 291.0]),
            "temp_sol_new_pft": np.full((2, 14), 289.0),
            "qsurf": np.asarray([0.02, 0.03]),
        }
        empty = SimpleNamespace(_asdict=lambda: {})
        condveg = SimpleNamespace(
            _asdict=lambda: {
                "z0m": np.asarray([0.3, 0.4]),
                "z0h": np.asarray([0.03, 0.04]),
                "emis": np.asarray([0.95, 0.95]),
            }
        )
        return SimpleNamespace(
            diffuco_payload={"co2_flux": state["co2_flux"]},
            enerbil_payload=payload,
            hydrol_outputs=empty,
            hydrol_diagnostics=empty,
            condveg=condveg,
            thermosoil_payload={},
            slowproc=None,
        )

    result = sechiba_main_pft14_fixed(
        step_inputs={"marker": 17},
        state=state,
        dt_sechiba=1800.0,
        is_tree=_IS_TREE,
        natural=_NATURAL,
        step_owner=step_owner,
        xios_boundary=record("xios"),
        history_boundary=record("history"),
    )

    assert calls == ["step", "xios", "history"]
    assert result.tail.xios_result == "xios"
    assert result.tail.history_result == "history"
    np.testing.assert_array_equal(np.asarray(result.state["z0m_out"]), np.asarray([0.3, 0.4]))


def test_output_packet_preserves_fixed_path_call_order_provenance_and_inactive_streams():
    packet = build_sechiba_main_output_packet_pft14(
        state=_tail_state(),
        dt_sechiba=1800.0,
        is_tree=_IS_TREE,
        natural=_NATURAL,
    )

    assert [(field.name, field.source_line) for field in packet.xios[:5]] == [
        ("temp_sol_new", 1434),
        ("fluxsens", 1435),
        ("fluxlat", 1436),
        ("evapnu", 1437),
        ("snow", 1438),
    ]
    assert [(field.name, field.source_line) for field in packet.xios[-5:]] == [
        ("irrigation", 1512),
        ("ECanop", 1518),
        ("TVeg", 1523),
        ("ACond", 1524),
        ("ACond_pft", 1525),
    ]
    assert packet.xios[0].provenance.endswith("sechiba_main line 1434")
    assert len(packet.xios) == 63
    assert len(packet.history) == 58
    for field in packet.xios + packet.history:
        assert field.source_file == "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"
        assert field.source_subroutine == "sechiba_main"
        assert field.provenance.endswith(f"sechiba_main line {field.source_line}")
    assert [field.source_line for field in packet.xios] == sorted(
        field.source_line for field in packet.xios
    )
    assert [field.source_line for field in packet.history] == sorted(
        field.source_line for field in packet.history
    )
    assert packet.history2 == ()
    assert packet.history2_active is False

    xios_names = [field.name for field in packet.xios]
    assert xios_names[3] == "evapnu"
    assert [(field.name, field.source_line) for field in packet.xios if field.name == "vevapnu"] == [
        ("vevapnu", 1506)
    ]
    history_names = [field.name for field in packet.history]
    assert "rsol" not in history_names  # hydrol_cwrr=.TRUE., lines 1559-1561
    assert "evapflo" not in history_names  # do_floodplains=.FALSE., lines 1590-1593


def test_output_packet_matches_fortran_scaling_and_source_ordered_diagnostics():
    state = _tail_state()
    packet = build_sechiba_main_output_packet_pft14(
        state=state,
        dt_sechiba=1800.0,
        is_tree=_IS_TREE,
        natural=_NATURAL,
    )

    by_line = {field.source_line: np.asarray(field.value) for field in packet.xios}
    np.testing.assert_allclose(by_line[1437], np.asarray(state["vevapnu"]) * 48.0)
    np.testing.assert_allclose(by_line[1453], np.asarray(state["co2_flux"]) / 1800.0)
    np.testing.assert_allclose(by_line[1507], np.asarray(state["vevapnu"]) / 1800.0)
    np.testing.assert_allclose(by_line[1508], np.asarray(state["transpir"]) * 48.0)
    np.testing.assert_allclose(by_line[1512], np.asarray(state["irrigation"])[:, 0] * 48.0)

    veget_max = np.asarray(state["veget_max"])
    expected_tree = veget_max[:, 1]
    expected_crop = np.sum(veget_max[:, [2] + list(range(4, 14))], axis=1)
    np.testing.assert_allclose(np.asarray(packet.diagnostics["sum_treefrac"]), expected_tree)
    np.testing.assert_allclose(np.asarray(packet.diagnostics["sum_grassfrac"]), veget_max[:, 3])
    np.testing.assert_allclose(np.asarray(packet.diagnostics["sum_cropfrac"]), expected_crop)

    expected_lai = np.zeros(veget_max.shape[0])
    denominator = np.sum(veget_max, axis=1)
    for jv in range(1, 14):
        expected_lai += veget_max[:, jv] * np.asarray(state["lai"])[:, jv] / denominator
    np.testing.assert_allclose(np.asarray(packet.diagnostics["LAImean"]), expected_lai)
