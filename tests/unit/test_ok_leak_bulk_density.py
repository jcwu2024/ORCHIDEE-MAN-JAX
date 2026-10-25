"""C8 source-unit regression through raw producer, controls and real adapter.

Scalar expectations transcribe litter1971-1973 then soilcarbon1476-1478,
and soilcarbon1794/1798/1801/1802/1809, independently of the JAX helper.
"""
import math
from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jax_orchidee.coupled as coupled
from jax_orchidee.driver.static import slowproc_soilt_from_explicit_overlap
from jax_orchidee.stomate.integration import stomate_ok_leak_explicit
from jax_orchidee.stomate.carbon_kernels import littercalc_aboveground_controls


def source_density(value, stages):
    for _ in range(stages):
        if value < 500:
            value *= 1000
    return value


def source_adsorbed(density, z, clay=.2):
    result = []
    for layer in range(len(z) - 1):
        depth = z[1] * 100 / 2 if layer == 0 else (z[layer + 1] + z[layer]) / 2
        kd = 10 ** (-3.1 + .2 * math.log10(clay * 100) + math.log10(depth))
        kd = min(10 ** -1.5, max(10 ** -3.2, kd))
        result.append(kd * density / (kd * density + 1))
    return result


def raw_static_density(value, *, fallback=False):
    # An invalid source class selects slowproc_soilt's real default1.65.
    raw = slowproc_soilt_from_explicit_overlap(
        np.array([[0 if fallback else 1.]]), np.array([[value]]),
        np.array([[7.]]), np.array([[0.]]),
        np.array([[[1, 1]]]), np.array([[1.]]), soil_classif="zobler",
    )
    driver = SimpleNamespace(
        kjpindex=1, nbindex=1, kindex=np.array([1]), lalo=np.array([[21., 109.]]),
        contfrac=np.ones(1), precip_rain=np.zeros(1), precip_snow=np.zeros(1),
        swdown=np.zeros(1), pb=np.array([101325.]), u=np.ones(1), v=np.zeros(1),
        resolution=None, neighbours=None, **{k: raw[k] for k in ("clay_frac", "bulk_dens", "soil_ph", "poor_soils")},
    )
    routing = coupled.stomate_static_routing_ok_leak_inputs(
        driver_payload=driver, kjit=1, nflow=2, river_routing=False, nbp_glo=1,
    )
    # Conversion belongs at litter/soil call time, not in raw static storage.
    np.testing.assert_array_equal(routing.ok_leak_inputs["bulk_dens"], raw["bulk_dens"])
    return routing.ok_leak_inputs["bulk_dens"]


def adapter_inputs(density):
    n, v, d, tiles = 1, 2, 5, 1
    z = np.array([0., .01, .1, .2, .3, .4])
    zero_pft = np.zeros((n, v))
    doc = np.zeros((n, v, d, 2, 7, 1))
    doc[0, 1, :, 0, 0, 0] = 1.
    args = dict(
        litter_above=np.zeros((n, 2, v, 1)), litter_below=np.zeros((n, 2, v, d, 1)),
        lignin_struc_above=zero_pft.copy(), lignin_struc_below=np.zeros((n, v, d)),
        litterpart=np.zeros((n, v, 2)), dead_leaves=np.zeros((n, v, 2)),
        **{f"fuel_{size}hr": np.zeros((n, v, 2, 1)) for size in (1, 10, 100, 1000)},
        bm_to_litter=np.zeros((n, v, 12, 1)), turnover=np.zeros((n, v, 12, 1)),
        rprof=np.ones((n, v)), veget_max=np.array([[0., 1.]]), sla_calc=np.ones((n, v)),
        fbact_litter=np.zeros((n, d, v)), poor_soils=np.zeros(n), flood_frac=np.zeros(n),
        carbon_32l=np.zeros((n, 3, v, d)), doc=doc,
        **{name: np.zeros((n, v, 1)) for name in ("doc_precip2ground", "doc_precip2canopy", "dry_dep_canopy", "interception_storage", "canopy2ground")},
        doc_to_topsoil=np.zeros((n, 2)), doc_to_subsoil=np.zeros((n, 2)),
        fbact_doc=np.zeros((n, d, v)), fbact_soilcarbon=np.zeros((n, d, v)),
        soil_mc=np.full((n, d, tiles), .5), soil_mc_32l=np.full((n, d, tiles), .5),
        wat_flux=np.zeros((n, d, tiles)), soilwater_31mm=np.ones((n, tiles)),
        runoff_per_soil=np.zeros((n, tiles)), drainage_per_soil=np.zeros((n, tiles)),
        runoff2peat=np.zeros((n, tiles)), fastr=np.zeros(n), pref_soil_veg=np.zeros(v, dtype=int),
        tprof=np.full((n, d, v), 280.), clay=np.array([.2]), bulk_dens=np.asarray(density),
        z_soil=z, zf_soil_b=z, flux_red=np.ones(n), natural=np.array([False, True]),
        is_peat=np.zeros(v, dtype=bool), is_c4=np.zeros(v, dtype=bool),
        resp_maint_part_radia=np.zeros((n, v, 12)), flood_root_radia=zero_pft.copy(),
        dt_days=1/48, control_temp_above=np.ones((n, 2)),
        control_moist_above=np.ones((n, v)), soil_mc_top_by_pft=np.ones((n, v)),
        dif_doc=np.zeros(n), nslm=d, ndeep=d, sro_bottom=5,
    )
    args["canopy2ground"] = np.zeros((n, v))
    return args


@pytest.mark.parametrize("density,fallback", [(1.65, False), (1.65, True), (1650., False), (500., False), (.5, False), (.0001, False)])
def test_raw_producer_to_real_adapter_matches_two_source_density_updates(density, fallback):
    raw = raw_static_density(density, fallback=fallback)
    args = adapter_inputs(raw)
    result = stomate_ok_leak_explicit(**args)
    expected = source_adsorbed(source_density(float(raw[0]), 2), args["z_soil"])
    np.testing.assert_allclose(np.asarray(result.soilcarbon.doc)[0, 1, :, 1, 0, 0], expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(args["bulk_dens"], raw)


@pytest.mark.parametrize("density", [1.65, 1650., 500., .0001])
def test_controls_and_adapter_share_litter_prepared_density(monkeypatch, density):
    raw = raw_static_density(density)
    args = adapter_inputs(raw)
    args["carbon_32l"][0, :, 1, :] = 1.
    seen = []
    original = coupled.littercalc_aboveground_controls
    def observe_controls(**kwargs):
        seen.append(np.asarray(kwargs["bulk_dens"]).copy())
        return original(**kwargs)
    monkeypatch.setattr(coupled, "littercalc_aboveground_controls", observe_controls)
    # Same-step builder exercised with genuine raw static/entry output and local
    # component payloads, without running a SECHIBA/model day.
    hydro = SimpleNamespace(**{name: args[name] for name in ("wat_flux", "runoff_per_soil", "drainage_per_soil", "runoff2peat", "canopy2ground")})
    sechiba = SimpleNamespace(hydrol_outputs=hydro,
        hydrol_diagnostics=SimpleNamespace(mc_layh_s=args["soil_mc"]), slowproc=None)
    base = {name: value for name, value in args.items() if name not in {
        "soil_mc", "wat_flux", "runoff_per_soil", "drainage_per_soil", "runoff2peat", "canopy2ground",
        "control_temp_above", "control_moist_above", "soil_mc_top_by_pft"}}
    result = coupled.stomate_same_step_ok_leak_boundary_inputs(
        sechiba=sechiba, post_npp=None, dt_sechiba=1800., ok_leak_inputs=base,
        litter_controls_boundary=dict(tsurf=np.array([280.]), pref_soil_veg=np.array([1, 1]),
            frozen_respiration_func=1, moist_func_moyano=True, veget_max=args["veget_max"],
            zz_coef_deep=args["zf_soil_b"][1:]),
        is_tree=np.array([False, True]),
    )
    expected_litter = source_density(density, 1)
    np.testing.assert_array_equal(seen[0], [expected_litter])
    forwarded = result.boundary_inputs.ok_leak_inputs
    np.testing.assert_array_equal(forwarded["bulk_dens"], [expected_litter])
    assert forwarded["bulk_dens_litter_prepared"] is True
    # Real Moyano component comparison is separate from parameter capture.
    controls_expected = littercalc_aboveground_controls(np.array([280.]), args["soil_mc"],
        args["z_soil"], np.array([0, 0]), frozen_respiration_func=1, moist_func_moyano=True,
        zz_coef_deep=args["zf_soil_b"][1:], bulk_dens=np.array([expected_litter]),
        clay=args["clay"], carbon_32l=args["carbon_32l"], veget_max=args["veget_max"])
    np.testing.assert_allclose(forwarded["control_moist_above"], controls_expected.control_moist_above)
    numeric = stomate_ok_leak_explicit(**forwarded)
    np.testing.assert_allclose(np.asarray(numeric.soilcarbon.doc)[0, 1, :, 1, 0, 0],
        source_adsorbed(source_density(density, 2), args["z_soil"]), rtol=1e-12, atol=1e-14)


def test_density_literal_update_is_jittable_and_not_arbitrarily_idempotent():
    from jax_orchidee.stomate.soil_properties import bulk_density_source_update
    values = jnp.array([1.65, 1650., 500., .5, .0001])
    update = jax.jit(bulk_density_source_update)
    first = update(values)
    second = update(first)
    np.testing.assert_allclose(first, [1650., 1650., 500., 500., .1])
    np.testing.assert_allclose(second, [1650., 1650., 500., 500., 100.])


def test_prepared_density_replaces_raw_generic_bundle_without_touching_peat_density():
    raw = np.array([.0001])
    litter = np.array([.1])
    peat = np.array([.12])
    result = coupled.stomate_ok_leak_boundary_inputs(
        restart_state=SimpleNamespace(ok_leak_state_inputs={}, output_inputs={}, provenance=()),
        static_routing=SimpleNamespace(ok_leak_inputs={"bulk_dens": raw, "peat_bulk_density": peat},
                                       output_inputs={}, provenance=()),
        litter_controls=SimpleNamespace(ok_leak_inputs={"bulk_dens": litter, "bulk_dens_litter_prepared": True},
                                        provenance=()),
    )
    np.testing.assert_array_equal(result.ok_leak_inputs["bulk_dens"], litter)
    assert result.ok_leak_inputs["bulk_dens_litter_prepared"] is True
    assert result.ok_leak_inputs["peat_bulk_density"] is peat
    np.testing.assert_array_equal(raw, [.0001])


@pytest.mark.parametrize("density", [1.65, 1650., 500., .0001])
@pytest.mark.parametrize("prepared", [False, True])
def test_compiled_scan_body_shares_one_litter_prepared_density(monkeypatch, density, prepared):
    """Real narrow scan body; controls/core seams only expose arguments.

    This compiles the producer scan, not the full scientific carbon core.
    Independent real controls and adapter numerics are tested above.
    """
    from jax_orchidee.driver import orchestration as driver

    class Probe(NamedTuple):
        control_density: object
        core_density: object
        marker: object

    args = adapter_inputs(np.array([density]))
    state = {name: args[name] for name in driver.DriverCompiledOkLeakCarry._fields}
    initial = driver.DriverCompiledOkLeakCarry(**state)
    payload = {name: args[name] for name in (
        "soil_mc", "wat_flux", "runoff_per_soil", "drainage_per_soil", "runoff2peat", "canopy2ground",
        "resp_maint_part_radia")}
    payload.update(precip2ground=np.zeros((1, 2)), precip2canopy=np.zeros((1, 2)),
                   temp_sol=np.array([280.]), tdeep=args["tprof"], hsdeep=np.ones((1, 5, 2)),
                   shumdiag_peat=np.ones((1, 5)))
    steps = driver.DriverCompiledOkLeakStepInputs(**{name: np.asarray(value)[None] for name, value in payload.items()})

    def controls(*_positional, **kwargs):
        return SimpleNamespace(control_temp_above=np.ones((1, 2)),
                               control_moist_above=kwargs["bulk_dens"],
                               soil_mc_top_by_pft=np.ones((1, 2)))

    monkeypatch.setattr(driver, "littercalc_aboveground_controls", controls)
    monkeypatch.setattr(driver, "stomate_permafrost_decomposition_controls", lambda *_a, **_k:
                        SimpleNamespace(prmfrst_soilc_tempctrl=jnp.ones((1, 5, 2)),
                                        prmfrst_soilc_tempctrl_doc=jnp.ones((1, 5, 2))))
    monkeypatch.setattr(driver, "stomate_ok_leak_explicit", lambda **a:
                        Probe(a["control_moist_above"], a["bulk_dens"], jnp.asarray(a["bulk_dens_litter_prepared"])))
    monkeypatch.setattr(driver, "_paper_half_hour_ok_leak_state_updates", lambda _result: state)
    static = dict(z_soil=args["z_soil"], zi_soil=np.arange(5) + .5,
                  poor_soils=args["poor_soils"], flood_frac=args["flood_frac"],
                  bulk_dens=np.array([source_density(density, 1) if prepared else density]),
                  bulk_dens_litter_prepared=jnp.asarray(prepared), clay=args["clay"])
    _, result, _ = driver._paper_compiled_ok_leak_fold(
        initial=initial, steps=steps, static_inputs=static,
        turnover_daily=args["turnover"], bm_to_litter_daily=args["bm_to_litter"],
        biomass=np.zeros((1, 2, 12, 1)), veget_max=args["veget_max"], sla_calc=args["sla_calc"],
        rprof=args["rprof"], pref_soil_veg_fortran=np.array([1, 1]), is_tree=np.array([False, True]),
        is_peat=args["is_peat"], nslm=5, ndeep=5, dt_sechiba=1800., tf_doc_dt_days=1/48,
        frozen_respiration_func=1, moist_func_moyano=True, ok_tf_doc=False, perma_peat=False,
        conc_doc_rain=3.02)
    expected = [source_density(density, 1)]
    np.testing.assert_array_equal(result.control_density, expected)
    np.testing.assert_array_equal(result.core_density, expected)
    assert bool(result.marker)
