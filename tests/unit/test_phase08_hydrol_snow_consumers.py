"""Portable same-input checks at the frozen-bin and local snow boundaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_orchidee.sechiba import hydrol
from jax_orchidee.stomate import permafrost
from jax_orchidee.stomate.main import stomate_activity_control_precomputation

ROOT = Path(__file__).resolve().parents[2]
ACTIVITY_INPUTS = ("tdeep", "hsdeep", "shumdiag_peat", "shumdiag_man", "zz_deep", "poor_soils")
ACTIVITY_FIELDS = ("mc_peat", "mc_man", "hsdeep_new", "prmfrst_soilc_tempctrl", "prmfrst_soilc_tempctrl_doc")
RTOL, ATOL = 1e-12, 1e-14


@pytest.fixture(scope="module")
def reference():
    with np.load(ROOT / "tests/fixtures/phase08_hydrol_snow_consumers.npz", allow_pickle=False) as data:
        yield data, json.loads(data["metadata"].item())


def frozen_writeback_chain(reference, case):
    data, metadata = reference
    settings = metadata["frozen_cases"][str(case)]
    mineral = metadata["frozen_cases"]["8"]
    texture = np.array([settings["texture"]])
    mcs = np.array([mineral["mcs"]])
    mcr = np.array([mineral["mcr"]])
    dz = data["dz_mm"]
    cwrr = hydrol.build_mineral_cwrr_tables(texture, mcs, data["z_m"])
    peat_tables = hydrol.build_peat_cwrr_tables(data["z_m"])
    peat = settings["peat"]
    dt = settings["dt_days"]
    active = np.ones(1)
    root = data[f"input_root_factor_case{case}"]

    def chain(mc, flux, solve):
        frozen = jnp.full_like(mc, settings["frozen_fraction"])

        def coefficient(value):
            if peat:
                return hydrol.hydrol_soil_coef_peat_profile_from_tables(value, frozen, root, peat_tables)
            return hydrol.hydrol_soil_coef_mineral_profile_from_tables(value, frozen, root, cwrr, mcr, mcs)

        coef = coefficient(mc)
        infiltrated = hydrol.hydrol_soil_infilt_explicit(
            mc=mc, flux_infilt=flux, k=coef.k, dz_mm=dz, dt_days=dt, mcs=mcs,
            ks=hydrol.USDA_KS, kfact=cwrr.kfact, kfact_root=root, njsc=texture,
            soil_tile_index=settings["tile"], peat_hydro=peat, mcs_peat=peat_tables.mcs,
            ks_peat=peat_tables.ks, kfact_peat=peat_tables.kfact, ok_freeze_cwrr=True,
            temp_hydro=jnp.full_like(mc, settings["temp_hydro_K"]),
        )
        updated = coefficient(infiltrated.mc)
        setup = hydrol.hydrol_soil_setup_coefficients(updated.a, updated.d, dz, dt, active)
        liquid_mcr = peat_tables.mcr if peat else mcr
        liquid = hydrol.hydrol_mc_to_mcl(infiltrated.mc, frozen, liquid_mcr)
        rhs = hydrol.hydrol_soil_rhs_main(setup, liquid, updated.b, dt, active)
        solved = hydrol.hydrol_soil_tridiag_solve(rhs.e, rhs.f, rhs.g1, rhs.rhs, resolv=solve, initial_mcl=liquid).mcl
        written = hydrol.hydrol_mcl_to_mc_after_solve(solved, infiltrated.mc, frozen, liquid_mcr)
        return updated.bin_i[0], written[0]

    args = data[f"input_mc_case{case}"], np.array([settings["flux"]]), np.array([settings["solve"]])
    return chain, args, data[f"original_post_bins_case{case}"], data[f"fortran_case{case}"][-11:]


@pytest.mark.parametrize("case", [8, 12], ids=["mineral", "peat"])
@pytest.mark.parametrize("batched", [False, True], ids=["jit", "jit_vmap"])
def test_frozen_coefficient_bin_reaches_actual_moisture_writeback(reference, case, batched):
    chain, inputs, source_bins, source_water = frozen_writeback_chain(reference, case)
    if batched:
        bins, written = jax.jit(jax.vmap(chain))(*(np.stack((arg, arg)) for arg in inputs))
        source_bins = np.stack((source_bins, source_bins))
        source_water = np.stack((source_water, source_water))
    else:
        bins, written = jax.jit(chain)(*inputs)
    np.testing.assert_array_equal(bins, source_bins)
    np.testing.assert_allclose(written, source_water, rtol=RTOL, atol=ATOL)


def activity_controls(inputs):
    result = stomate_activity_control_precomputation(
        **inputs, frozen_respiration_func=1, is_peat=jnp.array([False] * 13 + [True]),
    )
    return {name: getattr(result if name in ACTIVITY_FIELDS[:3] else result.controls, name) for name in ACTIVITY_FIELDS}


compiled_activity_controls = jax.jit(activity_controls)


@pytest.mark.parametrize("case", [3, 4, 5, 6])
def test_peat_bucket_and_activity_match_original(reference, case):
    data, _ = reference
    inputs = {name: data[f"case{case}/input/{name}"] for name in ACTIVITY_INPUTS}
    for name, actual in compiled_activity_controls(inputs).items():
        np.testing.assert_allclose(actual, data[f"case{case}/original/{name}"], rtol=RTOL, atol=ATOL, err_msg=name)


@pytest.mark.parametrize("order", [(4, 5), (6, 5, 4, 3)], ids=["controller_cases", "reordered_neighbors"])
def test_activity_controls_match_original_under_outer_vmap(reference, order):
    data, _ = reference
    inputs = {name: np.stack([data[f"case{case}/input/{name}"] for case in order]) for name in ACTIVITY_INPUTS}
    actual = jax.jit(jax.vmap(activity_controls))(inputs)
    for name in ACTIVITY_FIELDS:
        expected = np.stack([data[f"case{case}/original/{name}"] for case in order])
        np.testing.assert_allclose(actual[name], expected, rtol=RTOL, atol=ATOL, err_msg=name)


@pytest.fixture(scope="module")
def source_bins():
    with np.load(ROOT / "tests/fixtures/phase08_discrete_bin_boundaries.npz", allow_pickle=False) as data:
        yield data


def test_frozen_bins_match_source_at_boundaries_in_distinct_profiles(source_bins):
    def project(mc, freeze, residual, saturation, imin, imax):
        selected = hydrol.select_mineral_cwrr_bin(mc, freeze, residual, saturation, imin=imin, imax=imax)
        return selected.mc_used, selected.bin_i

    fields = ("mc", "frozen", "residual", "saturation", "imin", "imax")
    args = tuple(source_bins[f"cwrr/input_{key}"] for key in fields)
    for actual in (
        jax.jit(project)(*args),
        jax.jit(jax.vmap(project))(*(v.reshape(3, 1, -1) for v in args)),
    ):
        for field, value in zip(("original_used", "original_bins"), actual, strict=True):
            np.testing.assert_array_equal(np.asarray(value).ravel(), source_bins[f"cwrr/{field}"])


def test_frozen_bin_broadcast_preserves_assigned_moisture():
    mc = jnp.array([0.37], dtype=jnp.float32)
    frozen = jnp.array([0.42], dtype=jnp.float32)
    residual = jnp.array(0.067, dtype=jnp.float32)
    used = residual + (1.0 - frozen) * jnp.maximum(mc - residual, 0.0)
    select = jax.jit(hydrol.select_mineral_cwrr_bin)
    for saturation in (jnp.array([0.45], dtype=jnp.float64), jnp.empty((0,), dtype=jnp.float32)):
        actual = select(mc, frozen, residual, saturation)
        assert actual.mc_used.dtype == used.dtype
        np.testing.assert_array_equal(actual.mc_used, used)
        assert actual.bin_i.shape == saturation.shape


@pytest.mark.parametrize("width", [32, 64])
def test_moyano_source_indices_and_epsilons(source_bins, width):
    values = source_bins[f"moyano{width}/points"]
    expected = source_bins[f"moyano{width}/original"]
    index = permafrost._moyano_bin_index
    np.testing.assert_array_equal(jax.jit(index)(values), expected)
    np.testing.assert_array_equal(jax.jit(jax.vmap(index))(values.reshape(4, 1, -1)).ravel(), expected)
    dry = jnp.asarray(0, dtype=values.dtype)
    np.testing.assert_array_equal(permafrost.moyano_peat_moisture_lookup(dry), np.finfo(np.float32).eps)
    exact_eps = np.finfo(values.dtype).eps
    np.testing.assert_array_equal(permafrost.moyano_peat_moisture_lookup(dry, epsilon=exact_eps), exact_eps)


def test_smooth_control_derivatives_are_preserved(reference):
    def liquid(moisture, frozen):
        return hydrol.select_mineral_cwrr_bin(moisture, frozen, 0.067, 0.45).mc_used

    moisture = jnp.array([0.26, 0.38])
    frozen = jnp.array([0.5, 0.25])
    liquid_in_profiles = jax.jit(jax.vmap(liquid))
    _, tangent = jax.jvp(liquid_in_profiles, (moisture, frozen), (jnp.ones_like(moisture), jnp.zeros_like(frozen)))
    np.testing.assert_array_equal(tangent, 1.0 - frozen)
    np.testing.assert_array_equal(jax.jit(jax.vmap(jax.grad(liquid)))(moisture, frozen), tangent)

    data, _ = reference
    inputs = {name: data[f"case3/input/{name}"] for name in ACTIVITY_INPUTS}

    def rate(temperature):
        tdeep = jnp.full_like(inputs["tdeep"], temperature)
        return activity_controls({**inputs, "tdeep": tdeep})["prmfrst_soilc_tempctrl"][0, 0, 13]

    t = jnp.asarray(280.0)
    value, tangent = jax.jvp(jax.jit(rate), (t,), (jnp.ones_like(t),))
    expected = value * (np.log(2.0) / 10.0)
    assert value > 0
    np.testing.assert_allclose(tangent, expected, rtol=RTOL, atol=ATOL)
    _, vjp = jax.vjp(jax.jit(rate), t)
    np.testing.assert_allclose(vjp(jnp.ones_like(value))[0], expected, rtol=RTOL, atol=ATOL)


@jax.jit
def snow_melt_chain(inputs):
    keys = ("precip_rain", "pgflux", "soilcap", "snowtemp", "snowdz", "snowrho", "snowliq", "snowmelt",
            "grndflux", "temp_air", "soilflxresid")
    melt = hydrol.explicitsnow_melt_refrz_step(**{key: inputs[key] for key in keys})
    sub = hydrol.explicitsnow_sublimation_step(
        **{key: inputs[key] for key in ("vevapsno", "frac_nobio", "totfrac_nobio")},
        snowrho=melt.snowrho, snowdz=melt.snowdz, snowliq=melt.snowliq, snowtemp=melt.snowtemp,
    )
    mass = hydrol.explicitsnow_maxmass_step(soilcap=inputs["soilcap"], snowrho=melt.snowrho, snowdz=sub.snowdz)
    excess = hydrol.explicitsnow_liquid_heat_excess_step(
        snowliq=sub.snowliq, snowdz=mass.snowdz, snowrho=melt.snowrho, grndflux=melt.grndflux,
    )
    return {name: getattr(melt, name) for name in ("snowdz", "snowrho", "snowtemp", "snowliq", "snowmelt", "grndflux")} | {
        "consumer_snow": jnp.sum(melt.snowrho * mass.snowdz, axis=1),
        "consumer_heat": hydrol.snow3lheat_explicit(excess.snowliq, melt.snowrho, mass.snowdz, sub.snowtemp),
        "consumer_liq": excess.snowliq,
        "consumer_flux": excess.grndflux,
    }


@pytest.mark.parametrize("case", [28, 29, 30], ids=["below", "equal", "above"])
def test_minimum_depth_melt_and_actual_snow_consumers(reference, case):
    data, metadata = reference
    settings = metadata["snow_cases"][str(case)]
    actual = snow_melt_chain({name: np.asarray(value) for name, value in settings["input"].items()})
    for name in settings["fields"]:
        expected = data[f"c{case}_{name}_original"]
        value = np.asarray(actual[name])
        assert value.shape in (expected.shape, (1, *expected.shape)), name
        np.testing.assert_allclose(value.reshape(expected.shape), expected, rtol=RTOL, atol=ATOL, err_msg=name)


@pytest.mark.external_data
def test_original_source_identity(reference):
    _, metadata = reference
    if not (ROOT / "fortran_source/ORCHIDEE").is_dir():
        pytest.skip("Original Fortran source identity is external to portable numerical tests")
    for row in metadata["original_sources"]:
        raw = (ROOT / row["path"]).read_bytes()
        assert (len(raw), hashlib.sha256(raw).hexdigest()) == (row["bytes"], row["sha256"]), row["path"]
