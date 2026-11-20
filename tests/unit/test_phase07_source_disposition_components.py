"""Portable CPU/x64 checks from original Fortran Phase07 component outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_orchidee.sechiba import condveg, hydrol
from scripts.dev import oracle_phase06_growth_arithmetic as growth

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def reference():
    with np.load(ROOT / "tests/fixtures/phase07_source_disposition_components.npz", allow_pickle=False) as data:
        yield data


def test_explicit_snow_cover_threshold_on_original_binary64_neighbors(reference):
    inputs = {name: reference[f"canopy.{name}"] for name in ("snow", "snow_nobio", "snowrho", "snowdz")}
    component = jax.jit(condveg.condveg_frac_snow, static_argnames=("ok_explicitsnow",))
    result = component(**inputs, ok_explicitsnow=True)
    for name in ("frac_snow_veg", "frac_snow_nobio"):
        np.testing.assert_allclose(np.asarray(getattr(result, name)).ravel(), reference[f"canopy.{name}"], rtol=1e-11, atol=1e-12)
    assert result.frac_snow_veg[1] == 0
    assert result.frac_snow_veg[2] > 0


def test_source_snow_levels_feed_remap_and_temperature_liquid_consumers(reference):
    def chain(depth):
        old = depth * jnp.array([[0.2, 0.5, 0.3]], dtype=jnp.float64)
        rho = jnp.array([[180., 250., 320.]])
        grain = jnp.array([[0.00012, 0.0002, 0.0003]])
        heat = hydrol.snow3lheat_explicit(jnp.zeros_like(old), rho, old, jnp.array([[268., 270., 272.]]))
        levels = hydrol.explicitsnow_levels_step(depth.reshape(1)).snowdz
        remap = hydrol.explicitsnow_transf_step(snowdz_old=old, snowdz=levels, snowrho=rho, snowheat=heat, snowgrain=grain)
        present = jnp.sum(remap.snowdz, axis=1)[:, None] > 0
        temperature = hydrol.snow3ltemp_explicit(remap.snowheat, remap.snowrho, remap.snowdz)
        temperature = jnp.where(present, temperature, 273.15)
        liquid = hydrol.snow3lliq_explicit(remap.snowheat, remap.snowrho, remap.snowdz, temperature)
        liquid = jnp.where(present, liquid, 0.)
        return jnp.concatenate((levels, remap.snowdz, remap.snowrho, remap.snowheat, remap.snowgrain, temperature, liquid), axis=1).ravel()

    compiled = jax.jit(chain).lower(jnp.float64(0.0)).compile()
    actual = np.stack([np.asarray(compiled(jnp.float64(depth))) for depth in reference["snow.depths"]])
    np.testing.assert_allclose(actual, reference["snow.expected"], rtol=1e-12, atol=1e-14)


def test_normalized_humidity_original_upper_cap_matches_compiled_growth_and_vmax(reference):
    inputs = growth.base.decode(json.loads(reference["metadata"].item())["growth_inputs"])
    result = growth.current(inputs, compiled=True)
    actual = growth.base.chain_fields(result)
    assert {name for name in reference.files if name.startswith("growth.")} == {f"growth.{name}" for name in actual}
    for name, field in actual.items():
        np.testing.assert_allclose(field, reference[f"growth.{name}"], rtol=1e-12, atol=1e-14, err_msg=name)
    vmax = growth.actual_vmax_consumer(inputs, result)
    for name in ("leaf_age", "leaf_frac", "vcmax"):
        np.testing.assert_allclose(getattr(vmax, name), reference[f"growth.vmax.{name}"], rtol=1e-12, atol=1e-14, err_msg=name)


@pytest.mark.external_data
def test_original_source_identity(reference):
    paths = json.loads(reference["metadata"].item())["source_identities"]
    if not (ROOT / "fortran_source/ORCHIDEE").is_dir():
        pytest.skip("Original Fortran source identity is external to portable numerical tests")
    for row in paths:
        raw = (ROOT / row["path"]).read_bytes()
        assert (len(raw), hashlib.sha256(raw).hexdigest()) == (row["bytes"], row["sha256"]), row["path"]
