"""Native REAL*8 paper activity and generic epsilon compatibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from jax_orchidee.coupled import stomate_permafrost_activity_ok_leak_inputs
from jax_orchidee.stomate.main import stomate_activity_control_precomputation
from jax_orchidee.stomate.permafrost import FORTRAN_EPSILON_R4, FORTRAN_EPSILON_R8

ROOT = Path(__file__).resolve().parents[2]
INPUTS = ("tdeep", "hsdeep", "shumdiag_peat", "shumdiag_man", "zz_deep", "poor_soils")
PROFILES = ("mc_peat", "mc_man", "hsdeep_new")
RATES = ("prmfrst_soilc_tempctrl", "prmfrst_soilc_tempctrl_doc")
CASES = ("warm_dry", "cold_dry", "warm_zero", "warm_normal", "hot_wet", "freeze_band")
IS_PEAT = np.arange(14) == 13
RTOL, ATOL = 1e-12, 1e-14


@pytest.fixture(scope="module")
def reference():
    with np.load(ROOT / "tests/fixtures/phase09_paper_activity.npz", allow_pickle=False) as data:
        yield data


def activity(arguments):
    computed = stomate_activity_control_precomputation(
        **arguments, frozen_respiration_func=1, is_peat=IS_PEAT,
    )
    return {name: getattr(computed if name in PROFILES else computed.controls, name) for name in (*PROFILES, *RATES)}


def assert_activity_matches_native(actual, expected):
    for name, native in expected.items():
        np.testing.assert_allclose(actual[name], native, rtol=RTOL, atol=ATOL, err_msg=name)
        if name in RATES:
            # Observe the nonzero active-PFT floor below the absolute gate.
            np.testing.assert_allclose(actual[name][..., 13], native[..., 13], rtol=RTOL, atol=0, err_msg=name)


@pytest.mark.parametrize("case", CASES)
def test_paper_activity_matches_native_under_jit(reference, case):
    arguments = {name: reference[f"{case}/input/{name}"] for name in INPUTS}
    expected = {name: reference[f"{case}/original/{name}"] for name in (*PROFILES, *RATES)}
    assert_activity_matches_native(jax.jit(activity)(arguments), expected)


def test_paper_activity_matches_native_under_outer_vmap(reference):
    order = ("hot_wet", "warm_dry", "freeze_band", "warm_zero", "cold_dry", "warm_normal")
    arguments = {name: np.stack([reference[f"{case}/input/{name}"] for case in order]) for name in INPUTS}
    expected = {name: np.stack([reference[f"{case}/original/{name}"] for case in order])
                for name in (*PROFILES, *RATES)}
    assert_activity_matches_native(jax.jit(jax.vmap(activity))(arguments), expected)


def test_generic_activity_default_and_real8_override_reach_all_ok_leak_inputs(reference):
    data = {name: reference[f"warm_dry/input/{name}"] for name in ("tdeep", "hsdeep", "zz_deep", "poor_soils")}
    data["mc_peat"] = reference["warm_dry/original/mc_peat"]

    def controls(arguments, **precision):
        result = stomate_permafrost_activity_ok_leak_inputs(
            **arguments, perma_peat=True, frozen_respiration_func=1, is_peat=IS_PEAT, **precision,
        )
        return tuple(result.ok_leak_inputs[name] for name in ("fbact_litter", "fbact_soilcarbon", "fbact_doc"))

    repaired = jax.jit(lambda a: controls(a, epsilon=FORTRAN_EPSILON_R8))(data)
    default = jax.jit(controls)(data)
    explicit = jax.jit(lambda a: controls(a, epsilon=FORTRAN_EPSILON_R4))(data)
    for name, got, implicit_r4, explicit_r4 in zip((RATES[0], *RATES), repaired, default, explicit, strict=True):
        expected = reference[f"warm_dry/original/{name}"][..., 13]
        np.testing.assert_allclose(got[..., 13], expected, rtol=RTOL, atol=0)
        np.testing.assert_array_equal(implicit_r4, explicit_r4)
        np.testing.assert_allclose(implicit_r4[..., 13], expected * (2**29), rtol=RTOL, atol=0)


@pytest.mark.external_data
def test_native_activity_source_identity(reference):
    if not (ROOT / "fortran_source/ORCHIDEE").is_dir():
        pytest.skip("Original Fortran source identity is external to portable tests")
    metadata = json.loads(reference["metadata"].item())
    assert tuple(metadata["cases"]) == CASES
    for row in metadata["original_sources"]:
        raw = (ROOT / row["path"]).read_bytes()
        assert (len(raw), hashlib.sha256(raw).hexdigest()) == (row["bytes"], row["sha256"]), row["path"]
