"""Portable original-reference arrays; source identity is separately marked."""

import hashlib
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from jax_orchidee.stomate.soilcarbon_kernels import soilcarbon_leak_doc_diffusion
from scripts.dev.oracle_phase06_soilcarbon_diffusion import (
    ARMS,
    META,
    SPECS,
    actual_chain,
    inputs,
)

jax.config.update("jax_enable_x64", True)
REFERENCE_PATH = Path(__file__).parents[1] / "fixtures/phase06_soilcarbon_diffusion_reference.npz"
REFERENCE = np.load(REFERENCE_PATH)
METADATA = json.loads(META.read_text(encoding="utf-8"))


def assert_reference(actual, prefix):
    names = [name.removeprefix(prefix) for name in REFERENCE.files if name.startswith(prefix)]
    assert names
    for name in names:
        observed = np.asarray(actual[name])
        assert np.isfinite(observed).all(), name
        np.testing.assert_allclose(observed, REFERENCE[prefix + name], rtol=METADATA["rtol"], atol=METADATA["atol"], err_msg=name)


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec["id"])
def test_complete_original_component_and_real_writer_carry(spec):
    x, raw = inputs(spec)
    boundaries = x["zf_soil_b"]
    assert boundaries[0] == 0.0
    assert np.all(np.diff(boundaries) > 0.0)
    assert np.all((raw["zz"] > boundaries[:-1]) & (raw["zz"] < boundaries[1:]))
    for field in ["doc", "carbon_32l", "litter_below"]:
        assert np.isfinite(x[field]).all() and np.min(x[field]) >= 0.0
    cryo = spec["route"] == "cryoturbation"
    nonzero_seen = False
    for call in range(3 if cryo else 1):
        actual, result, next_x = actual_chain(x, cryo=cryo)
        assert_reference(actual, f"{spec['id']}/call{call}/")
        np.testing.assert_allclose(next_x["litter_below"], result.soilcarbon.litter_below, rtol=METADATA["rtol"], atol=METADATA["atol"])
        np.testing.assert_array_equal(next_x["doc"], result.soilcarbon.doc)
        if cryo:
            assert next_x["cryoturbation_coefficients"] is result.soilcarbon.cryoturbation_coefficients
            nonzero_seen |= np.max(np.abs(result.transfers.cryoturbation_litter_transfer)) > 1e-8
            assert np.max(result.soilcarbon.cryoturbation_coefficients.diff_k) > 0.0
        x = next_x
    if cryo:
        assert nonzero_seen


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "narrow_jit"])
@pytest.mark.parametrize("spec", [spec for spec in SPECS if spec["route"] == "diffusion"], ids=lambda spec: spec["id"])
def test_full_diffusion_boundary_tuple_against_original_observer(spec, compiled):
    x, _ = inputs(spec)
    prefix = f"{spec['id']}/observer/"
    function = jax.jit(soilcarbon_leak_doc_diffusion) if compiled else soilcarbon_leak_doc_diffusion
    result = function(REFERENCE[prefix + "entry"], x["tprof"], x["zf_soil_b"], np.full(2, 1e-5 * x["dt_days"]))
    np.testing.assert_allclose(result.doc, REFERENCE[prefix + "after"], rtol=METADATA["rtol"], atol=METADATA["atol"])
    np.testing.assert_allclose(result.doc_flux_diff, REFERENCE[prefix + "flux"], rtol=METADATA["rtol"], atol=METADATA["atol"])
    # Adsorbed DOC is outside this process and must retain the entire shape.
    np.testing.assert_array_equal(np.asarray(result.doc)[:, :, :, 1], REFERENCE[prefix + "entry"][:, :, :, 1])


def test_original_masks_witness_each_target_arm_without_assuming_physical_admission():
    counts = np.zeros(len(ARMS), np.int64)
    for spec in SPECS:
        if spec["route"] != "diffusion":
            continue
        arms = REFERENCE[f"{spec['id']}/observer/arms"]
        assert np.all((arms == 0) | (arms == 1))
        assert np.all(arms[..., :3].sum(axis=-1) <= 1)
        assert np.all(arms[:, 1:, :31, :, :, 3:].sum(axis=-1) == 1)
        counts += arms.sum(axis=tuple(range(arms.ndim - 1)))
    assert np.all(counts > 0), dict(zip(ARMS, counts.tolist()))
    # The literal source peak limiter has a negative intermediate outcome.
    # Numerical source conformance must not relabel that as physical closure.
    assert REFERENCE["thin_nonuniform_limiters/observer/after"].min() < 0.0
    frozen = REFERENCE["nonuniform_frozen/observer/flux"]
    np.testing.assert_array_equal(frozen[:, :, :2], 0.0)


@pytest.mark.external_data
def test_original_fortran_source_identity_is_separate_from_portable_numerics():
    source = Path(__file__).parents[2] / "fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90"
    if not source.exists():
        pytest.skip("Original Fortran source package is unavailable")
    data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == METADATA["source_sha256"]
    lines = data.splitlines(keepends=True)
    for span in METADATA["spans"].values():
        exact = b"".join(lines[span["start"] - 1:span["end"]])
        assert hashlib.sha256(exact).hexdigest() == span["sha256"]
