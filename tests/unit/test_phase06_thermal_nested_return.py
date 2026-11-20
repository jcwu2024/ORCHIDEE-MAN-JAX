"""Portable preservation of the saved complete-return thermal strict gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from jax_orchidee.sechiba import thermosoil
from scripts.dev import diagnose_phase06_thermal_nested_return as diagnostic
from scripts.dev import oracle_physics_branch_thermal as oracle

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/phase06_thermal_nested_return_reference.json"


def reference():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_portable_frozen_complete_input_layout_and_bytes():
    record = reference()
    inputs = {name: oracle.decode(value) for name, value in record["inputs"].items()}
    assert diagnostic.identity(inputs) == record["input_identity"]
    assert record["rtol"] == record["atol"] == 1e-12
    assert record["preserved_failed_flat_indices"] == list(range(13))


@pytest.mark.xfail(strict=True, raises=AssertionError, reason="P05-PB03-NESTED-RETURN-STRICT: preserved complete-return 13/14 flux discrepancy")
def test_complete_module_inner_jit_preserves_standalone_strict_return():
    """Keep the original complete return obligation beyond getdiff material."""
    record = reference()
    inputs = {name: oracle.decode(value) for name, value in record["inputs"].items()}
    kwargs = oracle.accepted.owner_kwargs(inputs)
    baseline = thermosoil.run_thermosoil_first_step_module(**kwargs, use_jit=True).module

    def owner(ptn):
        restart = SimpleNamespace(**{**vars(kwargs["thermosoil_restart"]), "ptn": ptn})
        return thermosoil.run_thermosoil_first_step_module(
            **{**kwargs, "thermosoil_restart": restart}, use_jit=True,
        ).module

    source_error, nested = jax.jit(checkify.checkify(owner))(jnp.asarray(inputs["ptn"]))
    checkify.check_error(source_error)
    actual, expected = diagnostic.flatten(nested), diagnostic.flatten(baseline)
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        np.testing.assert_allclose(value, expected[name], rtol=record["rtol"], atol=record["atol"], err_msg=name)


@pytest.mark.external_data
def test_original_coefficient_source_identity_separate_from_portable_gate():
    record = reference()["external_source_identity"]
    source = ROOT / record["path"]
    if not source.is_file():
        pytest.skip("Original source identity only; portable complete-return gate remains active")
    raw = source.read_bytes()
    span = b"".join(raw.splitlines(keepends=True)[record["start"] - 1:record["end"]])
    assert hashlib.sha256(raw).hexdigest() == record["source_sha256"]
    assert hashlib.sha256(span).hexdigest() == record["sha256"]
