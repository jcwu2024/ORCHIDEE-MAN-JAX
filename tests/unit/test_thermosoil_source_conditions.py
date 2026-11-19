"""Source aborts and safely overwritten initialization markers at real owners."""

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

from jax_orchidee.sechiba import coupling, thermosoil
from scripts.dev import oracle_physics_branch_thermal as oracle

ROOT = Path(__file__).resolve().parents[2]


def reference(name):
    return json.loads((ROOT / "tests/fixtures" / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize("mode", ("eager", "jit"))
def test_source_abort_rejects_actual_component_before_downstream_payload(mode, monkeypatch):
    record = reference("thermosoil_source_abort_reference.json")
    inputs = {name: oracle.decode(value) for name, value in record["inputs"].items()}
    published = []
    original = coupling.thermosoil_downstream_payload

    def observe(result):
        published.append(result)
        return original(result)

    monkeypatch.setattr(coupling, "thermosoil_downstream_payload", observe)
    with pytest.raises(ValueError, match=record["expected_error"]):
        oracle.run_production(inputs, mode)
    assert not published


def test_functional_error_survives_nonfinal_scan_iteration():
    record = reference("thermosoil_source_abort_reference.json")
    bad = jnp.asarray(oracle.decode(record["guard_input"]["profil_froz"]))
    good = jnp.broadcast_to(bad[:, :, :1], bad.shape)

    def scan(profiles):
        def body(previous, profile):
            shared = thermosoil.thermosoil_frozen_profile_mean(profile)
            return shared, previous
        return jax.lax.scan(body, jnp.zeros(bad.shape[:2]), profiles)

    executable = jax.jit(checkify.checkify(scan))
    error, _ = executable(jnp.stack((good, good)))
    checkify.check_error(error)
    error, _ = executable(jnp.stack((bad, good)))
    with pytest.raises(ValueError, match=record["expected_error"]):
        checkify.check_error(error)


def test_unfunctionalized_jit_cannot_silently_skip_source_check():
    with pytest.raises(ValueError, match="checkify.checkify"):
        jax.jit(thermosoil.thermosoil_frozen_profile_mean)(jnp.zeros((1, 2, 14)))


@pytest.mark.parametrize("rejects", (False, True), ids=("valid-same-input", "source-abort"))
def test_real_module_inner_jit_composes_with_checkified_owner(rejects, monkeypatch):
    """Compose the real component/Error-result/owner seam, without a model step."""
    if rejects:
        record = reference("thermosoil_source_abort_reference.json")
        inputs = {name: oracle.decode(value) for name, value in record["inputs"].items()}
    else:
        record = oracle.load_fixture()
        inputs = oracle.decode_inputs(record["cases"][0])
    owner_kwargs = oracle.accepted.owner_kwargs(inputs)
    if not rejects:
        baseline = thermosoil.run_thermosoil_first_step_module(**owner_kwargs, use_jit=True).module.getdiff

    actual_inner = thermosoil._thermosoil_explicit_step_jit
    error_pairs = []

    def observe_inner(**kwargs):
        output = actual_inner(**kwargs)
        error_pairs.append(isinstance(output[0], checkify.Error))
        return output

    monkeypatch.setattr(thermosoil, "_thermosoil_explicit_step_jit", observe_inner)

    def narrow_owner(ptn):
        restart = SimpleNamespace(**{**vars(owner_kwargs["thermosoil_restart"]), "ptn": ptn})
        closure = thermosoil.run_thermosoil_first_step_module(
            **{**owner_kwargs, "thermosoil_restart": restart}, use_jit=True)
        # This interface owner consumes material fields. Full-state numerical
        # acceptance remains governed by the unchanged component strict gates.
        return closure.module.getdiff

    source_error, module = jax.jit(checkify.checkify(narrow_owner))(jnp.asarray(inputs["ptn"]))
    assert error_pairs and all(error_pairs)
    if rejects:
        with pytest.raises(ValueError, match=record["expected_error"]):
            checkify.check_error(source_error)
    else:
        checkify.check_error(source_error)
        actual = jax.tree_util.tree_leaves(module)
        expected = jax.tree_util.tree_leaves(baseline)
        assert len(actual) == len(expected)
        for value, unchanged in zip(actual, expected, strict=True):
            np.testing.assert_allclose(value, unchanged, rtol=record["rtol"], atol=record["atol"])


def test_partial_markers_are_overwritten_before_actual_consumers():
    # This maintained harness calls the public initializer and actual profile /
    # energy kernels; it reads only the portable fixture, with no Fortran run.
    import oracle_thermosoil_initialization as initialization

    record = reference("thermosoil_partial_marker_reference.json")
    inputs = {name: oracle.decode(value) for name, value in record["inputs"].items()}
    actual, flags = initialization.jax_results(inputs, {"cold_default_boundary": False})
    assert flags["calculate_coef"] and flags["real_owner_assembler_executed"]
    # The harness also exposes adapter-input witnesses with no numerical
    # oracle; every one of the 43 original output fields remains required.
    assert record["expected"].keys() <= actual.keys()
    for name, expected in record["expected"].items():
        np.testing.assert_allclose(actual[name], oracle.decode(expected),
            rtol=record["rtol"], atol=record["atol"], err_msg=name)
    for name in initialization.COEFFICIENTS:
        assert not np.any(actual["initial." + name] == 999999.0)
    for name in ("surfheat_incr", "coldcont_incr", "ptn_beg", "temp_sol_beg"):
        np.testing.assert_allclose(actual["jit_energy." + name],
            oracle.decode(record["expected"]["consumer." + name]),
            rtol=record["rtol"], atol=record["atol"], err_msg=name)
    for boundary, output in (("pcapa_en_previous", "pcapa_en"),
                              ("temp_sol_beg", "temp_sol_beg"),
                              ("soilcap_initial", "soilcap")):
        np.testing.assert_array_equal(actual["owner." + boundary], actual["initial." + output])


def test_partial_markers_in_retained_group_remain_rejected():
    import oracle_thermosoil_initialization as initialization

    record = reference("thermosoil_partial_marker_reference.json")
    inputs = {name: oracle.decode(value) for name, value in record["inputs"].items()}
    # No wholly missing member: the source would retain this raw group.
    inputs["rest_cgrnd"] = np.zeros_like(inputs["rest_cgrnd"])
    with pytest.raises(ValueError, match="restart soilcap_pft mixes val_exp"):
        initialization.jax_results(inputs, {"cold_default_boundary": False})


def test_source_identity_for_original_abort_and_group_overwrite():
    record = reference("thermosoil_source_abort_reference.json")
    source = ROOT / record["source_file"]
    if not source.is_file():
        pytest.skip("Original source identity only; numerical and input checks remain active")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record["source_sha256"]
