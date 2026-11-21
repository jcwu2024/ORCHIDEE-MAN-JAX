"""Original DIFFUCO full leaf CO2 state and maintained restart writer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from jax_orchidee.driver.orchestration import _updated_leaf_ci
from jax_orchidee.sechiba.diffuco import run_pft14_local_enerbil_precall_from_first_step_precall
from jax_orchidee.sechiba.science_completion import diffuco_finalize_restart_packet
from scripts.dev.oracle_canopy_component_fixture import make_canopy_fixture

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/phase10_diffuco_leaf_ci.npz"
FIELDS = ("gpp", "gsmean", "rveget", "rstruct", "cimean", "humrel", "vbeta", "vbeta_pft", "valpha", "vbeta3", "vbeta3pot")


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURE, allow_pickle=False) as saved:
        yield saved, json.loads(saved["metadata"].item())


def production(saved, metadata, use_jit, indices=slice(None)):
    base = make_canopy_fixture(indices=(0,))
    inputs = {key.removeprefix("input/"): saved[key][indices] for key in saved.files if key.startswith("input/")}
    precall = base.diffuco_precall._replace(payload=inputs)
    static = {**base.static_kwargs, **metadata["static"], "trans_co2_pft_params": metadata["params"],
              "ok_laidev": np.zeros(14, dtype=bool)}
    return run_pft14_local_enerbil_precall_from_first_step_precall(
        precall, run_def_values={}, static_kwargs=static, use_jit=use_jit,
    )


def packet(result, previous):
    return diffuco_finalize_restart_packet(
        rstruct=result.result.payload.payload["rstruct"], q_cdrag_pft=result.result.boundary.drag.q_cdrag_pft,
        ok_co2=True, leaf_ci=_updated_leaf_ci(previous, result),
    )


@pytest.mark.parametrize("use_jit", [False, True], ids=["eager", "jit"])
def test_full_leaf_ci_packet_matches_original(reference, use_jit):
    saved, metadata = reference
    local = production(saved, metadata, use_jit)
    native = saved["source_leaf_ci"]
    received = packet(local, saved["previous_leaf_ci"])["leaf_ci"]
    for name in FIELDS:
        actual = local.result.payload.payload[name]
        np.testing.assert_allclose(actual, saved[f"source_public/{name}"],
                                   rtol=metadata["rtol"], atol=metadata["atol"], err_msg=name)
    actual_layer = local.result.boundary.process_chain.trans_co2.canopy
    assimilate = saved["assimilate"]
    uncalculated = assimilate[:, None] & ~saved["consumed_mask"]
    assert np.any(uncalculated) and np.any(~assimilate)
    np.testing.assert_array_equal(actual_layer.leaf_ci[uncalculated], native[:, 13, :][uncalculated])
    np.testing.assert_array_equal(received[~assimilate, 13, :], native[~assimilate, 13, :])
    np.testing.assert_array_equal(received[:, :13, :], native[:, :13, :])
    np.testing.assert_allclose(received, native, rtol=metadata["rtol"], atol=metadata["atol"])


def test_restart_leaf_state_is_independent_of_point_neighbours(reference):
    saved, metadata = reference
    selection = [3, 0, 1]
    grouped = production(saved, metadata, True, selection)
    independent = [production(saved, metadata, True, [index]) for index in selection]
    previous = saved["previous_leaf_ci"]
    all_points = packet(grouped, previous[selection])["leaf_ci"]
    separate = np.concatenate([packet(local, previous[[index]])["leaf_ci"]
                               for index, local in zip(selection, independent, strict=True)])
    np.testing.assert_allclose(all_points, separate, rtol=metadata["rtol"], atol=metadata["atol"])
    np.testing.assert_allclose(all_points, saved["source_leaf_ci"][selection],
                               rtol=metadata["rtol"], atol=metadata["atol"])


@pytest.mark.external_data
def test_original_leaf_ci_reference_source_identity(reference):
    source = ROOT / "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90"
    if not source.exists():
        pytest.skip("Original Fortran source identity is external to portable checks")
    _, metadata = reference
    assert hashlib.sha256(source.read_bytes()).hexdigest() == metadata["source_sha256"]
