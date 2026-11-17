"""Portable actual first-step component regression, with original Fortran truth."""

from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np

from scripts.dev.oracle_canopy_component_fixture import make_canopy_fixture, run_canopy_components


def test_canopy_component_first_step_jit_matches_original_fortran():
    reference = json.loads((Path(__file__).parents[1] / "fixtures/canopy_local_component_reference.json").read_text())
    fixture = make_canopy_fixture(indices=(reference["point_index"],))
    # This supported explicit-parameter entry used to fail before compilation:
    # nlai became a tracer when no precomputed layer table was supplied.
    assert "lai_light" not in fixture.static_kwargs["trans_co2_pft_params"]
    result = run_canopy_components(fixture, diffuco_use_jit=True, enerbil_use_jit=True)
    assert jax.default_backend() == "cpu"
    assert jax.config.jax_enable_x64
    assert result.enerbil_precall.missing_external_inputs == ()
    assert type(result.diffuco).__name__ == "DiffucoPFT14FirstStepLocalEnerbilPrecallAssembly"
    assert type(result.enerbil).__name__ == "EnerbilFirstStepLocalAssembly"
    payloads = {"diffuco": result.diffuco.result.payload.payload, "enerbil": result.enerbil.payload}
    for name, expected in reference["expected"].items():
        owner, field = name.split(".")
        array = np.asarray(payloads[owner][field])
        actual = array[0, 13] if array.ndim == 2 else array[0]
        np.testing.assert_allclose(actual, expected, rtol=reference["rtol"], atol=reference["atol"], err_msg=name)
