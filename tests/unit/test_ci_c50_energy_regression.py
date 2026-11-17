"""Tracked clean-checkout regression for the accepted C50 energy allocation."""
from __future__ import annotations

import numpy as np
import pytest
from test_thermal_first_consumer import energy_operands, fixture

from jax_orchidee.sechiba import thermosoil


@pytest.mark.parametrize(
    "caller", ["thermosoil_explicit_step", "thermosoil_no_explicit_snow_step"]
)
def test_c50_original_fortran_energy_reference_at_both_callers(caller: str) -> None:
    record = fixture()
    operands = energy_operands(caller)
    np.testing.assert_array_equal(operands["veget_max"], record["veget_max_bg"])
    actual = thermosoil.thermosoil_energy_diagnostics(**operands)

    np.testing.assert_allclose(
        actual.surfheat_incr,
        record["expected_surfheat_incr"],
        rtol=record["rtol"],
        atol=record["atol"],
    )
    np.testing.assert_allclose(
        actual.coldcont_incr,
        record["expected_coldcont_incr"],
        rtol=record["rtol"],
        atol=record["atol"],
    )
