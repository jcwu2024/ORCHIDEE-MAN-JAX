import numpy as np
import pytest

from jax_orchidee.sechiba.enerbil import (
    EnerbilFirstStepPrecallAssembly,
    run_enerbil_first_step_local_from_precall,
)


@pytest.mark.parametrize("use_jit", [False, True])
def test_enerbil_carries_bare_temperature_from_cold_or_restart_input(use_jit):
    # Two land points exercise the ENERBIL_TSURF cold default and a valid
    # nondefault restart value, with the same forcing and noncrop controls.
    old_temperature = [[280.0, 280.0], [286.125, 280.0]]
    one = np.ones(2)
    zero = np.zeros(2)
    pft = np.ones((2, 2))
    payload = {
        "temp_sol": 280.0 * one,
        "temp_sol_pft": old_temperature,
        "lwdown": 338.0 * one,
        "swnet": 120.0 * one,
        "pb": 1010.0 * one,
        "emis": 0.96 * one,
        "epot_air": 282000.0 * one,
        "petAcoef": 0.01 * one,
        "petBcoef": 281500.0 * one,
        "qair": 0.006 * one,
        "peqAcoef": 0.1 * one,
        "peqBcoef": 0.012 * one,
        "soilflx": 8.0 * one,
        "soilflx_pft": 8.0 * pft,
        "rau": 1.2 * one,
        "u": one,
        "v": zero,
        "q_cdrag": 0.01 * one,
        "q_cdrag_pft": 0.01 * pft,
        "vbeta": 0.4 * one,
        "vbeta_pft": 0.4 * pft,
        "valpha": one,
        "vbeta1": 0.2 * one,
        "vbeta2": 0.1 * pft,
        "vbeta3": 0.3 * pft,
        "vbeta3pot": 0.4 * pft,
        "vbeta4": 0.3 * one,
        "vbeta4_pft": 0.3 * pft,
        "vbeta5": 0.05 * one,
        "soilcap": 5e5 * one,
        "soilcap_pft": 5e5 * pft,
        "veget_max": np.array([[0.0, 1.0], [0.0, 1.0]]),
    }
    precall = EnerbilFirstStepPrecallAssembly(
        payload=payload, missing_inputs=(), driver_inputs=(),
        source_kernel_inputs=(), restart_inputs=(), diffuco_inputs=(),
        provenance=(), notes=(),
    )
    result = run_enerbil_first_step_local_from_precall(
        precall, ok_laidev=[False, False], dt_sechiba=1800.0, ok_explicitsnow=False, use_jit=use_jit,
    )

    np.testing.assert_array_equal(result.payload["temp_sol_new_pft"][:, 0], [280.0, 286.125])
    np.testing.assert_array_equal(result.payload["temp_sol_new_pft"][:, 1], result.payload["temp_sol_new"])
    assert float(result.payload["temp_sol_new"][0]) != 280.0
    assert result.payload["temp_sol_new"][0] == result.payload["temp_sol_new"][1]
