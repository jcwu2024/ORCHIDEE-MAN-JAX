import jax
import numpy as np
import pytest

from jax_orchidee.coupled import stomate_active_layer_ok_leak_inputs
from jax_orchidee.stomate.soilcarbon_kernels import AltCalcDocResult


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_ok_leak_active_layer_carries_maximum_and_previous_year_through_day_two(compiled):
    """The original altcalc_DOC saves both the maximum and its layer index."""
    def calculate_arrays(**kwargs):
        return stomate_active_layer_ok_leak_inputs(**kwargs)._replace(provenance=())

    calculate = calculate_arrays
    if compiled:
        calculate = jax.jit(calculate, static_argnames=("firstcall_soilcarbon",))
    inputs = {
        "zi_soil": np.array([0.1, 0.3, 0.7]),
        "restart_altmax": np.array([[0.0, 0.1]]),
        "restart_fixed_cryoturbation_depth": np.zeros((1, 2)),
    }
    warm = np.full((1, 3, 2), 274.0)
    cold = np.array([[[274.0, 274.0], [270.0, 270.0], [270.0, 270.0]]])

    first = calculate(tprof=warm, dayno=1, firstcall_soilcarbon=True, **inputs)
    second = calculate(
        tprof=warm, dayno=1, firstcall_soilcarbon=False,
        previous_state=AltCalcDocResult(**first.state_updates), **inputs,
    )
    np.testing.assert_array_equal(second.output_inputs["altmax"], [[0.3, 0.3]])
    np.testing.assert_array_equal(second.ok_leak_inputs["altmax_ind"], [[2, 2]])
    np.testing.assert_array_equal(second.ok_leak_inputs["altmax_lastyear"], inputs["restart_altmax"])
    np.testing.assert_array_equal(second.state_updates["altmax_ind_lastyear"], [[0, 1]])

    day_two = calculate(
        tprof=cold, dayno=2, firstcall_soilcarbon=False,
        previous_state=AltCalcDocResult(**second.state_updates), **inputs,
    )
    np.testing.assert_array_equal(day_two.output_inputs["altmax"], [[0.1, 0.1]])
    np.testing.assert_array_equal(day_two.ok_leak_inputs["altmax_ind"], [[1, 1]])
    np.testing.assert_array_equal(day_two.ok_leak_inputs["altmax_lastyear"], [[0.3, 0.3]])
    np.testing.assert_array_equal(day_two.state_updates["altmax_ind_lastyear"], [[2, 2]])

    new_run = calculate(
        tprof=cold, dayno=1, firstcall_soilcarbon=True,
        previous_state=AltCalcDocResult(**day_two.state_updates), **inputs,
    )
    np.testing.assert_array_equal(new_run.ok_leak_inputs["altmax_lastyear"], inputs["restart_altmax"])
    np.testing.assert_array_equal(new_run.state_updates["altmax_ind_lastyear"], [[0, 1]])
