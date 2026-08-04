from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from jax_orchidee.stomate.modelout import ModeloutResult
from research.daily_coarse_graining.teacher_parameter_gradient_gate import (
    TeacherGradientOutputs,
    TeacherGradientPairSpec,
    TeacherGradientRuntime,
    _output_scalar,
    _replace_diffuco_parameter,
    _replace_stomate_parameter,
    parameter_reference_value,
    physical_parameter_vector_objective,
    run_teacher_gradient_matrix,
)


def _runtime():
    stomate = teacher.DriverCompiledStomateParameterValues(
        alloc_min=np.arange(14, dtype=np.float64),
        residence_time=np.arange(14, dtype=np.float64) + 20.0,
        vcmax25=np.arange(14, dtype=np.float64) + 40.0,
        maint_resp_slope=np.arange(42, dtype=np.float64).reshape(14, 3),
    )
    diffuco = teacher.DriverCompiledDiffucoParameterValues(
        trans_co2={"g0": 0.00625, "alpha_ll": 0.3},
        rstruct_const=100.0,
    )
    return TeacherGradientRuntime(
        transition=lambda *args: None,
        initial_values=(),
        forcing_days=(),
        science_day_numbers=np.asarray([2]),
        stomate_parameters=stomate,
        hydrol_table_arrays=(),
        landpoint_payload={},
        stomate_restart_template=(),
        stomate_season_values={},
        diffuco_parameters=diffuco,
        day_indices=(2,),
        year=1961,
    )


def test_stomate_parameter_replacement_changes_only_named_pft14_leaf():
    runtime = _runtime()
    updated = _replace_stomate_parameter(runtime.stomate_parameters, "maint_resp_slope_b", 0.125)

    expected = np.asarray(runtime.stomate_parameters.maint_resp_slope).copy()
    expected[13, 1] = 0.125
    np.testing.assert_array_equal(np.asarray(updated.maint_resp_slope), expected)
    np.testing.assert_array_equal(np.asarray(updated.alloc_min), runtime.stomate_parameters.alloc_min)


def test_diffuco_parameter_replacement_preserves_other_tree_leaves():
    runtime = _runtime()
    updated = _replace_diffuco_parameter(runtime.diffuco_parameters, "g0", jnp.asarray(0.01))

    assert float(updated.trans_co2["g0"]) == 0.01
    assert updated.trans_co2["alpha_ll"] == runtime.diffuco_parameters.trans_co2["alpha_ll"]
    assert updated.rstruct_const == runtime.diffuco_parameters.rstruct_const


def test_parameter_reference_value_supports_paper_and_extension_parameters():
    runtime = _runtime()

    assert parameter_reference_value(runtime, "alloc_min") == 13.0
    assert parameter_reference_value(runtime, "maint_resp_slope_c") == 39.0
    assert parameter_reference_value(runtime, "maint_resp_slope_b") == 40.0
    assert parameter_reference_value(runtime, "g0") == 0.00625


def test_output_scalar_selects_real_modelout_and_history_field_axes():
    modelout = ModeloutResult(
        AGB_model=np.arange(28, dtype=np.float64).reshape(2, 1, 14),
        BGB_model=np.zeros((2, 1, 14)),
        GPP_model=np.ones((2, 1, 14)),
        NPP_model=np.ones((2, 1, 14)) * 2.0,
    )
    outputs = TeacherGradientOutputs(
        modelout_fields={"MAINT_RESP": np.arange(28, dtype=np.float64).reshape(2, 1, 14) + 100.0},
        modelout=modelout,
    )

    assert float(_output_scalar(outputs, "AGB_model", output_day=-1, pft_index=13)) == 27.0
    assert float(_output_scalar(outputs, "MAINT_RESP", output_day=0, pft_index=13)) == 113.0


def test_vector_objective_updates_multiple_parameter_families_in_one_transition():
    runtime = _runtime()

    def transition(
        initial_values,
        forcing_days,
        science_day_numbers,
        stomate_parameters,
        hydrol_table_arrays,
        landpoint_payload,
        stomate_restart_template,
        stomate_season_values,
        diffuco_parameters,
    ):
        del (
            initial_values,
            forcing_days,
            science_day_numbers,
            hydrol_table_arrays,
            landpoint_payload,
            stomate_restart_template,
            stomate_season_values,
        )
        shape = (1, 1, 14)
        gpp = jnp.zeros(shape).at[0, 0, 13].set(diffuco_parameters.trans_co2["g0"])
        npp = jnp.zeros(shape).at[0, 0, 13].set(stomate_parameters.alloc_min[13])
        modelout = ModeloutResult(
            AGB_model=jnp.zeros(shape),
            BGB_model=jnp.zeros(shape),
            GPP_model=gpp,
            NPP_model=npp,
        )
        return (), ({}, modelout)

    runtime = TeacherGradientRuntime(**{**runtime.__dict__, "transition": transition})
    objective = physical_parameter_vector_objective(
        runtime,
        parameter_ids=("alloc_min", "g0"),
        output_keys=(("NPP_model", 0), ("GPP_model", 0)),
    )
    result = np.asarray(objective(jnp.asarray([0.3, 0.01], dtype=jnp.float64)))

    np.testing.assert_allclose(result, [0.3, 0.01])


def test_projected_jvp_vjp_matrix_matches_finite_differences_without_wide_jacobian():
    runtime = _runtime()

    def transition(*args):
        stomate_parameters = args[3]
        diffuco_parameters = args[8]
        shape = (1, 1, 14)
        alloc = stomate_parameters.alloc_min[13]
        g0 = diffuco_parameters.trans_co2["g0"]
        gpp = jnp.zeros(shape).at[0, 0, 13].set(2.0 * g0)
        npp = jnp.zeros(shape).at[0, 0, 13].set(3.0 * alloc + g0)
        modelout = ModeloutResult(
            AGB_model=jnp.zeros(shape),
            BGB_model=jnp.zeros(shape),
            GPP_model=gpp,
            NPP_model=npp,
        )
        return (), ({}, modelout)

    runtime = TeacherGradientRuntime(**{**runtime.__dict__, "transition": transition})
    results = run_teacher_gradient_matrix(
        runtime,
        (
            TeacherGradientPairSpec(
                "mock.npp__alloc",
                "smooth_active",
                "alloc_min",
                "NPP_model",
                1.0e-5,
                0,
            ),
            TeacherGradientPairSpec(
                "mock.gpp__g0",
                "smooth_active",
                "g0",
                "GPP_model",
                1.0e-6,
                0,
            ),
        ),
    )

    assert all(result.passed for result in results)
    np.testing.assert_allclose([result.forward_ad for result in results], [3.0, 2.0])
    np.testing.assert_allclose([result.reverse_ad for result in results], [3.0, 2.0])
