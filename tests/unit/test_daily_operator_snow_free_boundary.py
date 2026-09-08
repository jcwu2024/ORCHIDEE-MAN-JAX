from dataclasses import replace
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import bind_daily_operator, plumbing_test_retained_tail_adapter
from research.daily_coarse_graining.daily_operator.conservative_update import validate_conservative_result
from research.daily_coarse_graining.daily_operator.mechanistic_output_attainability import oracle_learned_outputs
from research.daily_coarse_graining.daily_operator.process_derivations import NativeForcingFieldLayout
from research.daily_coarse_graining.daily_operator.snow_free_boundary import (
    DENSITY_KEY,
    HYDROL,
    THERMAL_REFERENCES,
    ZERO_OUTPUT_KEYS,
    ZERO_STATE_KEYS,
    apply_warm_snow_free_boundary,
    warm_snow_free_regime,
)
from research.daily_coarse_graining.daily_operator.types import NativeForcingInput
from tests.unit.test_daily_operator_mechanistic_output import _paper_vertical_grid, _set_static
from tests.unit.test_daily_operator_mechanistic_output import mechanistic_setup as _mechanistic_setup

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def mechanistic_setup():
    return _mechanistic_setup.__wrapped__()


def simple_case():
    keys = (*ZERO_STATE_KEYS, *(f"{HYDROL}.{name}" for name in THERMAL_REFERENCES))
    contract = NS(state_leaves=tuple(NS(key=k, start=i, stop=i + 1) for i, k in enumerate(keys)))
    state = jnp.zeros(len(keys)).at[-1].set(273.15)
    # Predecessor snowfall and padded NaNs must not count as today's snowfall.
    forcing = NativeForcingInput(jnp.array([[9., 280.], [0., 285.], [jnp.nan, jnp.nan]]),
                                jnp.array([0., 1., 2.]), jnp.ones(3),
                                jnp.array([True, False, False]), jnp.array([True, True, False]))
    layout = NativeForcingFieldLayout(0, 1, 0, 1, 1, 2)
    outputs = {k: jnp.ones(1) for k in (*ZERO_OUTPUT_KEYS, DENSITY_KEY)}
    outputs.update({f"thermal.{HYDROL}.{n}.normalized_tendency": jnp.array([0.05]) for n in THERMAL_REFERENCES})
    outputs["daily.fast_gpp"] = jnp.array([7.0])
    inputs = NS(canonical_input=NS(continuous_state=state), forcing=forcing)
    return contract, inputs, layout, outputs


@pytest.mark.parametrize("change", ("snow", "depth", "liquid", "nobio_snow", "nobio_fraction", "snowfall", "frost", "cold_predecessor"))
def test_active_regimes_keep_original_outputs(change):
    contract, inputs, layout, outputs = simple_case()
    if change in ("snow", "depth", "liquid", "nobio_snow", "nobio_fraction"):
        i = ("snow", "depth", "liquid", "nobio_snow", "nobio_fraction").index(change)
        inputs.canonical_input.continuous_state = inputs.canonical_input.continuous_state.at[i].set(1e-12)
    else:
        record, field, value = {"snowfall": (1, 0, 1e-9), "frost": (1, 1, 273.15),
                                "cold_predecessor": (0, 1, 270.0)}[change]
        inputs.forcing = inputs.forcing._replace(values=inputs.forcing.values.at[record, field].set(value))
    actual = apply_warm_snow_free_boundary(outputs, inputs, contract, layout)
    for key in outputs:
        np.testing.assert_array_equal(actual[key], outputs[key])


def test_warm_zero_snow_branch_and_thermal_forward_reverse_fd():
    contract, inputs, layout, outputs = simple_case()
    assert bool(warm_snow_free_regime(inputs.canonical_input.continuous_state, inputs.forcing, contract, layout))
    corrected = apply_warm_snow_free_boundary(outputs, inputs, contract, layout)
    assert all(np.all(np.asarray(corrected[k]) == 0) for k in ZERO_OUTPUT_KEYS)
    assert float(corrected[DENSITY_KEY][0]) == 50.0
    np.testing.assert_array_equal(corrected["daily.fast_gpp"], outputs["daily.fast_gpp"])

    def endpoint(values):
        state = inputs.canonical_input.continuous_state.at[-2:].set(values[:2])
        changed = NS(canonical_input=NS(continuous_state=state), forcing=inputs.forcing)
        predicted = dict(outputs)
        predicted[f"thermal.{HYDROL}.snowheat.normalized_tendency"] = values[2:3]
        predicted[f"thermal.{HYDROL}.snowtemp.normalized_tendency"] = values[3:4]
        final = apply_warm_snow_free_boundary(predicted, changed, contract, layout)
        return jnp.stack([values[i] + bound * final[f"thermal.{HYDROL}.{name}.normalized_tendency"][0]
                          for i, (name, (_, bound)) in enumerate(THERMAL_REFERENCES.items())])

    point = jnp.array([12345.0, 280.0, 0.1, 0.2])
    direction = jnp.array([12.0, 0.2, 0.01, 0.01])
    primal, tangent = jax.jit(lambda x, v: jax.jvp(endpoint, (x,), (v,)))(point, direction)
    reverse = jax.jacrev(endpoint)(point) @ direction
    fd = (endpoint(point + 0.01 * direction) - endpoint(point - 0.01 * direction)) / 0.02
    np.testing.assert_allclose(primal, [0., 273.15], atol=1e-10)
    np.testing.assert_allclose(tangent, 0., atol=1e-10)
    np.testing.assert_allclose(reverse, tangent, atol=1e-10)
    np.testing.assert_allclose(fd, tangent, atol=1e-8)


def test_canonical_binding_snow_mirrors_and_budgets(mechanistic_setup):
    setup = mechanistic_setup
    head = replace(setup["head"], warm_snow_free_boundary=True)
    assembler = head.input_assembler
    inputs = setup["assembled"]
    grid = _paper_vertical_grid()
    for name, value in {
        "measurement_heights": [2., 10.], "hydrol_zz_mm": np.asarray(grid.znh) * 1000.,
        "hydrol_dz_mm": np.asarray(grid.dnh) * 1000., "hydrol_cwrr_ks": np.ones(12),
        "soiltile": [[1., 0., 0., 0., 0., 0.]], "njsc": [6],
        "pref_soil_veg": np.array([1] * 13 + [6]),
        "ext_coeff_vegetfrac": np.array([0.] + [0.5] * 13), "sechiba_qsint": 0.1,
    }.items():
        inputs = _set_static(inputs, assembler, name, value)
    canonical = inputs.canonical_input
    state = jnp.asarray(canonical.continuous_state)
    targets = {**{k: 0. for k in ZERO_STATE_KEYS},
               f"{HYDROL}.snowheat": 0., f"{HYDROL}.snowtemp": 273.15}
    for leaf in head.contract.state_leaves:
        if leaf.key in targets:
            state = state.at[leaf.start:leaf.stop].set(targets[leaf.key])
    forcing = canonical.forcing._replace(values=jnp.asarray(canonical.forcing.values)
        .at[:, head.forcing_layout.snow_start:head.forcing_layout.snow_stop].set(0.)
        .at[:, head.forcing_layout.temperature_start:head.forcing_layout.temperature_stop].set(280.))
    canonical = canonical._replace(continuous_state=state, forcing=forcing)
    inputs = assembler.assemble(canonical)
    learned, _ = oracle_learned_outputs(head, setup["targets"], inputs)
    for key in ZERO_OUTPUT_KEYS:
        learned[key] = jnp.full_like(learned[key], 0.1)
    for name in THERMAL_REFERENCES:
        key = f"thermal.{HYDROL}.{name}.normalized_tendency"
        learned[key] = jnp.full_like(learned[key], 0.02)

    class FixedLearnedHead:
        def assemble_context(self, parameters, assembled):
            return None

        def __call__(self, parameters, context, assembled):
            return head.prediction_from_learned_outputs(parameters, assembled)

        materialize_post_update = head.materialize_post_update

    transition = bind_daily_operator(input_assembler=assembler, process_head=FixedLearnedHead(),
                                    retained_tail=plumbing_test_retained_tail_adapter())
    result = transition(learned, canonical)
    validate_conservative_result(result.fast_update)
    fields, _ = head.reconstructor.updated_fast_owner_fields(result.prediction, result.fast_update, inputs)
    for owner in (HYDROL, "diffuco_previous_step_state"):
        for name, value in (("snow", 0.), ("snowdz", 0.), ("snowheat", 0.), ("snowtemp", 273.15), ("snowrho", 50.)):
            np.testing.assert_allclose(fields[owner][name], value, atol=1e-10)
    np.testing.assert_array_equal(fields["enerbil_previous_step_state"]["snowdz"], 0.)
    assert abs(float(result.fast_update.water.budget_residual)) < 2e-12
    assert abs(float(result.fast_update.carbon.budget_residual)) < 2e-12
