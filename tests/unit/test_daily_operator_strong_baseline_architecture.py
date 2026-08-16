from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import (
    StrongBaselineProcessHead,
    bind_daily_operator,
    plumbing_test_retained_tail_adapter,
    validate_conservative_result,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
    assemble_strong_baseline_context,
    build_strong_baseline_layout,
    encode_strong_baseline_experts,
    initialize_strong_baseline_parameters,
    strong_baseline_parameter_count,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_contract import (
    load_strong_baseline_contract,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_loss import (
    masked_equal_expert_label_macro_mse,
)
from tests.unit.test_daily_operator_successor_process_head import (
    extended_successor_setup,
    successor_setup,
)


@pytest.fixture(scope="module")
def strong_baseline_setup():
    base = successor_setup.__wrapped__()
    extended = extended_successor_setup.__wrapped__(base)
    contract = load_strong_baseline_contract()
    layout = build_strong_baseline_layout(
        adapter=base["adapter"],
        schema=extended["schema"],
        owner_contract=extended["owner_contract"],
        contract=contract,
        inputs=base["assembled"],
    )
    parameters = initialize_strong_baseline_parameters(
        jax.random.PRNGKey(20260816),
        schema=extended["schema"],
        layout=layout,
        inputs=base["assembled"],
        contract=contract,
    )
    head = StrongBaselineProcessHead(
        contract=base["adapter"].contract,
        architecture_contract=contract,
        input_assembler=base["adapter"].input_assembler,
        schema=extended["schema"],
        layout=layout,
        input_normalization=base["input_normalization"],
        output_normalization=extended["output_normalization"],
        forcing_layout=base["head"].forcing_layout,
        forcing_indices=base["head"].forcing_indices,
    )
    return (
        base
        | extended
        | {
            "architecture_contract": contract,
            "head": head,
            "layout": layout,
            "parameters": parameters,
        }
    )


def test_strong_baseline_capacity_precision_and_axis_wiring(strong_baseline_setup):
    setup = strong_baseline_setup
    layout = setup["layout"]
    parameters = setup["parameters"]
    count = strong_baseline_parameter_count(parameters)
    capacity = setup["architecture_contract"].payload["capacity"]
    assert capacity["parameter_count_minimum"] <= count <= capacity["parameter_count_target_maximum"]
    assert count == 49_750_681
    assert {leaf.dtype for leaf in jax.tree_util.tree_leaves(parameters)} == {jnp.dtype("float32")}
    assert parameters["axis_position_embedding"][layout.axis_names.index("pft")] is None
    assert int(layout.static_axis_pool.sum()) == 412
    assert int(layout.process_static_axis_pool.sum()) == 294
    assert len(parameters["axial_blocks"]) == 1
    assert layout.pft_state_indices.shape == np.asarray(setup["assembled"].pft.pft_state).shape
    assert np.all(layout.pft_state_indices >= 0)

    field_index = next(
        index
        for index, field in enumerate(layout.ledger)
        if field.name == "thermal.thermosoil_previous_step_state.ptn.normalized_tendency"
    )
    local = layout.local_state_indices[field_index].reshape(layout.ledger[field_index].shape)
    assert np.all(local[..., 0] >= 0)
    assert np.all(local[..., 13] >= 0)
    assert np.all(local[..., 1:13] == -1)


def test_strong_baseline_factorized_axial_block_propagates_within_one_axis(
    strong_baseline_setup,
):
    setup = strong_baseline_setup
    parameters = setup["parameters"]
    layout = setup["layout"]
    assembled = setup["assembled"]
    expert_index = layout.expert_names.index("hydrology_snow")
    axis_index = layout.axis_names.index("soil_layer")
    offset = layout.axis_offsets[axis_index]
    extent = layout.axis_extents[axis_index]
    first_pool = layout.state_axis_pool[expert_index, offset]
    second_pool = layout.state_axis_pool[expert_index, offset + 1]
    candidates = np.flatnonzero(first_pool & ~second_pool)
    assert candidates.size

    state = np.asarray(assembled.state.other).copy()
    state[candidates[0]] += 10.0
    perturbed = assembled._replace(
        state=assembled.state._replace(other=state),
        canonical_input=assembled.canonical_input._replace(continuous_state=state),
    )
    baseline = assemble_strong_baseline_context(
        parameters,
        assembled,
        normalization=setup["input_normalization"],
        layout=layout,
    )
    observed = assemble_strong_baseline_context(
        parameters,
        perturbed,
        normalization=setup["input_normalization"],
        layout=layout,
    )
    delta = np.asarray(
        observed.expert_axis_tokens[expert_index, offset : offset + extent]
        - baseline.expert_axis_tokens[expert_index, offset : offset + extent]
    )
    assert np.linalg.norm(delta[0]) > 0.0
    assert np.any(np.linalg.norm(delta[1:], axis=1) > 0.0)


def test_strong_baseline_eager_jit_and_bound_one_day_are_finite(strong_baseline_setup):
    setup = strong_baseline_setup
    head = setup["head"]
    parameters = setup["parameters"]
    assembled = setup["assembled"]
    context = head.assemble_context(parameters, assembled)
    eager = head.named_outputs(parameters, context, assembled)
    compiled = jax.jit(lambda values, encoded, inputs: head.named_outputs(values, encoded, inputs))(
        parameters, context, assembled
    )
    jax.block_until_ready(compiled["daily.fast_gpp"])
    assert set(eager) == set(setup["targets"])
    assert sum(np.asarray(value).size for value in eager.values()) == 39_371
    assert all(np.asarray(value).dtype == np.dtype(np.float64) for value in eager.values())
    assert all(np.all(np.isfinite(np.asarray(value))) for value in eager.values())
    assert all(np.all(np.isfinite(np.asarray(value))) for value in compiled.values())

    transition = bind_daily_operator(
        input_assembler=setup["adapter"].input_assembler,
        process_head=head,
        retained_tail=plumbing_test_retained_tail_adapter(),
    )
    result = transition(parameters, setup["example"].inputs)
    validate_conservative_result(result.fast_update)
    assert result.next_state.other.shape == (3_854,)
    assert abs(float(result.fast_update.water.budget_residual)) <= 2.0e-12
    assert abs(float(result.fast_update.carbon.budget_residual)) <= 2.0e-12


def test_strong_baseline_loss_is_equal_experts_then_equal_labels(strong_baseline_setup):
    setup = strong_baseline_setup
    head = setup["head"]
    parameters = setup["parameters"]
    assembled = setup["assembled"]
    outputs = head.named_outputs(
        parameters,
        head.assemble_context(parameters, assembled),
        assembled,
    )
    loss = masked_equal_expert_label_macro_mse(
        outputs,
        setup["targets"],
        inputs=assembled,
        schema=setup["schema"],
        scales=setup["scales"],
        contract=setup["architecture_contract"],
    )
    active = tuple(
        expert for expert in setup["architecture_contract"].expert_names if float(loss.expert_label_count[expert]) > 0.0
    )
    expected = sum(float(loss.expert_mse[expert]) for expert in active) / len(active)
    assert len(active) == 7
    assert np.isfinite(float(loss.total))
    assert float(loss.total) == pytest.approx(expected)


def test_strong_baseline_forcing_families_do_not_leak_before_expert_mask(
    strong_baseline_setup,
):
    setup = strong_baseline_setup
    parameters = setup["parameters"]
    layout = setup["layout"]
    assembled = setup["assembled"]
    forcing = np.asarray(assembled.forcing.values).copy()
    for name in ("Rainf", "Snowf"):
        forcing[:, setup["head"].forcing_indices[name]] += 1.0e3
    perturbed = assembled._replace(
        forcing=assembled.forcing._replace(values=forcing),
    )
    baseline_context = assemble_strong_baseline_context(
        parameters,
        assembled,
        normalization=setup["input_normalization"],
        layout=layout,
    )
    perturbed_context = assemble_strong_baseline_context(
        parameters,
        perturbed,
        normalization=setup["input_normalization"],
        layout=layout,
    )
    baseline = encode_strong_baseline_experts(
        parameters,
        baseline_context,
        layout=layout,
        message_gate_scale=0.0,
    )
    observed = encode_strong_baseline_experts(
        parameters,
        perturbed_context,
        layout=layout,
        message_gate_scale=0.0,
    )
    canopy = layout.expert_names.index("canopy_gpp")
    hydrology = layout.expert_names.index("hydrology_snow")
    np.testing.assert_allclose(np.asarray(observed[canopy]), np.asarray(baseline[canopy]))
    assert not np.allclose(np.asarray(observed[hydrology]), np.asarray(baseline[hydrology]))


def test_strong_baseline_zero_message_scale_bypasses_all_graph_parameters(
    strong_baseline_setup,
):
    setup = strong_baseline_setup
    parameters = setup["parameters"]
    context = assemble_strong_baseline_context(
        parameters,
        setup["assembled"],
        normalization=setup["input_normalization"],
        layout=setup["layout"],
    )
    changed = dict(parameters)
    for name in (
        "graph_sender",
        "graph_gate_logits",
        "graph_norm_scale",
        "graph_norm_bias",
    ):
        changed[name] = jax.tree_util.tree_map(lambda value: value + 100.0, parameters[name])

    baseline = encode_strong_baseline_experts(
        parameters,
        context,
        layout=setup["layout"],
        message_gate_scale=0.0,
    )
    observed = encode_strong_baseline_experts(
        changed,
        context,
        layout=setup["layout"],
        message_gate_scale=0.0,
    )
    np.testing.assert_array_equal(np.asarray(observed), np.asarray(baseline))


def test_strong_baseline_inactive_pft_inputs_are_isolated(strong_baseline_setup):
    setup = strong_baseline_setup
    head = setup["head"]
    parameters = setup["parameters"]
    layout = setup["layout"]
    assembled = setup["assembled"]
    active = np.asarray([True, False], dtype=np.bool_)
    fraction = np.asarray(assembled.pft.pft_fraction).copy()
    fraction[1] = 0.0
    inactive_pft = assembled.pft._replace(
        pft_fraction=fraction,
        active_pft_mask=active,
    )
    inactive = assembled._replace(
        pft=inactive_pft,
        canonical_input=assembled.canonical_input._replace(pft=inactive_pft),
    )

    state = np.asarray(inactive.state.other).copy()
    state[np.isin(layout.state_element_pft_index, (1, -2))] = 1.0e8
    static = np.asarray(inactive.static_conditions).copy()
    static[np.isin(layout.static_element_pft_index, (1, -2))] = 1.0e8
    process_values = []
    offset = 0
    for name in inactive.process_static._fields:
        value = np.asarray(getattr(inactive.process_static, name)).copy()
        width = value.size
        local = layout.process_static_element_pft_index[offset : offset + width]
        value.reshape(-1)[np.isin(local, (1, -2))] = 1.0e8
        process_values.append(value)
        offset += width
    discrete = {}
    offset = 0
    for name in layout.discrete_state_names:
        value = np.asarray(inactive.discrete_state[name]).copy()
        width = value.size
        local = layout.discrete_state_element_pft_index[offset : offset + width]
        value.reshape(-1)[np.isin(local, (1, -2))] = 1
        discrete[name] = value
        offset += width
    pft_state = np.asarray(inactive.pft.pft_state).copy()
    pft_traits = np.asarray(inactive.pft.pft_traits).copy()
    pft_parameters = np.asarray(inactive.pft.pft_parameters).copy()
    pft_state[1] = 1.0e8
    pft_traits[1] = 1.0e8
    pft_parameters[1] = 1.0e8
    perturbed_pft = inactive.pft._replace(
        pft_state=pft_state,
        pft_traits=pft_traits,
        pft_parameters=pft_parameters,
    )
    perturbed = inactive._replace(
        state=inactive.state._replace(other=state),
        discrete_state=discrete,
        static_conditions=static,
        process_static=inactive.process_static.__class__(*process_values),
        pft=perturbed_pft,
        canonical_input=inactive.canonical_input._replace(
            continuous_state=state,
            discrete_state=discrete,
            pft=perturbed_pft,
        ),
    )
    baseline = head.named_outputs(
        parameters,
        head.assemble_context(parameters, inactive),
        inactive,
    )
    observed = head.named_outputs(
        parameters,
        head.assemble_context(parameters, perturbed),
        perturbed,
    )
    for name in baseline:
        np.testing.assert_allclose(
            np.asarray(observed[name]),
            np.asarray(baseline[name]),
            rtol=1.0e-6,
            atol=1.0e-7,
            err_msg=name,
        )


def test_strong_baseline_undefined_continuous_values_are_isolated(
    strong_baseline_setup,
):
    setup = strong_baseline_setup
    head = setup["head"]
    parameters = setup["parameters"]
    assembled = setup["assembled"]
    undefined = np.zeros_like(np.asarray(assembled.state_defined.other), dtype=np.bool_)
    undefined[0] = True
    defined = np.asarray(assembled.state_defined.other).copy()
    defined[undefined] = False
    baseline_state = np.asarray(assembled.state.other).copy()
    baseline_state[undefined] = 1.0e20
    baseline_inputs = assembled._replace(
        state=assembled.state._replace(other=baseline_state),
        state_defined=assembled.state_defined._replace(other=defined),
        canonical_input=assembled.canonical_input._replace(continuous_state=baseline_state),
    )
    perturbed_state = baseline_state.copy()
    perturbed_state[undefined] = 1.0e30
    perturbed = baseline_inputs._replace(
        state=baseline_inputs.state._replace(other=perturbed_state),
        canonical_input=baseline_inputs.canonical_input._replace(continuous_state=perturbed_state),
    )
    baseline = head.named_outputs(
        parameters,
        head.assemble_context(parameters, baseline_inputs),
        baseline_inputs,
    )
    observed = head.named_outputs(
        parameters,
        head.assemble_context(parameters, perturbed),
        perturbed,
    )
    for name in baseline:
        np.testing.assert_allclose(
            np.asarray(observed[name]),
            np.asarray(baseline[name]),
            rtol=1.0e-6,
            atol=1.0e-7,
            err_msg=name,
        )


def test_strong_baseline_representative_loss_jvp_and_vjp_are_finite(strong_baseline_setup):
    setup = strong_baseline_setup
    head = setup["head"]
    parameters = setup["parameters"]
    assembled = setup["assembled"]
    context = head.assemble_context(parameters, assembled)

    def loss_for_query_kernel(kernel):
        query = dict(
            parameters["expert_blocks"][0][0]["attention"]["query"],
            kernel=kernel,
        )
        attention = dict(
            parameters["expert_blocks"][0][0]["attention"],
            query=query,
        )
        block = dict(parameters["expert_blocks"][0][0], attention=attention)
        expert = (block,) + parameters["expert_blocks"][0][1:]
        changed = dict(
            parameters,
            expert_blocks=(expert,) + parameters["expert_blocks"][1:],
        )
        outputs = head.named_outputs(changed, context, assembled)
        return masked_equal_expert_label_macro_mse(
            outputs,
            setup["targets"],
            inputs=assembled,
            schema=setup["schema"],
            scales=setup["scales"],
            contract=setup["architecture_contract"],
        ).total

    kernel = parameters["expert_blocks"][0][0]["attention"]["query"]["kernel"]
    loss, gradient = jax.value_and_grad(loss_for_query_kernel)(kernel)
    direction = jnp.ones_like(kernel) / jnp.sqrt(kernel.size)
    jvp_loss, directional_derivative = jax.jvp(
        loss_for_query_kernel,
        (kernel,),
        (direction,),
    )
    assert np.isfinite(float(loss))
    assert np.isfinite(float(jvp_loss))
    assert np.isfinite(float(directional_derivative))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0
