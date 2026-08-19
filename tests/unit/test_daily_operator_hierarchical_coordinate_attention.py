from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import (
    BRANCH_NAMES,
    MechanisticHierarchicalCoordinateAttentionProcessHead,
    bind_daily_operator,
    hierarchical_coordinate_attention_parameter_count,
    initialize_hierarchical_coordinate_attention_parameters,
    litter_coordinate_branch,
    masked_mechanistic_output_loss,
    plumbing_test_retained_tail_adapter,
    validate_conservative_result,
)
from research.daily_coarse_graining.daily_operator.output_transform import (
    normalized_output_to_physical,
)
from research.daily_coarse_graining.thermal_coefficient_exact_reconstruction_panel import (
    _paper_vertical_grid,
)
from tests.unit.test_daily_operator_mechanistic_output import (
    _set_static,
)
from tests.unit.test_daily_operator_mechanistic_output import (
    mechanistic_setup as _mechanistic_setup_fixture,
)


@pytest.fixture(scope="module")
def hierarchical_setup():
    setup = _mechanistic_setup_fixture.__wrapped__()
    parameters = initialize_hierarchical_coordinate_attention_parameters(
        jax.random.PRNGKey(20260819),
        schema=setup["learned_schema"],
        layout=setup["layout"],
        inputs=setup["assembled"],
        contract=setup["architecture_contract"],
    )
    baseline = setup["head"]
    head = MechanisticHierarchicalCoordinateAttentionProcessHead(
        contract=baseline.contract,
        architecture_contract=baseline.architecture_contract,
        mechanism_contract=baseline.mechanism_contract,
        input_assembler=baseline.input_assembler,
        complete_schema=baseline.complete_schema,
        learned_schema=baseline.learned_schema,
        layout=baseline.layout,
        input_normalization=baseline.input_normalization,
        output_normalization=baseline.output_normalization,
        forcing_layout=baseline.forcing_layout,
        forcing_indices=baseline.forcing_indices,
        message_gate_scale=0.0,
    )
    return setup | {"candidate_parameters": parameters, "candidate_head": head}


def _litter_field_indices(setup):
    return tuple(
        index
        for index, ownership in enumerate(setup["layout"].ledger)
        if ownership.expert == "litter_poc_doc"
    )


def _host_l2(value):
    array = np.asarray(value, dtype=np.float64)
    return float(np.sqrt(np.sum(array * array, dtype=np.float64)))


def _swap_compact_elements(value, element_pft_index):
    source = np.asarray(value)
    result = source.copy()
    first = np.asarray(element_pft_index) == 0
    second = np.asarray(element_pft_index) == 1
    assert np.count_nonzero(first) == np.count_nonzero(second)
    result.reshape(-1)[first] = source.reshape(-1)[second]
    result.reshape(-1)[second] = source.reshape(-1)[first]
    return result


def _permuted_candidate_inputs(setup):
    assembled = setup["assembled"]
    layout = setup["layout"]
    permutation = np.asarray([1, 0])
    pft = assembled.pft._replace(
        **{
            name: np.asarray(getattr(assembled.pft, name))[permutation]
            for name in assembled.pft._fields
        }
    )
    state_other = _swap_compact_elements(
        assembled.state.other,
        layout.state_element_pft_index,
    )
    state_defined_other = _swap_compact_elements(
        assembled.state_defined.other,
        layout.state_element_pft_index,
    )
    static = _swap_compact_elements(
        assembled.static_conditions,
        layout.static_element_pft_index,
    )
    discrete = {}
    offset = 0
    for name in layout.discrete_state_names:
        value = np.asarray(assembled.discrete_state[name])
        width = value.size
        indices = layout.discrete_state_element_pft_index[offset : offset + width]
        discrete[name] = _swap_compact_elements(value, indices)
        offset += width
    process_values = []
    offset = 0
    for name in assembled.process_static._fields:
        value = np.asarray(getattr(assembled.process_static, name))
        width = value.size
        indices = layout.process_static_element_pft_index[offset : offset + width]
        process_values.append(_swap_compact_elements(value, indices))
        offset += width
    process_static = assembled.process_static.__class__(*process_values)
    day_start = assembled.day_start_process_state._replace(
        **{
            name: np.flip(
                np.asarray(getattr(assembled.day_start_process_state, name)),
                axis=1,
            )
            for name in assembled.day_start_process_state._fields
        }
    )
    registry = assembled.canonical_input.static_registry._replace(
        context_features=static,
        rprof=process_static.rprof,
    )
    canonical = assembled.canonical_input._replace(
        continuous_state=state_other,
        discrete_state=discrete,
        pft=pft,
        static_registry=registry,
    )
    return assembled._replace(
        state=assembled.state._replace(other=state_other),
        state_defined=assembled.state_defined._replace(other=state_defined_other),
        discrete_state=discrete,
        pft=pft,
        day_start_process_state=day_start,
        process_static=process_static,
        static_conditions=static,
        canonical_input=canonical,
    )


def _complete_prediction_for_loss(setup, parameters, context):
    head = setup["candidate_head"]
    prediction = head(parameters, context, setup["assembled"])
    complete = dict(prediction.named_outputs)
    complete.pop("__mechanistic.doc_residual_coordinate")
    fields = {field.name: field for field in setup["schema"].fields}
    for name in setup["mechanism_contract"].exact_fields:
        complete.setdefault(name, jnp.zeros(fields[name].shape, dtype=jnp.float64))
    return prediction, complete


def _production_valid_inputs(setup):
    head = setup["candidate_head"]
    assembler = setup["adapter"].input_assembler
    inputs = setup["assembled"]
    grid = _paper_vertical_grid()
    statics = {
        "measurement_heights": [2.0, 10.0],
        "hydrol_zz_mm": np.asarray(grid.znh) * 1000.0,
        "hydrol_dz_mm": np.asarray(grid.dnh) * 1000.0,
        "hydrol_cwrr_ks": np.ones((12,)),
        "soiltile": np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        "njsc": [6],
        "pref_soil_veg": np.asarray([1] * 13 + [6]),
        "ext_coeff_vegetfrac": np.asarray([0.0] + [0.5] * 13),
        "sechiba_qsint": 0.1,
    }
    for name, value in statics.items():
        inputs = _set_static(inputs, assembler, name, value)
    forcing_values = jnp.asarray(inputs.forcing.values)
    forcing_values = forcing_values.at[:, head.forcing_indices["Tair"]].set(280.0)
    forcing_values = forcing_values.at[:, head.forcing_indices["PSurf"]].set(100000.0)
    forcing_values = forcing_values.at[:, head.forcing_indices["Wind_N"]].set(2.0)
    forcing_values = forcing_values.at[:, head.forcing_indices["Wind_E"]].set(1.0)
    forcing = inputs.forcing._replace(values=forcing_values)
    return inputs._replace(
        forcing=forcing,
        canonical_input=inputs.canonical_input._replace(forcing=forcing),
    )


def test_hierarchical_candidate_capacity_and_complete_field_routing(
    hierarchical_setup,
):
    setup = hierarchical_setup
    parameters = setup["candidate_parameters"]
    indices = _litter_field_indices(setup)
    assert len(indices) == 42
    assert hierarchical_coordinate_attention_parameter_count(parameters) == 58_736_698
    assert hierarchical_coordinate_attention_parameter_count(parameters) <= 60_000_000
    assert {leaf.dtype for leaf in jax.tree_util.tree_leaves(parameters)} == {
        jnp.dtype("float32")
    }

    routed = {
        name: tuple(
            setup["learned_schema"].fields[index].name
            for index in indices
            if litter_coordinate_branch(setup["learned_schema"].fields[index].name)
            == name
        )
        for name in BRANCH_NAMES
    }
    assert {name: len(fields) for name, fields in routed.items()} == {
        "litter_composition": 12,
        "doc_reaction": 5,
        "doc_transport": 15,
        "inventory_transition": 10,
    }
    assert all(parameters["hierarchical_litter"]["field_heads"][index] for index in indices)
    litter_expert = setup["layout"].expert_names.index("litter_poc_doc")
    assert parameters["expert_blocks"][litter_expert] == ()
    litter_labels = {
        setup["learned_schema"].fields[index].label_index for index in indices
    }
    assert all(not parameters["label_heads"][index] for index in litter_labels)


def test_hierarchical_candidate_constrained_origins_have_usable_gradients(
    hierarchical_setup,
):
    setup = hierarchical_setup
    field = next(
        field
        for field in setup["learned_schema"].fields
        if field.name == "inventory.carbon.litter_above.incoming_amount"
    )
    raw = jnp.zeros((int(np.prod(field.shape)),), dtype=jnp.float32)
    active = jnp.ones_like(raw, dtype=jnp.bool_)

    def total(value):
        return jnp.sum(
            normalized_output_to_physical(
                value,
                active,
                field=field,
                normalization=setup["output_normalization"],
                inputs=setup["assembled"],
                minimum_nonnegative_ratio=0.05,
            )
        )

    gradient = np.asarray(jax.grad(total)(raw), dtype=np.float64)
    assert np.all(gradient > 0.04)

    bounded_field = next(
        field
        for field in setup["learned_schema"].fields
        if field.name == "endpoint.ok_leak_aux.lignin_struc_below"
    )
    bounded_raw = jnp.zeros(
        (int(np.prod(bounded_field.shape)),),
        dtype=jnp.float32,
    )
    bounded_active = jnp.ones_like(bounded_raw, dtype=jnp.bool_)

    def bounded_total(value):
        return jnp.sum(
            normalized_output_to_physical(
                value,
                bounded_active,
                field=bounded_field,
                normalization=setup["output_normalization"],
                inputs=setup["assembled"],
                minimum_unit_interval_center=0.05,
                rescale_unit_interval_delta=False,
            )
        )

    bounded_gradient = np.asarray(
        jax.grad(bounded_total)(bounded_raw),
        dtype=np.float64,
    )
    assert np.all(bounded_gradient > 0.04)


def test_hierarchical_candidate_eager_jit_and_bound_day_are_finite(
    hierarchical_setup,
):
    setup = hierarchical_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    inputs = _production_valid_inputs(setup)
    context = head.assemble_context(parameters, inputs)
    eager = head.learned_outputs(parameters, context, inputs)
    compiled = jax.jit(
        lambda values, encoded, assembled: head.learned_outputs(
            values,
            encoded,
            assembled,
        )
    )(parameters, context, inputs)
    jax.block_until_ready(compiled["typed.ok_leak.doc_advective_interface"])
    assert set(eager) == {field.name for field in setup["learned_schema"].fields}
    assert sum(np.asarray(value).size for value in eager.values()) == 39_006
    assert all(np.all(np.isfinite(np.asarray(value))) for value in eager.values())
    assert all(np.all(np.isfinite(np.asarray(value))) for value in compiled.values())

    transition = bind_daily_operator(
        input_assembler=setup["adapter"].input_assembler,
        process_head=head,
        retained_tail=plumbing_test_retained_tail_adapter(),
    )
    result = transition(parameters, inputs.canonical_input)
    validate_conservative_result(result.fast_update)
    assert result.next_state.other.shape == (3_854,)
    assert abs(float(result.fast_update.water.budget_residual)) <= 2.0e-12
    assert abs(float(result.fast_update.carbon.budget_residual)) <= 2.0e-12


def test_hierarchical_candidate_official_loss_reaches_every_active_litter_head(
    hierarchical_setup,
):
    setup = hierarchical_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    context = head.assemble_context(parameters, setup["assembled"])

    def isolated_loss(candidate_tree):
        changed = dict(parameters, hierarchical_litter=candidate_tree)
        prediction, complete = _complete_prediction_for_loss(
            setup,
            changed,
            context,
        )
        result = masked_mechanistic_output_loss(
            complete,
            setup["targets"],
            inputs=setup["assembled"],
            complete_schema=setup["schema"],
            learned_schema=setup["learned_schema"],
            scales=setup["scales"],
            architecture_contract=setup["architecture_contract"],
            mechanism_contract=setup["mechanism_contract"],
            auxiliary_outputs=prediction.auxiliary_outputs,
        )
        return result.expert_mse["litter_poc_doc"], result

    (loss, details), gradients = jax.value_and_grad(
        isolated_loss,
        has_aux=True,
    )(parameters["hierarchical_litter"])
    assert np.isfinite(float(loss))
    assert all(
        np.all(np.isfinite(np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(gradients)
    )
    for branch_index in range(len(BRANCH_NAMES)):
        for block in gradients["branch_blocks"][branch_index]:
            assert (
                _host_l2(block["attention"]["query"]["kernel"])
                > 0.0
            )
        assert (
            _host_l2(gradients["branch_decoders"][branch_index]["kernel"])
            > 0.0
        )

    active_head_count = 0
    for field_index in _litter_field_indices(setup):
        name = setup["learned_schema"].fields[field_index].name
        if float(details.field_count[name]) <= 0.0:
            continue
        active_head_count += 1
        assert (
            _host_l2(gradients["field_heads"][field_index]["kernel"])
            > 0.0
        ), name
    assert active_head_count >= 30


def test_hierarchical_candidate_does_not_embed_physical_parameter_channels(
    hierarchical_setup,
):
    setup = hierarchical_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    inputs = setup["assembled"]
    changed_pft = inputs.pft._replace(
        pft_parameters=inputs.pft.pft_parameters
        + jnp.arange(inputs.pft.pft_parameters.size, dtype=jnp.float64).reshape(
            inputs.pft.pft_parameters.shape
        )
    )
    changed = inputs._replace(
        pft=changed_pft,
        canonical_input=inputs.canonical_input._replace(pft=changed_pft),
    )
    baseline_context = head.assemble_context(parameters, inputs)
    changed_context = head.assemble_context(parameters, changed)
    np.testing.assert_array_equal(
        np.asarray(changed_context.pft_tokens),
        np.asarray(baseline_context.pft_tokens),
    )


def test_hierarchical_candidate_inactive_pft_does_not_enter_litter_context(
    hierarchical_setup,
):
    setup = hierarchical_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    inputs = setup["assembled"]
    fraction = np.asarray(inputs.pft.pft_fraction).copy()
    fraction[1] = 0.0
    active = np.asarray([True, False], dtype=np.bool_)
    inactive_pft = inputs.pft._replace(
        pft_fraction=fraction,
        active_pft_mask=active,
    )
    inactive = inputs._replace(
        pft=inactive_pft,
        canonical_input=inputs.canonical_input._replace(pft=inactive_pft),
    )
    pft_state = np.asarray(inactive.pft.pft_state).copy()
    pft_parameters = np.asarray(inactive.pft.pft_parameters).copy()
    pft_traits = np.asarray(inactive.pft.pft_traits).copy()
    pft_state[1] = 1.0e8
    pft_parameters[1] = -1.0e8
    pft_traits[1] = 1.0e8
    perturbed_pft = inactive.pft._replace(
        pft_state=pft_state,
        pft_parameters=pft_parameters,
        pft_traits=pft_traits,
    )
    perturbed = inactive._replace(
        pft=perturbed_pft,
        canonical_input=inactive.canonical_input._replace(pft=perturbed_pft),
    )
    expected = head.learned_outputs(
        parameters,
        head.assemble_context(parameters, inactive),
        inactive,
    )
    observed = head.learned_outputs(
        parameters,
        head.assemble_context(parameters, perturbed),
        perturbed,
    )
    for field_index in _litter_field_indices(setup):
        field = setup["learned_schema"].fields[field_index]
        left = np.asarray(expected[field.name])
        right = np.asarray(observed[field.name])
        if field.pft_axis is None:
            np.testing.assert_array_equal(right, left, err_msg=field.name)
            continue
        active_index = (
            setup["layout"].active_pft_indices[0]
            if field.full_pft_axis
            else 0
        )
        selected = [slice(None)] * left.ndim
        selected[field.pft_axis] = active_index
        np.testing.assert_array_equal(
            right[tuple(selected)],
            left[tuple(selected)],
            err_msg=field.name,
        )


def test_hierarchical_candidate_is_pft_permutation_equivariant(
    hierarchical_setup,
):
    setup = hierarchical_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    assembled = setup["assembled"]
    original = head.learned_outputs(
        parameters,
        head.assemble_context(parameters, assembled),
        assembled,
    )
    permuted = _permuted_candidate_inputs(setup)
    observed = head.learned_outputs(
        parameters,
        head.assemble_context(parameters, permuted),
        permuted,
    )
    for field_index in _litter_field_indices(setup):
        field = setup["learned_schema"].fields[field_index]
        expected = np.asarray(original[field.name]).copy()
        if field.pft_axis is not None:
            if field.full_pft_axis:
                first = [slice(None)] * expected.ndim
                second = [slice(None)] * expected.ndim
                first[field.pft_axis] = setup["layout"].active_pft_indices[0]
                second[field.pft_axis] = setup["layout"].active_pft_indices[1]
                first_value = expected[tuple(first)].copy()
                expected[tuple(first)] = expected[tuple(second)]
                expected[tuple(second)] = first_value
            else:
                expected = np.flip(expected, axis=field.pft_axis)
        np.testing.assert_allclose(
            np.asarray(observed[field.name]),
            expected,
            rtol=1.0e-12,
            atol=1.0e-10,
            err_msg=field.name,
        )
