from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import (
    AUXILIARY_SUPERVISION,
    DOMAIN_REFINEMENT_DEPTHS,
    PRIMARY_SCIENCE_DIRECT,
    SCIENCE_TOKEN_NAMES,
    STATE_BUDGET_ESSENTIAL,
    MechanisticTargetOrientedAttentionProcessHead,
    bind_daily_operator,
    initialize_target_oriented_attention_parameters,
    load_target_oriented_contract,
    masked_mechanistic_output_loss,
    necessary_process_supervision_loss,
    plumbing_test_retained_tail_adapter,
    target_oriented_attention_parameter_count,
    target_oriented_field_categories,
    validate_conservative_result,
)
from tests.unit.test_daily_operator_hierarchical_coordinate_attention import (
    _complete_prediction_for_loss,
    _host_l2,
    _permuted_candidate_inputs,
    _production_valid_inputs,
)
from tests.unit.test_daily_operator_mechanistic_output import (
    mechanistic_setup as _mechanistic_setup_fixture,
)




@pytest.fixture(scope="module")
def target_attention_setup():
    setup = _mechanistic_setup_fixture.__wrapped__()
    target_contract = load_target_oriented_contract()
    parameters = initialize_target_oriented_attention_parameters(
        jax.random.PRNGKey(20260821),
        schema=setup["learned_schema"],
        layout=setup["layout"],
        inputs=setup["assembled"],
        architecture_contract=setup["architecture_contract"],
        target_contract=target_contract,
    )
    baseline = setup["head"]
    head = MechanisticTargetOrientedAttentionProcessHead(
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
        message_gate_scale=1.0,
        target_contract=target_contract,
    )
    return setup | {
        "candidate_parameters": parameters,
        "candidate_head": head,
        "target_contract": target_contract,
    }


def test_target_attention_capacity_and_category_isolation(target_attention_setup):
    setup = target_attention_setup
    parameters = setup["candidate_parameters"]
    target = parameters["target_oriented"]
    categories = target_oriented_field_categories(
        setup["learned_schema"],
        setup["target_contract"],
    )
    assert {category: categories.count(category) for category in set(categories)} == {
        PRIMARY_SCIENCE_DIRECT: 2,
        STATE_BUDGET_ESSENTIAL: 93,
        AUXILIARY_SUPERVISION: 66,
    }
    count = target_oriented_attention_parameter_count(parameters)
    assert count == 83_755_173
    assert "label_heads" not in parameters
    assert tuple(len(blocks) for blocks in target["domain_blocks"]) == tuple(
        DOMAIN_REFINEMENT_DEPTHS[name] for name in setup["layout"].expert_names
    )
    assert target["science_query"].shape == (len(SCIENCE_TOKEN_NAMES), 384)
    assert all(
        bool(target["primary_heads"][index]) == (category == PRIMARY_SCIENCE_DIRECT)
        for index, category in enumerate(categories)
    )
    assert all(
        bool(target["state_heads"][index]) == (category == STATE_BUDGET_ESSENTIAL)
        for index, category in enumerate(categories)
    )
    auxiliary_labels = {
        field.label_index
        for field, category in zip(
            setup["learned_schema"].fields,
            categories,
            strict=True,
        )
        if category == AUXILIARY_SUPERVISION
    }
    assert all(bool(head) == (index in auxiliary_labels) for index, head in enumerate(target["auxiliary_label_heads"]))
    assert {leaf.dtype for leaf in jax.tree_util.tree_leaves(parameters)} == {jnp.dtype("float32")}


def test_target_attention_eager_jit_and_bound_day_are_finite(
    target_attention_setup,
):
    setup = target_attention_setup
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
    jax.block_until_ready(compiled["daily.fast_gpp"])
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


def test_target_attention_loss_reaches_all_head_families(target_attention_setup):
    setup = target_attention_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    assembled = setup["assembled"]
    pft_parameters = jnp.asarray(assembled.pft.pft_parameters).at[1, 0].set(60.0)
    active_pft = assembled.pft._replace(pft_parameters=pft_parameters)
    active_inputs = assembled._replace(
        pft=active_pft,
        canonical_input=assembled.canonical_input._replace(pft=active_pft),
    )
    active_setup = setup | {"assembled": active_inputs}
    context = head.assemble_context(parameters, active_inputs)

    def target_proxy(candidate_tree):
        changed = dict(parameters, target_oriented=candidate_tree)
        prediction, complete = _complete_prediction_for_loss(
            active_setup,
            changed,
            context,
        )
        process_loss = masked_mechanistic_output_loss(
            complete,
            setup["targets"],
            inputs=active_inputs,
            complete_schema=setup["schema"],
            learned_schema=setup["learned_schema"],
            scales=setup["scales"],
            architecture_contract=setup["architecture_contract"],
            mechanism_contract=setup["mechanism_contract"],
            auxiliary_outputs=prediction.auxiliary_outputs,
        )
        necessary = necessary_process_supervision_loss(
            process_loss,
            setup["target_contract"],
        )
        return (
            process_loss.label_mse["carbon.fast_gpp"]
            + necessary.category_loss[STATE_BUDGET_ESSENTIAL]
            + 0.25 * necessary.category_loss[AUXILIARY_SUPERVISION]
        )

    loss, gradients = jax.value_and_grad(target_proxy)(parameters["target_oriented"])
    assert np.isfinite(float(loss))
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(gradients))
    assert _host_l2(gradients["domain_blocks"][0][0]["attention"]["query"]["kernel"]) > 0.0
    assert _host_l2(gradients["science_cross_blocks"][0]["attention"]["query"]["kernel"]) > 0.0
    assert _host_l2(gradients["science_mixer_blocks"][0]["attention"]["query"]["kernel"]) > 0.0
    primary_index = next(
        index for index, field in enumerate(setup["learned_schema"].fields) if field.name == "daily.fast_gpp"
    )
    state_index = next(
        index
        for index, category in enumerate(
            target_oriented_field_categories(
                setup["learned_schema"],
                setup["target_contract"],
            )
        )
        if category == STATE_BUDGET_ESSENTIAL and _host_l2(gradients["state_heads"][index]["kernel"]) > 0.0
    )
    auxiliary_label = next(
        index
        for index, value in enumerate(gradients["auxiliary_label_heads"])
        if value and _host_l2(value["second"]["kernel"]) > 0.0
    )
    assert _host_l2(gradients["primary_heads"][primary_index]["kernel"]) > 0.0
    assert _host_l2(gradients["state_heads"][state_index]["kernel"]) > 0.0
    assert auxiliary_label >= 0


def test_target_attention_inactive_pft_isolation_and_permutation(
    target_attention_setup,
):
    setup = target_attention_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    inputs = setup["assembled"]
    fraction = np.asarray(inputs.pft.pft_fraction).copy()
    fraction[1] = 0.0
    inactive_pft = inputs.pft._replace(
        pft_fraction=fraction,
        active_pft_mask=np.asarray([True, False]),
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
    for field, ownership in zip(
        setup["learned_schema"].fields,
        setup["layout"].ledger,
        strict=True,
    ):
        left = np.asarray(expected[field.name])
        right = np.asarray(observed[field.name])
        if ownership.pft_axis is None:
            np.testing.assert_array_equal(right, left, err_msg=field.name)
            continue
        active_index = setup["layout"].active_pft_indices[0] if field.full_pft_axis else 0
        selected = [slice(None)] * left.ndim
        selected[ownership.pft_axis] = active_index
        np.testing.assert_array_equal(
            right[tuple(selected)],
            left[tuple(selected)],
            err_msg=field.name,
        )

    original = head.learned_outputs(
        parameters,
        head.assemble_context(parameters, inputs),
        inputs,
    )
    permuted = _permuted_candidate_inputs(setup)
    permuted_outputs = head.learned_outputs(
        parameters,
        head.assemble_context(parameters, permuted),
        permuted,
    )
    for field, ownership in zip(
        setup["learned_schema"].fields,
        setup["layout"].ledger,
        strict=True,
    ):
        expected_permuted = np.asarray(original[field.name]).copy()
        if ownership.pft_axis is None:
            pass
        elif field.full_pft_axis:
            first = [slice(None)] * expected_permuted.ndim
            second = [slice(None)] * expected_permuted.ndim
            first[ownership.pft_axis] = setup["layout"].active_pft_indices[0]
            second[ownership.pft_axis] = setup["layout"].active_pft_indices[1]
            first_value = expected_permuted[tuple(first)].copy()
            expected_permuted[tuple(first)] = expected_permuted[tuple(second)]
            expected_permuted[tuple(second)] = first_value
        else:
            first = [slice(None)] * expected_permuted.ndim
            second = [slice(None)] * expected_permuted.ndim
            first[ownership.pft_axis] = 0
            second[ownership.pft_axis] = 1
            first_value = expected_permuted[tuple(first)].copy()
            expected_permuted[tuple(first)] = expected_permuted[tuple(second)]
            expected_permuted[tuple(second)] = first_value
        np.testing.assert_allclose(
            np.asarray(permuted_outputs[field.name]),
            expected_permuted,
            rtol=1.0e-6,
            atol=2.0e-9,
            err_msg=field.name,
        )
