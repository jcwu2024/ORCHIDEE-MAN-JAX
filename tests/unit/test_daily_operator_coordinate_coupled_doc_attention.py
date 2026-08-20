from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import (
    DOC_ALIGNED_SOURCE_FIELDS,
    DOC_INCOMING_FIELD,
    DOC_OUTGOING_FIELD,
    MechanisticCoordinateCoupledDocTransitionProcessHead,
    NeuralOutputNormalization,
    bind_daily_operator,
    build_doc_coordinate_alignment,
    coordinate_coupled_doc_transition_parameter_count,
    initialize_coordinate_coupled_doc_transition_parameters,
    masked_mechanistic_output_loss,
    materialize_joint_doc_inventory_transition,
    plumbing_test_retained_tail_adapter,
    validate_conservative_result,
)
from research.daily_coarse_graining.daily_operator.output_transform import (
    field_coordinates,
)
from tests.unit.test_daily_operator_hierarchical_coordinate_attention import (
    _complete_prediction_for_loss,
    _host_l2,
    _litter_field_indices,
    _permuted_candidate_inputs,
    _production_valid_inputs,
)
from tests.unit.test_daily_operator_mechanistic_output import (
    mechanistic_setup as _mechanistic_setup_fixture,
)


@pytest.fixture(scope="module")
def coupled_doc_setup():
    setup = _mechanistic_setup_fixture.__wrapped__()
    parameters = initialize_coordinate_coupled_doc_transition_parameters(
        jax.random.PRNGKey(20260820),
        schema=setup["learned_schema"],
        layout=setup["layout"],
        inputs=setup["assembled"],
        contract=setup["architecture_contract"],
    )
    baseline = setup["head"]
    head = MechanisticCoordinateCoupledDocTransitionProcessHead(
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


def _field_index(setup, name):
    return next(index for index, field in enumerate(setup["learned_schema"].fields) if field.name == name)


def _inactive_candidate_inputs(setup):
    inputs = setup["assembled"]
    fraction = np.asarray(inputs.pft.pft_fraction).copy()
    fraction[1] = 0.0
    inactive_pft = inputs.pft._replace(
        pft_fraction=fraction,
        active_pft_mask=np.asarray([True, False], dtype=np.bool_),
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
    return inactive, perturbed


def test_coordinate_coupled_doc_contract_matches_executable_identity():
    root = Path(__file__).resolve().parents[2]
    payload = json.loads(
        (root / "manifests" / "coarse_graining" / "coordinate_coupled_doc_transition_v4.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["candidate_name"] == "coordinate_coupled_doc_transition_attention_v4"
    assert tuple(payload["aligned_source_fields"]) == DOC_ALIGNED_SOURCE_FIELDS
    assert payload["joint_transition_fields"] == {
        "incoming": DOC_INCOMING_FIELD,
        "outgoing": DOC_OUTGOING_FIELD,
    }
    assert payload["capacity"]["expected_parameter_count"] == 59_971_257
    assert payload["inference_policy"]["reconstruct_48_forcing_records"] is False
    assert payload["inference_policy"]["scan_48_state_steps"] is False


def test_doc_alignment_matches_every_common_physical_coordinate(coupled_doc_setup):
    setup = coupled_doc_setup
    schema = setup["learned_schema"]
    layout = setup["layout"]
    alignment = build_doc_coordinate_alignment(schema=schema, layout=layout)
    assert tuple(schema.fields[index].name for index in alignment.source_field_indices) == DOC_ALIGNED_SOURCE_FIELDS
    assert len(alignment.source_gather_indices) == 5
    assert all(gather.shape == (896,) for gather in alignment.source_gather_indices)

    target = schema.fields[alignment.incoming_field_index]
    target_axes = layout.ledger[alignment.incoming_field_index].axes
    target_coordinates = field_coordinates(target.shape)
    target_pft = target_axes.index("pft")
    expected_full_pft = np.asarray(layout.active_pft_indices)[target_coordinates[:, target_pft]]
    for source_index, gather in zip(
        alignment.source_field_indices,
        alignment.source_gather_indices,
        strict=True,
    ):
        source = schema.fields[source_index]
        source_axes = layout.ledger[source_index].axes
        selected = field_coordinates(source.shape)[gather]
        np.testing.assert_array_equal(
            selected[:, source_axes.index("pft")],
            expected_full_pft,
        )
        for axis in set(source_axes) - {"pft"}:
            np.testing.assert_array_equal(
                selected[:, source_axes.index(axis)],
                target_coordinates[:, target_axes.index(axis)],
            )


def test_coupled_doc_candidate_capacity_and_joint_head(coupled_doc_setup):
    setup = coupled_doc_setup
    parameters = setup["candidate_parameters"]
    assert coordinate_coupled_doc_transition_parameter_count(parameters) == 59_971_257
    assert {leaf.dtype for leaf in jax.tree_util.tree_leaves(parameters)} == {jnp.dtype("float32")}
    alignment = build_doc_coordinate_alignment(
        schema=setup["learned_schema"],
        layout=setup["layout"],
    )
    heads = parameters["hierarchical_litter"]["field_heads"]
    assert heads[alignment.incoming_field_index] == {}
    assert heads[alignment.outgoing_field_index] == {}
    coupling = parameters["hierarchical_litter"]["doc_coupling"]
    assert len(coupling["source_projection"]) == len(DOC_ALIGNED_SOURCE_FIELDS)
    assert coupling["pair_projection"]["kernel"].shape == (768, 384)
    assert coupling["transition_decoder"]["kernel"].shape == (384, 512)


def test_joint_doc_transition_is_exclusive_bounded_and_differentiable(
    coupled_doc_setup,
):
    setup = coupled_doc_setup
    incoming_source = setup["learned_schema"].fields[_field_index(setup, DOC_INCOMING_FIELD)]
    outgoing_source = setup["learned_schema"].fields[_field_index(setup, DOC_OUTGOING_FIELD)]
    incoming = replace(incoming_source, shape=(3,), pft_axis=None)
    outgoing = replace(outgoing_source, shape=(3,), pft_axis=None)
    normalization = NeuralOutputNormalization(
        mean={incoming.name: np.zeros(3), outgoing.name: np.zeros(3)},
        scale={incoming.name: np.ones(3), outgoing.name: np.full(3, 0.1)},
    )
    active = jnp.ones((3,), dtype=jnp.bool_)
    day_start = jnp.asarray([10.0, 10.0, 0.0], dtype=jnp.float64)
    raw = jnp.asarray([-1.0, 1.0, 0.0], dtype=jnp.float64)
    incoming_value, outgoing_value = materialize_joint_doc_inventory_transition(
        raw,
        active,
        day_start,
        incoming_field=incoming,
        outgoing_field=outgoing,
        normalization=normalization,
    )
    incoming_value = np.asarray(incoming_value)
    outgoing_value = np.asarray(outgoing_value)
    assert incoming_value[0] == 0.0 and outgoing_value[0] > 0.0
    assert incoming_value[1] > 0.0 and outgoing_value[1] == 0.0
    assert outgoing_value[2] == 0.0
    assert np.all((outgoing_value >= 0.0) & (outgoing_value <= 1.0))
    assert not np.any((incoming_value > 0.0) & (outgoing_value > 0.0))
    reconstructed_end = np.asarray(day_start) * (1.0 - outgoing_value) + incoming_value
    assert np.all(reconstructed_end >= 0.0)

    def incoming_total(value):
        return jnp.sum(
            materialize_joint_doc_inventory_transition(
                value,
                active,
                day_start,
                incoming_field=incoming,
                outgoing_field=outgoing,
                normalization=normalization,
            )[0]
        )

    def outgoing_total(value):
        return jnp.sum(
            materialize_joint_doc_inventory_transition(
                value,
                active,
                day_start,
                incoming_field=incoming,
                outgoing_field=outgoing,
                normalization=normalization,
            )[1]
        )

    assert np.any(np.asarray(jax.grad(incoming_total)(jnp.ones(3))) != 0.0)
    assert np.any(np.asarray(jax.grad(outgoing_total)(-jnp.ones(3))) != 0.0)


def test_coupled_doc_candidate_eager_jit_and_bound_day(coupled_doc_setup):
    setup = coupled_doc_setup
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
    jax.block_until_ready(compiled[DOC_INCOMING_FIELD])
    assert set(eager) == {field.name for field in setup["learned_schema"].fields}
    assert sum(np.asarray(value).size for value in eager.values()) == 39_006
    assert all(np.all(np.isfinite(np.asarray(value))) for value in eager.values())
    assert all(np.all(np.isfinite(np.asarray(value))) for value in compiled.values())
    incoming = np.asarray(eager[DOC_INCOMING_FIELD])
    outgoing = np.asarray(eager[DOC_OUTGOING_FIELD])
    assert not np.any((incoming > 0.0) & (outgoing > 0.0))

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


def test_official_loss_reaches_aligned_sources_inventory_attention_and_joint_head(
    coupled_doc_setup,
):
    setup = coupled_doc_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    context = head.assemble_context(parameters, setup["assembled"])
    original = parameters["hierarchical_litter"]
    probe = {
        "doc_coupling": original["doc_coupling"],
        "inventory_blocks": original["branch_blocks"][3],
    }

    def isolated_loss(candidate_probe):
        hierarchical = dict(original, doc_coupling=candidate_probe["doc_coupling"])
        blocks = list(hierarchical["branch_blocks"])
        blocks[3] = candidate_probe["inventory_blocks"]
        hierarchical["branch_blocks"] = tuple(blocks)
        changed = dict(parameters, hierarchical_litter=hierarchical)
        prediction, complete = _complete_prediction_for_loss(setup, changed, context)
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
        return result.expert_mse["litter_poc_doc"]

    loss, gradients = jax.value_and_grad(isolated_loss)(probe)
    assert np.isfinite(float(loss))
    coupling = gradients["doc_coupling"]
    assert all(_host_l2(projection["kernel"]) > 0.0 for projection in coupling["source_projection"])
    assert _host_l2(coupling["scalar_input"]["kernel"]) > 0.0
    assert _host_l2(coupling["pair_projection"]["kernel"]) > 0.0
    assert _host_l2(coupling["transition_decoder"]["kernel"]) > 0.0
    assert _host_l2(coupling["transition_head"]["kernel"]) > 0.0
    assert all(_host_l2(block["attention"]["query"]["kernel"]) > 0.0 for block in gradients["inventory_blocks"])


def test_coupled_doc_candidate_preserves_pft_isolation_and_equivariance(
    coupled_doc_setup,
):
    setup = coupled_doc_setup
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
            rtol=1.0e-7,
            atol=5.0e-10,
            err_msg=field.name,
        )

    inactive, perturbed = _inactive_candidate_inputs(setup)
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
    for name in (DOC_INCOMING_FIELD, DOC_OUTGOING_FIELD):
        field = setup["learned_schema"].fields[_field_index(setup, name)]
        selected = [slice(None)] * len(field.shape)
        selected[field.pft_axis] = 0
        np.testing.assert_array_equal(
            np.asarray(observed[name])[tuple(selected)],
            np.asarray(expected[name])[tuple(selected)],
        )
