from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import (
    DOC_INCOMING_FIELD,
    DOC_OUTGOING_FIELD,
    DOC_STATUS_LOGITS_AUX,
    DOC_STATUS_PROBABILITIES_AUX,
    MechanisticDocStatusBranchMagnitudeProcessHead,
    NeuralOutputNormalization,
    bind_daily_operator,
    doc_status_branch_magnitude_parameter_count,
    initialize_doc_status_branch_magnitude_parameters,
    materialize_doc_status_branch_magnitudes,
    plumbing_test_retained_tail_adapter,
    validate_conservative_result,
)
from tests.unit.test_daily_operator_coordinate_coupled_doc_attention import (
    _field_index,
    _inactive_candidate_inputs,
)
from tests.unit.test_daily_operator_hierarchical_coordinate_attention import (
    _litter_field_indices,
    _permuted_candidate_inputs,
    _production_valid_inputs,
)
from tests.unit.test_daily_operator_mechanistic_output import (
    mechanistic_setup as _mechanistic_setup_fixture,
)


@pytest.fixture(scope="module")
def doc_status_setup():
    setup = _mechanistic_setup_fixture.__wrapped__()
    parameters = initialize_doc_status_branch_magnitude_parameters(
        jax.random.PRNGKey(20260820),
        schema=setup["learned_schema"],
        layout=setup["layout"],
        inputs=setup["assembled"],
        contract=setup["architecture_contract"],
    )
    baseline = setup["head"]
    head = MechanisticDocStatusBranchMagnitudeProcessHead(
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


def test_doc_status_candidate_capacity_and_head_shapes(doc_status_setup):
    parameters = doc_status_setup["candidate_parameters"]
    assert doc_status_branch_magnitude_parameter_count(parameters) == 59_973_309
    assert {leaf.dtype for leaf in jax.tree_util.tree_leaves(parameters)} == {
        jnp.dtype("float32")
    }
    coupling = parameters["hierarchical_litter"]["doc_coupling"]
    assert "transition_head" not in coupling
    assert coupling["status_head"]["kernel"].shape == (512, 3)
    assert coupling["incoming_magnitude_head"]["kernel"].shape == (512, 1)
    assert coupling["outgoing_magnitude_head"]["kernel"].shape == (512, 1)
    np.testing.assert_allclose(
        jax.nn.softmax(coupling["status_head"]["bias"]),
        np.asarray([0.07485037284144427, 0.8762663118131869, 0.04888331534536892]),
        rtol=1.0e-6,
        atol=0.0,
    )


def test_doc_status_materializer_is_differentiable_bounded_and_simplex(
    doc_status_setup,
):
    setup = doc_status_setup
    incoming_source = setup["learned_schema"].fields[
        _field_index(setup, DOC_INCOMING_FIELD)
    ]
    outgoing_source = setup["learned_schema"].fields[
        _field_index(setup, DOC_OUTGOING_FIELD)
    ]
    incoming = replace(incoming_source, shape=(3,), pft_axis=None)
    outgoing = replace(outgoing_source, shape=(3,), pft_axis=None)
    normalization = NeuralOutputNormalization(
        mean={incoming.name: np.full(3, 0.2), outgoing.name: np.full(3, 0.1)},
        scale={incoming.name: np.ones(3), outgoing.name: np.full(3, 0.1)},
    )
    logits = jnp.asarray(
        [[8.0, -8.0, -8.0], [-8.0, -8.0, 8.0], [-8.0, 8.0, -8.0]],
        dtype=jnp.float64,
    )
    active = jnp.ones((3,), dtype=jnp.bool_)

    def decode(incoming_raw, outgoing_raw):
        return materialize_doc_status_branch_magnitudes(
            logits,
            incoming_raw,
            outgoing_raw,
            active,
            incoming_field=incoming,
            outgoing_field=outgoing,
            normalization=normalization,
        )

    decoded = decode(jnp.zeros(3), jnp.zeros(3))
    probabilities = np.asarray(decoded.status_probabilities)
    np.testing.assert_allclose(np.sum(probabilities, axis=-1), 1.0)
    assert int(np.argmax(probabilities[0])) == 0
    assert int(np.argmax(probabilities[1])) == 2
    assert int(np.argmax(probabilities[2])) == 1
    assert np.all(np.asarray(decoded.incoming) >= 0.0)
    assert np.all(
        (np.asarray(decoded.outgoing) >= 0.0)
        & (np.asarray(decoded.outgoing) <= 1.0)
    )
    incoming_gradient = jax.grad(
        lambda raw: jnp.sum(decode(raw, jnp.zeros(3)).incoming)
    )(jnp.zeros(3))
    outgoing_gradient = jax.grad(
        lambda raw: jnp.sum(decode(jnp.zeros(3), raw).outgoing)
    )(jnp.zeros(3))
    assert np.any(np.asarray(incoming_gradient) > 0.0)
    assert np.any(np.asarray(outgoing_gradient) > 0.0)


def test_doc_status_candidate_eager_jit_and_bound_day(doc_status_setup):
    setup = doc_status_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    inputs = _production_valid_inputs(setup)
    context = head.assemble_context(parameters, inputs)
    eager, auxiliary = head.learned_outputs_with_auxiliary(
        parameters,
        context,
        inputs,
    )
    compiled = jax.jit(
        lambda values, encoded, assembled: head.learned_outputs_with_auxiliary(
            values,
            encoded,
            assembled,
        )
    )(parameters, context, inputs)
    jax.block_until_ready(compiled[0][DOC_INCOMING_FIELD])
    assert set(eager) == {field.name for field in setup["learned_schema"].fields}
    assert sum(np.asarray(value).size for value in eager.values()) == 39_006
    assert all(np.all(np.isfinite(np.asarray(value))) for value in eager.values())
    probabilities = np.asarray(auxiliary[DOC_STATUS_PROBABILITIES_AUX])
    np.testing.assert_allclose(
        np.sum(probabilities, axis=-1),
        1.0,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert auxiliary[DOC_STATUS_LOGITS_AUX].shape == (*eager[DOC_INCOMING_FIELD].shape, 3)
    assert np.all(np.asarray(eager[DOC_INCOMING_FIELD]) >= 0.0)
    assert np.all(
        (np.asarray(eager[DOC_OUTGOING_FIELD]) >= 0.0)
        & (np.asarray(eager[DOC_OUTGOING_FIELD]) <= 1.0)
    )

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




def test_doc_status_candidate_preserves_pft_isolation_and_equivariance(
    doc_status_setup,
):
    setup = doc_status_setup
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
