from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
)
from research.daily_coarse_graining.daily_operator.conservative_update import (
    apply_conservative_update,
    validate_conservative_result,
)
from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
from research.daily_coarse_graining.daily_operator.neural_architecture import (
    build_neural_output_schema,
    unit_input_normalization,
    unit_output_normalization,
)
from research.daily_coarse_graining.daily_operator.process_derivations import (
    NativeForcingFieldLayout,
)
from research.daily_coarse_graining.daily_operator.retained_tail import (
    plumbing_test_retained_tail_adapter,
)
from research.daily_coarse_graining.daily_operator.successor_architecture import (
    SOURCE_ROUTED_ATTENTION,
    SOURCE_ROUTED_GRAPH_CONTROL,
    assemble_successor_process_context,
    build_successor_token_layout,
    initialize_successor_parameters,
    successor_parameter_count,
)
from research.daily_coarse_graining.daily_operator.successor_contract import (
    FROZEN_CONTRACT_SHA256,
    contract_sha256,
    load_successor_candidate_contract,
)
from research.daily_coarse_graining.daily_operator.successor_process_head import (
    SuccessorProcessHead,
)
from research.daily_coarse_graining.daily_operator.successor_training import (
    CUMULATIVE_FLUX_FIELDS,
    LONG_HORIZON_COMPONENTS,
    TERMINAL_STOCK_FIELDS,
    calibrate_gradient_ratio_coefficients,
    extract_biomass_productivity_trajectories,
    rollout_daily_operator,
    rollout_input_from_template,
    successor_training_objective,
)
from research.daily_coarse_graining.daily_operator.training_data import (
    DailyProcessTrainingAdapter,
)
from research.daily_coarse_graining.daily_operator.training_loss import (
    masked_equal_label_macro_mse,
    unit_process_loss_scales,
)
from research.daily_coarse_graining.daily_operator.types import DailyOperatorResult

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_METADATA_PATH = (
    ROOT
    / "outputs"
    / "research"
    / "daily_coarse_graining"
    / "dataset-manifest-v5"
    / "dataset_manifest.json"
)


def _markov_contract():
    # The versioned schema metadata is shared by the accepted v6 release; no
    # historical training row or target is read by these synthetic tests.
    metadata = json.loads(CONTRACT_METADATA_PATH.read_text(encoding="utf-8"))[
        "markov_contract"
    ]
    contract = daily_markov_contract_from_metadata(metadata)
    assert contract.sha256 == (
        "6813065b51e95a2ec6663d8bc92f5336148704aea502632a05fd51d7e2e9d442"
    )
    return contract


def _set_condition(vector, leaves, name, value):
    leaf = next(item for item in leaves if item.name == name)
    vector[leaf.start : leaf.stop] = np.asarray(value, dtype=np.float64).reshape(-1)


def _synthetic_parent(adapter: DailyProcessTrainingAdapter):
    contract = adapter.contract
    state = np.zeros((contract.continuous_state_width,), dtype=np.float64)
    veget = next(
        leaf
        for leaf in contract.state_leaves
        if leaf.key == "slowproc_stomate_previous_step_state.veget_max"
    )
    state[veget.start : veget.stop] = np.asarray([0.5, 0.5], dtype=np.float64)
    discrete = {
        leaf.key: np.zeros(leaf.shape, dtype=np.dtype(leaf.dtype))
        for leaf in contract.discrete_leaves
    }
    parameters = np.zeros(
        (contract.static_conditions.parameter_width,), dtype=np.float64
    )
    _set_condition(
        parameters,
        contract.static_conditions.parameter_leaves,
        "maint_resp_slope",
        np.ones((14, 3), dtype=np.float64),
    )
    static = np.zeros(
        (contract.static_conditions.landpoint_static_width,), dtype=np.float64
    )
    _set_condition(
        static,
        contract.static_conditions.landpoint_static_leaves,
        "hydrol_humcste",
        np.ones((14,), dtype=np.float64),
    )
    _set_condition(
        static,
        contract.static_conditions.landpoint_static_leaves,
        "diaglev",
        np.arange(1.0, 12.0, dtype=np.float64),
    )
    _set_condition(
        static,
        contract.static_conditions.landpoint_static_leaves,
        "clay_frac",
        np.asarray([0.3], dtype=np.float64),
    )
    annual = np.zeros(
        (contract.static_conditions.annual_condition_width,), dtype=np.float64
    )
    parent = {
        "state": state,
        "fast_day_target": np.zeros(
            (contract.fast_day_target_width,), dtype=np.float64
        ),
        "forcing_native": np.zeros(
            (
                contract.native_forcing.source_records_per_day,
                contract.native_forcing.width,
            ),
            dtype=np.float64,
        ),
        "parameters": parameters,
        "landpoint_static": static,
        "annual_conditions": annual,
        "next_state": state.copy(),
        "diagnostics": np.zeros((contract.diagnostic_width,), dtype=np.float64),
        "year": np.asarray(1961, dtype=np.int32),
        "day_index": np.asarray(2, dtype=np.int32),
        "discrete_state": discrete,
        "next_discrete_state": {
            name: value.copy() for name, value in discrete.items()
        },
    }
    values = {
        field.path: np.zeros(field.feature_shape, dtype=np.float64)
        for field in adapter.typed_contract.fields
    }
    defined = {
        name: np.ones(value.shape, dtype=np.bool_) for name, value in values.items()
    }
    return parent, values, defined


@pytest.fixture(scope="module")
def successor_setup():
    adapter = DailyProcessTrainingAdapter.from_contract(_markov_contract())
    parent, values, defined = _synthetic_parent(adapter)
    example = adapter.assemble(
        parent,
        supplemental_values=values,
        supplemental_defined=defined,
    )
    assembled = adapter.input_assembler.assemble(example.inputs)
    schema = build_neural_output_schema(
        example.targets.supervision,
        field_labels=adapter.field_labels(),
        typed_axes={
            field.path: field.axes[1:] for field in adapter.typed_contract.fields
        },
        active_pft_indices=adapter.active_pft_indices,
    )
    contract = load_successor_candidate_contract()
    layout = build_successor_token_layout(adapter.input_assembler, schema, contract)
    input_normalization = unit_input_normalization(assembled)
    output_normalization = unit_output_normalization(schema)
    forcing_fields = adapter.contract.native_forcing.fields
    forcing_layout = NativeForcingFieldLayout(
        forcing_fields.index("Rainf"),
        forcing_fields.index("Rainf") + 1,
        forcing_fields.index("Snowf"),
        forcing_fields.index("Snowf") + 1,
        forcing_fields.index("Tair"),
        forcing_fields.index("Tair") + 1,
    )
    parameters = initialize_successor_parameters(
        jax.random.PRNGKey(20260809),
        schema=schema,
        layout=layout,
        inputs=assembled,
        arm=SOURCE_ROUTED_ATTENTION,
    )
    head = SuccessorProcessHead(
        arm=SOURCE_ROUTED_ATTENTION,
        contract=adapter.contract,
        input_assembler=adapter.input_assembler,
        schema=schema,
        token_layout=layout,
        input_normalization=input_normalization,
        output_normalization=output_normalization,
        forcing_layout=forcing_layout,
        forcing_indices={name: index for index, name in enumerate(forcing_fields)},
    )
    return {
        "adapter": adapter,
        "assembled": assembled,
        "contract": contract,
        "example": example,
        "head": head,
        "input_normalization": input_normalization,
        "layout": layout,
        "parameters": parameters,
        "schema": schema,
    }


def test_successor_contract_is_externally_frozen(tmp_path):
    contract = load_successor_candidate_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.candidate_id == SOURCE_ROUTED_ATTENTION
    assert contract.control_id == SOURCE_ROUTED_GRAPH_CONTROL

    payload = json.loads(contract.source_path.read_text(encoding="utf-8"))
    payload["candidate"]["d_model"] = 512
    payload["contract_sha256"] = contract_sha256(payload)
    tampered = tmp_path / "tampered_successor.json"
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen hash drift"):
        load_successor_candidate_contract(tampered)


def test_successor_layout_parameter_match_and_named_output_shapes(successor_setup):
    setup = successor_setup
    layout = setup["layout"]
    schema = setup["schema"]
    parameters = setup["parameters"]
    contract = setup["contract"].payload
    assembled = setup["assembled"]

    assert len(schema.fields) == 101
    assert len(schema.labels) == 40
    assert int(np.prod(np.asarray(assembled.forcing.values).shape)) == 45
    assert np.all(np.any(layout.field_allowed_family_mask, axis=1))
    assert np.all(np.any(layout.field_graph_mask, axis=1))
    assert layout.state.pool_matrix.shape[1] == assembled.state.other.size
    assert layout.static.pool_matrix.shape[1] == assembled.static_conditions.size
    assert layout.annual.pool_matrix.shape[1] == assembled.annual_conditions.size

    candidate_count = successor_parameter_count(parameters)
    control_parameters = initialize_successor_parameters(
        jax.random.PRNGKey(20260810),
        schema=schema,
        layout=layout,
        inputs=assembled,
        arm=SOURCE_ROUTED_GRAPH_CONTROL,
    )
    control_count = successor_parameter_count(control_parameters)
    assert candidate_count == control_count == 8_679_681
    assert contract["candidate"]["parameter_count_minimum"] <= candidate_count
    assert candidate_count <= contract["candidate"]["parameter_count_maximum"]
    assert jax.tree_util.tree_structure(parameters) == jax.tree_util.tree_structure(
        control_parameters
    )

    context = assemble_successor_process_context(
        parameters,
        assembled,
        normalization=setup["input_normalization"],
        layout=layout,
    )
    assert context.input_tokens.shape == (770, 256)
    assert int(np.asarray(context.input_mask).sum()) == 770
    outputs = setup["head"].named_outputs(parameters, context, assembled)
    assert set(outputs) == set(setup["example"].targets.supervision)
    assert sum(np.asarray(value).size for value in outputs.values()) == 38_530
    assert all(np.all(np.isfinite(np.asarray(value))) for value in outputs.values())
    compiled_outputs = jax.jit(
        lambda values, encoded, inputs: setup["head"].named_outputs(
            values, encoded, inputs
        )
    )(parameters, context, assembled)
    jax.block_until_ready(compiled_outputs["daily.fast_gpp"])
    assert all(
        np.all(np.isfinite(np.asarray(value)))
        for value in compiled_outputs.values()
    )


def test_successor_attention_forward_loss_and_gradient_are_finite(successor_setup):
    setup = successor_setup
    head = setup["head"]
    parameters = setup["parameters"]
    assembled = setup["assembled"]
    context = head.assemble_context(parameters, assembled)
    targets = setup["example"].targets.supervision
    scales = unit_process_loss_scales(targets)
    labels = setup["adapter"].field_labels()

    def loss_for_first_attention_query(kernel):
        query = dict(
            parameters["process_blocks"][0]["input_mixer"]["query"],
            kernel=kernel,
        )
        input_mixer = dict(
            parameters["process_blocks"][0]["input_mixer"],
            query=query,
        )
        first_block = dict(
            parameters["process_blocks"][0],
            input_mixer=input_mixer,
        )
        changed = dict(
            parameters,
            process_blocks=(first_block,) + parameters["process_blocks"][1:],
        )
        predicted = head.named_outputs(changed, context, assembled)
        return masked_equal_label_macro_mse(
            predicted,
            targets,
            scales=scales,
            field_labels=labels,
        ).total

    kernel = parameters["process_blocks"][0]["input_mixer"]["query"]["kernel"]
    loss, gradient = jax.value_and_grad(loss_for_first_attention_query)(kernel)
    direction = jnp.ones_like(kernel) / jnp.sqrt(kernel.size)
    jvp_loss, directional_derivative = jax.jvp(
        loss_for_first_attention_query,
        (kernel,),
        (direction,),
    )
    assert np.isfinite(float(loss))
    assert np.isfinite(float(jvp_loss))
    assert np.isfinite(float(directional_derivative))
    np.testing.assert_allclose(float(jvp_loss), float(loss), rtol=1.0e-12)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_successor_fast_parameter_routes_and_retained_channels_are_isolated(
    successor_setup,
):
    setup = successor_setup
    head = setup["head"]
    parameters = setup["parameters"]
    assembled = setup["assembled"]
    layout = setup["layout"]
    continuous = np.asarray(assembled.canonical_input.continuous_state).copy()
    continuous[layout.t2m_longterm_start] = 283.15
    canonical = assembled.canonical_input._replace(continuous_state=continuous)
    current = assembled._replace(
        canonical_input=canonical,
        state=assembled.state._replace(other=continuous),
    )
    context = head.assemble_context(parameters, current)
    pft_parameters = jnp.asarray(
        [
            [50.0, 0.0, 0.0, -0.001, 0.2, 100.0],
            [60.0, 0.0, 0.0, -0.001, 0.3, 120.0],
        ],
        dtype=jnp.float64,
    )

    def fast_objective(values):
        inputs = current._replace(
            pft=current.pft._replace(pft_parameters=values)
        )
        outputs = head.named_outputs(parameters, context, inputs)
        return jnp.sum(outputs["daily.fast_gpp"]) + jnp.sum(
            outputs["daily.fast_maintenance_respiration"]
        )

    gradient = np.asarray(jax.grad(fast_objective)(pft_parameters))
    assert np.all(np.isfinite(gradient))
    np.testing.assert_array_equal(gradient[0], 0.0)
    assert np.all(gradient[1, :4] != 0.0)
    np.testing.assert_array_equal(gradient[:, 4:], 0.0)


def _swap_compact_elements(value, element_pft_index):
    source = np.asarray(value)
    result = source.copy()
    first = np.asarray(element_pft_index) == 0
    second = np.asarray(element_pft_index) == 1
    assert np.count_nonzero(first) == np.count_nonzero(second)
    result[first] = source[second]
    result[second] = source[first]
    return result


def test_successor_candidate_is_pft_permutation_equivariant(successor_setup):
    setup = successor_setup
    head = setup["head"]
    parameters = setup["parameters"]
    assembled = setup["assembled"]
    layout = setup["layout"]
    original = head.named_outputs(
        parameters,
        head.assemble_context(parameters, assembled),
        assembled,
    )
    permutation = np.asarray([1, 0])
    pft = assembled.pft._replace(
        **{
            name: np.asarray(getattr(assembled.pft, name))[permutation]
            for name in assembled.pft._fields
        }
    )
    state = assembled.state._replace(
        **{
            domain.name: _swap_compact_elements(
                getattr(assembled.state, domain.name), domain.element_pft_index
            )
            for domain in layout.axis_layout.domains
        },
        other=_swap_compact_elements(
            assembled.state.other, layout.state.element_pft_index
        ),
    )
    state_defined = assembled.state_defined._replace(
        **{
            domain.name: _swap_compact_elements(
                getattr(assembled.state_defined, domain.name),
                domain.element_pft_index,
            )
            for domain in layout.axis_layout.domains
        },
        other=_swap_compact_elements(
            assembled.state_defined.other, layout.state.element_pft_index
        ),
    )
    static = _swap_compact_elements(
        assembled.static_conditions, layout.static.element_pft_index
    )
    process_static = assembled.process_static._replace(
        rprof=np.flip(np.asarray(assembled.process_static.rprof), axis=-1)
    )
    permuted = assembled._replace(
        pft=pft,
        state=state,
        state_defined=state_defined,
        static_conditions=static,
        process_static=process_static,
    )
    observed = head.named_outputs(
        parameters,
        head.assemble_context(parameters, permuted),
        permuted,
    )
    for field in setup["schema"].fields:
        expected = np.asarray(original[field.name]).copy()
        if field.pft_axis is not None:
            if field.full_pft_axis:
                first = [slice(None)] * expected.ndim
                second = [slice(None)] * expected.ndim
                first[field.pft_axis] = setup["adapter"].active_pft_indices[0]
                second[field.pft_axis] = setup["adapter"].active_pft_indices[1]
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


def test_successor_candidate_isolates_an_inactive_pft_slot(successor_setup):
    setup = successor_setup
    head = setup["head"]
    parameters = setup["parameters"]
    assembled = setup["assembled"]
    layout = setup["layout"]
    fraction = np.asarray(assembled.pft.pft_fraction).copy()
    fraction[1] = 0.0
    inactive_pft = assembled.pft._replace(
        active_pft_mask=np.asarray([True, False], dtype=np.bool_),
        pft_fraction=fraction,
    )
    inactive = assembled._replace(pft=inactive_pft)

    perturbed_pft_values = {}
    for name in ("pft_state", "pft_parameters", "pft_traits"):
        value = np.asarray(getattr(inactive_pft, name)).copy()
        value[1] = 1.0e8
        perturbed_pft_values[name] = value
    perturbed_fraction = fraction.copy()
    perturbed_fraction[1] = 1.0e8
    perturbed_pft = inactive_pft._replace(
        **perturbed_pft_values,
        pft_fraction=perturbed_fraction,
    )
    perturbed_state = inactive.state
    for domain in layout.axis_layout.domains:
        value = np.asarray(getattr(perturbed_state, domain.name)).copy()
        value[domain.element_pft_index == 1] = 1.0e8
        perturbed_state = perturbed_state._replace(**{domain.name: value})
    other = np.asarray(perturbed_state.other).copy()
    other[layout.state.element_pft_index == 1] = 1.0e8
    perturbed_state = perturbed_state._replace(other=other)
    static = np.asarray(inactive.static_conditions).copy()
    static[layout.static.element_pft_index == 1] = 1.0e8
    rprof = np.asarray(inactive.process_static.rprof).copy()
    rprof[..., 1] = 1.0e8
    perturbed = inactive._replace(
        pft=perturbed_pft,
        state=perturbed_state,
        static_conditions=static,
        process_static=inactive.process_static._replace(rprof=rprof),
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
            rtol=1.0e-12,
            atol=1.0e-10,
            err_msg=name,
        )


def test_successor_head_binds_one_day_conservative_transition(successor_setup):
    setup = successor_setup
    head = setup["head"]
    parameters = setup["parameters"]
    assembled = setup["assembled"]
    context = head.assemble_context(parameters, assembled)
    prediction = head(parameters, context, assembled)
    update = apply_conservative_update(assembled, prediction)
    validate_conservative_result(update)
    assert abs(float(update.water.budget_residual)) <= 2.0e-12
    assert abs(float(update.carbon.budget_residual)) <= 2.0e-12

    transition = bind_daily_operator(
        input_assembler=setup["adapter"].input_assembler,
        process_head=head,
        retained_tail=plumbing_test_retained_tail_adapter(),
    )
    result = transition(parameters, setup["example"].inputs)
    validate_conservative_result(result.fast_update)
    assert int(np.asarray(result.diagnostics["retained_tail_executed"])) == 1
    assert set(result.prediction.named_outputs) == set(
        setup["example"].targets.supervision
    )


def test_rollout_refreshes_state_derived_pft_views_and_ignores_future_teacher_state(
    successor_setup,
):
    setup = successor_setup
    source = setup["example"].inputs
    poisoned_pft = source.pft._replace(
        pft_state=np.full_like(np.asarray(source.pft.pft_state), 1.0e8),
        pft_fraction=np.full_like(np.asarray(source.pft.pft_fraction), 0.25),
    )
    template = source._replace(pft=poisoned_pft)
    refreshed = rollout_input_from_template(
        template,
        source.continuous_state,
        source.discrete_state,
        setup["adapter"].contract,
    )
    np.testing.assert_allclose(
        np.asarray(refreshed.pft.pft_state),
        np.asarray(source.pft.pft_state),
    )
    np.testing.assert_allclose(
        np.asarray(refreshed.pft.pft_fraction),
        np.asarray(source.pft.pft_fraction),
    )
    np.testing.assert_array_equal(
        np.asarray(refreshed.pft.pft_parameters),
        np.asarray(template.pft.pft_parameters),
    )

    templates = jax.tree_util.tree_map(
        lambda value: jnp.stack((value, value, value)),
        source,
    )
    future_state = np.asarray(templates.continuous_state).copy()
    future_state[1:] = 1.0e8
    templates = templates._replace(continuous_state=future_state)

    def increment_transition(increment, current):
        assembled = setup["adapter"].input_assembler.assemble(current)
        next_continuous = jnp.asarray(current.continuous_state).at[0].add(increment)
        return DailyOperatorResult(
            prediction=None,
            fast_update=None,
            next_state=assembled.state._replace(other=next_continuous),
            next_state_defined=assembled.state_defined._replace(
                other=jnp.ones_like(next_continuous, dtype=jnp.bool_)
            ),
            next_discrete_state=current.discrete_state,
            diagnostics={},
        )

    trace = rollout_daily_operator(
        jnp.asarray(1.0, dtype=jnp.float64),
        templates,
        transition=increment_transition,
        contract=setup["adapter"].contract,
    )
    expected = float(np.asarray(source.continuous_state)[0]) + np.arange(1.0, 4.0)
    np.testing.assert_allclose(
        np.asarray(trace.result.next_state.other)[:, 0],
        expected,
    )


def test_thirty_day_trajectory_extraction_and_training_objective(successor_setup):
    setup = successor_setup
    assembled = setup["assembled"]
    days = 30
    continuous = jnp.tile(
        jnp.asarray(assembled.state.other)[None, :],
        (days, 1),
    )
    defined = jnp.ones_like(continuous, dtype=jnp.bool_)
    fraction = jnp.tile(
        jnp.asarray(assembled.pft.pft_fraction)[None, :],
        (days, 1),
    )
    active = jnp.tile(
        jnp.asarray(assembled.pft.active_pft_mask)[None, :],
        (days, 1),
    )
    gpp = jnp.ones((days, 1, fraction.shape[-1]), dtype=jnp.float64)
    trajectories = extract_biomass_productivity_trajectories(
        continuous,
        defined,
        daily_fast_gpp=gpp,
        daily_fast_gpp_defined=jnp.ones_like(gpp, dtype=jnp.bool_),
        pft_fraction=fraction,
        active_pft_mask=active,
        contract=setup["adapter"].contract,
    )
    assert tuple(trajectories.fluxes) == CUMULATIVE_FLUX_FIELDS
    assert tuple(trajectories.stocks) == TERMINAL_STOCK_FIELDS
    for trajectory in (*trajectories.fluxes.values(), *trajectories.stocks.values()):
        assert trajectory.value.shape == (days, 1)
        assert np.all(np.asarray(trajectory.defined))
        assert np.all(np.isfinite(np.asarray(trajectory.value)))
    np.testing.assert_allclose(
        np.asarray(trajectories.fluxes["gpp"].value),
        np.sum(np.asarray(fraction), axis=1, keepdims=True),
    )

    coefficients = {
        "cumulative_flux_mse": 0.10,
        "cumulative_flux_bias_squared": 0.15,
        "terminal_stock_mse": 0.10,
        "terminal_stock_bias_squared": 0.15,
    }
    objective = successor_training_objective(
        jnp.asarray(2.0, dtype=jnp.float64),
        trajectories,
        trajectories,
        flux_scales={name: 1.0 for name in CUMULATIVE_FLUX_FIELDS},
        stock_scales={name: 1.0 for name in TERMINAL_STOCK_FIELDS},
        coefficients=coefficients,
    )
    assert float(objective.total) == 2.0
    assert float(objective.long_horizon.total) == 0.0


def test_train_train_gradient_ratio_preflight_is_single_and_bounded():
    parameters = {"weight": jnp.asarray([1.0, -2.0], dtype=jnp.float64)}

    def quadratic(values):
        return jnp.sum(values["weight"] ** 2)

    components = {name: quadratic for name in LONG_HORIZON_COMPONENTS}
    targets = {
        "cumulative_flux_mse": 0.10,
        "cumulative_flux_bias_squared": 0.15,
        "terminal_stock_mse": 0.10,
        "terminal_stock_bias_squared": 0.15,
    }
    preflight = calibrate_gradient_ratio_coefficients(
        parameters,
        quadratic,
        components,
        target_gradient_ratios=targets,
        coefficient_minimum=0.001,
        coefficient_maximum=10.0,
        split_label="train/train",
    )
    assert preflight.split_label == "train/train"
    assert preflight.resolved_coefficients == targets
    assert preflight.achieved_gradient_ratios == targets
    assert all(
        np.isfinite(float(value))
        for value in preflight.component_gradient_norms.values()
    )

    with pytest.raises(ValueError, match="restricted to train/train"):
        calibrate_gradient_ratio_coefficients(
            parameters,
            quadratic,
            components,
            target_gradient_ratios=targets,
            coefficient_minimum=0.001,
            coefficient_maximum=10.0,
            split_label="validation",
        )
