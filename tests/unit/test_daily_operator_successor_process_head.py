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
from research.daily_coarse_graining.daily_operator.fast_owner_reconstruction import (
    build_successor_fast_owner_materializer,
    load_fast_owner_reconstruction_contract,
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
    SUCCESSOR_FINAL_DECODER_KERNEL_GAIN,
    _final_decoder_parameters,
    _linear_parameters,
    _sqrt_with_zero_tangent,
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
from research.daily_coarse_graining.daily_operator.successor_fast_owner_decoder import (
    DAY_START_DELTA,
    DEFINED_LOGIT,
    FAST_OWNER_LABELS,
    RVEGET_LOGIT_KEY,
    build_successor_fast_owner_schema,
    extend_successor_output_normalization,
    schema_field_labels,
    training_representation_outputs,
    validate_successor_output_normalization,
)
from research.daily_coarse_graining.daily_operator.successor_fast_owner_training import (
    build_successor_fast_owner_targets,
    successor_fast_owner_loss,
)
from research.daily_coarse_graining.daily_operator.successor_process_head import (
    SuccessorProcessHead,
)
from research.daily_coarse_graining.daily_operator.successor_training import (
    CUMULATIVE_FLUX_FIELDS,
    LONG_HORIZON_COMPONENTS,
    TERMINAL_STOCK_FIELDS,
    GradientRatioPreflightError,
    calibrate_gradient_ratio_coefficients,
    derive_long_horizon_scales,
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


@pytest.fixture(scope="module")
def extended_successor_setup(successor_setup):
    base = successor_setup
    owner_contract = load_fast_owner_reconstruction_contract(
        markov_contract=base["adapter"].contract,
    )
    schema = build_successor_fast_owner_schema(
        base["schema"],
        markov_contract=base["adapter"].contract,
        owner_contract=owner_contract,
    )
    layout = build_successor_token_layout(
        base["adapter"].input_assembler,
        schema,
        base["contract"],
    )
    output_normalization = unit_output_normalization(schema)
    parameters = initialize_successor_parameters(
        jax.random.PRNGKey(20260811),
        schema=schema,
        layout=layout,
        inputs=base["assembled"],
        arm=SOURCE_ROUTED_ATTENTION,
    )
    head = SuccessorProcessHead(
        arm=SOURCE_ROUTED_ATTENTION,
        contract=base["adapter"].contract,
        input_assembler=base["adapter"].input_assembler,
        schema=schema,
        token_layout=layout,
        input_normalization=base["input_normalization"],
        output_normalization=output_normalization,
        forcing_layout=base["head"].forcing_layout,
        forcing_indices=base["head"].forcing_indices,
    )
    targets = build_successor_fast_owner_targets(
        base["example"],
        schema=schema,
        markov_contract=base["adapter"].contract,
        owner_contract=owner_contract,
    )
    scales = {
        name: np.ones_like(np.asarray(target.value), dtype=np.float64)
        for name, target in targets.items()
        if name != RVEGET_LOGIT_KEY
    }
    return base | {
        "base_schema": base["schema"],
        "head": head,
        "layout": layout,
        "output_normalization": output_normalization,
        "owner_contract": owner_contract,
        "parameters": parameters,
        "schema": schema,
        "scales": scales,
        "targets": targets,
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
    forcing_indices = setup["head"].forcing_indices
    forcing_values = np.asarray(assembled.forcing.values).copy()
    forcing_values[:, forcing_indices["Tair"]] = 280.0
    forcing_values[:, forcing_indices["PSurf"]] = 101325.0
    forcing_values[:, forcing_indices["Qair"]] = 0.01
    driver_inputs = assembled._replace(
        forcing=assembled.forcing._replace(values=forcing_values)
    )
    prediction = setup["head"].prediction_from_named_outputs(
        outputs,
        driver_inputs,
    )

    def get_path(root, path):
        value = root
        for name in path:
            value = value[name]
        return value

    for leaf in setup["adapter"].contract.fast_day_target_leaves:
        if leaf.component is not None:
            value = get_path(
                prediction.retained_boundary.fast_state_fields[leaf.component],
                leaf.path,
            )
            assert np.asarray(value).shape == leaf.full_shape

    driver = prediction.retained_boundary.fast_state_fields[
        "driver_previous_step_state"
    ]
    forcing = np.asarray(driver_inputs.forcing.values)
    height_leaf = next(
        leaf
        for leaf in setup["adapter"].contract.static_conditions.landpoint_static_leaves
        if leaf.name == "measurement_heights"
    )
    heights = np.asarray(assembled.static_conditions)[
        height_leaf.start : height_leaf.stop
    ].reshape(height_leaf.shape)
    temperature = forcing[-1, forcing_indices["Tair"]]
    pressure = forcing[-1, forcing_indices["PSurf"]]
    humidity = forcing[-1, forcing_indices["Qair"]]
    np.testing.assert_allclose(driver["old_zlev"], heights[:1])
    np.testing.assert_allclose(driver["old_qair"], [humidity])
    np.testing.assert_allclose(
        driver["old_eair"],
        [1004.675 * temperature + 9.80665 * heights[0]],
    )
    np.testing.assert_allclose(
        driver["rau_old"],
        [pressure / (287.05 * temperature)],
    )


def test_fast_owner_schema_routes_and_parameter_match_are_frozen(
    extended_successor_setup,
):
    setup = extended_successor_setup
    schema = setup["schema"]
    added = schema.fields[101:]
    values = tuple(field for field in added if field.value_representation != DEFINED_LOGIT)
    logits = tuple(field for field in added if field.value_representation == DEFINED_LOGIT)

    assert len(schema.fields) == 181
    assert len(schema.labels) == 48
    assert schema.labels[-8:] == FAST_OWNER_LABELS
    assert len(values) == 79
    assert len(logits) == 1
    assert logits[0].name == RVEGET_LOGIT_KEY
    assert sum(int(np.prod(field.shape)) for field in values) == 839
    assert int(np.prod(logits[0].shape)) == 2
    assert sum(field.value_representation == DAY_START_DELTA for field in values) == 65
    assert all(
        field.day_start_owner is not None
        and field.day_start_state_start is not None
        and field.day_start_state_stop is not None
        for field in values
        if field.value_representation == DAY_START_DELTA
    )
    assert {
        domain: sum(field.process_domain == domain for field in values)
        for domain in ("water", "energy", "carbon")
    } == {"water": 42, "energy": 21, "carbon": 16}
    by_name = {field.name: field for field in added}
    assert by_name[
        "endpoint.fast_state.diffuco_previous_step_state.control_inudate"
    ].source_route == "carbon.fast_gpp"
    assert by_name["endpoint.daily_summary.resp_maint_radia"].source_route == (
        "carbon.fast_maintenance_respiration"
    )
    assert by_name["endpoint.ok_leak_aux.litterpart"].source_route == "carbon.other"
    labels = schema_field_labels(schema)
    assert {labels[field.name] for field in added} == set(FAST_OWNER_LABELS)
    assert np.all(np.any(setup["layout"].field_allowed_family_mask, axis=1))
    assert np.all(np.any(setup["layout"].field_graph_mask, axis=1))

    control = initialize_successor_parameters(
        jax.random.PRNGKey(20260812),
        schema=schema,
        layout=setup["layout"],
        inputs=setup["assembled"],
        arm=SOURCE_ROUTED_GRAPH_CONTROL,
    )
    assert successor_parameter_count(setup["parameters"]) == 8_722_689
    assert successor_parameter_count(control) == 8_722_689
    assert jax.tree_util.tree_structure(setup["parameters"]) == (
        jax.tree_util.tree_structure(control)
    )

    base_normalization = unit_output_normalization(setup["base_schema"])
    added_mean = {
        field.name: np.zeros(field.shape, dtype=np.float64) for field in values
    }
    added_scale = {
        field.name: np.ones(field.shape, dtype=np.float64) for field in values
    }
    extended = extend_successor_output_normalization(
        base_normalization,
        schema,
        added_mean=added_mean,
        added_scale=added_scale,
    )
    validate_successor_output_normalization(schema, extended)
    with pytest.raises(ValueError, match="fitted fast-owner normalization registry drift"):
        extend_successor_output_normalization(
            base_normalization,
            schema,
            added_mean=dict(list(added_mean.items())[1:]),
            added_scale=added_scale,
        )


def test_successor_final_decoder_only_scales_the_kernel():
    key = jax.random.PRNGKey(20260812)
    width = 256
    standard = _linear_parameters(key, width, 1)
    final = _final_decoder_parameters(key, width)

    np.testing.assert_array_equal(final["bias"], np.zeros((1,), dtype=np.float64))
    np.testing.assert_allclose(
        final["kernel"],
        standard["kernel"] * SUCCESSOR_FINAL_DECODER_KERNEL_GAIN,
        rtol=0.0,
        atol=0.0,
    )
    assert np.any(np.asarray(final["kernel"]) != 0.0)


def test_fast_owner_delta_targets_and_rveget_status_are_explicit(
    extended_successor_setup,
):
    setup = extended_successor_setup
    example = setup["example"]
    state_leaf = next(
        leaf
        for leaf in setup["adapter"].contract.state_leaves
        if leaf.key == "diffuco_previous_step_state.rveget"
    )
    state = np.asarray(example.inputs.continuous_state).copy()
    state[state_leaf.start : state_leaf.stop] = np.asarray([1.0e20, 4.0])
    inputs = example.inputs._replace(continuous_state=state)

    boundary = example.targets.retained_boundary
    fields = dict(boundary.fast_state_fields)
    diffuco = dict(fields["diffuco_previous_step_state"])
    rveget = np.asarray(diffuco["rveget"]).copy()
    active = setup["adapter"].active_pft_indices
    rveget[0, active[0]] = 3.0
    rveget[0, active[1]] = 1.0e20
    diffuco["rveget"] = rveget
    fields["diffuco_previous_step_state"] = diffuco
    targets = example.targets._replace(
        retained_boundary=boundary._replace(fast_state_fields=fields)
    )
    changed = example._replace(inputs=inputs, targets=targets)
    observed = build_successor_fast_owner_targets(
        changed,
        schema=setup["schema"],
        markov_contract=setup["adapter"].contract,
        owner_contract=setup["owner_contract"],
    )
    value_key = "endpoint.fast_state.diffuco_previous_step_state.rveget"
    np.testing.assert_array_equal(observed[value_key].value, [[3.0, 0.0]])
    np.testing.assert_array_equal(observed[value_key].defined, [[True, False]])
    np.testing.assert_array_equal(observed[RVEGET_LOGIT_KEY].value, [[1.0, 0.0]])
    np.testing.assert_array_equal(observed[RVEGET_LOGIT_KEY].defined, [[True, True]])

    physical = {
        name: np.zeros(field.shape, dtype=np.float64)
        for name, field in ((field.name, field) for field in setup["schema"].fields)
    }
    physical[value_key] = np.asarray([[3.0, 7.0]], dtype=np.float64)
    represented = training_representation_outputs(physical, inputs, setup["schema"])
    np.testing.assert_array_equal(represented[value_key], [[3.0, 3.0]])


@pytest.mark.parametrize(
    "arm,seed",
    (
        (SOURCE_ROUTED_ATTENTION, 20260813),
        (SOURCE_ROUTED_GRAPH_CONTROL, 20260814),
    ),
)
def test_fast_owner_forward_loss_jvp_and_vjp_are_finite(
    extended_successor_setup,
    arm,
    seed,
):
    setup = extended_successor_setup
    parameters = initialize_successor_parameters(
        jax.random.PRNGKey(seed),
        schema=setup["schema"],
        layout=setup["layout"],
        inputs=setup["assembled"],
        arm=arm,
    )
    head = SuccessorProcessHead(
        arm=arm,
        contract=setup["adapter"].contract,
        input_assembler=setup["adapter"].input_assembler,
        schema=setup["schema"],
        token_layout=setup["layout"],
        input_normalization=setup["input_normalization"],
        output_normalization=setup["output_normalization"],
        forcing_layout=setup["head"].forcing_layout,
        forcing_indices=setup["head"].forcing_indices,
    )
    context = head.assemble_context(parameters, setup["assembled"])
    eager = head.named_outputs(parameters, context, setup["assembled"])
    compiled = jax.jit(
        lambda values, encoded, inputs: head.named_outputs(values, encoded, inputs)
    )(parameters, context, setup["assembled"])
    jax.block_until_ready(compiled[RVEGET_LOGIT_KEY])
    assert set(eager) == set(setup["targets"])
    assert sum(np.asarray(value).size for value in eager.values()) == 39_371
    assert all(np.all(np.isfinite(np.asarray(value))) for value in eager.values())
    assert all(np.all(np.isfinite(np.asarray(value))) for value in compiled.values())
    for field in setup["schema"].fields[101:]:
        value = np.asarray(eager[field.name])
        if field.constraint == "nonnegative":
            assert np.all(value >= 0.0), field.name
        elif field.constraint == "unit_interval":
            assert np.all((value >= 0.0) & (value <= 1.0)), field.name

    def loss_for_query_kernel(kernel):
        query = dict(
            parameters["process_blocks"][0]["input_mixer"]["query"],
            kernel=kernel,
        )
        mixer = dict(
            parameters["process_blocks"][0]["input_mixer"],
            query=query,
        )
        block = dict(parameters["process_blocks"][0], input_mixer=mixer)
        changed = dict(
            parameters,
            process_blocks=(block,) + parameters["process_blocks"][1:],
        )
        outputs = head.named_outputs(changed, context, setup["assembled"])
        return successor_fast_owner_loss(
            outputs,
            setup["targets"],
            inputs=setup["assembled"],
            schema=setup["schema"],
            scales=setup["scales"],
        ).total

    kernel = parameters["process_blocks"][0]["input_mixer"]["query"]["kernel"]
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
    np.testing.assert_allclose(float(jvp_loss), float(loss), rtol=1.0e-12)
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0


def test_fast_owner_inactive_pft_outputs_and_one_day_budgets(
    extended_successor_setup,
):
    setup = extended_successor_setup
    head = setup["head"]
    parameters = setup["parameters"]
    assembled = setup["assembled"]
    inactive = assembled._replace(
        pft=assembled.pft._replace(
            active_pft_mask=np.asarray([True, False], dtype=np.bool_),
        )
    )
    inactive_outputs = head.named_outputs(
        parameters,
        head.assemble_context(parameters, inactive),
        inactive,
    )
    for field in setup["schema"].fields[101:]:
        if field.pft_axis is None:
            continue
        selected = np.take(np.asarray(inactive_outputs[field.name]), 1, axis=field.pft_axis)
        np.testing.assert_array_equal(selected, 0.0, err_msg=field.name)

    forcing = np.asarray(assembled.forcing.values).copy()
    forcing[:, head.forcing_indices["Tair"]] = 280.0
    forcing[:, head.forcing_indices["PSurf"]] = 101325.0
    forcing[:, head.forcing_indices["Qair"]] = 0.01
    driven = assembled._replace(forcing=assembled.forcing._replace(values=forcing))
    outputs = head.named_outputs(
        parameters,
        head.assemble_context(parameters, driven),
        driven,
    )
    prediction = head.prediction_from_named_outputs(outputs, driven)
    update = apply_conservative_update(driven, prediction)
    validate_conservative_result(update)
    assert abs(float(update.water.budget_residual)) <= 2.0e-12
    assert abs(float(update.carbon.budget_residual)) <= 2.0e-12

    materializer = build_successor_fast_owner_materializer(
        setup["adapter"].contract,
        forcing_layout=head.forcing_layout,
    )
    height_leaf = next(
        leaf
        for leaf in setup["adapter"].contract.static_conditions.landpoint_static_leaves
        if leaf.name == "measurement_heights"
    )
    height = np.asarray(driven.static_conditions)[height_leaf.start : height_leaf.stop][0]
    boundary = prediction.retained_boundary
    materialized = materializer.materialize(
        fields_by_component=boundary.fast_state_fields,
        daily_fields=boundary.daily_fields,
        ok_leak_updates=boundary.ok_leak_result,
        final_diagnostics=boundary.final_entry_diagnostics,
        named_outputs=prediction.named_outputs,
        forcing=driven.forcing,
        measurement_height=height,
    )
    assert len(materialized.fields_by_component) > 0

    transition = bind_daily_operator(
        input_assembler=setup["adapter"].input_assembler,
        process_head=head,
        retained_tail=plumbing_test_retained_tail_adapter(),
    )
    result = transition(parameters, setup["example"].inputs)
    validate_conservative_result(result.fast_update)
    assert abs(float(result.fast_update.water.budget_residual)) <= 2.0e-12
    assert abs(float(result.fast_update.carbon.budget_residual)) <= 2.0e-12
    assert set(result.prediction.named_outputs) == set(setup["targets"])


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
    rematerialized = rollout_daily_operator(
        jnp.asarray(1.0, dtype=jnp.float64),
        templates,
        transition=increment_transition,
        contract=setup["adapter"].contract,
        rematerialize=True,
    )
    expected = float(np.asarray(source.continuous_state)[0]) + np.arange(1.0, 4.0)
    np.testing.assert_allclose(
        np.asarray(trace.result.next_state.other)[:, 0],
        expected,
    )
    np.testing.assert_array_equal(
        np.asarray(rematerialized.result.next_state.other),
        np.asarray(trace.result.next_state.other),
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


def test_long_horizon_scales_are_derived_from_frozen_column_scales(successor_setup):
    setup = successor_setup
    contract = setup["adapter"].contract
    gpp_shape = next(
        field.shape for field in setup["schema"].fields if field.name == "daily.fast_gpp"
    )
    scales = derive_long_horizon_scales(
        np.ones((contract.continuous_state_width,), dtype=np.float64),
        np.ones(gpp_shape, dtype=np.float64),
        contract,
    )
    assert tuple(scales.fluxes) == CUMULATIVE_FLUX_FIELDS
    assert tuple(scales.stocks) == TERMINAL_STOCK_FIELDS
    assert all(np.isfinite(value) and value > 0.0 for value in scales.fluxes.values())
    assert all(np.isfinite(value) and value > 0.0 for value in scales.stocks.values())
    assert (
        scales.stocks["ecosystem_carbon"]
        > scales.stocks["total_live_biomass_carbon"]
    )


def test_train_train_gradient_ratio_preflight_is_single_and_bounded():
    parameters = {"weight": jnp.asarray([1.0, -2.0], dtype=jnp.float64)}

    def quadratic(values):
        return jnp.sum(values["weight"] ** 2)

    components = {name: quadratic for name in LONG_HORIZON_COMPONENTS}
    def combined(values):
        return jnp.stack(
            tuple(quadratic(values) for _name in LONG_HORIZON_COMPONENTS)
        )
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
        compile_gradients=True,
        combined_component_loss=combined,
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


def test_gradient_ratio_preflight_reports_nonfinite_parameter_leaves():
    parameters = {"weight": jnp.asarray(2.0)}
    components = {
        name: (lambda values: values["weight"] ** 2)
        for name in LONG_HORIZON_COMPONENTS
    }
    components["cumulative_flux_mse"] = lambda values: values["weight"] * jnp.nan
    targets = {name: 0.1 for name in LONG_HORIZON_COMPONENTS}

    with pytest.raises(GradientRatioPreflightError) as caught:
        calibrate_gradient_ratio_coefficients(
            parameters,
            lambda values: values["weight"] ** 2,
            components,
            target_gradient_ratios=targets,
            coefficient_minimum=0.001,
            coefficient_maximum=10.0,
            split_label="train/train",
        )

    assert caught.value.diagnostic == {
        "component": "cumulative_flux_mse",
        "loss": "nan",
        "gradient_norm": "nan",
        "gradient_element_count": 1,
        "gradient_nonfinite_count": 1,
        "gradient_nonfinite_leaves": [
            {
                "path": "['weight']",
                "shape": [],
                "nonfinite_count": 1,
                "element_count": 1,
            }
        ],
    }


def test_successor_summary_standard_deviation_has_zero_tangent_at_zero_variance():
    def singleton_standard_deviation(value):
        centered = value - jnp.mean(value)
        return _sqrt_with_zero_tangent(jnp.mean(centered * centered))

    value, tangent = jax.jvp(
        _sqrt_with_zero_tangent,
        (jnp.asarray(4.0),),
        (jnp.asarray(1.0),),
    )
    gradient = jax.grad(singleton_standard_deviation)(jnp.asarray([2.0]))

    assert float(value) == pytest.approx(2.0)
    assert float(tangent) == pytest.approx(0.25)
    assert float(singleton_standard_deviation(jnp.asarray([2.0]))) == 0.0
    np.testing.assert_array_equal(np.asarray(gradient), np.zeros(1))
