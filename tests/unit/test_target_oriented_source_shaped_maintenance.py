from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import (
    MAINTENANCE_FIELD,
    MechanisticTargetOrientedSourceShapedMaintenanceProcessHead,
    bind_daily_operator,
    compact_coeff_maint_zero_from_retained_tail,
    initialize_source_shaped_maintenance_parameters,
    load_source_shaped_maintenance_contract,
    maintenance_temperature_corrections,
    masked_mechanistic_output_loss,
    plumbing_test_retained_tail_adapter,
    resolve_maintenance_field_index,
    source_shaped_maintenance_parameter_count,
    validate_conservative_result,
)
from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_maintenance import (
    target_oriented_source_shaped_maintenance_process_outputs,
)
from tests.unit.test_daily_operator_hierarchical_coordinate_attention import (
    _complete_prediction_for_loss,
    _host_l2,
    _permuted_candidate_inputs,
    _production_valid_inputs,
)
from tests.unit.test_daily_operator_target_oriented_attention import (
    target_attention_setup as _target_attention_setup_fixture,
)


@pytest.fixture(scope="module")
def source_shaped_setup():
    setup = _target_attention_setup_fixture.__wrapped__()
    parent_parameters = setup["candidate_parameters"]
    source_contract = load_source_shaped_maintenance_contract()
    parameters = initialize_source_shaped_maintenance_parameters(
        parent_parameters,
        jax.random.PRNGKey(20260827),
        schema=setup["learned_schema"],
        target_contract=setup["target_contract"],
        source_contract=source_contract,
    )
    coefficient = np.zeros((2, 12), dtype=np.float64)
    coefficient[1] = 1.0e-4
    parent = setup["candidate_head"]
    head = MechanisticTargetOrientedSourceShapedMaintenanceProcessHead(
        contract=parent.contract,
        architecture_contract=parent.architecture_contract,
        mechanism_contract=parent.mechanism_contract,
        input_assembler=parent.input_assembler,
        complete_schema=parent.complete_schema,
        learned_schema=parent.learned_schema,
        layout=parent.layout,
        input_normalization=parent.input_normalization,
        output_normalization=parent.output_normalization,
        forcing_layout=parent.forcing_layout,
        forcing_indices=parent.forcing_indices,
        message_gate_scale=parent.message_gate_scale,
        config=parent.config,
        target_contract=parent.target_contract,
        target_config=parent.target_config,
        source_contract=source_contract,
        coeff_maint_zero=coefficient,
    )
    candidate = setup | {
        "candidate_parameters": parameters,
        "candidate_head": head,
        "parent_parameters": parent_parameters,
        "source_contract": source_contract,
    }
    inputs = _production_valid_inputs(candidate)
    state = np.asarray(inputs.pft.pft_state).copy()
    state[1, 0] = 0.8
    state[1, 1] = 2.0
    element_count = (state.shape[1] - 2) // 12
    biomass = state[1, 2:].reshape((12, element_count))
    biomass[:, 0] = np.linspace(2.0, 13.0, 12)
    state[1, 2:] = biomass.reshape(-1)
    fraction = np.asarray(inputs.pft.pft_fraction).copy()
    fraction[:] = (0.2, 0.8)
    pft_parameters = np.asarray(inputs.pft.pft_parameters).copy()
    pft_parameters[1, 0] = 60.0
    active_pft = inputs.pft._replace(
        pft_state=state,
        pft_parameters=pft_parameters,
        pft_fraction=fraction,
        active_pft_mask=np.asarray([True, True]),
    )
    inputs = inputs._replace(
        pft=active_pft,
        canonical_input=inputs.canonical_input._replace(pft=active_pft),
    )
    return candidate | {"assembled": inputs}


def test_source_shaped_contract_and_parameter_tree_are_exact(source_shaped_setup):
    setup = source_shaped_setup
    contract = setup["source_contract"]
    parameters = setup["candidate_parameters"]
    parent = setup["parent_parameters"]
    index = resolve_maintenance_field_index(
        setup["learned_schema"],
        setup["target_contract"],
    )

    assert contract.sha256 == "3c4d0c085c767f14b001837ccefb76aca291a56b8fd9d3a5e33279602af57fc4"
    assert source_shaped_maintenance_parameter_count(parameters) == 84_148_646
    assert parameters["target_oriented"]["primary_heads"][index] == {}
    parent_heads = parent["target_oriented"]["primary_heads"]
    candidate_heads = parameters["target_oriented"]["primary_heads"]
    assert all(candidate_heads[i] is parent_heads[i] for i in range(len(parent_heads)) if i != index)
    branches = parameters["target_oriented"]["source_shaped_maintenance_heads"]
    assert set(branches) == {"aboveground", "belowground"}
    for branch in branches.values():
        assert branch["first"]["kernel"].shape == (768, 256)
        assert branch["second"]["kernel"].shape == (256, 1)
        np.testing.assert_array_equal(np.asarray(branch["second"]["kernel"]), 0.0)
        np.testing.assert_array_equal(np.asarray(branch["second"]["bias"]), 0.0)


def test_exact_coefficient_is_selected_from_retained_static_owner(source_shaped_setup):
    full = np.arange(14 * 12, dtype=np.float64).reshape((14, 12))
    owner = SimpleNamespace(
        context=SimpleNamespace(stomate_static=SimpleNamespace(parameters=SimpleNamespace(coeff_maint_zero=full))),
        contract=SimpleNamespace(active_pft_indices=(0, 13)),
    )
    selected = compact_coeff_maint_zero_from_retained_tail(SimpleNamespace(static_owner=owner))
    np.testing.assert_array_equal(np.asarray(selected), full[[0, 13]])


def test_zero_correction_is_source_shaped_and_forward_is_finite(source_shaped_setup):
    setup = source_shaped_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    inputs = setup["assembled"]
    context = head.assemble_context(parameters, inputs)
    eager = head.learned_outputs(parameters, context, inputs)
    compiled = jax.jit(
        lambda values, encoded, assembled: head.learned_outputs(
            values,
            encoded,
            assembled,
        )
    )(parameters, context, inputs)
    jax.block_until_ready(compiled[MAINTENANCE_FIELD])

    hidden = jnp.ones((1, 2, 12, 768), dtype=jnp.float32)
    above, below = maintenance_temperature_corrections(
        parameters,
        hidden,
        np.asarray([False, True]),
    )
    np.testing.assert_array_equal(np.asarray(above), 0.0)
    np.testing.assert_array_equal(np.asarray(below), 0.0)
    maintenance = np.asarray(eager[MAINTENANCE_FIELD])
    assert maintenance.shape == (1, 2, 12)
    np.testing.assert_array_equal(maintenance[:, 0], 0.0)
    assert np.all(maintenance[:, 1] >= 0.0)
    assert np.any(maintenance[:, 1] > 0.0)
    assert all(np.all(np.isfinite(np.asarray(value))) for value in eager.values())
    assert all(np.all(np.isfinite(np.asarray(value))) for value in compiled.values())


def test_complete_loss_and_both_new_branch_gradients_are_finite(source_shaped_setup):
    setup = source_shaped_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    inputs = setup["assembled"]
    context = head.assemble_context(parameters, inputs)

    prediction, complete = _complete_prediction_for_loss(setup, parameters, context)
    complete_loss = masked_mechanistic_output_loss(
        complete,
        setup["targets"],
        inputs=inputs,
        complete_schema=setup["schema"],
        learned_schema=setup["learned_schema"],
        scales=setup["scales"],
        architecture_contract=setup["architecture_contract"],
        mechanism_contract=setup["mechanism_contract"],
        auxiliary_outputs=prediction.auxiliary_outputs,
    )
    assert np.isfinite(float(complete_loss.total))

    def proxy(target_tree):
        changed = dict(parameters, target_oriented=target_tree)
        outputs = head.learned_outputs(changed, context, inputs)
        return jnp.sum(outputs[MAINTENANCE_FIELD]) + 1.0e-4 * jnp.sum(outputs["daily.fast_gpp"])

    value, gradients = jax.value_and_grad(proxy)(parameters["target_oriented"])
    assert np.isfinite(float(value))
    leaves = jax.tree_util.tree_leaves(gradients)
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)
    branches = gradients["source_shaped_maintenance_heads"]
    assert _host_l2(branches["aboveground"]["second"]["kernel"]) > 0.0
    assert _host_l2(branches["belowground"]["second"]["kernel"]) > 0.0
    assert any(_host_l2(blocks[0]["attention"]["query"]["kernel"]) > 0.0 for blocks in gradients["domain_blocks"])

    hidden = jnp.ones((1, 2, 12, 768), dtype=jnp.float32)

    def branch_proxy(branch_tree):
        target = dict(parameters["target_oriented"])
        target["source_shaped_maintenance_heads"] = branch_tree
        changed = dict(parameters, target_oriented=target)
        above, below = maintenance_temperature_corrections(
            changed,
            hidden,
            np.asarray([False, True]),
        )
        return jnp.sum(above) + jnp.sum(below)

    branch_gradients = jax.grad(branch_proxy)(parameters["target_oriented"]["source_shaped_maintenance_heads"])
    assert _host_l2(branch_gradients["aboveground"]["second"]["kernel"]) > 0.0
    assert _host_l2(branch_gradients["belowground"]["second"]["kernel"]) > 0.0


def test_bare_inactive_permutation_and_bound_day(source_shaped_setup):
    setup = source_shaped_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    inputs = setup["assembled"]
    target = dict(parameters["target_oriented"])
    branches = {}
    for index, (name, branch) in enumerate(target["source_shaped_maintenance_heads"].items()):
        second = dict(branch["second"])
        second["kernel"] = jnp.full_like(second["kernel"], 1.0e-3)
        second["bias"] = jnp.full_like(second["bias"], 0.05 * (index + 1))
        branches[name] = dict(branch, second=second)
    target["source_shaped_maintenance_heads"] = branches
    routed_parameters = dict(parameters, target_oriented=target)
    original = head.learned_outputs(
        routed_parameters,
        head.assemble_context(routed_parameters, inputs),
        inputs,
    )
    np.testing.assert_array_equal(np.asarray(original[MAINTENANCE_FIELD])[:, 0], 0.0)

    inactive_pft = inputs.pft._replace(
        active_pft_mask=np.asarray([True, False]),
        pft_fraction=np.asarray([0.2, 0.0]),
    )
    inactive = inputs._replace(
        pft=inactive_pft,
        canonical_input=inputs.canonical_input._replace(pft=inactive_pft),
    )
    inactive_outputs = head.learned_outputs(
        routed_parameters,
        head.assemble_context(routed_parameters, inactive),
        inactive,
    )
    np.testing.assert_array_equal(
        np.asarray(inactive_outputs[MAINTENANCE_FIELD])[:, 1],
        0.0,
    )

    permuted = _permuted_candidate_inputs(setup)
    permuted_outputs = head.learned_outputs(
        routed_parameters,
        head.assemble_context(routed_parameters, permuted),
        permuted,
    )
    np.testing.assert_allclose(
        np.asarray(permuted_outputs[MAINTENANCE_FIELD]),
        np.asarray(original[MAINTENANCE_FIELD])[:, ::-1],
        rtol=1.0e-6,
        atol=2.0e-12,
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


def test_source_shaped_inference_contains_no_half_hour_reconstruction_or_scan():
    source = inspect.getsource(target_oriented_source_shaped_maintenance_process_outputs)
    assert "lax.scan" not in source
    assert "48" not in source

    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/run_target_oriented_source_shaped_maintenance_gpu_admission.py").read_text(
        encoding="utf-8"
    )
    launcher = (root / "scripts/hpc/run_target_oriented_source_shaped_maintenance_gpu_admission.sh").read_text(
        encoding="utf-8"
    )
    assert '"training_updates": 0' in runner
    assert '"optimizer_created": False' in runner
    assert "initialize_target_oriented_adamw" not in runner
    assert "load_target_oriented_stage_checkpoint" in runner
    assert "initialize_source_shaped_maintenance_parameters" in runner
    assert "build_target_oriented_source_shaped_maintenance_head" in runner
    assert "free gln01" in launcher
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-120m}" in launcher
