from __future__ import annotations

import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import (
    GPP_FIELD,
    MechanisticTargetOrientedCoupledGPPCarbonFeedbackProcessHead,
    bind_daily_operator,
    build_target_oriented_coupled_gpp_carbon_feedback_head,
    carbon_feedback_query_residuals,
    coupled_gpp_carbon_feedback_parameter_count,
    initialize_coupled_gpp_carbon_feedback_parameters,
    load_coupled_gpp_carbon_feedback_contract,
    masked_mechanistic_output_loss,
    plumbing_test_retained_tail_adapter,
    target_oriented_coupled_gpp_carbon_feedback_process_outputs,
    validate_conservative_result,
)
from research.daily_coarse_graining.daily_operator.parameter_routing import (
    DEFAULT_PFT_PARAMETER_ROUTING,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
    strong_baseline_parameter_count,
)
from tests.unit.test_daily_operator_hierarchical_coordinate_attention import (
    _complete_prediction_for_loss,
    _host_l2,
    _permuted_candidate_inputs,
)
from tests.unit.test_target_oriented_structured_gpp import (
    structured_gpp_setup as _structured_gpp_setup_fixture,
)


@pytest.fixture(scope="module")
def coupled_gpp_setup():
    parent = _structured_gpp_setup_fixture.__wrapped__()
    contract = load_coupled_gpp_carbon_feedback_contract()
    head = build_target_oriented_coupled_gpp_carbon_feedback_head(
        parent["candidate_head"],
        coupled_contract=contract,
    )
    parameters = initialize_coupled_gpp_carbon_feedback_parameters(
        parent["candidate_parameters"],
        jax.random.PRNGKey(20260831),
        state_layout=head.carbon_state_layout,
        coupled_contract=contract,
        structured_contract=parent["structured_contract"],
    )
    assert isinstance(head, MechanisticTargetOrientedCoupledGPPCarbonFeedbackProcessHead)
    return parent | {
        "candidate_parameters": parameters,
        "candidate_head": head,
        "parent_parameters": parent["candidate_parameters"],
        "parent_head": parent["candidate_head"],
        "coupled_contract": contract,
    }


def test_coupled_contract_layout_and_parameter_budget_are_exact(coupled_gpp_setup):
    setup = coupled_gpp_setup
    contract = setup["coupled_contract"]
    layout = setup["candidate_head"].carbon_state_layout
    parameters = setup["candidate_parameters"]
    adapter = parameters["target_oriented"]["coupled_gpp_carbon_feedback"]

    assert contract.sha256 == "c7f81436ff6ba1179b01adad0f76b47ddd2aab68208e97719c2c5464f3e7e74d"
    assert len(layout.field_names) == 107
    assert layout.state_indices.shape == (2_386,)
    assert layout.field_index.shape == (2_386,)
    assert layout.pft_owner_index.shape == (2_386,)
    assert set(np.unique(layout.pft_owner_index)) == {-1, 0, 1}
    assert "pft" not in layout.axis_role_names
    assert layout.axis_role_index.shape == (2_386, 5)
    assert layout.axis_position_index.shape == (2_386, 5)
    assert strong_baseline_parameter_count(adapter) == 3_080_832
    assert coupled_gpp_carbon_feedback_parameter_count(parameters) == 109_464_359
    assert len(adapter["blocks"]) == 2
    assert adapter["blocks"][0]["attention"]["query"]["kernel"].shape == (384, 384)
    assert adapter["blocks"][0]["ffn"]["first"]["kernel"].shape == (384, 768)
    assert len(adapter["residual_injections"]) == 3
    assert all(projection["kernel"].shape == (384, 384) for projection in adapter["residual_injections"])
    assert parameters["target_oriented"]["structured_gpp"] is setup["parent_parameters"]["target_oriented"][
        "structured_gpp"
    ]


def test_coupled_forward_loss_gradients_and_parent_isolation(coupled_gpp_setup):
    setup = coupled_gpp_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    inputs = setup["assembled"]
    context = head.assemble_context(parameters, inputs)
    eager = head.learned_outputs(parameters, context, inputs)
    compiled = jax.jit(
        lambda values, encoded, assembled: head.learned_outputs(values, encoded, assembled)
    )(parameters, context, inputs)
    jax.block_until_ready(compiled[GPP_FIELD])
    assert all(np.all(np.isfinite(np.asarray(value))) for value in eager.values())
    assert all(np.all(np.isfinite(np.asarray(value))) for value in compiled.values())

    parent_head = setup["parent_head"]
    parent_parameters = setup["parent_parameters"]
    parent_outputs = parent_head.learned_outputs(
        parent_parameters,
        parent_head.assemble_context(parent_parameters, inputs),
        inputs,
    )
    for name in parent_outputs:
        if name != GPP_FIELD:
            np.testing.assert_array_equal(np.asarray(eager[name]), np.asarray(parent_outputs[name]))

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

    def proxy(adapter_parameters):
        target = dict(parameters["target_oriented"])
        target["coupled_gpp_carbon_feedback"] = adapter_parameters
        changed = dict(parameters, target_oriented=target)
        return jnp.sum(head.learned_outputs(changed, context, inputs)[GPP_FIELD])

    value, gradients = jax.value_and_grad(proxy)(
        parameters["target_oriented"]["coupled_gpp_carbon_feedback"]
    )
    assert np.isfinite(float(value))
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(gradients))
    assert _host_l2(gradients["value_input"]["kernel"]) > 0.0
    assert _host_l2(gradients["field_embedding"]) > 0.0
    assert _host_l2(gradients["blocks"][0]["attention"]["query"]["kernel"]) > 0.0
    assert _host_l2(gradients["blocks"][1]["ffn"]["first"]["kernel"]) > 0.0
    assert all(_host_l2(projection["kernel"]) > 0.0 for projection in gradients["residual_injections"])


def test_coupled_carbon_state_pft_masks_permutation_and_daily_budget(coupled_gpp_setup):
    setup = coupled_gpp_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    inputs = setup["assembled"]
    outputs = head.learned_outputs(parameters, head.assemble_context(parameters, inputs), inputs)

    carbon_layout = head.carbon_state_layout
    owned = carbon_layout.state_indices[carbon_layout.pft_owner_index == 1]
    changed_state = np.asarray(inputs.state.other).copy()
    changed_state[int(owned[0])] += 0.5
    changed = inputs._replace(
        state=inputs.state._replace(other=changed_state),
        canonical_input=inputs.canonical_input._replace(continuous_state=changed_state),
    )
    changed_outputs = head.learned_outputs(
        parameters,
        head.assemble_context(parameters, changed),
        changed,
    )
    assert not np.array_equal(np.asarray(outputs[GPP_FIELD]), np.asarray(changed_outputs[GPP_FIELD]))

    inactive_pft = inputs.pft._replace(
        active_pft_mask=np.asarray([True, False]),
        pft_fraction=np.asarray([0.2, 0.0]),
    )
    inactive = inputs._replace(
        pft=inactive_pft,
        canonical_input=inputs.canonical_input._replace(pft=inactive_pft),
    )
    inactive_outputs = head.learned_outputs(
        parameters,
        head.assemble_context(parameters, inactive),
        inactive,
    )
    np.testing.assert_array_equal(np.asarray(inactive_outputs[GPP_FIELD])[:, 1], 0.0)

    permuted = _permuted_candidate_inputs(setup)
    permuted_outputs = head.learned_outputs(
        parameters,
        head.assemble_context(parameters, permuted),
        permuted,
    )
    np.testing.assert_allclose(
        np.asarray(permuted_outputs[GPP_FIELD]),
        np.asarray(outputs[GPP_FIELD])[:, ::-1],
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


def test_coupled_gpp_parameter_routes_remain_isolated(coupled_gpp_setup):
    setup = coupled_gpp_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    inputs = setup["assembled"]

    def gpp_total(pft_parameters):
        pft = inputs.pft._replace(pft_parameters=pft_parameters)
        changed = inputs._replace(
            pft=pft,
            canonical_input=inputs.canonical_input._replace(pft=pft),
        )
        context = head.assemble_context(parameters, changed)
        return jnp.sum(head.learned_outputs(parameters, context, changed)[GPP_FIELD])

    gradients = jax.grad(gpp_total)(jnp.asarray(inputs.pft.pft_parameters))
    vcmax = DEFAULT_PFT_PARAMETER_ROUTING.index("vcmax25")
    assert abs(float(gradients[1, vcmax])) > 0.0
    for name in (
        "maint_resp_slope_a",
        "maint_resp_slope_b",
        "maint_resp_slope_c",
        "alloc_min",
        "residence_time",
    ):
        np.testing.assert_array_equal(
            np.asarray(gradients[:, DEFAULT_PFT_PARAMETER_ROUTING.index(name)]),
            0.0,
        )


def test_coupled_inference_has_no_subdaily_reconstruction_or_scan():
    source = inspect.getsource(target_oriented_coupled_gpp_carbon_feedback_process_outputs)
    attention_source = inspect.getsource(carbon_feedback_query_residuals)
    assert "lax.scan" not in source
    assert "lax.scan" not in attention_source
    assert "48" not in source
    assert "48" not in attention_source
    assert "optimizer" not in source
    assert "checkpoint" not in source

    root = Path(__file__).resolve().parents[2]
    runner = (
        root
        / "scripts/hpc/run_target_oriented_coupled_gpp_carbon_feedback_gpu_admission.py"
    ).read_text(encoding="utf-8")
    launcher = (
        root
        / "scripts/hpc/run_target_oriented_coupled_gpp_carbon_feedback_gpu_admission.sh"
    ).read_text(encoding="utf-8")
    assert '"training_updates": 0' in runner
    assert '"optimizer_created": False' in runner
    assert '"checkpoint_written": False' in runner
    assert "initialize_target_oriented_adamw" not in runner
    assert "write_broad_checkpoint" not in runner
    assert "argnums=1" in runner
    assert "free gln01" in launcher
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-120m}" in launcher
