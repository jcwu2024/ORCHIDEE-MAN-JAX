from __future__ import annotations

import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import (
    GPP_FIELD,
    MAINTENANCE_FIELD,
    MechanisticTargetOrientedStructuredGPPProcessHead,
    bind_daily_operator,
    build_target_oriented_structured_gpp_head,
    initialize_structured_gpp_parameters,
    load_structured_gpp_contract,
    masked_mechanistic_output_loss,
    plumbing_test_retained_tail_adapter,
    resolve_gpp_field_index,
    target_oriented_structured_gpp_process_outputs,
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
from tests.unit.test_target_oriented_distribution_aware_maintenance import (
    distribution_aware_setup as _distribution_aware_setup_fixture,
)


@pytest.fixture(scope="module")
def structured_gpp_setup():
    parent = _distribution_aware_setup_fixture.__wrapped__()
    contract = load_structured_gpp_contract()
    parameters = initialize_structured_gpp_parameters(
        parent["candidate_parameters"],
        jax.random.PRNGKey(20260829),
        schema=parent["learned_schema"],
        target_contract=parent["target_contract"],
        structured_contract=contract,
    )
    head = build_target_oriented_structured_gpp_head(
        parent["candidate_head"],
        structured_contract=contract,
    )
    assert isinstance(head, MechanisticTargetOrientedStructuredGPPProcessHead)
    return parent | {
        "candidate_parameters": parameters,
        "candidate_head": head,
        "parent_parameters": parent["candidate_parameters"],
        "parent_head": parent["candidate_head"],
        "structured_contract": contract,
    }


def test_structured_gpp_contract_and_parameter_tree_are_exact(structured_gpp_setup):
    setup = structured_gpp_setup
    parameters = setup["candidate_parameters"]
    target = parameters["target_oriented"]
    index = resolve_gpp_field_index(setup["learned_schema"], setup["target_contract"])

    assert (
        setup["structured_contract"].sha256
        == "ce211fa7c6f5a09bd900e02e31a864b61fb32f6b06ee96d4262402c513d620ea"
    )
    assert target["primary_heads"][index] == {}
    assert strong_baseline_parameter_count(target["structured_gpp"]) == 12_879_363
    assert strong_baseline_parameter_count(parameters) == 106_383_527
    assert len(target["structured_gpp"]["forcing_blocks"]) == 2
    assert len(target["structured_gpp"]["potential_blocks"]) == 2
    assert len(target["structured_gpp"]["limitation_blocks"]) == 2
    assert len(target["structured_gpp"]["correction_blocks"]) == 1


def test_structured_gpp_forward_loss_gradients_and_parent_isolation(structured_gpp_setup):
    setup = structured_gpp_setup
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

    def proxy(specialist):
        target = dict(parameters["target_oriented"])
        target["structured_gpp"] = specialist
        changed = dict(parameters, target_oriented=target)
        outputs = head.learned_outputs(changed, context, inputs)
        return jnp.sum(outputs[GPP_FIELD])

    value, gradients = jax.value_and_grad(proxy)(
        parameters["target_oriented"]["structured_gpp"]
    )
    assert np.isfinite(float(value))
    assert all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(gradients))
    for branch in ("potential_blocks", "limitation_blocks", "correction_blocks"):
        assert _host_l2(gradients[branch][0]["attention"]["query"]["kernel"]) > 0.0
        assert _host_l2(gradients[branch][0]["ffn"]["first"]["kernel"]) > 0.0
    for output in ("potential_output", "limitation_output", "correction_output"):
        assert _host_l2(gradients[output]["kernel"]) > 0.0


def test_structured_gpp_parameter_routes_are_isolated(structured_gpp_setup):
    setup = structured_gpp_setup
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


def test_structured_gpp_pft_masks_permutation_and_daily_budget(structured_gpp_setup):
    setup = structured_gpp_setup
    head = setup["candidate_head"]
    parameters = setup["candidate_parameters"]
    inputs = setup["assembled"]
    outputs = head.learned_outputs(parameters, head.assemble_context(parameters, inputs), inputs)
    np.testing.assert_array_equal(np.asarray(outputs[GPP_FIELD])[:, 0], 0.0)

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


def test_structured_gpp_inference_has_no_subdaily_scan():
    source = inspect.getsource(target_oriented_structured_gpp_process_outputs)
    assert "lax.scan" not in source
    assert "48" not in source
    assert "structured_gpp" in source
    assert MAINTENANCE_FIELD in source or "MAINTENANCE_FIELD" in source

    root = Path(__file__).resolve().parents[2]
    runner = (
        root / "scripts/hpc/run_target_oriented_structured_gpp_gpu_admission.py"
    ).read_text(encoding="utf-8")
    launcher = (
        root / "scripts/hpc/run_target_oriented_structured_gpp_gpu_admission.sh"
    ).read_text(encoding="utf-8")
    assert '"training_updates": 0' in runner
    assert '"optimizer_created": False' in runner
    assert '"parent_parameters_trainable": False' in runner
    assert "initialize_target_oriented_adamw" not in runner
    assert "argnums=1" in runner
    assert "free gln01" in launcher
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-120m}" in launcher
