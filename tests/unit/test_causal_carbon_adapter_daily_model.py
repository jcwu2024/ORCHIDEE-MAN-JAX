from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.axis_process_coupled_daily_model import (
    axis_process_coupled_model_apply,
    axis_process_spec_from_contract,
    initialize_axis_process_coupled_model,
)
from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalDayBatch,
    CanonicalModelConfig,
)
from research.daily_coarse_graining.causal_carbon_adapter_daily_model import (
    CAUSAL_CARBON_ADAPTER_V1,
    assemble_causal_carbon_adapter_parameters,
    causal_carbon_adapter_model_apply,
    causal_carbon_adapter_spec_from_contract,
    causal_carbon_adapter_trainable_parameters,
    causal_carbon_interface_layout_from_contract,
    disable_causal_carbon_adapter_groups,
    initialize_causal_carbon_adapter,
)
from research.daily_coarse_graining.daily_model_architecture import (
    AXIS_PROCESS_COUPLED_V1,
    build_daily_model_definition,
    initialize_causal_adapter_from_parent,
    verify_checkpoint_architecture,
)


def _leaf(component, field, index, *, axes=("npts",)):
    return {
        "component": component,
        "path": [field],
        "shape": [1],
        "start": index,
        "stop": index + 1,
        "axis_names": list(axes),
    }


def _target(
    family,
    component,
    field,
    start,
    shape,
    *,
    axes,
    selected=(),
):
    stop = start + int(np.prod(shape))
    return {
        "family": family,
        "component": component,
        "path": [field],
        "shape": list(shape),
        "full_shape": list(shape),
        "start": start,
        "stop": stop,
        "axis_names": list(axes),
        "selected_pft_indices": list(selected),
        "owner": "source-backed test owner",
    }


def _contract_metadata():
    return {
        "continuous_state_width": 8,
        "fast_day_target_width": 22,
        "state_leaves": [
            _leaf("slowproc_stomate_previous_step_state", "gpp_daily", 0),
            _leaf("slowproc_stomate_previous_step_state", "biomass", 1),
            _leaf("slowproc_stomate_previous_step_state", "litter", 2),
            _leaf("slowproc_stomate_previous_step_state", "carbon", 3),
            _leaf("slowproc_stomate_previous_step_state", "herbivores", 4),
            _leaf("hydrol_previous_step_state", "soil_moisture", 5),
            _leaf("thermosoil_previous_step_state", "soil_temperature", 6),
            _leaf("diffuco_previous_step_state", "rveget", 7),
        ],
        "fast_day_target_leaves": [
            _target("driver", "driver_previous_step_state", "albedo", 0, (1,), axes=("npts",)),
            _target(
                "diffuco_enerbil",
                "diffuco_previous_step_state",
                "gpp",
                1,
                (1,),
                axes=("npts",),
            ),
            _target("hydrol", "hydrol_previous_step_state", "mc", 2, (1,), axes=("npts",)),
            _target(
                "thermosoil",
                "thermosoil_previous_step_state",
                "ptn",
                3,
                (1,),
                axes=("npts",),
            ),
            _target(
                "daily_interface",
                None,
                "gpp_daily",
                4,
                (1, 2),
                axes=("npts", "nvm"),
                selected=(0, 13),
            ),
            _target(
                "daily_interface",
                None,
                "resp_maint_part",
                6,
                (1, 2, 3),
                axes=("npts", "nvm", "nparts"),
                selected=(0, 13),
            ),
            _target(
                "ok_leak",
                None,
                "carbon_32l",
                12,
                (1, 2, 2),
                axes=("npts", "nvm", "ndeep"),
                selected=(0, 13),
            ),
            _target(
                "ok_leak",
                None,
                "deepC_peat",
                16,
                (1, 2, 2),
                axes=("npts", "ndeep", "nvm"),
                selected=(0, 13),
            ),
            _target(
                "final_diagnostics",
                None,
                "t2mdiag",
                20,
                (1,),
                axes=(),
            ),
            _target(
                "sechiba_finalize",
                "sechiba_finalize_state",
                "leaf_ci",
                21,
                (1,),
                axes=("nlai",),
            ),
        ],
    }


def _config():
    return CanonicalModelConfig(
        state_width=8,
        forcing_width=2,
        parameter_width=4,
        landpoint_static_width=5,
        annual_condition_width=1,
        fast_day_target_width=22,
        dynamic_undefined_width=2,
        state_latent_width=7,
        forcing_latent_width=5,
        condition_latent_width=4,
        hidden_width=9,
    )


def _batch():
    batch_size = 3
    state = jnp.arange(batch_size * 8, dtype=jnp.float32).reshape(batch_size, 8)
    forcing = jnp.arange(
        batch_size * 5 * 2,
        dtype=jnp.float32,
    ).reshape(batch_size, 5, 2)
    return CanonicalDayBatch(
        state=state / 20.0,
        state_finite=jnp.ones_like(state, dtype=bool),
        normalized_fast_day_baseline=jnp.zeros((batch_size, 22)),
        forcing_native=forcing / 10.0,
        forcing_finite=jnp.ones_like(forcing, dtype=bool),
        parameters=jnp.ones((batch_size, 4)),
        parameters_finite=jnp.ones((batch_size, 4), dtype=bool),
        landpoint_static=jnp.ones((batch_size, 5)),
        landpoint_static_finite=jnp.ones((batch_size, 5), dtype=bool),
        annual_conditions=jnp.ones((batch_size, 1)),
        annual_conditions_finite=jnp.ones((batch_size, 1), dtype=bool),
        calendar=jnp.zeros((batch_size, 4)),
    )


def _spec_and_parameters():
    base_spec = axis_process_spec_from_contract(_config(), _contract_metadata())
    base_parameters = initialize_axis_process_coupled_model(base_spec, seed=7)
    adapter_spec = causal_carbon_adapter_spec_from_contract(
        base_spec,
        _contract_metadata(),
    )
    parameters = initialize_causal_carbon_adapter(
        adapter_spec,
        base_parameters=base_parameters,
        seed=11,
    )
    return base_spec, adapter_spec, base_parameters, parameters


def test_causal_carbon_layout_selects_only_pft14_causal_columns():
    base_spec = axis_process_spec_from_contract(_config(), _contract_metadata())
    layout = causal_carbon_interface_layout_from_contract(
        _contract_metadata(),
        base_spec,
    )

    assert layout.groups[0].target_indices == (5, 9, 10, 11)
    assert layout.groups[1].target_indices == (14, 15, 17, 19)
    assert layout.metadata["protected_target_width"] == 14
    assert (
        layout.sha256
        == causal_carbon_interface_layout_from_contract(
            _contract_metadata(),
            base_spec,
        ).sha256
    )


def test_causal_carbon_layout_rejects_missing_or_malformed_owner_fields():
    base_spec = axis_process_spec_from_contract(_config(), _contract_metadata())
    missing = _contract_metadata()
    missing["fast_day_target_leaves"] = [
        leaf for leaf in missing["fast_day_target_leaves"] if leaf["path"] != ["deepC_peat"]
    ]
    with pytest.raises(ValueError, match="missing causal carbon target"):
        causal_carbon_interface_layout_from_contract(missing, base_spec)

    malformed = _contract_metadata()
    malformed["fast_day_target_leaves"][4]["selected_pft_indices"] = [0]
    with pytest.raises(ValueError, match="does not carry PFT14"):
        causal_carbon_interface_layout_from_contract(malformed, base_spec)


def test_zero_initialized_adapter_is_bit_exact_with_frozen_base():
    base_spec, adapter_spec, base_parameters, parameters = _spec_and_parameters()
    batch = _batch()

    expected = jax.jit(
        lambda value, inputs: axis_process_coupled_model_apply(
            value,
            inputs,
            base_spec,
        )
    )(base_parameters, batch)
    observed = jax.jit(
        lambda value, inputs: causal_carbon_adapter_model_apply(
            value,
            inputs,
            adapter_spec,
        )
    )(parameters, batch)

    np.testing.assert_array_equal(
        observed.normalized_fast_day_target,
        expected.normalized_fast_day_target,
    )
    np.testing.assert_array_equal(
        observed.dynamic_undefined_flip_logits,
        expected.dynamic_undefined_flip_logits,
    )


def test_adapter_registry_verifies_parent_and_roundtrips_trainable_leaves():
    parent = build_daily_model_definition(
        AXIS_PROCESS_COUPLED_V1,
        _config(),
        _contract_metadata(),
    )
    parent_parameters = parent.initialize(seed=37)
    adapter = build_daily_model_definition(
        CAUSAL_CARBON_ADAPTER_V1,
        _config(),
        _contract_metadata(),
    )
    parameters = initialize_causal_adapter_from_parent(
        adapter,
        parent_identity={"model_architecture": parent.identity()},
        parent_parameters=parent_parameters,
        seed=41,
    )
    verify_checkpoint_architecture(
        {"model_architecture": adapter.identity()},
        adapter,
    )

    trainable = causal_carbon_adapter_trainable_parameters(parameters)
    restored = assemble_causal_carbon_adapter_parameters(
        parent_parameters,
        trainable,
    )
    expected = adapter.apply(parameters, _batch())
    observed = adapter.apply(restored, _batch())

    assert adapter.identity()["id"] == CAUSAL_CARBON_ADAPTER_V1
    np.testing.assert_array_equal(
        observed.normalized_fast_day_target,
        expected.normalized_fast_day_target,
    )


def test_adapter_registry_rejects_wrong_parent_architecture():
    adapter = build_daily_model_definition(
        CAUSAL_CARBON_ADAPTER_V1,
        _config(),
        _contract_metadata(),
    )
    base_parameters = initialize_axis_process_coupled_model(
        adapter.causal_carbon_adapter_spec.base_spec,
        seed=43,
    )

    with pytest.raises(ValueError, match="does not match requested model"):
        initialize_causal_adapter_from_parent(
            adapter,
            parent_identity={"model_architecture": {"id": "canonical_flat_v1"}},
            parent_parameters=base_parameters,
            seed=47,
        )


def test_adapter_gradients_freeze_base_and_reach_declared_outputs():
    _, adapter_spec, _, parameters = _spec_and_parameters()
    indices = jnp.asarray(adapter_spec.interface_layout.target_indices)

    def loss(value):
        prediction = causal_carbon_adapter_model_apply(
            value,
            _batch(),
            adapter_spec,
        )
        return jnp.sum(jnp.take(prediction.normalized_fast_day_target, indices, axis=1))

    gradients = jax.jit(jax.grad(loss))(parameters)

    assert all(np.all(np.asarray(gradient) == 0.0) for gradient in jax.tree_util.tree_leaves(gradients.base))
    assert all(np.any(np.asarray(output.weight) != 0.0) for output in gradients.group_outputs)
    assert all(np.all(np.isfinite(np.asarray(gradient))) for gradient in jax.tree_util.tree_leaves(gradients))


def test_nonzero_adapter_cannot_modify_protected_target_columns():
    _, adapter_spec, _, parameters = _spec_and_parameters()
    outputs = tuple(
        output._replace(
            weight=jnp.ones_like(output.weight) * 0.01,
            bias=jnp.ones_like(output.bias) * 0.02,
        )
        for output in parameters.group_outputs
    )
    changed_parameters = parameters._replace(group_outputs=outputs)
    baseline = causal_carbon_adapter_model_apply(
        parameters,
        _batch(),
        adapter_spec,
    ).normalized_fast_day_target
    changed = causal_carbon_adapter_model_apply(
        changed_parameters,
        _batch(),
        adapter_spec,
    ).normalized_fast_day_target

    adapted = set(adapter_spec.interface_layout.target_indices)
    protected = [index for index in range(22) if index not in adapted]
    np.testing.assert_array_equal(changed[:, protected], baseline[:, protected])
    assert np.any(np.asarray(changed[:, sorted(adapted)]) != np.asarray(baseline[:, sorted(adapted)]))


def test_group_ablation_exactly_disables_only_selected_correction():
    _, adapter_spec, _, parameters = _spec_and_parameters()
    outputs = tuple(
        output._replace(
            weight=jnp.ones_like(output.weight) * (0.01 + index * 0.01),
            bias=jnp.ones_like(output.bias) * (0.02 + index * 0.01),
        )
        for index, output in enumerate(parameters.group_outputs)
    )
    changed_parameters = parameters._replace(group_outputs=outputs)
    ablated_parameters = disable_causal_carbon_adapter_groups(
        changed_parameters,
        adapter_spec,
        ("carbon_stock_interface",),
    )
    baseline = causal_carbon_adapter_model_apply(
        parameters,
        _batch(),
        adapter_spec,
    ).normalized_fast_day_target
    changed = causal_carbon_adapter_model_apply(
        changed_parameters,
        _batch(),
        adapter_spec,
    ).normalized_fast_day_target
    ablated = causal_carbon_adapter_model_apply(
        ablated_parameters,
        _batch(),
        adapter_spec,
    ).normalized_fast_day_target

    flux_indices = adapter_spec.interface_layout.groups[0].target_indices
    stock_indices = adapter_spec.interface_layout.groups[1].target_indices
    np.testing.assert_array_equal(ablated[:, flux_indices], changed[:, flux_indices])
    np.testing.assert_array_equal(ablated[:, stock_indices], baseline[:, stock_indices])

    with pytest.raises(ValueError, match="unknown causal carbon adapter groups"):
        disable_causal_carbon_adapter_groups(
            changed_parameters,
            adapter_spec,
            ("not_a_group",),
        )
