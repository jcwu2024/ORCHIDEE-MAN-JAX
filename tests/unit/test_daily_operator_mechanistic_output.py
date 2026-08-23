from __future__ import annotations

from inspect import signature

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import (
    MechanisticStrongBaselineProcessHead,
    bind_daily_operator,
    build_mechanistic_learned_schema,
    build_mechanistic_output_layout,
    initialize_strong_baseline_parameters,
    load_mechanistic_output_contract,
    load_strong_baseline_contract,
    masked_mechanistic_output_loss,
    plumbing_test_retained_tail_adapter,
    strong_baseline_parameter_count,
    validate_conservative_result,
)
from research.daily_coarse_graining.daily_operator.conservative_update import (
    apply_conservative_update,
)
from research.daily_coarse_graining.daily_operator.mechanistic_output_attainability import (
    materialize_attainability_oracle,
    oracle_learned_outputs,
)
from research.daily_coarse_graining.daily_operator.mechanistic_output_reconstruction import (
    _paper_zf_soil_b,
    compose_doc_baseline_residual,
    invert_doc_baseline_residual,
    reconstruct_thermal_coefficients,
    signed_endpoint_to_transfers,
    support_aware_nroot_simplex,
)
from research.daily_coarse_graining.thermal_coefficient_exact_reconstruction_panel import (
    _paper_vertical_grid,
    reconstruct_thermal_coefficient_fields,
)
from tests.unit.test_daily_operator_successor_process_head import (
    extended_successor_setup,
    successor_setup,
)


@pytest.fixture(scope="module")
def mechanistic_setup():
    base = successor_setup.__wrapped__()
    extended = extended_successor_setup.__wrapped__(base)
    architecture = load_strong_baseline_contract()
    mechanism = load_mechanistic_output_contract()
    learned_schema = build_mechanistic_learned_schema(
        extended["schema"],
        mechanism,
    )
    layout = build_mechanistic_output_layout(
        adapter=base["adapter"],
        complete_schema=extended["schema"],
        learned_schema=learned_schema,
        owner_contract=extended["owner_contract"],
        architecture_contract=architecture,
        mechanism_contract=mechanism,
        inputs=base["assembled"],
    )
    parameters = initialize_strong_baseline_parameters(
        jax.random.PRNGKey(20260817),
        schema=learned_schema,
        layout=layout,
        inputs=base["assembled"],
        contract=architecture,
    )
    head = MechanisticStrongBaselineProcessHead(
        contract=base["adapter"].contract,
        architecture_contract=architecture,
        mechanism_contract=mechanism,
        input_assembler=base["adapter"].input_assembler,
        complete_schema=extended["schema"],
        learned_schema=learned_schema,
        layout=layout,
        input_normalization=base["input_normalization"],
        output_normalization=extended["output_normalization"],
        forcing_layout=base["head"].forcing_layout,
        forcing_indices=base["head"].forcing_indices,
    )
    return (
        base
        | extended
        | {
            "architecture_contract": architecture,
            "mechanism_contract": mechanism,
            "learned_schema": learned_schema,
            "layout": layout,
            "parameters": parameters,
            "head": head,
        }
    )


def test_mechanistic_schema_capacity_and_coupled_energy_routing(mechanistic_setup):
    setup = mechanistic_setup
    learned = setup["learned_schema"]
    mechanism = setup["mechanism_contract"]
    assert len(learned.fields) == 161
    assert sum(np.prod(field.shape) for field in learned.fields) == 39_006
    assert strong_baseline_parameter_count(setup["parameters"]) == 49_743_001
    assert set(mechanism.exact_fields).isdisjoint(field.name for field in learned.fields)
    doc = next(field for field in learned.fields if field.name == mechanism.doc_field)
    nroot = next(field for field in learned.fields if field.name == mechanism.nroot_coordinate_field)
    assert doc.constraint == "logit"
    assert nroot.constraint == "logit"
    assert {
        field.constraint
        for field in learned.fields
        if field.name in set(mechanism.canopy_transfer_fields)
    } == {"logit"}
    ledger = {field.name: field for field in setup["layout"].ledger}
    assert {ledger[name].expert for name in mechanism.surface_energy_fields} == {"surface_energy"}


def test_paper_vertical_grid_does_not_leak_between_jit_traces():
    first = jax.jit(lambda offset: _paper_zf_soil_b() + offset)(0.0)
    second = jax.jit(lambda offset: _paper_zf_soil_b() + offset)(1.0)
    np.testing.assert_allclose(np.asarray(second), np.asarray(first) + 1.0)


def test_mechanistic_primitive_forward_is_finite(mechanistic_setup):
    setup = mechanistic_setup
    head = setup["head"]
    inputs = setup["assembled"]
    context = head.assemble_context(setup["parameters"], inputs)
    learned = head.learned_outputs(setup["parameters"], context, inputs)
    compiled = jax.jit(
        lambda values, encoded, assembled: head.learned_outputs(
            values,
            encoded,
            assembled,
        )
    )(setup["parameters"], context, inputs)
    jax.block_until_ready(compiled["daily.fast_gpp"])
    prediction = head(setup["parameters"], context, inputs)
    assert len(learned) == 161
    assert sum(np.asarray(value).size for value in learned.values()) == 39_006
    assert len(prediction.named_outputs) == 163
    assert "thermal.thermosoil_previous_step_state.stempdiag.normalized_tendency" in (prediction.named_outputs)
    assert "__mechanistic.doc_residual_coordinate" in prediction.named_outputs
    assert all(np.all(np.isfinite(np.asarray(value))) for value in learned.values())
    assert all(np.all(np.isfinite(np.asarray(value))) for value in compiled.values())
    assert np.all(np.isfinite(np.asarray(prediction.thermal.normalized_tendency)))


def test_support_aware_nroot_simplex_preserves_baseline_and_can_deactivate_layers():
    start = jnp.asarray(
        [[[0.0, 0.0, 0.0], [0.0, 0.6, 0.4]]],
        dtype=jnp.float64,
    )
    active = jnp.asarray([True, True])
    zero = support_aware_nroot_simplex(jnp.zeros_like(start), start, active)
    np.testing.assert_allclose(np.asarray(zero), np.asarray(start), atol=0.0, rtol=0.0)

    residual = jnp.zeros_like(start).at[0, 1, 1].set(-2.0).at[0, 1, 2].set(0.2)
    changed = support_aware_nroot_simplex(residual, start, active)
    assert float(changed[0, 1, 1]) == 0.0
    assert float(changed[0, 1, 2]) == 1.0
    assert float(changed[0, 1, 0]) == 0.0
    gradient = jax.grad(
        lambda raw: jnp.sum(support_aware_nroot_simplex(raw, start, active) ** 2)
    )(jnp.zeros_like(start))
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_signed_canopy_endpoint_has_unique_conservative_transfers():
    start = jnp.asarray([[0.0, 0.4, -0.2]], dtype=jnp.float64)
    end = jnp.asarray([[0.3, -0.1, 0.0]], dtype=jnp.float64)
    transfers = signed_endpoint_to_transfers(start, end)
    start_storage = jnp.maximum(start, 0.0)
    start_debt = jnp.maximum(-start, 0.0)
    end_storage = start_storage * (1.0 - transfers["storage.outgoing_fraction"]) + transfers[
        "storage.incoming_amount"
    ]
    end_debt = start_debt * (1.0 - transfers["debt.outgoing_fraction"]) + transfers[
        "debt.incoming_amount"
    ]
    np.testing.assert_allclose(np.asarray(end_storage - end_debt), np.asarray(end))
    assert np.all(np.asarray(transfers["storage.outgoing_fraction"]) <= 1.0)
    assert np.all(np.asarray(transfers["debt.outgoing_fraction"]) <= 1.0)
    gradient = jax.grad(
        lambda target: sum(
            jnp.sum(value)
            for value in signed_endpoint_to_transfers(start, target).values()
        )
    )(end)
    assert np.all(np.isfinite(np.asarray(gradient)))


def test_doc_composition_is_baseline_centered_nonnegative_and_differentiable():
    baseline = jnp.asarray([0.0, 0.2, 2.0], dtype=jnp.float64)
    scale = jnp.asarray([1.0, 0.5, 2.0], dtype=jnp.float64)
    zero = compose_doc_baseline_residual(baseline, jnp.zeros_like(baseline), scale)
    np.testing.assert_allclose(np.asarray(zero), np.asarray(baseline), atol=2.0e-15)
    positive = compose_doc_baseline_residual(
        baseline,
        jnp.asarray([1.0, 1.0, 1.0]),
        scale,
    )
    assert np.all(np.asarray(positive) >= 0.0)
    assert float(positive[0]) > 0.0
    gradient = jax.grad(lambda residual: jnp.sum(compose_doc_baseline_residual(baseline, residual, scale)))(
        jnp.zeros_like(baseline)
    )
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert np.all(np.asarray(gradient) > 0.0)


def test_doc_residual_coordinate_has_finite_exact_inverse():
    baseline = jnp.asarray([0.0, 0.2, 2.0], dtype=jnp.float64)
    target = jnp.asarray([0.0, 0.5, 1.0], dtype=jnp.float64)
    scale = jnp.asarray([1.0, 0.5, 2.0], dtype=jnp.float64)
    residual = invert_doc_baseline_residual(target, baseline, scale)
    reconstructed = compose_doc_baseline_residual(baseline, residual, scale)
    assert np.all(np.isfinite(np.asarray(residual)))
    np.testing.assert_allclose(np.asarray(reconstructed), np.asarray(target), atol=2.0e-15, rtol=0.0)


def test_zero_training_attainability_oracle_materializes_complete_boundary(mechanistic_setup):
    setup = mechanistic_setup
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
        inputs = _set_static(inputs, setup["adapter"].input_assembler, name, value)
    forcing_values = jnp.asarray(inputs.forcing.values)
    forcing_values = forcing_values.at[:, setup["head"].forcing_indices["Tair"]].set(280.0)
    forcing_values = forcing_values.at[:, setup["head"].forcing_indices["PSurf"]].set(100000.0)
    forcing_values = forcing_values.at[:, setup["head"].forcing_indices["Wind_N"]].set(2.0)
    forcing_values = forcing_values.at[:, setup["head"].forcing_indices["Wind_E"]].set(1.0)
    forcing = inputs.forcing._replace(values=forcing_values)
    inputs = inputs._replace(
        forcing=forcing,
        canonical_input=inputs.canonical_input._replace(forcing=forcing),
    )
    learned, target_doc = oracle_learned_outputs(
        setup["head"],
        setup["targets"],
        inputs,
    )
    assert set(learned) == {field.name for field in setup["learned_schema"].fields}
    assert target_doc.shape == learned[setup["mechanism_contract"].doc_field].shape

    result = materialize_attainability_oracle(
        setup["head"],
        setup["targets"],
        inputs,
        doc_fixed_point_iterations=2,
    )
    validate_conservative_result(result.fast_update)
    assert set(result.prediction.named_outputs) == {field.name for field in setup["schema"].fields}
    nonfinite = []
    vegetation = np.asarray(inputs.pft.pft_traits)[:, 0] < 0.5
    for name, value in result.prediction.named_outputs.items():
        defined = np.asarray(setup["targets"][name].defined, dtype=np.bool_)
        if name.endswith("roughheight_pft"):
            defined = defined & vegetation[None, :]
        if not np.all(np.isfinite(np.asarray(value))[defined]):
            nonfinite.append(name)
    assert not nonfinite, nonfinite
    assert np.isfinite(float(result.doc_coordinate_fixed_point_residual))


def test_exact_reconstruction_sees_materialized_learned_day_end_owners(mechanistic_setup):
    setup = mechanistic_setup
    head = setup["head"]
    inputs = setup["assembled"]
    learned, _target_doc = oracle_learned_outputs(
        head,
        setup["targets"],
        inputs,
    )
    primitive = head.prediction_from_learned_outputs(learned, inputs)
    fast_update = apply_conservative_update(inputs, primitive)
    fields, _ok_leak = head.reconstructor.updated_fast_owner_fields(
        primitive,
        fast_update,
        inputs,
    )
    thermal_sources = head.reconstructor.thermal_coefficient_inputs(fields, inputs)
    assert set(thermal_sources) == set(signature(reconstruct_thermal_coefficients).parameters)
    active = np.asarray(head.input_assembler.active_pft_indices, dtype=np.int32)
    np.testing.assert_allclose(
        np.asarray(thermal_sources["ptn"]),
        np.take(
            np.asarray(fields["thermosoil_previous_step_state"]["ptn"]),
            active,
            axis=2,
        ),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(fields["hydrol_previous_step_state"]["nroot"])[:, active],
        np.asarray(primitive.named_outputs[setup["mechanism_contract"].nroot_coordinate_field]),
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(fields["hydrol_previous_step_state"]["evap_bare_lim_ns"]),
        np.asarray(
            primitive.named_outputs[
                setup["mechanism_contract"].payload["hydrology_boundary"]["tile_beta_primitive_field"]
            ]
        ),
        atol=0.0,
        rtol=0.0,
    )


def _thermal_inputs():
    batch = 2
    nslm = 11
    ngrnd = 32
    nvm = 2
    nstm = 6
    grid = _paper_vertical_grid()
    return {
        "ptn": np.broadcast_to(
            np.linspace(268.0, 280.0, ngrnd)[None, :, None],
            (batch, ngrnd, nvm),
        ).copy(),
        "shum_ngrnd_permalong": np.full((batch, ngrnd, nvm), 0.5),
        "ref_soc": np.full((batch, ngrnd), 10.0),
        "mc": np.full((batch, nslm, nstm), 0.25),
        "mcl": np.full((batch, nslm, nstm), 0.2),
        "veget_max": np.asarray([[0.7, 0.3], [0.6, 0.4]]),
        "frac_nobio": np.zeros((batch, 1)),
        "lai": np.asarray([[0.0, 3.0], [0.0, 2.0]]),
        "temp_sol": np.asarray([274.0, 275.0]),
        "temp_sol_pft": np.asarray([[274.0, 274.5], [275.0, 275.5]]),
        "snow": np.asarray([10.0, 5.0]),
        "snow_nobio": np.zeros((batch, 1)),
        "snowrho": np.full((batch, 3), 250.0),
        "snowtemp": np.full((batch, 3), 270.0),
        "snowdz": np.full((batch, 3), 0.02),
        "pressure_hpa": np.asarray([1000.0, 990.0]),
        "njsc": np.asarray([6, 6], dtype=np.int32),
        "hydrol_zz_mm": np.asarray(grid.znh) * 1000.0,
        "hydrol_dz_mm": np.asarray(grid.dnh) * 1000.0,
        "pref_soil_veg": np.asarray([1, 6], dtype=np.int32),
        "ext_coeff_vegetfrac": np.asarray([0.0, 0.5]),
        "model_interval_seconds": 1800.0,
    }


def test_runtime_thermal_reconstruction_matches_accepted_panel_kernel():
    inputs = _thermal_inputs()
    expected, _metadata = reconstruct_thermal_coefficient_fields(**inputs)
    observed = reconstruct_thermal_coefficients(**inputs)
    assert set(observed) == set(expected)
    for name in expected:
        np.testing.assert_allclose(
            np.asarray(observed[name]),
            np.asarray(expected[name]),
            rtol=0.0,
            atol=0.0,
            err_msg=name,
        )


def _set_static(inputs, assembler, name, value):
    leaf = next(leaf for leaf in assembler.contract.static_conditions.landpoint_static_leaves if leaf.name == name)
    static = jnp.asarray(inputs.static_conditions).at[leaf.start : leaf.stop].set(jnp.asarray(value).reshape(-1))
    registry = inputs.canonical_input.static_registry._replace(context_features=static)
    return inputs._replace(
        static_conditions=static,
        canonical_input=inputs.canonical_input._replace(static_registry=registry),
    )


def test_post_update_materializes_complete_boundary(mechanistic_setup):
    setup = mechanistic_setup
    head = setup["head"]
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
    forcing_values = jnp.asarray(inputs.forcing.values).at[:, :].set(inputs.forcing.values)
    forcing_values = forcing_values.at[:, head.forcing_indices["Tair"]].set(280.0)
    forcing_values = forcing_values.at[:, head.forcing_indices["PSurf"]].set(100000.0)
    forcing_values = forcing_values.at[:, head.forcing_indices["Wind_N"]].set(2.0)
    forcing_values = forcing_values.at[:, head.forcing_indices["Wind_E"]].set(1.0)
    forcing = inputs.forcing._replace(values=forcing_values)
    inputs = inputs._replace(
        forcing=forcing,
        canonical_input=inputs.canonical_input._replace(forcing=forcing),
    )

    context = head.assemble_context(setup["parameters"], inputs)
    prediction = head(setup["parameters"], context, inputs)
    fields = {component: dict(values) for component, values in prediction.retained_boundary.fast_state_fields.items()}
    hydrol = fields["hydrol_previous_step_state"]
    hydrol.update(
        mc=jnp.full((1, 11, 6), 0.25),
        mcl=jnp.full((1, 11, 6), 0.2),
        snow=jnp.asarray([10.0]),
        snow_nobio=jnp.zeros((1, 1)),
        snowrho=jnp.full((1, 3), 250.0),
        snowtemp=jnp.full((1, 3), 270.0),
        snowdz=jnp.full((1, 3), 0.02),
    )
    diffuco = fields["diffuco_previous_step_state"]
    veget = jnp.asarray([[0.7] + [0.0] * 12 + [0.3]])
    lai = jnp.asarray([[0.0] + [0.0] * 12 + [3.0]])
    diffuco.update(
        veget=veget,
        veget_max=veget,
        lai=lai,
        height=jnp.asarray([[0.0] + [0.0] * 12 + [10.0]]),
        frac_nobio=jnp.zeros((1, 1)),
        temp_sol_pft=jnp.full((1, 14), 274.0),
    )
    thermal = fields["thermosoil_previous_step_state"]
    thermal.update(
        ptn=jnp.full((1, 32, 14), 274.0),
        stempdiag=jnp.full((1, 11), 274.0),
        shum_ngrnd_permalong=jnp.full((1, 32, 14), 0.5),
        refSOC=jnp.full((1, 32), 10.0),
    )
    fields["enerbil_previous_step_state"]["temp_sol"] = jnp.asarray([274.0])
    ok_leak = dict(prediction.retained_boundary.ok_leak_result)
    ok_leak["DOC"] = jnp.zeros((1, 14, 32, 2, 7, 1))
    boundary = prediction.retained_boundary._replace(
        fast_state_fields=fields,
        ok_leak_result=ok_leak,
    )
    prediction = prediction._replace(retained_boundary=boundary)
    fast_update = apply_conservative_update(inputs, prediction)
    water = assembler._pack_water(hydrol)
    carbon = assembler._pack_carbon(ok_leak)
    thermal_vector = assembler._pack_thermal(fields)
    state = fast_update.state._replace(
        water=water,
        carbon=carbon,
        thermal=thermal_vector,
    )
    fast_update = fast_update._replace(
        state=state,
        water=fast_update.water._replace(inventory=water),
        carbon=fast_update.carbon._replace(inventory=carbon),
    )
    complete = head.materialize_post_update(
        setup["parameters"],
        context,
        inputs,
        prediction,
        fast_update,
    )
    compiled_complete = jax.jit(
        lambda primitive, update, assembled: head.materialize_post_update(
            {},
            None,
            assembled,
            primitive,
            update,
        )
    )(prediction, fast_update, inputs)
    jax.block_until_ready(compiled_complete.named_outputs["typed.ok_leak.doc_diffusive_interface"])
    assert set(complete.named_outputs) == {field.name for field in setup["schema"].fields}
    assert sum(np.asarray(value).size for value in complete.named_outputs.values()) == 39_371
    assert set(compiled_complete.named_outputs) == set(complete.named_outputs)
    for name in setup["mechanism_contract"].exact_fields:
        value = np.asarray(complete.named_outputs[name])
        if name.endswith("roughheight_pft"):
            assert np.all(np.isfinite(value[:, 1]))
        else:
            assert np.all(np.isfinite(value)), name

    water_prediction = prediction.water_inventory._replace(
        outgoing_fraction=jnp.zeros_like(prediction.water_inventory.outgoing_fraction),
        external_input_amount=water - inputs.state.water,
        external_output_share=jnp.ones_like(prediction.water_inventory.external_output_share),
        destination_shares=jnp.zeros_like(prediction.water_inventory.destination_shares),
    )
    carbon_prediction = prediction.carbon_inventory._replace(
        outgoing_fraction=jnp.zeros_like(prediction.carbon_inventory.outgoing_fraction),
        external_input_amount=carbon - inputs.state.carbon,
        external_output_share=jnp.ones_like(prediction.carbon_inventory.external_output_share),
        destination_shares=jnp.zeros_like(prediction.carbon_inventory.destination_shares),
    )
    thermal_prediction = prediction.thermal._replace(
        normalized_tendency=(thermal_vector - inputs.state.thermal) / prediction.thermal.bounds,
    )
    driven_prediction = prediction._replace(
        water_inventory=water_prediction,
        carbon_inventory=carbon_prediction,
        thermal=thermal_prediction,
    )

    class FixedPrimitiveHead:
        def assemble_context(self, _parameters, _inputs):
            return None

        def __call__(self, _parameters, _context, _inputs):
            return driven_prediction

        def materialize_post_update(
            self,
            parameters,
            context,
            assembled,
            primitive,
            update,
        ):
            return head.materialize_post_update(
                parameters,
                context,
                assembled,
                primitive,
                update,
            )

    transition = bind_daily_operator(
        input_assembler=assembler,
        process_head=FixedPrimitiveHead(),
        retained_tail=plumbing_test_retained_tail_adapter(),
    )
    result = transition({}, inputs.canonical_input)
    validate_conservative_result(result.fast_update)
    assert set(result.prediction.named_outputs) == {field.name for field in setup["schema"].fields}
    assert abs(float(result.fast_update.water.budget_residual)) <= 2.0e-12
    assert abs(float(result.fast_update.carbon.budget_residual)) <= 2.0e-12


def test_mechanistic_loss_excludes_exact_fields_from_optimization(mechanistic_setup):
    setup = mechanistic_setup
    predicted = {field.name: jnp.zeros(field.shape, dtype=jnp.float64) for field in setup["schema"].fields}
    loss = masked_mechanistic_output_loss(
        predicted,
        setup["targets"],
        inputs=setup["assembled"],
        complete_schema=setup["schema"],
        learned_schema=setup["learned_schema"],
        scales=setup["scales"],
        architecture_contract=setup["architecture_contract"],
        mechanism_contract=setup["mechanism_contract"],
    )
    assert np.isfinite(float(loss.total))
    assert set(loss.exact_diagnostic_mse) == set(setup["mechanism_contract"].exact_fields)
    assert not set(loss.exact_diagnostic_mse) & set(loss.field_mse)


def test_mechanistic_loss_jvp_and_vjp_reach_attention(mechanistic_setup):
    setup = mechanistic_setup
    head = setup["head"]
    inputs = setup["assembled"]
    parameters = setup["parameters"]
    context = head.assemble_context(parameters, inputs)
    fields = {field.name: field for field in setup["schema"].fields}

    def loss_for_query_kernel(kernel):
        query = dict(
            parameters["expert_blocks"][0][0]["attention"]["query"],
            kernel=kernel,
        )
        attention = dict(
            parameters["expert_blocks"][0][0]["attention"],
            query=query,
        )
        block = dict(parameters["expert_blocks"][0][0], attention=attention)
        expert = (block,) + parameters["expert_blocks"][0][1:]
        changed = dict(
            parameters,
            expert_blocks=(expert,) + parameters["expert_blocks"][1:],
        )
        prediction = head(changed, context, inputs)
        primitive = dict(prediction.named_outputs)
        primitive.pop("__mechanistic.doc_residual_coordinate")
        for name in setup["mechanism_contract"].exact_fields:
            primitive.setdefault(name, jnp.zeros(fields[name].shape, dtype=jnp.float64))
        return masked_mechanistic_output_loss(
            primitive,
            setup["targets"],
            inputs=inputs,
            complete_schema=setup["schema"],
            learned_schema=setup["learned_schema"],
            scales=setup["scales"],
            architecture_contract=setup["architecture_contract"],
            mechanism_contract=setup["mechanism_contract"],
            auxiliary_outputs=prediction.auxiliary_outputs,
        ).total

    kernel = parameters["expert_blocks"][0][0]["attention"]["query"]["kernel"]
    loss, gradient = jax.value_and_grad(loss_for_query_kernel)(kernel)
    direction = jnp.ones_like(kernel) / jnp.sqrt(kernel.size)
    jvp_loss, derivative = jax.jvp(
        loss_for_query_kernel,
        (kernel,),
        (direction,),
    )
    assert np.isfinite(float(loss))
    assert np.isfinite(float(jvp_loss))
    assert np.isfinite(float(derivative))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(jnp.linalg.norm(gradient)) > 0.0
