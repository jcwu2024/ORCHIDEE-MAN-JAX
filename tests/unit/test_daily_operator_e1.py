from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.constrained_daily_replay import (
    encode_inventory_endpoints,
)
from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
)
from research.daily_coarse_graining.daily_operator.audit import (
    audit_executable_jaxpr,
    audit_inference_source_graph,
    audit_process_label_bindings,
)
from research.daily_coarse_graining.daily_operator.conservative_update import (
    apply_conservative_inventory_update,
    combine_signed_inventory,
    split_signed_inventory,
    validate_conservative_result,
)
from research.daily_coarse_graining.daily_operator.input_assembly import (
    CanonicalDailyOperatorInputAssembler,
)
from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
from research.daily_coarse_graining.daily_operator.native_forcing import (
    encode_native_forcing,
    validate_native_forcing,
)
from research.daily_coarse_graining.daily_operator.process_derivations import (
    NativeForcingFieldLayout,
)
from research.daily_coarse_graining.daily_operator.process_heads import (
    classify_differentiability_case,
    parameterize_inventory_prediction,
    run_minimal_process_heads,
)
from research.daily_coarse_graining.daily_operator.retained_tail import (
    plumbing_test_retained_tail_adapter,
)
from research.daily_coarse_graining.daily_operator.state_features import (
    validate_pft_input,
    validate_state_feature_groups,
)
from research.daily_coarse_graining.daily_operator.types import (
    AssembledDailyOperatorInput,
    CompactHeadParameters,
    DailyOperatorInput,
    DailyOperatorParameters,
    DayStartProcessState,
    DomainHeadParameters,
    FluxHeadParameters,
    ForcingEncoderParameters,
    InventoryPrediction,
    NativeForcingInput,
    PFTAxisInput,
    PFTHeadParameters,
    ProcessStaticConditions,
    RawInventoryPrediction,
    StateDefinedMasks,
    StateFeatureGroups,
    StaticRegistryInput,
    ThermalHeadParameters,
)

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[2]


def _forcing(n_record: int = 5, *, masked_tail: int = 0) -> NativeForcingInput:
    active_count = n_record - masked_tail
    values = np.arange(n_record * 3, dtype=np.float64).reshape(n_record, 3) / 20.0
    times = np.arange(n_record, dtype=np.float64) * 21600.0 - 21600.0
    durations = np.full((n_record,), 21600.0, dtype=np.float64)
    predecessor = np.zeros((n_record,), dtype=np.bool_)
    predecessor[0] = True
    mask = np.arange(n_record) < active_count
    if masked_tail:
        values[active_count:] = 1.0e12
        times[active_count:] = np.nan
        durations[active_count:] = np.nan
    return NativeForcingInput(values, times, durations, predecessor, mask)


def _state() -> tuple[StateFeatureGroups, StateDefinedMasks]:
    state = StateFeatureGroups(
        canopy=np.asarray([0.2, 0.4], dtype=np.float64),
        soil=np.asarray([[0.3, 0.7], [0.2, 0.8]], dtype=np.float64),
        snow=np.asarray([0.1, 0.0], dtype=np.float64),
        water=np.asarray([2.0, 4.0, 1.0], dtype=np.float64),
        carbon=np.asarray([10.0, 20.0], dtype=np.float64),
        thermal=np.asarray([280.0, 275.0], dtype=np.float64),
        other=np.asarray([1.0], dtype=np.float64),
    )
    masks = StateDefinedMasks(
        **{
            name: np.ones_like(getattr(state, name), dtype=np.bool_)
            for name in StateFeatureGroups._fields
        }
    )
    return state, masks


def _pft(*, inactive_middle: bool = False) -> PFTAxisInput:
    state = np.asarray([[0.2, 0.5], [0.8, 0.3]], dtype=np.float64)
    parameters = np.asarray(
        [
            [2.0, 1.0e-3, 1.0e-2, 1.0, 0.2, 10.0],
            [3.0, 2.0e-3, 2.0e-2, 1.2, 0.3, 12.0],
        ],
        dtype=np.float64,
    )
    traits = np.zeros((2, 10), dtype=np.float64)
    traits[0, 0] = 1.0
    fraction = np.asarray([0.1, 0.9], dtype=np.float64)
    active = np.asarray([True, True])
    if inactive_middle:
        state = np.insert(state, 1, [1.0e9, -1.0e9], axis=0)
        parameters = np.insert(parameters, 1, np.full((6,), 1.0e9), axis=0)
        traits = np.insert(traits, 1, np.full((10,), 1.0e9), axis=0)
        fraction = np.asarray([0.1, 0.0, 0.9], dtype=np.float64)
        active = np.asarray([True, False, True])
    return PFTAxisInput(state, parameters, traits, fraction, active)


class _FixtureInputAssembler:
    """Small-array plumbing fixture; scientific assembly is tested separately."""

    def assemble(self, value: DailyOperatorInput) -> AssembledDailyOperatorInput:
        values = jnp.asarray(value.continuous_state, dtype=jnp.float64)
        n_pft = value.pft.pft_state.shape[0]
        cursor = 0

        def take(shape):
            nonlocal cursor
            size = int(np.prod(shape, dtype=np.int64))
            result = values[cursor : cursor + size].reshape(shape)
            cursor += size
            return result

        state = StateFeatureGroups(
            canopy=take((2,)),
            soil=take((2, 2)),
            snow=take((2,)),
            water=take((3,)),
            carbon=take((2,)),
            thermal=take((2,)),
            other=take((1,)),
        )
        bm_to_litter = take((1, n_pft, 12, 1))
        turnover = take((1, n_pft, 12, 1))
        lignin_above = take((1, n_pft))
        lignin_below = take((1, n_pft, 2))
        if cursor != values.size:
            raise ValueError("fixture canonical state width drift")
        masks = StateDefinedMasks(
            **{
                name: jnp.isfinite(getattr(state, name))
                for name in StateFeatureGroups._fields
            }
        )
        static = jnp.asarray(value.static_registry.context_features, dtype=jnp.float64)
        return AssembledDailyOperatorInput(
            state=state,
            state_defined=masks,
            discrete_state=value.discrete_state,
            forcing=value.forcing,
            pft=value.pft,
            day_start_process_state=DayStartProcessState(
                bm_to_litter=bm_to_litter,
                turnover_daily=turnover,
                lignin_struc_above=lignin_above,
                lignin_struc_below=lignin_below,
            ),
            process_static=ProcessStaticConditions(
                rprof=value.static_registry.rprof,
                z_soil=value.static_registry.z_soil,
                litterfrac=jnp.full((12, 2), 0.5, dtype=jnp.float64),
                cue=jnp.asarray(0.3, dtype=jnp.float64),
                frac_carb=static[2:].reshape((1, 3, 3)),
                sro_bottom=jnp.asarray(1, dtype=jnp.int32),
            ),
            static_conditions=static[:2],
            annual_conditions=value.annual_conditions,
            year=value.year,
            day_index=value.day_index,
            canonical_input=value,
        )


_FIXTURE_ASSEMBLER = _FixtureInputAssembler()


def _input(*, n_record: int = 5, masked_tail: int = 0, inactive_middle: bool = False) -> DailyOperatorInput:
    state, masks = _state()
    pft = _pft(inactive_middle=inactive_middle)
    n_pft = pft.pft_state.shape[0]
    bm_to_litter = np.zeros((1, n_pft, 12, 1), dtype=np.float64)
    turnover = np.zeros_like(bm_to_litter)
    bm_to_litter[:, :, 0, 0] = 0.2
    turnover[:, :, 7, 0] = 0.1
    frac_carb = np.zeros((1, 3, 3), dtype=np.float64)
    frac_carb[:, 0, 1] = 0.2
    frac_carb[:, 0, 2] = 0.1
    frac_carb[:, 1, 0] = 0.3
    frac_carb[:, 1, 2] = 0.1
    frac_carb[:, 2, 0] = 0.2
    frac_carb[:, 2, 1] = 0.2
    continuous = np.concatenate(
        [
            *(np.asarray(getattr(state, name)).reshape(-1) for name in StateFeatureGroups._fields),
            bm_to_litter.reshape(-1),
            turnover.reshape(-1),
            np.full((1, n_pft), 0.3, dtype=np.float64).reshape(-1),
            np.full((1, n_pft, 2), 0.4, dtype=np.float64).reshape(-1),
        ]
    )
    return DailyOperatorInput(
        continuous_state=continuous,
        discrete_state={"phenology_status": np.asarray([1, 2], dtype=np.int32)},
        forcing=_forcing(n_record, masked_tail=masked_tail),
        pft=pft,
        static_registry=StaticRegistryInput(
            context_features=np.concatenate(
                (np.asarray([0.4, 0.6], dtype=np.float64), frac_carb.reshape(-1))
            ),
            rprof=np.full((1, n_pft), 1.0, dtype=np.float64),
            z_soil=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
        ),
        annual_conditions=np.asarray([410.0], dtype=np.float64),
        year=np.asarray(1961, dtype=np.int32),
        day_index=np.asarray(2, dtype=np.int32),
    )


def _parameters(inputs: DailyOperatorInput) -> DailyOperatorParameters:
    assembled = _FIXTURE_ASSEMBLER.assemble(inputs)
    hidden = 4
    forcing_feature_count = inputs.forcing.values.shape[1] + 3
    local_pft_width = int(np.prod(inputs.pft.pft_state.shape[1:])) + inputs.pft.pft_traits.shape[1] + 1
    context_width = hidden + 2 * len(StateFeatureGroups._fields) + local_pft_width + 2 + 1

    def domain(n_inventory: int, offset: float) -> DomainHeadParameters:
        return DomainHeadParameters(
            outgoing_kernel=np.full((n_inventory, context_width), 1.0e-4 + offset, dtype=np.float64),
            outgoing_bias=np.full((n_inventory,), -2.0, dtype=np.float64),
            input_kernel=np.full((n_inventory, context_width), 2.0e-5 + offset, dtype=np.float64),
            input_bias=np.full((n_inventory,), -3.0, dtype=np.float64),
            external_share_kernel=np.full((n_inventory, context_width), 1.0e-5 + offset, dtype=np.float64),
            external_share_bias=np.zeros((n_inventory,), dtype=np.float64),
            destination_kernel=np.full(
                (n_inventory, n_inventory, context_width),
                3.0e-5 + offset,
                dtype=np.float64,
            ),
            destination_bias=np.zeros((n_inventory, n_inventory), dtype=np.float64),
        )

    return DailyOperatorParameters(
        forcing=ForcingEncoderParameters(
            value_scale=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
            input_kernel=np.arange(forcing_feature_count * hidden, dtype=np.float64).reshape(
                forcing_feature_count, hidden
            )
            / 100.0,
            recurrent_kernel=np.eye(hidden, dtype=np.float64) * 0.2,
            bias=np.zeros((hidden,), dtype=np.float64),
        ),
        water=domain(assembled.state.water.size, 0.0),
        carbon=domain(assembled.state.carbon.size, 1.0e-6),
        thermal=ThermalHeadParameters(
            kernel=np.full((assembled.state.thermal.size, context_width), 1.0e-5, dtype=np.float64),
            bias=np.zeros((assembled.state.thermal.size,), dtype=np.float64),
        ),
        fluxes=FluxHeadParameters(
            water_kernel=np.full((10, context_width), 1.0e-5, dtype=np.float64),
            water_bias=np.zeros((10,), dtype=np.float64),
            carbon_kernel=np.full((13, context_width), 1.0e-5, dtype=np.float64),
            carbon_bias=np.zeros((13,), dtype=np.float64),
            energy_kernel=np.full((8, context_width), 1.0e-5, dtype=np.float64),
            energy_bias=np.zeros((8,), dtype=np.float64),
        ),
        pft=PFTHeadParameters(
            kernel=np.full((6, local_pft_width + context_width), 1.0e-4, dtype=np.float64),
            bias=np.zeros((6,), dtype=np.float64),
        ),
        compact=CompactHeadParameters(
            kernel=np.full((3, context_width), 1.0e-5, dtype=np.float64),
            bias=np.zeros((3,), dtype=np.float64),
        ),
    )


def _operator(inputs: DailyOperatorInput):
    parameters = _parameters(inputs)

    def process_head(current_parameters, context, current_inputs):
        return run_minimal_process_heads(
            current_parameters,
            context,
            current_inputs,
            thermal_bounds=np.asarray([100.0, 100.0], dtype=np.float64),
            forcing_layout=NativeForcingFieldLayout(0, 1, 1, 2),
        )

    transition = bind_daily_operator(
        input_assembler=_FIXTURE_ASSEMBLER,
        process_head=process_head,
        retained_tail=plumbing_test_retained_tail_adapter(),
    )
    return parameters, transition


def _contract_input_and_assembler():
    manifest_path = (
        ROOT
        / "outputs"
        / "research"
        / "daily_coarse_graining"
        / "dataset-manifest-v5"
        / "dataset_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    assembler = CanonicalDailyOperatorInputAssembler.from_contract(contract)
    discrete = {
        leaf.key: np.zeros(leaf.shape, dtype=np.dtype(leaf.dtype))
        for leaf in contract.discrete_leaves
    }
    static = np.zeros(
        (contract.static_conditions.landpoint_static_width,),
        dtype=np.float64,
    )
    clay = next(
        leaf
        for leaf in contract.static_conditions.landpoint_static_leaves
        if leaf.name == "clay_frac"
    )
    static[clay.start : clay.stop] = 0.2
    inputs = DailyOperatorInput(
        continuous_state=np.zeros(
            (contract.continuous_state_width,),
            dtype=np.float64,
        ),
        discrete_state=discrete,
        forcing=NativeForcingInput(
            values=np.zeros((5, contract.native_forcing.width), dtype=np.float64),
            record_time_seconds=np.arange(5, dtype=np.float64) * 21600.0,
            duration_seconds=np.full((5,), 21600.0, dtype=np.float64),
            predecessor_mask=np.asarray([True, False, False, False, False]),
            record_mask=np.ones((5,), dtype=np.bool_),
        ),
        pft=PFTAxisInput(
            pft_state=np.zeros((2, 1), dtype=np.float64),
            pft_parameters=np.zeros((2, 1), dtype=np.float64),
            pft_traits=np.zeros((2, 1), dtype=np.float64),
            pft_fraction=np.asarray([0.0, 1.0], dtype=np.float64),
            active_pft_mask=np.asarray([True, True]),
        ),
        static_registry=StaticRegistryInput(
            context_features=static,
            rprof=np.zeros((1, 14, 11), dtype=np.float64),
            z_soil=np.arange(11, dtype=np.float64),
        ),
        annual_conditions=np.zeros(
            (contract.static_conditions.annual_condition_width,),
            dtype=np.float64,
        ),
        year=np.asarray(1961, dtype=np.int32),
        day_index=np.asarray(0, dtype=np.int32),
    )
    return inputs, assembler


def _tamper_named_path(value, path: tuple[str, ...]):
    name = path[0]
    current = getattr(value, name)
    if len(path) > 1:
        replacement = _tamper_named_path(current, path[1:])
    else:
        replacement = np.asarray(current).copy()
        replacement.flat[0] += 1
    return value._replace(**{name: replacement})


@pytest.mark.parametrize("n_record,masked_tail", [(3, 0), (5, 0), (7, 2)])
def test_native_forcing_accepts_variable_masked_record_lengths(n_record: int, masked_tail: int) -> None:
    sequence = _forcing(n_record, masked_tail=masked_tail)
    validate_native_forcing(sequence)
    inputs = _input(n_record=n_record, masked_tail=masked_tail)
    parameters = _parameters(inputs).forcing

    encoded = encode_native_forcing(sequence, parameters)

    assert encoded.shape == (4,)
    assert np.isfinite(np.asarray(encoded)).all()


def test_native_forcing_encoder_is_order_sensitive_and_ignores_masked_padding() -> None:
    inputs = _input(n_record=5)
    parameters = _parameters(inputs).forcing
    sequence = inputs.forcing
    reversed_sequence = sequence._replace(values=sequence.values[::-1])
    padded = _forcing(7, masked_tail=2)

    original = encode_native_forcing(sequence, parameters)
    reversed_result = encode_native_forcing(reversed_sequence, parameters)
    padded_result = encode_native_forcing(padded, parameters)

    assert not np.allclose(original, reversed_result)
    np.testing.assert_allclose(original, padded_result, rtol=0.0, atol=1.0e-14)


def test_interface_validation_and_named_pytree_boundary() -> None:
    inputs = _input()
    assembled = _FIXTURE_ASSEMBLER.assemble(inputs)
    validate_native_forcing(inputs.forcing)
    validate_state_feature_groups(assembled.state, assembled.state_defined)
    validate_pft_input(inputs.pft)

    assert "landpoint" not in DailyOperatorInput._fields
    assert "pft_id" not in DailyOperatorInput._fields
    assert "next_state" not in DailyOperatorInput._fields
    assert "state" not in DailyOperatorInput._fields
    assert "state_defined" not in DailyOperatorInput._fields
    assert "day_start_process_state" not in DailyOperatorInput._fields
    assert "process_static" not in DailyOperatorInput._fields
    assert all(np.asarray(leaf).dtype != np.dtype(np.float32) for leaf in jax.tree_util.tree_leaves(inputs))


@pytest.mark.parametrize(
    "path",
    [
        ("state", "water"),
        ("state", "carbon"),
        ("state", "thermal"),
        ("day_start_process_state", "bm_to_litter"),
        ("day_start_process_state", "turnover_daily"),
        ("day_start_process_state", "lignin_struc_above"),
        ("day_start_process_state", "lignin_struc_below"),
        ("process_static", "rprof"),
        ("process_static", "z_soil"),
        ("process_static", "litterfrac"),
        ("process_static", "cue"),
        ("process_static", "frac_carb"),
        ("process_static", "sro_bottom"),
    ],
)
def test_canonical_assembler_rejects_contradictory_cached_views(
    path: tuple[str, ...],
) -> None:
    inputs, assembler = _contract_input_and_assembler()
    cached = assembler.assemble(inputs)
    tampered = _tamper_named_path(cached, path)

    with pytest.raises(ValueError, match="contradicts its canonical owner"):
        assembler.validate_cached_views(inputs, tampered)


def test_conservative_parameterization_and_budget_close_without_clipping() -> None:
    raw = RawInventoryPrediction(
        outgoing_rate_logits=np.asarray([-1.0, 0.5, 2.0], dtype=np.float64),
        external_input_logits=np.asarray([-2.0, -1.0, 0.0], dtype=np.float64),
        external_output_share_logits=np.asarray([0.0, 1.0, -1.0], dtype=np.float64),
        destination_logits=np.asarray(
            [[0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [1.0, 2.0, 0.0]],
            dtype=np.float64,
        ),
        destination_mask=~np.eye(3, dtype=np.bool_),
        defined=np.ones((3,), dtype=np.bool_),
    )
    prediction = parameterize_inventory_prediction(raw)
    result = apply_conservative_inventory_update(
        np.asarray([2.0, 4.0, 1.0], dtype=np.float64),
        prediction,
    )

    assert np.all((np.asarray(prediction.outgoing_fraction) >= 0.0) & (np.asarray(prediction.outgoing_fraction) <= 1.0))
    assert np.all(np.asarray(prediction.external_input_amount) >= 0.0)
    np.testing.assert_allclose(np.sum(prediction.destination_shares, axis=1), 1.0, atol=1.0e-15)
    assert np.min(np.asarray(result.inventory)) >= 0.0
    np.testing.assert_allclose(result.budget_residual, 0.0, atol=2.0e-15)


def test_allowed_input_boundary_and_in_graph_exact_derivations() -> None:
    inputs = _input()
    parameters, transition = _operator(inputs)

    result = transition(parameters, inputs)

    forbidden = {
        "rain_input",
        "snowfall_input",
        "litter_input",
        "poc_respiration",
        "poc_to_doc",
        "doc_to_poc",
        "doc_respiration",
        "daily_temperature_context",
    }
    assert forbidden.isdisjoint(DailyOperatorInput._fields)
    current = inputs.forcing.values[1:] * inputs.forcing.duration_seconds[1:, None]
    np.testing.assert_allclose(result.prediction.fluxes.water.rain_input, np.sum(current[:, 0:1], axis=0))
    np.testing.assert_allclose(result.prediction.fluxes.water.snowfall_input, np.sum(current[:, 1:2], axis=0))
    poc = result.prediction.fluxes.carbon.poc_gross_decomposition
    expected_poc_resp = 0.7 * np.sum(np.asarray(poc.ordinary) + np.asarray(poc.flooded), axis=(1, 2, 4))
    np.testing.assert_allclose(result.prediction.fluxes.carbon.poc_respiration, expected_poc_resp)
    doc = result.prediction.fluxes.carbon.doc_gross_decomposition
    expected_doc_resp = 0.7 * np.sum(np.asarray(doc.ordinary) + np.asarray(doc.flooded), axis=(2, 3))
    np.testing.assert_allclose(result.prediction.fluxes.carbon.doc_respiration, expected_doc_resp)
    litter = result.prediction.fluxes.carbon.litter_input
    np.testing.assert_array_equal(np.asarray(litter.above)[:, :, 0], 0.0)
    assert np.max(np.asarray(litter.above)[:, :, -1]) > 0.0


@pytest.mark.parametrize(
    "start,end",
    [
        ([0.0, 2.0, 4.0], [3.0, 1.0, 4.5]),
        ([1.0, 1.0, 0.0], [0.2, 2.0, 0.5]),
        ([10.0, 0.1, 2.0], [9.0, 0.1, 3.0]),
    ],
)
def test_gate_c_true_endpoint_labels_flow_through_new_inference_updater(start, end) -> None:
    encoded = encode_inventory_endpoints(start, end, name="oracle.true_endpoint")
    size = len(start)
    prediction = InventoryPrediction(
        outgoing_fraction=encoded.outgoing_fraction,
        external_input_amount=encoded.incoming_amount,
        external_output_share=np.ones((size,), dtype=np.float64),
        destination_shares=np.zeros((size, size), dtype=np.float64),
        defined=encoded.defined,
    )

    result = apply_conservative_inventory_update(np.asarray(start, dtype=np.float64), prediction)

    np.testing.assert_allclose(result.inventory, end, rtol=1.0e-12, atol=1.0e-12)
    assert np.min(np.asarray(result.inventory)) >= 0.0


def test_signed_inventory_decomposition_is_exact_and_nonnegative() -> None:
    signed = np.asarray([-0.3, 0.0, 1.2], dtype=np.float64)
    storage, debt = split_signed_inventory(signed)

    assert np.min(np.asarray(storage)) >= 0.0
    assert np.min(np.asarray(debt)) >= 0.0
    np.testing.assert_array_equal(combine_signed_inventory(storage, debt), signed)


def test_pft_permutation_equivariance_and_bare_soil_masks() -> None:
    inputs = _input()
    parameters, transition = _operator(inputs)
    original = transition(parameters, inputs)
    order = np.asarray([1, 0])
    permuted_pft = PFTAxisInput(
        *(np.asarray(value)[order] for value in inputs.pft)
    )
    permuted = transition(parameters, inputs._replace(pft=permuted_pft))

    for field in (
        "gpp",
        "maintenance_respiration",
        "transpiration",
        "wet_canopy_evaporation",
        "bare_soil_evaporation",
        "vegetation_process_mask",
        "bare_soil_process_mask",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(permuted.prediction.pft, field)),
            np.asarray(getattr(original.prediction.pft, field))[order],
        )
    np.testing.assert_array_equal(original.prediction.pft.bare_soil_process_mask, [True, False])
    np.testing.assert_array_equal(original.prediction.pft.vegetation_process_mask, [False, True])


def test_inactive_zero_fraction_pft_cannot_change_active_or_aggregate_outputs() -> None:
    baseline = _input()
    expanded = _input(inactive_middle=True)
    baseline_parameters, baseline_transition = _operator(baseline)
    expanded_parameters, expanded_transition = _operator(expanded)

    baseline_result = baseline_transition(baseline_parameters, baseline)
    expanded_result = expanded_transition(expanded_parameters, expanded)

    np.testing.assert_allclose(
        np.asarray(expanded_result.prediction.pft.gpp)[[0, 2]],
        baseline_result.prediction.pft.gpp,
    )
    assert expanded_result.prediction.pft.gpp[1] == 0.0
    np.testing.assert_allclose(expanded_result.next_state.water, baseline_result.next_state.water)
    np.testing.assert_allclose(expanded_result.next_state.carbon, baseline_result.next_state.carbon)


def test_eager_jit_discrete_masks_and_restart_roundtrip_are_exact(tmp_path) -> None:
    inputs = _input()
    parameters, transition = _operator(inputs)
    eager = transition(parameters, inputs)
    compiled = jax.jit(transition)(parameters, inputs)

    for expected, observed in zip(jax.tree_util.tree_leaves(eager), jax.tree_util.tree_leaves(compiled), strict=True):
        np.testing.assert_allclose(np.asarray(observed), np.asarray(expected), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_array_equal(eager.next_discrete_state["phenology_status"], inputs.discrete_state["phenology_status"])
    for name in StateDefinedMasks._fields:
        np.testing.assert_array_equal(
            getattr(eager.next_state_defined, name),
            getattr(_FIXTURE_ASSEMBLER.assemble(inputs).state_defined, name),
        )
    path = tmp_path / "e1_restart.npz"
    np.savez(
        path,
        **{f"state__{name}": np.asarray(getattr(eager.next_state, name)) for name in StateFeatureGroups._fields},
        discrete__phenology_status=np.asarray(eager.next_discrete_state["phenology_status"]),
    )
    with np.load(path, allow_pickle=False) as restart:
        for name in StateFeatureGroups._fields:
            np.testing.assert_array_equal(restart[f"state__{name}"], getattr(eager.next_state, name))
        np.testing.assert_array_equal(restart["discrete__phenology_status"], inputs.discrete_state["phenology_status"])
    validate_conservative_result(eager.fast_update)


def test_forward_and_reverse_gradients_are_finite_for_required_boundaries() -> None:
    inputs = _input()
    parameters, transition = _operator(inputs)

    def loss(current_parameters, current_inputs):
        result = transition(current_parameters, current_inputs)
        return (
            jnp.sum(result.next_state.water)
            + jnp.sum(result.next_state.carbon)
            + jnp.sum(result.next_state.thermal) * 1.0e-3
            + jnp.sum(result.prediction.pft.gpp) * 1.0e-3
            + jnp.sum(result.prediction.pft.maintenance_respiration) * 1.0e-3
        )

    parameter_gradient, input_gradient = jax.grad(loss, argnums=(0, 1), allow_int=True)(parameters, inputs)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree_util.tree_leaves(parameter_gradient))
    assert np.isfinite(np.asarray(input_gradient.continuous_state)).all()
    assert np.isfinite(np.asarray(input_gradient.forcing.values)).all()
    assert np.isfinite(np.asarray(input_gradient.pft.pft_parameters)).all()
    tangent = jax.tree_util.tree_map(
        lambda leaf: jnp.ones_like(leaf) if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact) else jnp.zeros_like(leaf),
        parameters,
    )
    _, forward = jax.jvp(lambda value: loss(value, inputs), (parameters,), (tangent,))
    assert np.isfinite(np.asarray(forward))


def test_inactive_and_threshold_adjacent_cases_are_explicitly_classified() -> None:
    smooth = classify_differentiability_case(_FIXTURE_ASSEMBLER.assemble(_input()))
    inactive = classify_differentiability_case(
        _FIXTURE_ASSEMBLER.assemble(_input(inactive_middle=True))
    )
    adjacent_inputs = _input()
    adjacent_traits = np.asarray(adjacent_inputs.pft.pft_traits).copy()
    adjacent_traits[1, 0] = 0.5
    adjacent = classify_differentiability_case(
        _FIXTURE_ASSEMBLER.assemble(
            adjacent_inputs._replace(
                pft=adjacent_inputs.pft._replace(pft_traits=adjacent_traits)
            )
        )
    )

    assert bool(np.asarray(smooth["smooth_active"]))
    assert bool(np.asarray(inactive["contains_inactive_slots"]))
    assert bool(np.asarray(adjacent["threshold_adjacent"]))


def test_graph_and_frozen_label_owner_audits_pass() -> None:
    inputs = _input()
    parameters, transition = _operator(inputs)
    graph = audit_inference_source_graph()
    executable = audit_executable_jaxpr(transition, parameters, inputs)
    labels = audit_process_label_bindings()

    assert graph["passed"], graph["violations"]
    assert graph["actual_input_fields"] == [
        "continuous_state",
        "discrete_state",
        "forcing",
        "pft",
        "static_registry",
        "annual_conditions",
        "year",
        "day_index",
    ]
    assert graph["forbidden_input_fields"] == []
    assert graph["missing_input_fields"] == []
    assert graph["extra_input_fields"] == []
    assert graph["input_schema_drift"] == []
    assert graph["input_schemas"]["StaticRegistryInput"] == [
        "context_features",
        "rprof",
        "z_soil",
    ]
    assert executable["passed"], executable
    assert executable["scan_lengths"] == [5]
    assert labels["passed"], labels
    assert labels["required_label_count"] == 47
