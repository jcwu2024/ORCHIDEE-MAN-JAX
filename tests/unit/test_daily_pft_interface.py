from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.daily_coarse_graining.daily_pft_interface import (
    DEFAULT_CONTRACT_PATH,
    apply_active_pft_mask,
    load_daily_pft_interface_contract,
    pack_named_pft_channels,
    pft_fraction_weighted_sum,
    validate_pft_axis_inputs,
)


def _inputs(n_pft: int = 3):
    contract = load_daily_pft_interface_contract()
    state = np.arange(n_pft * 4, dtype=np.float64).reshape(n_pft, 4) / 10.0
    parameters = np.arange(
        n_pft * len(contract.parameter_channels), dtype=np.float64
    ).reshape(n_pft, len(contract.parameter_channels)) / 100.0
    traits = np.zeros((n_pft, len(contract.trait_channels)), dtype=np.float64)
    traits[:, 2] = np.arange(n_pft) % 2
    fraction = np.linspace(0.0, 1.0, n_pft, dtype=np.float64)
    active = np.ones((n_pft,), dtype=np.bool_)
    return contract, validate_pft_axis_inputs(
        contract,
        pft_state=state,
        pft_parameters=parameters,
        pft_traits=traits,
        pft_fraction=fraction,
        active_pft_mask=active,
        pft_ids=tuple(f"stable_{index}" for index in range(n_pft)),
    )


def _shared_probe(inputs):
    local = (
        inputs.pft_state[:, :2]
        + inputs.pft_parameters[:, :2] * np.asarray([0.3, -0.7])
        + inputs.pft_traits[:, :2] * np.asarray([0.2, 0.4])
    )
    return apply_active_pft_mask(local, inputs.active_pft_mask)


def test_frozen_contract_classifies_paper_parameters_without_identity_embedding() -> None:
    contract = load_daily_pft_interface_contract()

    assert contract.status == "frozen_preimplementation"
    assert tuple(field.name for field in contract.network_inputs) == (
        "pft_state",
        "pft_parameters",
        "pft_traits",
        "pft_fraction",
        "active_pft_mask",
    )
    ownership = {
        channel.name: channel.ownership_class for channel in contract.parameter_channels
    }
    assert ownership["vcmax25"] == "explicit_parameterized_fast_process_factor"
    assert ownership["maint_resp_slope_c"] == "explicit_parameterized_fast_process_factor"
    assert ownership["alloc_min"] == "retained_exact"
    assert ownership["residence_time"] == "retained_exact"
    assert all("pft14" not in name for name in contract.parameter_names + contract.trait_names)


def test_manifest_rejects_learned_parameter_without_controlled_perturbations(
    tmp_path: Path,
) -> None:
    payload = json.loads(DEFAULT_CONTRACT_PATH.read_text(encoding="utf-8"))
    channel = payload["parameter_channels"][0]
    channel["ownership_class"] = "learned_conditional"
    channel["controlled_perturbation"] = {"status": "not_varied"}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="controlled perturbation data"):
        load_daily_pft_interface_contract(path)


def test_shared_interface_probe_is_pft_permutation_equivariant() -> None:
    _, inputs = _inputs(4)
    order = np.asarray([2, 0, 3, 1])

    original = _shared_probe(inputs)
    permuted = _shared_probe(inputs.permuted(order))

    np.testing.assert_array_equal(permuted, original[order])
    np.testing.assert_allclose(
        pft_fraction_weighted_sum(permuted, inputs.permuted(order)),
        pft_fraction_weighted_sum(original, inputs),
        rtol=0.0,
        atol=1e-15,
    )


def test_named_channel_packing_is_complete_and_manifest_ordered() -> None:
    contract = load_daily_pft_interface_contract()
    values = {
        name: np.full((2,), index + 1.0, dtype=np.float64)
        for index, name in enumerate(reversed(contract.parameter_names))
    }

    packed = pack_named_pft_channels(contract, values, channel_kind="parameter")

    assert packed.shape == (2, len(contract.parameter_channels))
    for channel in contract.parameter_channels:
        np.testing.assert_array_equal(packed[:, channel.index], values[channel.name])
    with pytest.raises(ValueError, match="missing=.*vcmax25"):
        pack_named_pft_channels(
            contract,
            {name: value for name, value in values.items() if name != "vcmax25"},
            channel_kind="parameter",
        )


def test_inactive_pft_isolation_holds_for_outputs_and_aggregation() -> None:
    contract, inputs = _inputs(3)
    fraction = inputs.pft_fraction.copy()
    fraction[1] = 0.0
    active = inputs.active_pft_mask.copy()
    active[1] = False
    baseline = validate_pft_axis_inputs(
        contract,
        pft_state=inputs.pft_state,
        pft_parameters=inputs.pft_parameters,
        pft_traits=inputs.pft_traits,
        pft_fraction=fraction,
        active_pft_mask=active,
        pft_ids=inputs.pft_ids,
    )
    perturbed_state = baseline.pft_state.copy()
    perturbed_parameters = baseline.pft_parameters.copy()
    perturbed_traits = baseline.pft_traits.copy()
    perturbed_state[1] = 1e12
    perturbed_parameters[1] = -1e12
    perturbed_traits[1] = 1e12
    perturbed = validate_pft_axis_inputs(
        contract,
        pft_state=perturbed_state,
        pft_parameters=perturbed_parameters,
        pft_traits=perturbed_traits,
        pft_fraction=fraction,
        active_pft_mask=active,
        pft_ids=baseline.pft_ids,
    )

    baseline_output = _shared_probe(baseline)
    perturbed_output = _shared_probe(perturbed)
    np.testing.assert_array_equal(perturbed_output[active], baseline_output[active])
    np.testing.assert_array_equal(perturbed_output[~active], 0.0)
    np.testing.assert_array_equal(
        pft_fraction_weighted_sum(perturbed_output, perturbed),
        pft_fraction_weighted_sum(baseline_output, baseline),
    )


@pytest.mark.parametrize("n_pft", [1, 2, 5])
def test_interface_accepts_variable_legal_pft_axis(n_pft: int) -> None:
    _, inputs = _inputs(n_pft)

    assert inputs.n_pft == n_pft
    assert set(inputs.network_arrays()) == {
        "pft_state",
        "pft_parameters",
        "pft_traits",
        "pft_fraction",
        "active_pft_mask",
    }
    assert "pft_ids" not in inputs.network_arrays()


def test_interface_rejects_nonboolean_mask_and_nonzero_inactive_fraction() -> None:
    contract, inputs = _inputs(2)
    with pytest.raises(TypeError, match="exact bool"):
        validate_pft_axis_inputs(
            contract,
            pft_state=inputs.pft_state,
            pft_parameters=inputs.pft_parameters,
            pft_traits=inputs.pft_traits,
            pft_fraction=inputs.pft_fraction,
            active_pft_mask=np.asarray([1, 0], dtype=np.int8),
            pft_ids=inputs.pft_ids,
        )
    with pytest.raises(ValueError, match="exactly zero fraction"):
        validate_pft_axis_inputs(
            contract,
            pft_state=inputs.pft_state,
            pft_parameters=inputs.pft_parameters,
            pft_traits=inputs.pft_traits,
            pft_fraction=np.asarray([0.0, 0.5], dtype=np.float64),
            active_pft_mask=np.asarray([True, False]),
            pft_ids=inputs.pft_ids,
        )
