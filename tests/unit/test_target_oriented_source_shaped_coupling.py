from __future__ import annotations

from typing import Any, Mapping

import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_coupling import (
    carbon_bias_identity_report,
    classify_source_shaped_coupling,
    compare_update_states,
    load_source_shaped_coupling_attribution_contract,
)


def _key(rmse: float, bias: float = 0.0) -> Mapping[str, Any]:
    return {
        "normalized_rmse": rmse,
        "normalized_signed_bias": bias,
        "evaluated_values": 8.0,
        "passed": True,
    }


def _groups(
    contract: Any,
    *,
    maintenance_rmse: float,
    npp_rmse: float,
    biases: Mapping[str, float] | None = None,
) -> list[Mapping[str, Any]]:
    values = dict(biases or {})
    records = []
    for update in contract.evaluation_updates:
        acceptance = {}
        for horizon in ("7", "30"):
            keys = {
                name: _key(
                    maintenance_rmse if name == "maintenance_respiration" else npp_rmse if name == "npp" else 0.1,
                    values.get(name, 0.0),
                )
                for name in contract.comparison_keys
            }
            acceptance[horizon] = {"keys": keys, "all_key_gates_passed": True}
        records.append(
            {
                "update": update,
                "runtime_year": 1960 + update,
                "acceptance": acceptance,
            }
        )
    return records


def test_frozen_contract_loads() -> None:
    contract = load_source_shaped_coupling_attribution_contract()
    assert contract.sha256 == "12b327c81851551dfd7ef5e4878b03845290d78b5d51d26168046fe25dce93b2"
    assert contract.evaluation_updates == (251, 358, 465, 572, 679, 786, 893, 1000)
    assert "maintenance_respiration" in contract.comparison_keys


def test_comparison_and_joint_flux_classification() -> None:
    contract = load_source_shaped_coupling_attribution_contract()
    update_0 = _groups(contract, maintenance_rmse=0.2, npp_rmse=0.1)
    update_100 = _groups(contract, maintenance_rmse=0.05, npp_rmse=0.2)
    free = compare_update_states(update_0, update_100, contract)
    teacher = compare_update_states(update_0, update_100, contract)
    classification = classify_source_shaped_coupling(
        free,
        teacher,
        {"passed": True},
        contract,
    )
    assert free["all_finite"]
    assert free["horizons"]["7"]["keys"]["maintenance_respiration"]["median_relative_score_change"] == pytest.approx(
        -0.9375
    )
    assert free["horizons"]["7"]["keys"]["npp"]["median_relative_score_change"] == pytest.approx(3.0)
    assert classification["joint_flux_scope_required"]
    assert classification["classification"] == "direct_daily_carbon_coupling_requires_joint_flux_scope"


def test_raw_carbon_bias_identities_close() -> None:
    contract = load_source_shaped_coupling_attribution_contract()
    biases = {
        "gpp": 0.6,
        "maintenance_respiration": 0.1,
        "growth_respiration": 0.2,
        "autotrophic_respiration": 0.3,
        "npp": 0.3,
    }
    groups = _groups(
        contract,
        maintenance_rmse=0.2,
        npp_rmse=0.1,
        biases=biases,
    )
    report = carbon_bias_identity_report(
        {"update_0": groups, "update_100": groups},
        flux_scales={name: 1.0 for name in biases},
        contract=contract,
    )
    assert report["passed"]
    assert report["maximum_absolute_identity_residual"] < 1.0e-12


def test_group_order_drift_is_rejected() -> None:
    contract = load_source_shaped_coupling_attribution_contract()
    groups = _groups(contract, maintenance_rmse=0.2, npp_rmse=0.1)
    with pytest.raises(ValueError, match="identity or order"):
        compare_update_states(groups[::-1], groups, contract)
