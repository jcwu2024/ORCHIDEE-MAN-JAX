from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from research.daily_coarse_graining.daily_markov_contract import StateLeafSpec
from research.daily_coarse_graining.daily_operator.target_oriented_maintenance_state_recurrence import (
    FROZEN_CONTRACT_SHA256,
    INTERVENTION_STATE_KEYS,
    RESP_MAINT_PART_STATE_KEY,
    RESP_MAINT_STATE_KEY,
    WINDOW_NAMES,
    classify_maintenance_state_recurrence,
    load_target_oriented_maintenance_state_recurrence_contract,
    resolve_maintenance_intervention_state_leaves,
)


def _state_leaf(key: str, start: int) -> StateLeafSpec:
    component, field = key.split(".")
    shape = (2,) if field == "resp_maint" else (3,)
    return StateLeafSpec(
        component=component,
        path=(field,),
        shape=shape,
        full_shape=(1, *shape),
        dtype="float64",
        start=start,
        stop=start + shape[0],
        classification="prognostic",
        source_ref="source-backed-test",
        axis_names=("npts", "nvm") if field == "resp_maint" else ("npts", "nsnow"),
    )


def _records(gradient_reduction: float, head_reduction: float):
    return {
        "thirty_day_extreme": {
            "parent_to_treatment_gradient_reduction_ratio": gradient_reduction,
            "parent_to_treatment_maintenance_head_reduction_ratio": head_reduction,
        },
        "thirty_day_matched_control": {
            "parent_to_treatment_gradient_reduction_ratio": 1.0,
            "parent_to_treatment_maintenance_head_reduction_ratio": 1.0,
        },
    }


def test_frozen_contract_binds_resp_maint_but_excludes_reset_accumulator() -> None:
    contract = load_target_oriented_maintenance_state_recurrence_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert tuple(record["name"] for record in contract.windows) == WINDOW_NAMES
    assert INTERVENTION_STATE_KEYS[-1] == RESP_MAINT_STATE_KEY
    assert RESP_MAINT_PART_STATE_KEY not in INTERVENTION_STATE_KEYS


def test_state_resolution_requires_persisted_resp_maint() -> None:
    leaves = tuple(
        _state_leaf(key, start)
        for key, start in zip(INTERVENTION_STATE_KEYS, (12, 2, 17, 7, 30), strict=True)
    )
    resolved = resolve_maintenance_intervention_state_leaves(
        SimpleNamespace(state_leaves=leaves)
    )
    assert resolved == leaves


def test_classifier_separates_confirmed_mixed_and_no_effect() -> None:
    contract = load_target_oriented_maintenance_state_recurrence_contract()
    confirmed = classify_maintenance_state_recurrence(_records(2.1, 2.2), contract)
    assert confirmed["classification"] == "maintenance_state_recurrence_confirmed"
    mixed = classify_maintenance_state_recurrence(_records(1.5, 2.2), contract)
    assert mixed["classification"] == "maintenance_state_recurrence_mixed"
    absent = classify_maintenance_state_recurrence(_records(1.1, 1.2), contract)
    assert absent["classification"] == "maintenance_state_recurrence_not_explanatory"


def test_launcher_is_free_gpu_two_gradient_boundary() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root
        / "scripts"
        / "hpc"
        / "run_target_oriented_maintenance_state_recurrence_intervention.sh"
    ).read_text(encoding="utf-8")
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "audit_target_oriented_maintenance_state_recurrence" in launcher
    assert "target-oriented-snow-thermal-alias-recurrence-51bb538/report.json" in launcher
    assert "--prepared-updates 1000" in launcher
