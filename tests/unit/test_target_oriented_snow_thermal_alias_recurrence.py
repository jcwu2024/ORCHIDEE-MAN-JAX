from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from research.daily_coarse_graining.daily_markov_contract import StateLeafSpec
from research.daily_coarse_graining.daily_operator.target_oriented_snow_thermal_alias_recurrence import (
    FROZEN_CONTRACT_SHA256,
    SNOW_THERMAL_ALIAS_GROUPS,
    SNOW_THERMAL_ALIAS_STATE_KEYS,
    WINDOW_NAMES,
    classify_snow_thermal_alias_recurrence,
    load_target_oriented_snow_thermal_alias_recurrence_contract,
    resolve_snow_thermal_alias_state_leaves,
)


def _state_leaf(key: str, start: int) -> StateLeafSpec:
    component, field = key.split(".")
    return StateLeafSpec(
        component=component,
        path=(field,),
        shape=(3,),
        full_shape=(1, 3),
        dtype="float64",
        start=start,
        stop=start + 3,
        classification="prognostic",
        source_ref="explicit_snow",
        axis_names=("npts", "nsnow"),
    )


def _owner_leaf(key: str, *, role: str, canonical_owner: str):
    return SimpleNamespace(key=key, role=role, canonical_owner=canonical_owner)


def _records(gradient_reduction: float, snowheat_reduction: float):
    return {
        "thirty_day_extreme": {
            "parent_to_treatment_gradient_reduction_ratio": gradient_reduction,
            "parent_to_treatment_snowheat_head_reduction_ratio": snowheat_reduction,
        },
        "thirty_day_matched_control": {
            "parent_to_treatment_gradient_reduction_ratio": 1.0,
            "parent_to_treatment_snowheat_head_reduction_ratio": 1.0,
        },
    }


def test_frozen_contract_binds_four_alias_slices_and_two_windows() -> None:
    contract = load_target_oriented_snow_thermal_alias_recurrence_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert tuple(record["name"] for record in contract.windows) == WINDOW_NAMES
    assert tuple(contract.payload["intervention"]["state_keys"]) == SNOW_THERMAL_ALIAS_STATE_KEYS
    assert len(SNOW_THERMAL_ALIAS_STATE_KEYS) == 4


def test_state_resolution_requires_both_declared_mirror_relations() -> None:
    state_leaves = tuple(
        _state_leaf(key, start)
        for key, start in zip(SNOW_THERMAL_ALIAS_STATE_KEYS, (12, 2, 17, 7), strict=True)
    )
    owner_leaves = []
    for canonical, mirrors in SNOW_THERMAL_ALIAS_GROUPS:
        owner_leaves.append(
            _owner_leaf(
                canonical,
                role="existing_process_endpoint",
                canonical_owner=f"conservative_update.thermal.{canonical}",
            )
        )
        owner_leaves.extend(
            _owner_leaf(
                mirror,
                role="mirror_of_canonical_owner",
                canonical_owner=canonical,
            )
            for mirror in mirrors
        )
    resolved = resolve_snow_thermal_alias_state_leaves(
        SimpleNamespace(state_leaves=state_leaves),
        SimpleNamespace(leaves=tuple(owner_leaves)),
    )
    assert resolved == state_leaves


def test_classifier_separates_confirmed_mixed_and_no_effect() -> None:
    contract = load_target_oriented_snow_thermal_alias_recurrence_contract()
    confirmed = classify_snow_thermal_alias_recurrence(_records(3.0, 20.0), contract)
    assert confirmed["classification"] == "snow_thermal_alias_path_confirmed"
    mixed = classify_snow_thermal_alias_recurrence(_records(1.5, 20.0), contract)
    assert mixed["classification"] == "snow_thermal_alias_path_mixed"
    absent = classify_snow_thermal_alias_recurrence(_records(1.1, 1.5), contract)
    assert absent["classification"] == "snow_thermal_alias_path_not_explanatory"


def test_launcher_is_free_gpu_two_gradient_boundary() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (
        root
        / "scripts"
        / "hpc"
        / "run_target_oriented_snow_thermal_alias_recurrence_intervention.sh"
    ).read_text(encoding="utf-8")
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "audit_target_oriented_snow_thermal_alias_recurrence" in launcher
    assert "target-oriented-snow-thermal-recurrence-975ae9b/report.json" in launcher
    assert "--prepared-updates 1000" in launcher
