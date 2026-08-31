from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research.daily_coarse_graining.daily_markov_contract import StateLeafSpec
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_state_feedback_attribution import (
    FROZEN_CONTRACT_SHA256,
    SOURCE_COMPONENTS,
    classify_component_feedback,
    load_structured_gpp_state_feedback_attribution_contract,
    resolve_component_state_masks,
)

ROOT = Path(__file__).resolve().parents[2]


def _leaf(component: str, field: str, start: int, stop: int, *, discrete: bool = False) -> StateLeafSpec:
    return StateLeafSpec(
        component=component,
        path=(field,),
        shape=(stop - start,),
        full_shape=(1, stop - start),
        dtype="int32" if discrete else "float64",
        start=None if discrete else start,
        stop=None if discrete else stop,
        classification="discrete" if discrete else "continuous",
        source_ref="synthetic_source_component",
        axis_names=("npts", "axis"),
    )


def _key(score: float) -> dict[str, float]:
    return {
        "normalized_rmse": float(np.sqrt(2.0 * score)),
        "normalized_signed_bias": 0.0,
        "evaluated_values": 1.0,
    }


def _groups(score: float):
    return [
        {
            "update": update,
            "acceptance": {horizon: {"keys": {"gpp": _key(score)}} for horizon in ("7", "30")},
        }
        for update in (301, 601, 901)
    ]


def test_contract_freezes_train_only_six_component_scope() -> None:
    contract = load_structured_gpp_state_feedback_attribution_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.evaluation_updates == (301, 601, 901)
    assert contract.source_components == SOURCE_COMPONENTS
    assert contract.payload["execution"]["optimizer_updates"] == 0
    assert contract.payload["execution"]["fresh_validation_read_allowed"] is False


def test_source_components_resolve_all_continuous_leaves_into_disjoint_masks() -> None:
    leaves = []
    for index, (_name, component) in enumerate(SOURCE_COMPONENTS):
        leaves.append(_leaf(component, "continuous", index * 2, index * 2 + 2))
        leaves.append(_leaf(component, "flag", 0, 0, discrete=True))
    resolved = resolve_component_state_masks(SimpleNamespace(state_leaves=tuple(leaves), continuous_state_width=12))
    assert tuple(resolved) == tuple(name for name, _component in SOURCE_COMPONENTS)
    for index, (name, component) in enumerate(SOURCE_COMPONENTS):
        record = resolved[name]
        assert record["source_component"] == component
        assert record["state_coordinate_count"] == 2
        expected = np.zeros((12,), dtype=np.bool_)
        expected[index * 2 : index * 2 + 2] = True
        np.testing.assert_array_equal(record["mask"], expected)


def test_classifier_selects_only_components_with_repeated_material_capture() -> None:
    contract = load_structured_gpp_state_feedback_attribution_contract()
    component_groups = {
        name: _groups(0.6 if name == "state_canopy" else 0.96) for name, _component in SOURCE_COMPONENTS
    }
    result = classify_component_feedback(
        _groups(1.0),
        _groups(0.2),
        component_groups,
        contract,
    )
    assert result["selected_components"] == ["state_canopy"]
    assert result["classification"] == "material_source_components_identified"
    assert np.isclose(
        result["components"]["state_canopy"]["horizons"]["7"]["median_capture_fraction"],
        0.5,
    )
    assert result["components"]["state_water"]["material_at_either_horizon"] is False


def test_runner_uses_one_dynamic_mask_graph_and_no_paid_or_fresh_path() -> None:
    runner = (ROOT / "scripts" / "hpc" / "audit_target_oriented_structured_gpp_state_feedback.py").read_text(
        encoding="utf-8"
    )
    launcher = (ROOT / "scripts" / "hpc" / "run_target_oriented_structured_gpp_state_feedback.sh").read_text(
        encoding="utf-8"
    )
    shared = (ROOT / "scripts" / "hpc" / "run_target_oriented_full_step_preflight.py").read_text(encoding="utf-8")
    assert "diagnostic_teacher_carry_state_mask=True" in runner
    assert "masked_teacher_carry_acceptance_forward" in runner
    assert "fresh-validation-prepared-manifest" not in runner
    assert "write_structured_gpp_checkpoint" not in runner
    assert "sbatch" not in launcher
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "diagnostic_teacher_carry_state_mask: bool = False" in shared
    assert '"seven_masked_teacher_carry_acceptance_forward"' in shared
