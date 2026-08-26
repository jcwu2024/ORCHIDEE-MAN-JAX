from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_markov_contract import StateLeafSpec
from research.daily_coarse_graining.daily_operator.target_oriented_snow_thermal_recurrence import (
    FROZEN_CONTRACT_SHA256,
    SNOW_THERMAL_STATE_KEYS,
    WINDOW_NAMES,
    classify_snow_thermal_recurrence,
    load_target_oriented_snow_thermal_recurrence_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_snowheat_recurrence import (
    replace_state_leaves_with_teacher,
    resolve_continuous_state_leaves,
)


def _leaf(key: str, start: int, stop: int) -> StateLeafSpec:
    component, field = key.split(".")
    return StateLeafSpec(
        component=component,
        path=(field,),
        shape=(stop - start,),
        full_shape=(1, stop - start),
        dtype="float64",
        start=start,
        stop=stop,
        classification="continuous",
        source_ref="explicit_snow",
        axis_names=("npts", "nsnow"),
    )


def _records(gradient_reduction: float, snowtemp_reduction: float):
    return {
        "thirty_day_extreme": {
            "parent_to_treatment_gradient_reduction_ratio": gradient_reduction,
            "parent_to_treatment_snowtemp_head_reduction_ratio": snowtemp_reduction,
        },
        "thirty_day_matched_control": {
            "parent_to_treatment_gradient_reduction_ratio": 1.0,
            "parent_to_treatment_snowtemp_head_reduction_ratio": 1.0,
        },
    }


def test_frozen_contract_binds_parent_and_two_thirty_day_windows() -> None:
    contract = load_target_oriented_snow_thermal_recurrence_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert tuple(record["name"] for record in contract.windows) == WINDOW_NAMES
    assert tuple(contract.payload["intervention"]["state_keys"]) == SNOW_THERMAL_STATE_KEYS
    assert contract.payload["measurement"]["gradient_count"] == 2


def test_two_nonoverlapping_state_leaves_are_resolved_and_replaced() -> None:
    leaves = (_leaf(SNOW_THERMAL_STATE_KEYS[0], 2, 5), _leaf(SNOW_THERMAL_STATE_KEYS[1], 7, 10))
    resolved = resolve_continuous_state_leaves(
        SimpleNamespace(state_leaves=leaves),
        SNOW_THERMAL_STATE_KEYS,
    )
    assert resolved == leaves
    predicted = jnp.arange(12.0, dtype=jnp.float64)
    teacher = jnp.arange(100.0, 112.0, dtype=jnp.float64)
    treated = np.asarray(replace_state_leaves_with_teacher(predicted, teacher, resolved))
    np.testing.assert_array_equal(treated[2:5], np.asarray(teacher)[2:5])
    np.testing.assert_array_equal(treated[7:10], np.asarray(teacher)[7:10])
    np.testing.assert_array_equal(treated[[0, 1, 5, 6, 10, 11]], np.asarray(predicted)[[0, 1, 5, 6, 10, 11]])


def test_classifier_separates_causal_mixed_and_no_effect() -> None:
    contract = load_target_oriented_snow_thermal_recurrence_contract()
    causal = classify_snow_thermal_recurrence(_records(20.0, 12.0), contract)
    assert causal["classification"] == "coupled_snow_thermal_recurrence_causal"
    mixed = classify_snow_thermal_recurrence(_records(5.0, 12.0), contract)
    assert mixed["classification"] == "coupled_snow_thermal_recurrence_mixed"
    absent = classify_snow_thermal_recurrence(_records(1.5, 20.0), contract)
    assert absent["classification"] == "coupled_snow_thermal_recurrence_not_causal"


def test_launcher_is_free_gpu_two_gradient_boundary() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts" / "hpc" / "run_target_oriented_snow_thermal_recurrence_intervention.sh").read_text(
        encoding="utf-8"
    )
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "audit_target_oriented_snow_thermal_recurrence" in launcher
    assert "target-oriented-snowheat-recurrence-159f672/report.json" in launcher
    source = (root / "scripts" / "hpc" / "run_target_oriented_full_step_preflight.py").read_text(encoding="utf-8")
    assert "diagnostic_teacher_carry_state_keys: tuple[str, ...] | None = None" in source
    assert '"thirty_teacher_carry_flux_field_gradient"' in source
