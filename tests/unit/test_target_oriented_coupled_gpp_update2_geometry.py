from __future__ import annotations

from pathlib import Path

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    SEVEN_SOURCE,
    STATE_SOURCE,
    THIRTY_SOURCE,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_update2_geometry import (
    DISPLACEMENT_SCALES,
    FROZEN_CONTRACT_SHA256,
    UPDATE_TWO_ARTIFACTS,
    classify_update2_geometry,
    load_coupled_gpp_update2_geometry_contract,
)


def _cosines(value: float) -> dict[str, float]:
    return {
        DIRECT_SOURCE: value,
        STATE_SOURCE: value,
        SEVEN_SOURCE: value,
        THIRTY_SOURCE: value,
    }


def test_update2_geometry_contract_is_frozen_and_bounded():
    contract = load_coupled_gpp_update2_geometry_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.scales == DISPLACEMENT_SCALES
    assert tuple(contract.payload["data"]["artifact_updates"]) == UPDATE_TWO_ARTIFACTS
    assert contract.payload["retention"] == {
        "optimizer_updates_persisted": 0,
        "parameter_proposals_persisted": 0,
        "recovery_modified": False,
        "checkpoint_written": False,
        "training_continuation_authorized": False,
    }


def test_update2_geometry_classification_is_exclusive():
    positive = _cosines(0.1)
    negative = _cosines(-0.1)
    assert (
        classify_update2_geometry(
            inherited_cosines=positive,
            reset_cosines=positive,
            inherited_accepted_scales=(0.5,),
            reset_accepted_scales=(),
        )
        == "inherited_backtracking_admissible"
    )
    assert (
        classify_update2_geometry(
            inherited_cosines=negative,
            reset_cosines=positive,
            inherited_accepted_scales=(),
            reset_accepted_scales=(0.25,),
        )
        == "moment_reset_admissible"
    )
    assert (
        classify_update2_geometry(
            inherited_cosines=negative,
            reset_cosines=positive,
            inherited_accepted_scales=(),
            reset_accepted_scales=(),
        )
        == "inherited_moment_rotation"
    )
    assert (
        classify_update2_geometry(
            inherited_cosines=positive,
            reset_cosines=negative,
            inherited_accepted_scales=(),
            reset_accepted_scales=(),
        )
        == "common_direction_no_finite_scale"
    )
    assert (
        classify_update2_geometry(
            inherited_cosines=negative,
            reset_cosines=negative,
            inherited_accepted_scales=(),
            reset_accepted_scales=(),
        )
        == "no_common_optimizer_direction"
    )


def test_update2_geometry_launcher_is_free_and_zero_retention():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/run_target_oriented_coupled_gpp_update2_geometry.py").read_text(encoding="utf-8")
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_update2_geometry.sh").read_text(encoding="utf-8")
    assert "return_internal=True" in runner
    assert "write_coupled_gpp_recovery" not in runner
    assert "write_coupled_gpp_checkpoint" not in runner
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "PARENT_DIR" in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-120m}" in launcher
