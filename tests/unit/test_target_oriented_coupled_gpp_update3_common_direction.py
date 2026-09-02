from __future__ import annotations

from pathlib import Path

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    GRADIENT_SOURCES,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_update3_common_direction import (
    DISPLACEMENT_SCALES,
    FROZEN_CONTRACT_SHA256,
    UPDATE_THREE_ARTIFACTS,
    UPDATE_TWO_ARTIFACTS,
    classify_update3_common_direction,
    load_coupled_gpp_update3_common_direction_contract,
)


def _cosines(value: float) -> dict[str, float]:
    return {name: value for name in GRADIENT_SOURCES}


def test_update3_common_direction_contract_is_frozen_and_bounded():
    contract = load_coupled_gpp_update3_common_direction_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.scales == DISPLACEMENT_SCALES
    assert tuple(contract.payload["data"]["reconstruction_artifact_updates"]) == UPDATE_TWO_ARTIFACTS
    assert tuple(contract.payload["data"]["audit_artifact_updates"]) == UPDATE_THREE_ARTIFACTS
    assert contract.payload["retention"]["checkpoint_written"] is False
    assert contract.payload["execution"]["paid_partition_authorized"] is False


def test_update3_common_direction_classification_prefers_direct_geometry():
    positive = _cosines(0.1)
    negative = _cosines(-0.1)
    assert (
        classify_update3_common_direction(
            raw_source_cosines=negative,
            direct_accepted_scales=(),
            adamw_source_cosines=positive,
            adamw_accepted_scales=(1.0,),
        )
        == "no_common_first_order_direction"
    )
    assert (
        classify_update3_common_direction(
            raw_source_cosines=positive,
            direct_accepted_scales=(0.5,),
            adamw_source_cosines=positive,
            adamw_accepted_scales=(1.0,),
        )
        == "direct_common_finite_step"
    )
    assert (
        classify_update3_common_direction(
            raw_source_cosines=positive,
            direct_accepted_scales=(),
            adamw_source_cosines=positive,
            adamw_accepted_scales=(0.25,),
        )
        == "fresh_adamw_common_finite_step"
    )
    assert (
        classify_update3_common_direction(
            raw_source_cosines=positive,
            direct_accepted_scales=(),
            adamw_source_cosines=negative,
            adamw_accepted_scales=(0.125,),
        )
        == "adamw_finite_step_without_common_geometry"
    )
    assert (
        classify_update3_common_direction(
            raw_source_cosines=positive,
            direct_accepted_scales=(),
            adamw_source_cosines=positive,
            adamw_accepted_scales=(),
        )
        == "common_direction_no_finite_scale"
    )


def test_update3_common_direction_launcher_is_free_and_zero_retention():
    root = Path(__file__).resolve().parents[2]
    runner = (root / "scripts/hpc/run_target_oriented_coupled_gpp_update3_common_direction.py").read_text(
        encoding="utf-8"
    )
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_update3_common_direction.sh").read_text(
        encoding="utf-8"
    )
    assert "gradient_direction_diagnostics" in runner
    assert "return_internal=True" in runner
    assert "write_coupled_gpp_recovery" not in runner
    assert "write_coupled_gpp_checkpoint" not in runner
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-90m}" in launcher
