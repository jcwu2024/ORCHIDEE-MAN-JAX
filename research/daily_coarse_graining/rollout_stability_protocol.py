"""Audit the frozen post-architecture rollout-stability experiment protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.daily_coarse_graining.canonical_architecture_ab_run import (
    load_architecture_ab_experiment,
)
from research.daily_coarse_graining.production_training_protocol import (
    load_training_protocol,
)
from research.daily_coarse_graining.teacher_production import ROOT

SCHEMA_VERSION = "canonical_rollout_stability_protocol_v1"
REQUIRED_PARENT_DECISION = "advance_axis_process_to_rollout_stability_experiment"

_TOP_LEVEL_KEYS = {
    "schema_version",
    "protocol_id",
    "status",
    "parent_architecture_screen",
    "artifacts",
    "data_policy",
    "comparison",
    "optimization",
    "mixed_horizon_sampling",
    "loss",
    "science_inventory",
    "hard_constraints",
    "screening_validation",
    "post_screening_promotion",
    "stop_rules",
    "execution_manifest_requirements",
}
_EXPECTED_ARMS = [
    {
        "id": "one_step_continuation_control",
        "objective": "canonical_one_step_v1",
    },
    {
        "id": "mixed_horizon_stability_v1",
        "objective": "fast_next_rollout_bias_science_v1",
    },
]
_EXPECTED_HORIZONS = [
    {"days": 1, "probability": 0.4, "batch_size": 256, "rematerialize": False},
    {"days": 3, "probability": 0.3, "batch_size": 64, "rematerialize": False},
    {"days": 7, "probability": 0.2, "batch_size": 32, "rematerialize": False},
    {"days": 30, "probability": 0.1, "batch_size": 8, "rematerialize": True},
]
_EXPECTED_LOSS_COMPONENTS = [
    {
        "id": "L_fast",
        "definition": "complete_normalized_fast_day_boundary_masked_huber",
        "coefficient": 1.0,
        "present_in_both_arms": True,
    },
    {
        "id": "L_next",
        "definition": "complete_normalized_next_state_after_differentiable_retained_tail",
        "coefficient_source": "train_only_gradient_calibration",
        "present_in_both_arms": False,
    },
    {
        "id": "L_rollout",
        "definition": "free_rollout_state_error_from_day_2_through_sampled_horizon",
        "coefficient_source": "train_only_gradient_calibration",
        "present_in_both_arms": False,
    },
    {
        "id": "L_bias",
        "definition": "batch_time_mean_normalized_tendency_error_by_eight_process_groups",
        "coefficient_source": "train_only_gradient_calibration",
        "present_in_both_arms": False,
    },
    {
        "id": "L_science",
        "definition": "named_flux_stock_and_tendency_objective",
        "coefficient_source": "train_only_gradient_calibration",
        "present_in_both_arms": False,
    },
]
_EXPECTED_SCIENCE_FIELDS = {
    "daily_fluxes": [
        "gpp_daily",
        "npp_daily",
        "resp_growth",
        "resp_maint",
        "resp_hetero",
    ],
    "vegetation_stocks": ["biomass", "lai", "height"],
    "litter_and_soil_carbon": ["litter", "carbon_32l", "DOC", "deepC_peat"],
}
_EXPECTED_PROMOTION_ORDER = [
    "three_seed_confirmation",
    "annual_365_day",
    "complete_1961_2007_validation_chains",
    "sealed_1961_2010_test",
]
_EXPECTED_EXECUTION_REQUIREMENTS = [
    "parent_architecture_ab_report_sha256",
    "selected_parent_checkpoint_sha256",
    "resolved_model_architecture_identity",
    "resolved_loss_coefficients_and_calibration_asset_sha256",
    "dataset_statistics_acceptance_and_training_protocol_sha256",
    "optimizer_update_budget_and_seed",
    "environment_lock_sha256",
    "output_root",
]


@dataclass(frozen=True)
class RolloutStabilityProtocol:
    path: Path
    raw: Mapping[str, Any]
    sha256: str


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} key inventory drift")


def _validate_parent_and_artifacts(raw: Mapping[str, Any]) -> None:
    parent = raw["parent_architecture_screen"]
    _require_exact_keys(
        parent,
        {
            "experiment_manifest",
            "experiment_canonical_sha256",
            "required_report_status",
            "required_classification_status",
            "required_decision",
            "report_and_checkpoint_sha256_resolved_in_execution_manifest",
        },
        context="parent architecture screen",
    )
    if parent["required_report_status"] != "completed":
        raise ValueError("parent architecture report must be completed")
    if parent["required_classification_status"] != "passed":
        raise ValueError("parent architecture classification must pass")
    if parent["required_decision"] != REQUIRED_PARENT_DECISION:
        raise ValueError("parent architecture decision drift")
    if parent["report_and_checkpoint_sha256_resolved_in_execution_manifest"] is not True:
        raise ValueError("parent report and checkpoint identities must be resolved")

    experiment = load_architecture_ab_experiment(
        _resolve_repo_path(parent["experiment_manifest"])
    )
    if experiment.sha256 != parent["experiment_canonical_sha256"]:
        raise ValueError("parent architecture experiment hash drift")

    artifacts = raw["artifacts"]
    _require_exact_keys(
        artifacts,
        {
            "dataset_manifest_sha256",
            "training_statistics_sha256",
            "acceptance_report_sha256",
            "training_protocol",
            "training_protocol_sha256",
        },
        context="rollout stability artifact",
    )
    for name in (
        "dataset_manifest_sha256",
        "training_statistics_sha256",
        "acceptance_report_sha256",
        "training_protocol_sha256",
    ):
        if artifacts[name] != experiment.raw["artifacts"][name]:
            raise ValueError(f"rollout stability {name} differs from parent experiment")
    training_protocol = load_training_protocol(
        _resolve_repo_path(artifacts["training_protocol"])
    )
    if training_protocol.sha256 != artifacts["training_protocol_sha256"]:
        raise ValueError("rollout stability training protocol hash drift")


def _validate_data_and_comparison(raw: Mapping[str, Any]) -> None:
    data = raw["data_policy"]
    _require_exact_keys(
        data,
        {
            "coefficient_calibration",
            "training",
            "model_selection_slices",
            "sealed_test_used",
            "window_sampling",
            "cross_year_or_restart_windows",
            "landpoint_identity_as_feature",
        },
        context="rollout stability data policy",
    )
    train_only = {"spatial_split": "train", "temporal_split": "train"}
    if data["coefficient_calibration"] != train_only or data["training"] != train_only:
        raise ValueError("training and coefficient calibration must use train/train only")
    if data["model_selection_slices"] != ["temporal", "spatial", "joint"]:
        raise ValueError("rollout stability model-selection slice drift")
    if data["sealed_test_used"] is not False:
        raise ValueError("rollout stability test split must remain sealed")
    if data["window_sampling"] != "balanced_landpoint_year_start_day_v1":
        raise ValueError("rollout stability window sampler drift")
    if data["cross_year_or_restart_windows"] is not False:
        raise ValueError("screening windows must not cross year or restart boundaries")
    if data["landpoint_identity_as_feature"] is not False:
        raise ValueError("landpoint identity must not be a model feature")

    comparison = raw["comparison"]
    _require_exact_keys(
        comparison,
        {
            "initialization",
            "arms",
            "same_anchor_batches",
            "same_optimizer_updates",
            "same_seed",
            "extra_rollout_windows_are_the_treatment",
        },
        context="rollout stability comparison",
    )
    if comparison["initialization"] != "same_parent_best_checkpoint_and_optimizer":
        raise ValueError("rollout stability parent initialization drift")
    if comparison["arms"] != _EXPECTED_ARMS:
        raise ValueError("rollout stability arm inventory or order drift")
    for name in (
        "same_anchor_batches",
        "same_optimizer_updates",
        "same_seed",
        "extra_rollout_windows_are_the_treatment",
    ):
        if comparison[name] is not True:
            raise ValueError(f"rollout stability comparison must require {name}")


def _validate_optimization_and_horizons(raw: Mapping[str, Any]) -> None:
    optimization = raw["optimization"]
    _require_exact_keys(
        optimization,
        {
            "optimizer",
            "learning_rate",
            "screening_updates",
            "anchor_batch_size",
            "checkpoint_every_updates",
            "progress_every_updates",
            "screening_seed",
            "confirmation_seeds_after_pass",
            "confirmation_runs_forbidden_before_screening_pass",
        },
        context="rollout stability optimization",
    )
    if optimization["optimizer"] != "adam" or float(optimization["learning_rate"]) <= 0:
        raise ValueError("rollout stability optimizer configuration drift")
    for name in (
        "screening_updates",
        "anchor_batch_size",
        "checkpoint_every_updates",
        "progress_every_updates",
    ):
        if int(optimization[name]) < 1:
            raise ValueError(f"rollout stability {name} must be positive")
    confirmation_seeds = optimization["confirmation_seeds_after_pass"]
    if len(confirmation_seeds) != 3 or len(set(confirmation_seeds)) != 3:
        raise ValueError("rollout stability requires three distinct confirmation seeds")
    if optimization["confirmation_runs_forbidden_before_screening_pass"] is not True:
        raise ValueError("confirmation runs must wait for a screening pass")

    sampling = raw["mixed_horizon_sampling"]
    _require_exact_keys(
        sampling,
        {
            "one_step_anchor_on_every_update",
            "sample_one_rollout_window_on_every_candidate_update",
            "horizons",
            "loss_reduction",
            "horizon_choice_seeded_and_checkpointed",
        },
        context="mixed-horizon sampling",
    )
    if sampling["horizons"] != _EXPECTED_HORIZONS:
        raise ValueError("mixed-horizon inventory, probability, or batch-size drift")
    probability_sum = math.fsum(float(item["probability"]) for item in sampling["horizons"])
    if not math.isclose(probability_sum, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("mixed-horizon probabilities must sum to one")
    for name in (
        "one_step_anchor_on_every_update",
        "sample_one_rollout_window_on_every_candidate_update",
        "horizon_choice_seeded_and_checkpointed",
    ):
        if sampling[name] is not True:
            raise ValueError(f"mixed-horizon sampling must require {name}")
    if sampling["loss_reduction"] != "mean_over_valid_days_then_batch":
        raise ValueError("mixed-horizon loss reduction drift")


def _validate_loss_and_science(raw: Mapping[str, Any]) -> None:
    loss = raw["loss"]
    _require_exact_keys(
        loss,
        {
            "undefined_flip_binary_weight",
            "components",
            "coefficient_calibration",
            "state_weighting",
            "state_delta_scale",
            "complete_budget_penalties",
        },
        context="rollout stability loss",
    )
    if float(loss["undefined_flip_binary_weight"]) != 0.1:
        raise ValueError("undefined-flip binary weight drift")
    if loss["components"] != _EXPECTED_LOSS_COMPONENTS:
        raise ValueError("rollout stability loss component inventory or definition drift")

    calibration = loss["coefficient_calibration"]
    _require_exact_keys(
        calibration,
        {
            "method",
            "anchor_component",
            "fixed_calibration_seed",
            "calibration_batches_per_horizon",
            "target_gradient_norm_ratios",
            "coefficient_clip",
            "validation_data_forbidden",
            "coefficient_asset_frozen_before_training",
            "coefficient_search_after_validation_forbidden",
        },
        context="loss coefficient calibration",
    )
    if calibration["method"] != "train_only_median_gradient_norm_ratio_v1":
        raise ValueError("loss coefficient calibration method drift")
    if calibration["anchor_component"] != "L_fast":
        raise ValueError("loss coefficient anchor drift")
    if int(calibration["calibration_batches_per_horizon"]) != 32:
        raise ValueError("loss coefficient calibration sample budget drift")
    if calibration["target_gradient_norm_ratios"] != {
        "L_next": 1.0,
        "L_rollout": 1.0,
        "L_bias": 0.25,
        "L_science": 0.5,
    }:
        raise ValueError("loss coefficient target gradient ratios drift")
    if calibration["coefficient_clip"] != [0.001, 1000.0]:
        raise ValueError("loss coefficient clip drift")
    for name in (
        "validation_data_forbidden",
        "coefficient_asset_frozen_before_training",
        "coefficient_search_after_validation_forbidden",
    ):
        if calibration[name] is not True:
            raise ValueError(f"loss coefficient calibration must require {name}")
    if loss["state_weighting"] != "equal_total_weight_for_each_of_eight_source_process_groups":
        raise ValueError("rollout stability state weighting drift")
    if loss["state_delta_scale"] != "max_empirical_delta_scale_and_0.01_state_scale":
        raise ValueError("rollout stability state delta scale drift")
    budgets = loss["complete_budget_penalties"]
    if budgets != {
        "training_status": "disabled_until_complete_source_term_ledger",
        "reporting_status": "report_only",
        "invented_or_partial_conservation_identity_forbidden": True,
    }:
        raise ValueError("incomplete conservation penalties must remain report-only")

    science = raw["science_inventory"]
    _require_exact_keys(
        science,
        {
            "daily_fluxes",
            "vegetation_stocks",
            "litter_and_soil_carbon",
            "required_metrics",
            "low_biomass_cases_require_absolute_and_relative_metrics",
        },
        context="rollout stability science inventory",
    )
    for name, expected in _EXPECTED_SCIENCE_FIELDS.items():
        if science.get(name) != expected:
            raise ValueError(f"rollout stability science inventory drift for {name}")
    if science["required_metrics"] != [
        "normalized_rmse",
        "mean_bias",
        "absolute_error",
        "relative_error_where_denominator_is_scientifically_valid",
        "tendency_error",
        "multiannual_drift_slope",
    ]:
        raise ValueError("rollout stability required science metrics drift")
    if science.get("low_biomass_cases_require_absolute_and_relative_metrics") is not True:
        raise ValueError("low-biomass cases require absolute and relative metrics")


def _validate_gates_and_stop_rules(raw: Mapping[str, Any]) -> None:
    hard = raw["hard_constraints"]
    _require_exact_keys(
        hard,
        {
            "defined_status_mismatches",
            "discrete_state_mismatches",
            "nonfinite_defined_values",
            "negative_source_nonnegative_carbon_stocks",
            "restart_split_prediction",
            "hidden_cross_day_network_memory",
            "deterministic_mirrors_and_retained_tail_remain_source_backed",
        },
        context="rollout stability hard constraint",
    )
    for name in (
        "defined_status_mismatches",
        "discrete_state_mismatches",
        "nonfinite_defined_values",
        "negative_source_nonnegative_carbon_stocks",
    ):
        if hard.get(name) != 0:
            raise ValueError(f"hard constraint {name} must be zero")
    if hard.get("restart_split_prediction") != "exact":
        raise ValueError("restart-split prediction must be exact")
    if hard.get("hidden_cross_day_network_memory") is not False:
        raise ValueError("hidden cross-day network memory is forbidden")
    if hard.get("deterministic_mirrors_and_retained_tail_remain_source_backed") is not True:
        raise ValueError("deterministic mirrors and retained tail must remain source-backed")

    screening = raw["screening_validation"]
    _require_exact_keys(
        screening,
        {
            "horizons_days",
            "teacher_forced_and_free_rollout",
            "all_model_selection_samples",
            "required_slices",
            "gates_relative_to_one_step_continuation_control",
            "all_hard_constraints_required",
        },
        context="rollout stability screening validation",
    )
    if screening.get("horizons_days") != [1, 7, 30]:
        raise ValueError("screening validation horizon inventory drift")
    if screening.get("required_slices") != ["temporal", "spatial", "joint"]:
        raise ValueError("screening validation slice inventory drift")
    if screening.get("teacher_forced_and_free_rollout") is not True:
        raise ValueError("screening requires teacher-forced and free-rollout validation")
    if screening.get("all_model_selection_samples") is not True:
        raise ValueError("screening must evaluate all model-selection samples")
    if screening.get("all_hard_constraints_required") is not True:
        raise ValueError("all screening hard constraints are required")
    gates = screening["gates_relative_to_one_step_continuation_control"]
    expected_gate_keys = {
        "one_step_global_ratio_max",
        "one_step_family_ratio_max",
        "seven_day_free_rollout_global_ratio_max",
        "thirty_day_free_rollout_global_ratio_max",
        "tendency_bias_ratio_max",
        "science_metric_ratio_max",
    }
    _require_exact_keys(gates, expected_gate_keys, context="screening gate")
    if any(float(value) <= 0 for value in gates.values()):
        raise ValueError("screening gate ratios must be positive")

    promotion = raw["post_screening_promotion"]
    if [gate.get("id") for gate in promotion] != _EXPECTED_PROMOTION_ORDER:
        raise ValueError("post-screening promotion order drift")
    expected_promotion_keys = [
        {"id", "required_before_next_gate"},
        {
            "id",
            "free_rollout",
            "restart_split_equivalence",
            "required_slices",
            "annual_integrated_flux_median_relative_bias_max",
            "annual_integrated_flux_p95_relative_bias_max",
            "year_end_stock_median_relative_error_max",
            "year_end_stock_p95_relative_error_max",
        },
        {
            "id",
            "free_rollout",
            "model_and_hyperparameters_frozen_after_pass",
        },
        {"id", "run_once_after_freeze", "model_selection_forbidden"},
    ]
    for gate, expected_keys in zip(promotion, expected_promotion_keys, strict=True):
        _require_exact_keys(gate, expected_keys, context=f"promotion gate {gate['id']}")
    if promotion[0]["required_before_next_gate"] is not True:
        raise ValueError("three-seed confirmation must precede annual promotion")
    annual = promotion[1]
    if annual["free_rollout"] is not True or annual["restart_split_equivalence"] is not True:
        raise ValueError("annual promotion requires free rollout and restart equivalence")
    if annual["required_slices"] != ["temporal", "spatial", "joint"]:
        raise ValueError("annual promotion validation slice inventory drift")
    if promotion[2]["free_rollout"] is not True:
        raise ValueError("complete validation chains must use free rollout")
    if promotion[2]["model_and_hyperparameters_frozen_after_pass"] is not True:
        raise ValueError("model and hyperparameters must freeze after chain promotion")
    if promotion[-1].get("run_once_after_freeze") is not True:
        raise ValueError("sealed final test must run once after model freeze")
    if promotion[-1].get("model_selection_forbidden") is not True:
        raise ValueError("sealed final test must not be used for model selection")

    stop = raw["stop_rules"]
    _require_exact_keys(
        stop,
        {
            "parent_architecture_screen_rejected",
            "screening_gate_failed",
            "open_ended_weight_search",
            "inspect_sealed_test_after_failure",
            "train_through_365_days_or_50_years",
        },
        context="rollout stability stop rule",
    )
    if stop.get("parent_architecture_screen_rejected") != "do_not_run_experiment_b":
        raise ValueError("rejected parent architecture must stop Experiment B")
    if stop.get("open_ended_weight_search") is not False:
        raise ValueError("open-ended coefficient search is forbidden")
    if stop.get("inspect_sealed_test_after_failure") is not False:
        raise ValueError("sealed test inspection after failure is forbidden")
    if stop.get("train_through_365_days_or_50_years") is not False:
        raise ValueError("365-day or 50-year backpropagation is forbidden")
    if raw["execution_manifest_requirements"] != _EXPECTED_EXECUTION_REQUIREMENTS:
        raise ValueError("rollout stability execution-manifest requirements drift")


def load_rollout_stability_protocol(
    path: str | Path,
) -> RolloutStabilityProtocol:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    _require_exact_keys(raw, _TOP_LEVEL_KEYS, context="rollout stability protocol")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"rollout stability protocol schema must be {SCHEMA_VERSION!r}")
    if raw.get("protocol_id") != "canonical-669-rollout-stability-v1":
        raise ValueError("rollout stability protocol identity drift")
    if raw.get("status") != "frozen_before_parent_result":
        raise ValueError("rollout stability protocol was not frozen before parent result")

    _validate_parent_and_artifacts(raw)
    _validate_data_and_comparison(raw)
    _validate_optimization_and_horizons(raw)
    _validate_loss_and_science(raw)
    _validate_gates_and_stop_rules(raw)
    return RolloutStabilityProtocol(path=path, raw=raw, sha256=_canonical_sha256(raw))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol = load_rollout_stability_protocol(args.protocol)
    print(
        json.dumps(
            {
                "status": "passed",
                "protocol_id": protocol.raw["protocol_id"],
                "protocol_sha256": protocol.sha256,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
