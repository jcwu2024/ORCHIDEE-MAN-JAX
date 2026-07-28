"""Matched architecture-only A/B on the admitted 669-point Teacher dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.canonical_training_run import (
    train_experiment,
    verify_training_acceptance,
)
from research.daily_coarse_graining.daily_model_architecture import (
    AXIS_PROCESS_COUPLED_V1,
    CANONICAL_FLAT_V1,
)
from research.daily_coarse_graining.production_training_protocol import (
    load_training_protocol,
)

SCHEMA_VERSION = "canonical_architecture_ab_experiment_v1"
REPORT_SCHEMA_VERSION = "canonical_architecture_ab_report_v1"


@dataclass(frozen=True)
class ArchitectureABExperiment:
    path: Path
    raw: Mapping[str, Any]
    sha256: str


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_architecture_ab_experiment(
    path: str | Path,
) -> ArchitectureABExperiment:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"architecture A/B schema must be {SCHEMA_VERSION!r}")
    arms = raw.get("arms")
    if arms != [
        {"id": "flat", "model_architecture": CANONICAL_FLAT_V1},
        {"id": "axis_process", "model_architecture": AXIS_PROCESS_COUPLED_V1},
    ]:
        raise ValueError("architecture A/B arm inventory or order drift")
    training = raw["training"]
    if training.get("objective") != "canonical_one_step_v1":
        raise ValueError("first architecture A/B must use the canonical one-step objective")
    if int(training["epochs"]) < 1 or int(training["batch_size"]) < 1:
        raise ValueError("architecture A/B epochs and batch size must be positive")
    if float(training["learning_rate"]) <= 0.0:
        raise ValueError("architecture A/B learning rate must be positive")
    if int(training["prefetch_batches"]) < 1:
        raise ValueError("architecture A/B prefetch must be positive")
    if training.get("initialization") != "from_scratch_same_seed":
        raise ValueError("architecture A/B must initialize both arms from scratch")
    if training.get("sampling") != "exact_once_hierarchical_shard_pool_v1":
        raise ValueError("architecture A/B must use the frozen production sampler")
    validation = raw["validation"]
    if validation.get("sealed_test_used") is not False:
        raise ValueError("architecture screening must keep the test split sealed")
    if validation.get("model_selection_slices") != [
        "temporal",
        "spatial",
        "joint",
    ]:
        raise ValueError("architecture A/B validation slice inventory drift")
    gates = raw["screening_gates"]
    required_gates = {
        "parameter_ratio_min",
        "parameter_ratio_max",
        "temporal_score_ratio_max",
        "spatial_score_ratio_max",
        "joint_score_ratio_max",
        "family_score_ratio_max",
    }
    if set(gates) != required_gates:
        raise ValueError("architecture A/B screening gate inventory drift")
    if not 0.0 < float(gates["parameter_ratio_min"]) <= 1.0:
        raise ValueError("invalid minimum parameter ratio")
    if float(gates["parameter_ratio_max"]) < 1.0:
        raise ValueError("invalid maximum parameter ratio")
    return ArchitectureABExperiment(path, raw, _canonical_sha256(raw))


def _mean_family_score(report: Mapping[str, Any], split: str) -> float:
    best_epoch = int(report["best_epoch"])
    records = [
        record for record in report["history"] if int(record["epoch"]) == best_epoch
    ]
    if len(records) != 1:
        raise ValueError(f"training report has no unique best epoch {best_epoch}")
    families = records[0]["validation"][split]["families"]
    values = np.asarray(
        [float(value["normalized_rmse"]) for value in families.values()],
        dtype=np.float64,
    )
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"architecture report has invalid {split} family metrics")
    return float(np.mean(values))


def _family_ratios(
    flat: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, float]:
    result = {}
    flat_record = flat["history"][int(flat["best_epoch"]) - 1]["validation"]
    candidate_record = candidate["history"][
        int(candidate["best_epoch"]) - 1
    ]["validation"]
    for split in ("temporal", "spatial", "joint"):
        flat_families = flat_record[split]["families"]
        candidate_families = candidate_record[split]["families"]
        if set(flat_families) != set(candidate_families):
            raise ValueError(f"architecture A/B {split} family inventory drift")
        for family in sorted(flat_families):
            denominator = float(flat_families[family]["normalized_rmse"])
            numerator = float(candidate_families[family]["normalized_rmse"])
            if denominator <= 0.0 or not np.isfinite(denominator + numerator):
                raise ValueError(f"architecture A/B invalid family metric {split}/{family}")
            result[f"{split}/{family}"] = numerator / denominator
    return result


def classify_architecture_ab(
    flat: Mapping[str, Any],
    candidate: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> Mapping[str, Any]:
    parameter_ratio = float(candidate["parameter_count"]) / float(
        flat["parameter_count"]
    )
    split_ratios = {
        split: _mean_family_score(candidate, split)
        / _mean_family_score(flat, split)
        for split in ("temporal", "spatial", "joint")
    }
    family_ratios = _family_ratios(flat, candidate)
    checks = {
        "parameter_budget": (
            float(gates["parameter_ratio_min"])
            <= parameter_ratio
            <= float(gates["parameter_ratio_max"])
        ),
        "temporal_no_material_regression": (
            split_ratios["temporal"]
            <= float(gates["temporal_score_ratio_max"])
        ),
        "spatial_improvement": (
            split_ratios["spatial"] <= float(gates["spatial_score_ratio_max"])
        ),
        "joint_improvement": (
            split_ratios["joint"] <= float(gates["joint_score_ratio_max"])
        ),
        "no_family_regression": (
            max(family_ratios.values())
            <= float(gates["family_score_ratio_max"])
        ),
        "test_split_sealed": (
            flat.get("test_split_evaluated") is False
            and candidate.get("test_split_evaluated") is False
        ),
    }
    passed = all(checks.values())
    return {
        "status": "passed" if passed else "rejected",
        "decision": (
            "advance_axis_process_to_rollout_stability_experiment"
            if passed
            else "retain_flat_control_and_revise_architecture_hypothesis"
        ),
        "checks": checks,
        "parameter_ratio": parameter_ratio,
        "split_score_ratios": split_ratios,
        "family_score_ratios": family_ratios,
        "maximum_family_score_ratio": max(family_ratios.values()),
    }


def run_architecture_ab(
    *,
    experiment_path: str | Path,
    dataset_path: str | Path,
    statistics_path: str | Path,
    acceptance_path: str | Path,
    protocol_path: str | Path,
    output_root: str | Path,
) -> Path:
    experiment = load_architecture_ab_experiment(experiment_path)
    paths = {
        "dataset_manifest": Path(dataset_path).resolve(),
        "training_statistics": Path(statistics_path).resolve(),
        "acceptance_report": Path(acceptance_path).resolve(),
    }
    for name, path in paths.items():
        expected = experiment.raw["artifacts"][f"{name}_sha256"]
        if _sha256_file(path) != expected:
            raise ValueError(f"architecture A/B {name} hash drift")
    protocol = load_training_protocol(Path(protocol_path).resolve())
    if protocol.sha256 != experiment.raw["artifacts"]["training_protocol_sha256"]:
        raise ValueError("architecture A/B training protocol hash drift")
    accepted = verify_training_acceptance(
        paths["acceptance_report"],
        paths["dataset_manifest"],
        paths["training_statistics"],
    )
    training = experiment.raw["training"]
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports = {}
    for arm in experiment.raw["arms"]:
        arm_dir = output_root / str(arm["id"])
        reports[str(arm["id"])] = train_experiment(
            paths["dataset_manifest"],
            paths["training_statistics"],
            arm_dir,
            epochs=int(training["epochs"]),
            batch_size=int(training["batch_size"]),
            learning_rate=float(training["learning_rate"]),
            seed=int(training["seed"]),
            resume=True,
            max_eval_batches=None,
            acceptance_path=paths["acceptance_report"],
            model_architecture=str(arm["model_architecture"]),
            training_protocol_path=protocol.path,
            prefetch_batches=int(training["prefetch_batches"]),
            verified_acceptance=accepted,
        )
    expected_samples = int(
        accepted["training_statistics"]["sample_count"]
    )
    for arm, report in reports.items():
        observed = [int(item["training_samples"]) for item in report["history"]]
        if observed != [expected_samples] * int(training["epochs"]):
            raise ValueError(f"architecture A/B {arm} did not consume exact epochs")
    classification = classify_architecture_ab(
        reports["flat"],
        reports["axis_process"],
        experiment.raw["screening_gates"],
    )
    result = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "completed",
        "experiment": {
            "path": str(experiment.path),
            "sha256": experiment.sha256,
            "id": experiment.raw["experiment_id"],
        },
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in paths.items()
        }
        | {
            "training_protocol": {
                "path": str(protocol.path),
                "sha256": protocol.sha256,
            }
        },
        "training": dict(training),
        "arms": {
            arm: {
                "model_architecture": report["model_architecture"],
                "parameter_count": report["parameter_count"],
                "best_epoch": report["best_epoch"],
                "best_validation_score": report["best_validation_score"],
                "training_report": str(output_root / arm / "training_report.json"),
                "training_report_sha256": _sha256_file(
                    output_root / arm / "training_report.json"
                ),
                "best_checkpoint": report["best_checkpoint"],
                "best_checkpoint_sha256": _sha256_file(
                    Path(report["best_checkpoint"])
                ),
                "test_split_evaluated": report["test_split_evaluated"],
            }
            for arm, report in reports.items()
        },
        "classification": classification,
    }
    return _atomic_json(output_root / "architecture_ab_report.json", result)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run_architecture_ab(
        experiment_path=args.experiment,
        dataset_path=args.dataset,
        statistics_path=args.statistics,
        acceptance_path=args.acceptance,
        protocol_path=args.protocol,
        output_root=args.output_root,
    )
    print(json.dumps({"status": "completed", "report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
