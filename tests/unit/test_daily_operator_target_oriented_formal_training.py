from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.long_horizon_objective import MaskedTrajectory
from research.daily_coarse_graining.daily_operator.target_oriented_checkpoint import (
    load_target_oriented_recovery,
    load_target_oriented_stage_checkpoint,
    write_target_oriented_best_checkpoint,
    write_target_oriented_recovery,
    write_target_oriented_stage_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
    load_target_oriented_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    TargetOrientedAcceptanceScales,
    build_target_oriented_validation_panel,
    finalize_target_oriented_acceptance_statistics,
    load_target_oriented_training_contract,
    ordered_target_oriented_validation_entries,
    target_oriented_acceptance_sufficient_statistics,
    verify_target_oriented_validation_panel,
)
from scripts.hpc.merge_target_oriented_training_batches import run as merge_prepared_chunks
from scripts.hpc.prepare_target_oriented_training_batches import _trees_exact
from scripts.hpc.run_target_oriented_training import _stage_bounds
from scripts.hpc.target_oriented_training_prepared import (
    PREPARED_ARTIFACT_SCHEMA,
    PREPARED_MANIFEST_SCHEMA,
    PreparedTargetOrientedUpdateArtifact,
    atomic_json,
    load_prepared_target_manifest,
    load_prepared_target_manifest_range,
    write_prepared_target_artifact,
)


def _trajectory(value: float, days: int = 30) -> MaskedTrajectory:
    return MaskedTrajectory(
        value=jnp.full((days, 1), value, dtype=jnp.float64),
        defined=jnp.ones((days, 1), dtype=jnp.bool_),
    )


def test_target_oriented_acceptance_reports_all_derived_science_keys():
    fluxes = {
        "gpp": _trajectory(5.0),
        "maintenance_respiration": _trajectory(1.0),
        "growth_respiration": _trajectory(0.5),
        "heterotrophic_respiration": _trajectory(0.75),
        "npp": _trajectory(999.0),
    }
    stocks = {
        "aboveground_biomass_carbon": _trajectory(10.0),
        "belowground_biomass_carbon": _trajectory(4.0),
        "total_live_biomass_carbon": _trajectory(14.0),
        "ecosystem_carbon": _trajectory(20.0),
    }
    scales = TargetOrientedAcceptanceScales(
        fluxes={
            "gpp": 1.0,
            "maintenance_respiration": 1.0,
            "growth_respiration": 1.0,
            "autotrophic_respiration": 1.0,
            "npp": 1.0,
            "heterotrophic_respiration": 1.0,
            "nep": 1.0,
            "nee": 1.0,
        },
        stocks={
            "aboveground_biomass_carbon": 1.0,
            "belowground_biomass_carbon": 1.0,
            "total_live_biomass_carbon": 1.0,
            "nonliving_ecosystem_carbon": 1.0,
            "ecosystem_carbon": 1.0,
        },
    )
    statistics = target_oriented_acceptance_sufficient_statistics(
        fluxes,
        fluxes,
        stocks,
        stocks,
        scales=scales,
        horizon=7,
    )
    statistics = statistics._replace(
        fluxes=dict(sorted(statistics.fluxes.items())),
        stocks=dict(sorted(statistics.stocks.items())),
    )
    report = finalize_target_oriented_acceptance_statistics(
        statistics,
        horizon=7,
        target_contract=load_target_oriented_contract(),
    )

    assert tuple(report["keys"]) == (
        "gpp",
        "maintenance_respiration",
        "growth_respiration",
        "autotrophic_respiration",
        "npp",
        "heterotrophic_respiration",
        "nep",
        "nee",
        "aboveground_biomass_carbon",
        "belowground_biomass_carbon",
        "total_live_biomass_carbon",
        "nonliving_ecosystem_carbon",
        "ecosystem_carbon",
    )
    assert report["all_key_gates_passed"] is True
    assert report["science_macro_normalized_rmse"] == 0.0


@dataclass(frozen=True)
class Parent:
    landpoint_id: str
    year: int
    sha256: str
    spatial_split: str
    temporal_split: str


@dataclass(frozen=True)
class Reference:
    parent: Parent
    sha256: str
    day_count: int = 364


class Index:
    def __init__(self, references):
        self.references = tuple(references)

    def select(self, *, spatial_split: str, temporal_split: str):
        return tuple(
            reference
            for reference in self.references
            if reference.parent.spatial_split == spatial_split
            and reference.parent.temporal_split == temporal_split
        )


def test_target_oriented_validation_panel_is_fixed_diverse_and_non_test():
    references = []
    for slice_id, (spatial, temporal) in {
        "temporal": ("train", "validation"),
        "spatial": ("validation", "train"),
        "joint": ("validation", "validation"),
    }.items():
        references.extend(
            Reference(
                parent=Parent(
                    landpoint_id=f"{slice_id}-point-{index:02d}",
                    year=2001 + index % 3,
                    sha256=f"{slice_id}-parent-{index:02d}",
                    spatial_split=spatial,
                    temporal_split=temporal,
                ),
                sha256=f"{slice_id}-typed-{index:02d}",
            )
            for index in range(20)
        )
    index = Index(references)
    training_contract = load_target_oriented_training_contract()
    target_contract = load_target_oriented_contract()

    first = build_target_oriented_validation_panel(index, training_contract, target_contract)
    second = build_target_oriented_validation_panel(index, training_contract, target_contract)

    assert first == second
    assert first["sealed_test_used"] is False
    assert len(ordered_target_oriented_validation_entries(first)) == 96
    for slice_id in ("temporal", "spatial", "joint"):
        for horizon in (7, 30):
            entries = first["slices"][slice_id]["horizons"][str(horizon)]
            assert len(entries) == 16
            assert len({entry["landpoint_id"] for entry in entries}) == 16
            assert all(entry["start_rows"][0] >= 1 for entry in entries)
    verify_target_oriented_validation_panel(first, index, training_contract, target_contract)


def test_target_oriented_recovery_and_stage_checkpoint_roundtrip(tmp_path):
    identity = {
        "run": "formal",
        "git_head": "a" * 40,
        "training_contract_sha256": "training",
        "target_contract_sha256": "target",
        "state_anchor_policy_sha256": "anchor",
        "validation_panel_sha256": "panel",
    }
    parameters = {"weight": np.asarray([1.0, 2.0], dtype=np.float32)}
    optimizer = {
        "step": np.asarray(2, dtype=np.int32),
        "first": np.asarray([0.1, 0.2], dtype=np.float32),
        "second": np.asarray([0.01, 0.04], dtype=np.float32),
    }
    validation = {
        "full_frozen_panel": True,
        "hard_failure_count": 0,
        "science_macro_normalized_rmse": 0.5,
    }
    records = [{"update": 1}, {"update": 2}]
    recovery_path = tmp_path / "recovery.pkl"
    write_target_oriented_recovery(
        recovery_path,
        identity=identity,
        parameters=parameters,
        optimizer=optimizer,
        completed_updates=2,
        parameter_count=2,
        clip_threshold=3.0,
        initial_validation=validation,
        update_records=records,
    )
    loaded = load_target_oriented_recovery(
        recovery_path,
        identity=identity,
        parameter_count=2,
        clip_threshold=3.0,
        maximum_updates=250,
    )
    assert loaded is not None
    assert loaded["completed_updates"] == 2
    np.testing.assert_array_equal(loaded["parameters"]["weight"], parameters["weight"])

    stage = write_target_oriented_stage_checkpoint(
        tmp_path / "stage.pkl",
        identity=identity,
        parameters=parameters,
        optimizer=optimizer,
        completed_updates=2,
        parameter_count=2,
        clip_threshold=3.0,
        validation=validation,
        update_records=records,
    )
    best = write_target_oriented_best_checkpoint(
        tmp_path / "best.pkl",
        identity=identity,
        parameters=parameters,
        completed_updates=2,
        parameter_count=2,
        validation=validation,
    )
    assert Path(stage["path"]).is_file()
    assert Path(best["path"]).is_file()
    loaded_stage = load_target_oriented_stage_checkpoint(
        stage["path"],
        expected_sha256=stage["sha256"],
        completed_updates=2,
        parameter_count=2,
        clip_threshold=3.0,
        training_contract_sha256="training",
        target_contract_sha256="target",
        state_anchor_policy_sha256="anchor",
        validation_panel_sha256="panel",
    )
    assert loaded_stage["completed_updates"] == 2
    np.testing.assert_array_equal(loaded_stage["optimizer"]["step"], np.asarray(2))


def test_retained_landpoint_reuse_requires_exact_tree_equality():
    import jax

    left = {"a": np.asarray([1.0, np.nan]), "b": (np.asarray(2),)}
    same = {"a": np.asarray([1.0, np.nan]), "b": (np.asarray(2),)}
    different = {"a": np.asarray([1.0, np.nan]), "b": (np.asarray(3),)}

    assert _trees_exact(jax, left, same)
    assert not _trees_exact(jax, left, different)


def test_target_oriented_training_launcher_uses_portable_timeout():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/hpc/run_target_oriented_training.sh").read_text(
        encoding="utf-8"
    )
    slurm_source = (
        root / "scripts/hpc/slurm_target_oriented_training_250.sh"
    ).read_text(encoding="utf-8")

    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-290m}" in source
    assert 'timeout --signal=TERM "$RUN_TIMEOUT"' in source
    assert "5h50m" not in source
    assert "target-oriented-prepared-250-9b82b5a/prepared_manifest.json" in slurm_source
    assert "target-oriented-validation-9b82b5a/prepared_manifest.json" in slurm_source

    preparation = (
        root / "scripts/hpc/prepare_target_oriented_training_batches.sh"
    ).read_text(encoding="utf-8")
    assert "PREPARE_ARGUMENTS=(" in preparation
    assert '"${PREPARE_ARGUMENTS[@]}"' in preparation
    assert "REUSE_ARGS=()" not in preparation
    assert 'exec >>"$LOG_PATH" 2>&1' in preparation
    assert 'BACKGROUND_CHILD=1 nohup bash "$0"' in preparation


def test_target_oriented_update_1000_launcher_is_exact_continuation():
    root = Path(__file__).resolve().parents[2]
    source = (
        root / "scripts/hpc/slurm_target_oriented_training_1000.sh"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --partition=gnall" in source
    assert "#SBATCH --cpus-per-task=4" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --mem=32G" in source
    assert "#SBATCH --time=10:00:00" in source
    assert "target-oriented-prepared-1000-a243029/prepared_manifest.json" in source
    assert "target-oriented-validation-9b82b5a/prepared_manifest.json" in source
    assert "target-oriented-training-formal-e363482/stage-update-0250.pkl" in source
    assert (
        "STAGE_CHECKPOINT_SHA256="
        "099d48c76277cafb04daf868c492200d0271fd6a4a2c6fa29ca6e4bdbe7989a5"
    ) in source
    assert "STAGE_START=250" in source
    assert "STAGE_END=1000" in source
    assert "PREPARED_UPDATES=1000" in source
    assert "RUN_TIMEOUT=580m" in source


def test_target_oriented_continuation_stage_requires_exact_checkpoint_pair():
    valid = SimpleNamespace(
        mode="formal",
        stage_start=250,
        stage_end=1000,
        prepared_updates=1000,
        stage_checkpoint=Path("stage-update-0250.pkl"),
        stage_checkpoint_sha256="0" * 64,
    )
    assert _stage_bounds(valid) == (250, 1000, 1000)
    admission = SimpleNamespace(**(vars(valid) | {"mode": "admission"}))
    assert _stage_bounds(admission) == (250, 1000, 251)

    invalid = SimpleNamespace(**(vars(valid) | {"stage_checkpoint_sha256": None}))
    with np.testing.assert_raises_regex(ValueError, "exact parent stage checkpoint"):
        _stage_bounds(invalid)


def test_continuation_admission_evaluates_a_matched_parent_baseline():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/hpc/run_target_oriented_training.py").read_text(
        encoding="utf-8"
    )
    assert 'if mode == "admission":' in source
    assert 'label=f"update_{stage_start}_admission_baseline"' in source


def test_disjoint_prepared_chunks_merge_into_full_manifest(tmp_path):
    root = tmp_path / "prepared"
    identity = {
        "source_git_head": "producer",
        "training_contract_sha256": "contract",
        "schedule_sha256": "schedule",
        "dataset_manifest_sha256": "dataset",
    }
    manifests = []
    for update in (1, 2):
        chunk = root / "chunks" / str(update)
        chunk.mkdir(parents=True)
        entry = {"update": update, "runtime_year": 2000 + update, "components": {}}
        artifact = PreparedTargetOrientedUpdateArtifact(
            PREPARED_ARTIFACT_SCHEMA,
            "contract",
            entry,
            "trace",
            {"value": np.asarray(update)},
        )
        record = {
            "update": update,
            "runtime_year": 2000 + update,
            "trace_signature": "trace",
            **write_prepared_target_artifact(chunk / f"update-{update:04d}.pkl.gz", artifact),
        }
        manifest = {
            "schema_version": PREPARED_MANIFEST_SCHEMA,
            "status": "passed",
            "identity": identity,
            "updates": 2,
            "start_update": update,
            "end_update": update,
            "entries": [record],
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
        }
        path = chunk / "prepared_manifest.json"
        atomic_json(path, manifest)
        manifests.append(path)

    report = merge_prepared_chunks(
        SimpleNamespace(output_dir=root, updates=2, chunk_manifests=manifests)
    )
    assert report["status"] == "passed"
    merged = load_prepared_target_manifest(
        root / "prepared_manifest.json",
        training_contract_sha256="contract",
        schedule_sha256="schedule",
        updates=2,
        source_git_head="producer",
    )
    assert [record["update"] for record in merged["entries"]] == [1, 2]
    assert all(record["path"].startswith("chunks/") for record in merged["entries"])


def test_disjoint_prepared_chunks_merge_into_continuation_range(tmp_path):
    root = tmp_path / "prepared"
    identity = {
        "source_git_head": "producer",
        "training_contract_sha256": "contract",
        "schedule_sha256": "schedule-3",
        "dataset_manifest_sha256": "dataset",
    }
    manifests = []
    for update in (2, 3):
        chunk = root / "chunks" / str(update)
        chunk.mkdir(parents=True)
        entry = {"update": update, "runtime_year": 2000 + update, "components": {}}
        artifact = PreparedTargetOrientedUpdateArtifact(
            PREPARED_ARTIFACT_SCHEMA,
            "contract",
            entry,
            "trace",
            {"value": np.asarray(update)},
        )
        record = {
            "update": update,
            "runtime_year": 2000 + update,
            "trace_signature": "trace",
            **write_prepared_target_artifact(chunk / f"update-{update:04d}.pkl.gz", artifact),
        }
        manifest = {
            "schema_version": PREPARED_MANIFEST_SCHEMA,
            "status": "passed",
            "identity": identity,
            "updates": 3,
            "start_update": update,
            "end_update": update,
            "entries": [record],
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
        }
        path = chunk / "prepared_manifest.json"
        atomic_json(path, manifest)
        manifests.append(path)

    report = merge_prepared_chunks(
        SimpleNamespace(
            output_dir=root,
            updates=3,
            start_update=2,
            end_update=3,
            chunk_manifests=manifests,
        )
    )
    assert report["start_update"] == 2
    assert report["end_update"] == 3
    merged = load_prepared_target_manifest_range(
        root / "prepared_manifest.json",
        training_contract_sha256="contract",
        schedule_sha256="schedule-3",
        updates=3,
        start_update=2,
        end_update=3,
        source_git_head="producer",
    )
    assert [record["update"] for record in merged["entries"]] == [2, 3]
