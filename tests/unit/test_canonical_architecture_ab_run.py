from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.daily_coarse_graining import canonical_architecture_ab_run as ab

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = (
    ROOT
    / "manifests"
    / "coarse_graining"
    / "canonical_669_axis_process_architecture_ab.json"
)
LAUNCHER = ROOT / "scripts" / "hpc" / "slurm_canonical_669_architecture_ab.sh"


def _training_report(scale: float, *, parameters: int):
    families = {
        "hydrol": {"normalized_rmse": 1.0 * scale},
        "thermosoil": {"normalized_rmse": 2.0 * scale},
    }
    return {
        "parameter_count": parameters,
        "best_epoch": 1,
        "history": [
            {
                "epoch": 1,
                "validation": {
                    split: {"families": families}
                    for split in ("temporal", "spatial", "joint")
                },
            }
        ],
        "test_split_evaluated": False,
    }


def test_frozen_architecture_ab_manifest_is_valid_and_test_sealed():
    experiment = ab.load_architecture_ab_experiment(EXPERIMENT)

    assert experiment.raw["training"]["epochs"] == 1
    assert experiment.raw["training"]["batch_size"] == 256
    assert experiment.raw["validation"]["sealed_test_used"] is False
    assert experiment.raw["arms"][1]["model_architecture"] == (
        "axis_process_coupled_v1"
    )


def test_architecture_ab_manifest_rejects_arm_or_test_drift(tmp_path):
    raw = json.loads(EXPERIMENT.read_text(encoding="utf-8"))
    raw["arms"].reverse()
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="arm inventory"):
        ab.load_architecture_ab_experiment(path)

    raw = json.loads(EXPERIMENT.read_text(encoding="utf-8"))
    raw["validation"]["sealed_test_used"] = True
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="test split sealed"):
        ab.load_architecture_ab_experiment(path)


def test_architecture_ab_classification_applies_all_frozen_gates():
    gates = ab.load_architecture_ab_experiment(EXPERIMENT).raw[
        "screening_gates"
    ]
    flat = _training_report(1.0, parameters=1_963_369)
    passing = _training_report(0.95, parameters=1_954_041)
    rejected = _training_report(1.03, parameters=1_954_041)

    accepted = ab.classify_architecture_ab(flat, passing, gates)
    failed = ab.classify_architecture_ab(flat, rejected, gates)

    assert accepted["status"] == "passed"
    assert all(accepted["checks"].values())
    assert failed["status"] == "rejected"
    assert failed["checks"]["spatial_improvement"] is False


def test_parallel_preflight_is_hash_bound_idempotent_and_fails_on_drift(
    tmp_path,
    monkeypatch,
):
    experiment = ab.load_architecture_ab_experiment(EXPERIMENT)
    paths = {}
    for name in ("dataset_manifest", "training_statistics", "acceptance_report"):
        path = tmp_path / f"{name}.json"
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("protocol", encoding="utf-8")
    protocol = SimpleNamespace(path=protocol_path, sha256="protocol-sha")
    accepted = {
        "identity": {"dataset_id": "dataset"},
        "training_statistics": {"sample_count": 123},
    }
    payload = ab._preflight_payload(experiment, paths, protocol, accepted)
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(payload), encoding="utf-8")

    observed = ab._verify_preflight(
        preflight,
        experiment=experiment,
        paths=paths,
        protocol=protocol,
        accepted=accepted,
    )
    assert observed["expected_training_samples"] == 123

    accepted["training_statistics"]["sample_count"] = 124
    with pytest.raises(ValueError, match="identity drift"):
        ab._verify_preflight(
            preflight,
            experiment=experiment,
            paths=paths,
            protocol=protocol,
            accepted=accepted,
        )

    accepted["training_statistics"]["sample_count"] = 123
    monkeypatch.setattr(
        ab,
        "_verified_inputs",
        lambda **_: (experiment, paths, protocol, accepted),
    )
    output_root = tmp_path / "output"
    first = ab.prepare_architecture_ab(
        experiment_path=EXPERIMENT,
        dataset_path=paths["dataset_manifest"],
        statistics_path=paths["training_statistics"],
        acceptance_path=paths["acceptance_report"],
        protocol_path=protocol_path,
        output_root=output_root,
    )
    assert ab.prepare_architecture_ab(
        experiment_path=EXPERIMENT,
        dataset_path=paths["dataset_manifest"],
        statistics_path=paths["training_statistics"],
        acceptance_path=paths["acceptance_report"],
        protocol_path=protocol_path,
        output_root=output_root,
    ) == first
    tampered = json.loads(first.read_text(encoding="utf-8"))
    tampered["expected_training_samples"] = 999
    first.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="existing.*identity drift"):
        ab.prepare_architecture_ab(
            experiment_path=EXPERIMENT,
            dataset_path=paths["dataset_manifest"],
            statistics_path=paths["training_statistics"],
            acceptance_path=paths["acceptance_report"],
            protocol_path=protocol_path,
            output_root=output_root,
        )


def test_parallel_arm_report_requires_exact_samples_and_sealed_test():
    experiment = ab.load_architecture_ab_experiment(EXPERIMENT)
    arm = experiment.raw["arms"][0]
    training = experiment.raw["training"]
    report = _training_report(1.0, parameters=1_963_369)
    report["model_architecture"] = {"id": "canonical_flat_v1"}
    report["history"][0]["training_samples"] = 123

    ab._validate_arm_report(
        report,
        arm=arm,
        training=training,
        expected_samples=123,
    )
    report["history"][0]["training_samples"] = 122
    with pytest.raises(ValueError, match="exact epochs"):
        ab._validate_arm_report(
            report,
            arm=arm,
            training=training,
            expected_samples=123,
        )


def test_architecture_ab_launcher_uses_two_parallel_gpu_arms_and_no_test_inputs():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:2" in text
    assert "#SBATCH --cpus-per-task=8" in text
    assert "#SBATCH --time=12:00:00" in text
    assert "--phase prepare" in text
    assert "--phase arm --arm flat" in text
    assert "--phase arm --arm axis_process" in text
    assert "--phase finalize" in text
    assert 'SINGULARITYENV_CUDA_VISIBLE_DEVICES="$visible_device"' in text
    assert 'wait "$FLAT_PID"' in text
    assert 'wait "$AXIS_PID"' in text
    assert "test ! -e \"$OUTPUT_ROOT\"" not in text
    assert '>>"$OUTPUT_ROOT/flat_worker.log"' in text
    assert '>>"$OUTPUT_ROOT/axis_process_worker.log"' in text
    assert "canonical_669_axis_process_architecture_ab.json" in text
    assert "test -z" in text
    assert "test" not in "\n".join(
        line for line in text.splitlines() if "--dataset" in line
    )
