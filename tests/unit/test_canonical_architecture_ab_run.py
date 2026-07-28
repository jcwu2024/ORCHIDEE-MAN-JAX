from __future__ import annotations

import json
from pathlib import Path

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


def test_architecture_ab_launcher_uses_one_shared_gpu_run_and_no_test_inputs():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --cpus-per-task=4" in text
    assert "#SBATCH --time=12:00:00" in text
    assert "run_canonical_architecture_ab_gpu" in text
    assert "canonical_669_axis_process_architecture_ab.json" in text
    assert "test -z" in text
    assert "test" not in "\n".join(
        line for line in text.splitlines() if "--dataset" in line
    )
