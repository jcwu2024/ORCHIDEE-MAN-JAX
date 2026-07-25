from __future__ import annotations

import json

import pytest

from scripts.hpc.run_canonical_gpu_rollout_matrix import load_matrix


def _matrix():
    return {
        "schema_version": "canonical_rollout_matrix_v1",
        "start_day": 100,
        "days": 7,
        "cases": [
            {
                "id": "train-train",
                "landpoint_id": "281.0-095.0",
                "year": 2004,
                "allow_training_split_diagnostic": True,
            }
        ],
    }


def test_rollout_matrix_loader_accepts_exact_schema(tmp_path):
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(_matrix()), encoding="utf-8")

    loaded = load_matrix(path)

    assert loaded["cases"][0]["landpoint_id"] == "281.0-095.0"
    assert loaded["days"] == 7


@pytest.mark.parametrize("mutation", ("duplicate", "path", "extra", "bad_flag"))
def test_rollout_matrix_loader_rejects_ambiguous_cases(tmp_path, mutation):
    payload = _matrix()
    if mutation == "duplicate":
        payload["cases"].append(dict(payload["cases"][0]))
    elif mutation == "path":
        payload["cases"][0]["id"] = "../escape"
    elif mutation == "extra":
        payload["cases"][0]["unexpected"] = True
    else:
        payload["cases"][0]["allow_training_split_diagnostic"] = "yes"
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_matrix(path)
