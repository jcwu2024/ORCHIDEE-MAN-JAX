from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    AUDIT_FLUX_KEYS,
    _aggregation_matrix,
    _audit_family_trees,
    _classify_batch_effect,
    _summarize_gram,
)


def _gram(vectors: np.ndarray) -> np.ndarray:
    return vectors @ vectors.T


def _key_report(gram: np.ndarray):
    return {"all_parameters": _summarize_gram(gram)}


def _flatten(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _flatten(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _flatten(child)
    else:
        yield value


def test_disjoint_aggregation_counts_and_weights() -> None:
    for windows, rows in ((4, 8), (8, 4), (16, 2), (32, 1)):
        matrix = _aggregation_matrix(8, 4, windows)
        assert matrix.shape == (rows, 8)
        np.testing.assert_allclose(np.sum(matrix, axis=1), 1.0)
        assert np.all(np.sum(matrix != 0.0, axis=1) == windows // 4)
        assert np.all(np.sum(matrix != 0.0, axis=0) == 1)


def test_gram_summary_and_batch_decision_detect_noise_reduction() -> None:
    vectors = []
    for axis in (1, 1, 2, 2, 3, 3, 4, 4):
        vector = np.zeros(5, dtype=np.float64)
        vector[0] = 1.0
        vector[axis] = 3.0 if len(vectors) % 2 == 0 else -3.0
        vectors.append(vector)
    noisy = _gram(np.stack(vectors))
    stable = _gram(np.tile(np.asarray([[1.0, 0.0]]), (8, 1)))
    summary = _summarize_gram(noisy)
    assert summary["4"]["estimate_count"] == 8
    assert summary["8"]["estimate_count"] == 4
    assert summary["16"]["estimate_count"] == 2
    assert summary["32"]["estimate_count"] == 1
    assert summary["4"]["pairwise_cosine"]["median"] < 0.25
    assert summary["16"]["pairwise_cosine"]["median"] == 1.0

    reports = {
        name: _key_report(noisy if index < 3 else stable)
        for index, name in enumerate(AUDIT_FLUX_KEYS)
    }
    decision = _classify_batch_effect(reports)
    assert decision["action"] == "larger_effective_batch_supported"
    assert decision["larger_batch_supported_keys"] == list(AUDIT_FLUX_KEYS[:3])


def test_gradient_family_partition_is_complete() -> None:
    target_names = (
        "domain_embedding",
        "domain_blocks",
        "science_query",
        "science_embedding",
        "science_cross_blocks",
        "science_mixer_blocks",
        "category_embedding",
        "primary_blocks",
        "state_blocks",
        "primary_decoder",
        "state_decoder",
        "primary_heads",
        "state_heads",
        "auxiliary_science_projection",
        "auxiliary_bottleneck",
        "auxiliary_label_heads",
    )
    gradient = {
        "source": np.zeros((2,)),
        "target_oriented": {name: np.zeros((1,)) for name in target_names},
    }
    families = _audit_family_trees(gradient)
    leaves = [np.asarray(value) for family in families.values() for value in _flatten(family)]
    assert sum(value.size for value in leaves) == 18
    assert set(families) == {
        "base_source_encoder",
        "shared_science_backbone",
        "primary_path",
        "state_path",
        "auxiliary_path",
    }


def test_free_gpu_launcher_retains_no_optimizer_update() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts/hpc/run_target_oriented_flux_gradient_variance_audit.sh").read_text(
        encoding="utf-8"
    )
    assert "gln01|ibgln01" in launcher
    assert "sbatch" not in launcher
    assert "audit_target_oriented_flux_gradient_variance" in launcher
    assert "stage-update-1000.pkl" in launcher
    assert "target-oriented-prepared-1000-a243029" in launcher
