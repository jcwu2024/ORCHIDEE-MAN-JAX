from __future__ import annotations

import numpy as np
import pytest

from jax_orchidee.stomate.soilcarbon_kernels import (
    IACT,
    IFREE,
    NDOC,
    NPOOL,
    soilcarbon_leak_doc_diffusion,
)
from research.daily_coarse_graining.doc_diffusion_boundary_baseline_panel import (
    PROXY_NAMES,
    ProxyArrays,
    aggregate_proxy_arrays,
    compute_boundary_proxies,
    decide_boundary_baseline,
    parse_selection,
    split_name,
)


def _doc_profile(values: tuple[float, ...]) -> np.ndarray:
    doc = np.zeros((1, 1, len(values), NDOC, NPOOL, 1), dtype=np.float64)
    doc[0, 0, :, IFREE, IACT, 0] = values
    return doc


def test_boundary_proxies_are_not_an_exact_48_step_diffusion_fold():
    zf_soil_b = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    start_doc = _doc_profile((0.2, 2.0, 0.1))
    warm = np.full((1, 3, 1), 273.15, dtype=np.float64)
    cold = warm.copy()
    cold[:, 1, :] = 272.15
    current = start_doc
    cumulative = np.zeros((1, 1, 3, NPOOL, 1), dtype=np.float64)
    for step in range(48):
        temperature = warm if step < 20 else cold
        result = soilcarbon_leak_doc_diffusion(
            current,
            temperature,
            zf_soil_b,
            np.asarray([0.24 / 48.0], dtype=np.float64),
        )
        cumulative += np.asarray(result.doc_flux_diff)
        current = np.asarray(result.doc)

    proxies = compute_boundary_proxies(
        start_doc=start_doc,
        start_tprof=warm,
        end_doc=current,
        end_tprof=cold,
        zf_soil_b=zf_soil_b,
    )
    assert np.any(cumulative != 0.0)
    assert all(not np.allclose(cumulative, proxies[name], rtol=1.0e-8, atol=1.0e-12) for name in PROXY_NAMES)


def test_proxy_aggregation_and_predeclared_gate():
    target = np.asarray([1.0, 2.0, 0.0, 4.0], dtype=np.float64)
    defined = np.ones_like(target, dtype=np.bool_)
    predictions = {
        "day_start": target * 0.5,
        "day_end": target * 1.4,
        "endpoint_trapezoid": target * 0.95,
    }
    record = ProxyArrays(target=target, defined=defined, predictions=predictions)
    metrics = aggregate_proxy_arrays((record, record))
    assert metrics["endpoint_trapezoid"]["defined"]["value_count"] == 8
    assert metrics["endpoint_trapezoid"]["target_nonzero_fraction"] == 0.75

    slices = {name: metrics for name in ("train_train", "temporal_validation", "spatial_validation", "joint_validation")}
    decision = decide_boundary_baseline(slices)
    assert decision["selected_proxy"] == "endpoint_trapezoid"
    assert decision["status"] == "physical_boundary_plus_learned_residual_qualified"


def test_selection_parsing_and_test_split_guard():
    assert parse_selection("001.0-071.0:2006") == ("001.0-071.0", 2006)
    assert split_name("train", "validation") == "temporal_validation"
    assert split_name("validation", "train") == "spatial_validation"
    with pytest.raises(ValueError, match="sealed test"):
        split_name("train", "test")
    with pytest.raises(ValueError, match="LANDPOINT:YEAR"):
        parse_selection("001.0-071.0")
