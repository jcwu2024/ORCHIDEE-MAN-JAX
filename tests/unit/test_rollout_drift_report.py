from __future__ import annotations

from types import SimpleNamespace

import pytest

from research.daily_coarse_graining import canonical_rollout
from research.daily_coarse_graining.rollout_drift_report import (
    build_drift_report,
    render_markdown,
)


def _leaf(component, key, start, stop):
    return SimpleNamespace(
        component=component,
        key=key,
        start=start,
        stop=stop,
        discrete=False,
    )


def _rollout():
    days = []
    for day, biomass, temperature in ((100, 0.2, 0.05), (101, 0.6, 0.3)):
        days.append(
            {
                "day_index": day,
                "state": {
                    "normalized_rmse": biomass,
                    "components": {
                        "slow": {
                            "normalized_rmse": biomass,
                            "current_uniform_loss_share": 0.8,
                        },
                        "thermal": {
                            "normalized_rmse": temperature,
                            "current_uniform_loss_share": 0.2,
                        },
                    },
                    "leaves": [
                        {
                            "key": "slow.biomass",
                            "component": "slow",
                            "source_ref": "stomate",
                            "width": 1,
                            "normalized_rmse": biomass,
                            "current_uniform_loss_share": 0.8,
                        },
                        {
                            "key": "thermal.temperature",
                            "component": "thermal",
                            "source_ref": "thermosoil",
                            "width": 2,
                            "normalized_rmse": temperature,
                            "current_uniform_loss_share": 0.2,
                        },
                    ],
                },
            }
        )
    return {
        "contract_sha256": "contract",
        "dataset_id": "dataset",
        "checkpoint": {"sha256": "checkpoint"},
        "selection": {
            "landpoint_id": "215.0-119.0",
            "year": 2005,
            "start_day": 100,
            "days": 2,
            "state_feedback": "free",
        },
        "summary": {"last_day_normalized_state_rmse": 0.6},
        "days": days,
    }


def test_drift_report_exposes_dimension_weighting_and_first_divergence():
    contract = SimpleNamespace(
        sha256="contract",
        continuous_state_width=3,
        state_leaves=(
            _leaf("slow", "slow.biomass", 0, 1),
            _leaf("thermal", "thermal.temperature", 1, 3),
        ),
    )

    report = build_drift_report(_rollout(), contract)

    assert report["largest_first_day_leaves"][0] == "slow.biomass"
    assert report["leaves"]["slow.biomass"]["first_threshold_crossings"][
        "0.5"
    ] == 101
    assert report["state_loss_weighting"]["components"]["thermal"][
        "nominal_weight_share"
    ] == pytest.approx(2.0 / 3.0)
    markdown = render_markdown(report)
    assert "Largest First-Day Field Errors" in markdown
    assert "`slow.biomass`" in markdown


def test_drift_report_rejects_legacy_truncated_rollout():
    contract = SimpleNamespace(
        sha256="contract",
        continuous_state_width=1,
        state_leaves=(_leaf("slow", "slow.biomass", 0, 1),),
    )
    rollout = _rollout()
    del rollout["days"][0]["state"]["leaves"]

    with pytest.raises(ValueError, match="complete field-level"):
        build_drift_report(rollout, contract)


def test_rollout_training_diagnostic_is_explicit_and_never_opens_test(monkeypatch):
    reference = SimpleNamespace(
        landpoint_id="281.0-095.0",
        year=2004,
        spatial_split="train",
        temporal_split="train",
    )
    monkeypatch.setattr(
        canonical_rollout,
        "load_dataset_index",
        lambda *_args, **_kwargs: SimpleNamespace(shards=(reference,)),
    )

    with pytest.raises(ValueError, match="requires a validation split"):
        canonical_rollout._select_reference(
            SimpleNamespace(), landpoint_id=reference.landpoint_id, year=2004
        )
    assert (
        canonical_rollout._select_reference(
            SimpleNamespace(),
            landpoint_id=reference.landpoint_id,
            year=2004,
            allow_training_split_diagnostic=True,
        )
        is reference
    )

    reference.spatial_split = "test"
    with pytest.raises(ValueError, match="sealed test split"):
        canonical_rollout._select_reference(
            SimpleNamespace(),
            landpoint_id=reference.landpoint_id,
            year=2004,
            allow_training_split_diagnostic=True,
        )
