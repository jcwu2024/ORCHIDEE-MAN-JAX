from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining.daily_markov_contract import MarkovShard
from research.daily_coarse_graining.ok_leak_capture_selection import (
    ALL_SELECTION_METRICS,
    DYNAMIC_SELECTION_METRICS,
    build_bounded_capture_plan,
    capture_selection_metrics,
    select_bounded_capture_records,
)


def _leaf(key, start, stop, shape, *, axes=(), selected=()):
    return SimpleNamespace(
        key=key,
        start=start,
        stop=stop,
        shape=shape,
        axis_names=axes,
        selected_pft_indices=selected,
    )


def _synthetic_contract():
    return SimpleNamespace(
        state_leaves=(
            _leaf("hydrol_previous_step_state.soil_mc", 0, 2, (1, 2)),
            _leaf("slowproc_stomate_previous_step_state.soilc_total", 2, 4, (1, 2)),
            _leaf("slowproc_stomate_previous_step_state.fpeat", 4, 5, (1,)),
        ),
        fast_day_target_leaves=(
            _leaf("daily_interface.tsoil_daily", 0, 2, (1, 2)),
            _leaf(
                "daily_interface.gpp_daily",
                2,
                4,
                (1, 2),
                axes=("npts", "nvm"),
                selected=(0, 13),
            ),
            _leaf("daily_interface.precip_daily", 4, 5, (1,)),
            _leaf("hydrol_previous_step_state.runoff_per_soil", 5, 7, (1, 2)),
            _leaf("hydrol_previous_step_state.drainage_per_soil", 7, 9, (1, 2)),
        ),
    )


def _synthetic_shard():
    states = np.asarray(
        [
            [0.1, 0.3, 1.0, 2.0, 0.25],
            [0.2, 0.4, 2.0, 3.0, 0.25],
            [0.3, 0.5, 3.0, 4.0, 0.25],
            [0.4, 0.6, 4.0, 5.0, 0.25],
            [0.5, 0.7, 5.0, 6.0, 0.25],
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [
            [280.0, 282.0, 999.0, 1.0, 0.0, 0.0, 0.1, 0.0, 0.2],
            [282.0, 284.0, 999.0, 2.0, 1.0, 0.1, 0.2, 0.2, 0.3],
            [284.0, 286.0, 999.0, 3.0, 2.0, 0.2, 0.3, 0.3, 0.4],
            [286.0, 288.0, 999.0, 4.0, 3.0, 0.3, 0.4, 0.4, 0.5],
        ],
        dtype=np.float64,
    )
    return MarkovShard(
        state_trajectory=states,
        fast_day_target=targets,
        forcing_native=np.zeros((4, 1, 1)),
        forcing_record_indices=np.zeros((4, 1), dtype=np.int32),
        parameters=np.zeros((1,)),
        landpoint_static=np.zeros((1,)),
        annual_conditions=np.zeros((1,)),
        diagnostics=np.zeros((4, 1)),
        year=np.asarray(1961),
        day_index=np.arange(2, 6),
        discrete_trajectories={},
    )


def test_capture_selection_metrics_use_pft14_and_source_fields():
    metrics = capture_selection_metrics(_synthetic_shard(), _synthetic_contract())

    assert tuple(metrics) == ALL_SELECTION_METRICS
    np.testing.assert_allclose(metrics["soil_wetness"], [0.2, 0.3, 0.4, 0.5])
    np.testing.assert_allclose(metrics["soil_temperature_k"], [281.0, 283.0, 285.0, 287.0])
    np.testing.assert_allclose(metrics["pft14_gpp_daily"], [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(metrics["soil_carbon_proxy"], [3.0, 5.0, 7.0, 9.0])
    np.testing.assert_allclose(metrics["precip_daily"], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(metrics["hydrologic_export"], [0.3, 0.8, 1.2, 1.6])
    np.testing.assert_allclose(metrics["peat_fraction"], 0.25)


def _candidate_records():
    records = []
    for point_offset, landpoint_id in enumerate(("001.0-071.0", "069.0-119.0")):
        for index in range(30):
            metrics = {
                name: float((metric_index + 1) * index + point_offset)
                for metric_index, name in enumerate(DYNAMIC_SELECTION_METRICS)
            }
            metrics["peat_fraction"] = float(point_offset)
            records.append(
                {
                    "landpoint_id": landpoint_id,
                    "year": 1961 + index // 10,
                    "day_index": 2 + index % 10,
                    "row": index % 10,
                    "source_shard": f"{landpoint_id}/{1961 + index // 10}.npz",
                    "source_shard_sha256": "0" * 64,
                    "metrics": metrics,
                }
            )
    return records


def test_selection_is_bounded_deterministic_and_keeps_extrema():
    records = _candidate_records()
    selected = select_bounded_capture_records(records, samples_per_landpoint=16)
    reversed_selected = select_bounded_capture_records(
        list(reversed(records)), samples_per_landpoint=16
    )

    assert selected == reversed_selected
    assert len(selected) == 32
    for landpoint_id in ("001.0-071.0", "069.0-119.0"):
        point = [item for item in selected if item["landpoint_id"] == landpoint_id]
        assert len(point) == 16
        assert (point[0]["year"], point[0]["day_index"]) == (1961, 2)
        reasons = {reason for item in point for reason in item["selection_reasons"]}
        assert "earliest_train_day" in reasons
        for name in DYNAMIC_SELECTION_METRICS:
            assert f"{name}:minimum" in reasons
            assert f"{name}:maximum" in reasons


def test_selection_refuses_to_drop_mandatory_extrema():
    with pytest.raises(ValueError, match="cold anchor and all metric extrema"):
        select_bounded_capture_records(_candidate_records(), samples_per_landpoint=12)


def test_plan_builder_never_opens_sealed_split(tmp_path, monkeypatch):
    import research.daily_coarse_graining.ok_leak_capture_selection as selection

    root = tmp_path / "dataset"
    root.mkdir()
    references = []
    shards = {}
    for offset, landpoint_id in enumerate(("001.0-071.0", "069.0-119.0")):
        relative = f"shards/{landpoint_id}/1961.npz"
        path = root / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(f"train-{landpoint_id}".encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        references.append(
            {
                "landpoint_id": landpoint_id,
                "year": 1961,
                "spatial_split": "train",
                "temporal_split": "train",
                "shard": relative,
                "shard_sha256": digest,
            }
        )
        days = 30
        shards[path.resolve()] = MarkovShard(
            state_trajectory=np.arange((days + 1) * 5, dtype=np.float64).reshape(days + 1, 5),
            fast_day_target=np.arange(days * 9, dtype=np.float64).reshape(days, 9),
            forcing_native=np.zeros((days, 1, 1)),
            forcing_record_indices=np.zeros((days, 1), dtype=np.int32),
            parameters=np.zeros((1,)),
            landpoint_static=np.zeros((1,)),
            annual_conditions=np.zeros((1,)),
            diagnostics=np.zeros((days, 1)),
            year=np.asarray(1961),
            day_index=np.arange(2, days + 2),
            discrete_trajectories={},
        )
    references.append(
        {
            "landpoint_id": "sealed",
            "year": 2009,
            "spatial_split": "test",
            "temporal_split": "test",
            "shard": "missing-sealed-test.npz",
            "shard_sha256": "f" * 64,
        }
    )
    manifest = {
        "status": "complete",
        "dataset_id": "synthetic-v5",
        "teacher_git_head": "a" * 40,
        "markov_contract_sha256": "b" * 64,
        "markov_contract": {"synthetic": True},
        "shards": references,
    }
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    monkeypatch.setattr(selection, "daily_markov_contract_from_metadata", lambda _: object())
    monkeypatch.setattr(selection, "load_markov_shard", lambda path, contract: shards[path.resolve()])

    def fake_metrics(shard, _contract):
        base = np.arange(shard.days, dtype=np.float64)
        return {
            name: base * (index + 1)
            for index, name in enumerate(DYNAMIC_SELECTION_METRICS)
        } | {"peat_fraction": np.full(shard.days, 0.5)}

    monkeypatch.setattr(selection, "capture_selection_metrics", fake_metrics)

    plan = build_bounded_capture_plan(manifest_path, root, samples_per_landpoint=16)

    assert not plan["sealed_test_used"]
    assert plan["candidate_shard_count"] == 2
    assert plan["candidate_day_count"] == 60
    assert plan["selected_day_count"] == 32
    assert plan["selected_landpoints"] == ["001.0-071.0", "069.0-119.0"]
    assert len(plan["plan_sha256"]) == 64
    assert all(item["landpoint_id"] != "sealed" for item in plan["records"])
