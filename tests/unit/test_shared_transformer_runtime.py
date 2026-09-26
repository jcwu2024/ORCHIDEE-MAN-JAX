from __future__ import annotations

import gzip
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.long_horizon_objective import (
    MaskedTrajectory,
)
from research.daily_coarse_graining.daily_operator.shared_trajectory_loss import (
    FLUX_KEYS,
    STOCK_KEYS,
    SharedComponentTrajectories,
)
from scripts.hpc.shared_transformer_runtime import (
    load_shared_runtime,
    prepare_shared_trajectory_scales,
    select_artifact_ids,
    validate_shared_artifact_metadata,
)


def _identity(landpoint: str, year: int) -> dict[str, object]:
    return {
        "landpoint_id": landpoint,
        "year": year,
        "parent_shard_sha256": f"parent-{landpoint}-{year}",
        "typed_shard_sha256": f"typed-{landpoint}-{year}",
    }


def _reference(identity, *, spatial="train", temporal="train", days=40):
    parent = SimpleNamespace(
        landpoint_id=identity["landpoint_id"],
        year=identity["year"],
        sha256=identity["parent_shard_sha256"],
        spatial_split=spatial,
        temporal_split=temporal,
    )
    return SimpleNamespace(
        parent=parent,
        sha256=identity["typed_shard_sha256"],
        day_count=days,
    )


def _schedule(identifier: int, identity: dict[str, object], starts=(2, 3)):
    def record(rows):
        return {**identity, "start_rows": list(rows)}

    return {
        "update": identifier,
        "runtime_year": identity["year"],
        "components": {
            "one_day": [record(starts)],
            "seven_day": [record((4,))],
            "thirty_day": [record((5,))],
        },
    }


def test_select_artifact_ids_is_seeded_metadata_only_and_order_independent():
    records = [
        {
            "update": update,
            "runtime_year": 1961 + (update % 5),
            "path": f"update-{update:04d}.pkl.gz",
            "trace_signature": f"trace-{update % 3}",
        }
        for update in range(1, 41)
    ]
    selected = select_artifact_ids(records)
    assert selected == select_artifact_ids(tuple(reversed(records)))
    assert len(selected) == len(set(selected)) == 16
    assert set(selected) <= set(range(1, 41))
    assert select_artifact_ids(records[:3]) == select_artifact_ids(records[:3], count=10)


def test_select_artifact_ids_rejects_ambiguous_metadata():
    with pytest.raises(ValueError, match="positive"):
        select_artifact_ids([{"update": 1, "runtime_year": 1961}], count=0)
    with pytest.raises(ValueError, match="not unique"):
        select_artifact_ids(
            [
                {"update": 1, "runtime_year": 1961},
                {"update": 1, "runtime_year": 1962},
            ]
        )


def test_validate_shared_artifact_metadata_reports_bounded_identity_coverage():
    first = _identity("001.0-071.0", 1961)
    second = _identity("002.0-072.0", 1962)
    index = SimpleNamespace(shards=(_reference(first), _reference(second)))
    artifacts = (
        SimpleNamespace(schedule_entry=_schedule(11, first)),
        SimpleNamespace(schedule_entry=_schedule(12, second)),
    )
    metadata = validate_shared_artifact_metadata(index, (11, 12), artifacts)
    assert metadata["split"] == "train/train"
    assert "not_global_representativeness" in metadata["scope"]
    assert metadata["artifact_ids"] == (11, 12)
    assert len(metadata["calibration_daily_rows"]) == 4
    assert metadata["unique_identity_coverage"]["reference_count"] == 2
    assert metadata["unique_identity_coverage"]["point_year_count"] == 2


def test_validate_shared_artifact_metadata_rejects_nontraining_reference():
    identity = _identity("001.0-071.0", 1961)
    index = SimpleNamespace(shards=(_reference(identity, temporal="validation"),))
    artifact = SimpleNamespace(schedule_entry=_schedule(11, identity))
    with pytest.raises(ValueError, match="non-train/train"):
        validate_shared_artifact_metadata(index, (11,), (artifact,))


class _Pft(NamedTuple):
    pft_fraction: object
    active_pft_mask: object


class _Inputs(NamedTuple):
    continuous_state: object
    pft: _Pft
    day_index: object


class _State(NamedTuple):
    other: object


class _Target(NamedTuple):
    value: object
    defined: object


class _OneDay(NamedTuple):
    inputs: _Inputs
    process_targets: object
    expected_next_state: _State
    expected_next_state_defined: _State


def test_prepare_shared_trajectory_scales_uses_all_rows_and_same_input_stocks(monkeypatch):
    import jax

    from research.daily_coarse_graining.daily_operator import shared_trajectory_loss

    identity = _identity("001.0-071.0", 1961)
    batch = _OneDay(
        inputs=_Inputs(
            continuous_state=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            pft=_Pft(
                pft_fraction=np.asarray([[0.25, 0.75], [0.4, 0.6]]),
                active_pft_mask=np.asarray([[True, True], [True, True]]),
            ),
            day_index=np.asarray([102, 103]),
        ),
        process_targets={
            "daily.fast_gpp": _Target(
                value=np.asarray([[5.0, 6.0], [7.0, 8.0]]),
                defined=np.ones((2, 2), dtype=bool),
            )
        },
        expected_next_state=_State(np.asarray([[2.0, 4.0], [6.0, 8.0]])),
        expected_next_state_defined=_State(np.ones((2, 2), dtype=bool)),
    )
    artifact = SimpleNamespace(
        schedule_entry=_schedule(11, identity),
        batch=SimpleNamespace(one_day=batch),
    )
    assembler = SimpleNamespace(
        assemble=lambda inputs: SimpleNamespace(
            state_defined=SimpleNamespace(other=np.ones_like(inputs.continuous_state, dtype=bool))
        )
    )
    runtime = {
        "jax": jax,
        "contract": object(),
        "adapter": SimpleNamespace(input_assembler=assembler),
        "index": SimpleNamespace(shards=(_reference(identity),)),
    }

    def fake_extract(observation, _contract):
        state = np.asarray(observation["state"], dtype=np.float64)
        defined = np.asarray(observation["defined"], dtype=bool)
        scalar = state[:, :1]
        scalar_defined = defined[:, :1]
        fluxes = {
            key: MaskedTrajectory(scalar + offset, scalar_defined)
            for offset, key in enumerate(FLUX_KEYS)
        }
        stocks = {
            key: MaskedTrajectory(scalar + offset, scalar_defined)
            for offset, key in enumerate(STOCK_KEYS)
        }
        return SharedComponentTrajectories(fluxes, stocks)

    captured = {}

    def fake_build(samples, **kwargs):
        materialized = tuple(samples)
        captured["samples"] = materialized
        captured["kwargs"] = kwargs
        # Initial stocks must come from input states 1 and 3, not endpoints 2 and 6.
        assert float(materialized[0][1][STOCK_KEYS[0]].value[0]) == 1.0
        assert float(materialized[1][1][STOCK_KEYS[0]].value[0]) == 3.0
        return SimpleNamespace(
            flux_counts={key: 2 for key in FLUX_KEYS},
            stock_change_counts={key: 2 for key in STOCK_KEYS},
            flux_nonzero_counts={key: 2 for key in FLUX_KEYS},
            stock_change_nonzero_counts={key: 2 for key in STOCK_KEYS},
        )

    monkeypatch.setattr(shared_trajectory_loss, "shared_component_trajectories", fake_extract)
    monkeypatch.setattr(shared_trajectory_loss, "build_shared_trajectory_scales", fake_build)
    flux_floors = {key: 0.01 for key in FLUX_KEYS}
    stock_floors = {key: 0.02 for key in STOCK_KEYS}
    _, metadata = prepare_shared_trajectory_scales(
        runtime,
        (artifact,),
        flux_floors=flux_floors,
        stock_change_floors=stock_floors,
    )
    assert captured["kwargs"] == {
        "split_label": "train/train",
        "flux_floors": flux_floors,
        "stock_change_floors": stock_floors,
    }
    assert len(captured["samples"]) == metadata["row_count"] == 2
    assert metadata["heterotrophic_respiration_nonzero_count"] == 2
    assert "not_global_scale_qualification" in metadata["scope"]


@dataclass(frozen=True)
class _StaticOwner:
    include_modelout_diagnostics: bool = True


@dataclass(frozen=True)
class _Template:
    static_owner: _StaticOwner


def test_load_shared_runtime_disables_neural_initialization_and_uses_cpu_backend(
    monkeypatch, tmp_path: Path
):
    from research.daily_coarse_graining.daily_operator import (
        mechanistic_output_contract,
        mechanistic_process_head,
        strong_baseline_contract,
        target_oriented_contract,
    )
    from scripts.hpc import (
        canonical_training_runtime,
        retained_training_runtime,
    )

    identity = _identity("001.0-071.0", 1961)
    artifact = SimpleNamespace(
        schema_version="prepared",
        training_contract_sha256="training-contract",
        schedule_entry=_schedule(11, identity),
        trace_signature="trace",
        batch=object(),
    )
    artifact_path = tmp_path / "update-0011.pkl.gz"
    with gzip.open(artifact_path, "wb") as handle:
        pickle.dump(artifact, handle)
    manifest_path = tmp_path / "prepared_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "target_oriented_prepared_manifest_v1",
                "status": "passed",
                "identity": {"training_contract_sha256": "training-contract"},
                "entries": [
                    {
                        "update": 11,
                        "runtime_year": 1961,
                        "path": artifact_path.name,
                        "trace_signature": "trace",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    calls = {}
    runtime = {
        "schema": "complete-schema",
        "adapter": SimpleNamespace(input_assembler="assembler"),
        "owner_contract": "owner",
        "probe_assembled": "probe",
        "index": SimpleNamespace(shards=(_reference(identity),)),
        "parameters": "must-be-removed",
    }

    def fake_load(args, **kwargs):
        calls["initialize_probe_parameters"] = kwargs["initialize_probe_parameters"]
        calls["backend_initializer"] = kwargs["backend_initializer"]
        return dict(runtime)

    monkeypatch.setattr(canonical_training_runtime, "_load_extended_runtime", fake_load)
    monkeypatch.setattr(strong_baseline_contract, "load_strong_baseline_contract", lambda: "architecture")
    monkeypatch.setattr(mechanistic_output_contract, "load_mechanistic_output_contract", lambda: "mechanism")
    monkeypatch.setattr(
        mechanistic_output_contract,
        "build_mechanistic_learned_schema",
        lambda schema, mechanism: "learned-schema",
    )
    monkeypatch.setattr(mechanistic_process_head, "build_mechanistic_output_layout", lambda **kwargs: "layout")
    monkeypatch.setattr(target_oriented_contract, "load_target_oriented_contract", lambda: "target")
    monkeypatch.setattr(
        retained_training_runtime,
        "build_parent_process_head",
        lambda loaded, target: ("parent-head", target),
    )
    monkeypatch.setattr(
        retained_training_runtime,
        "_prepare_template",
        lambda loaded, entry: _Template(_StaticOwner()),
    )

    args = SimpleNamespace(
        shape_only=True,
        cpu_forward_only=False,
        prepared_manifest=manifest_path,
    )
    loaded, parent, template, artifacts, metadata = load_shared_runtime(args, (11,))
    assert calls["initialize_probe_parameters"] is False
    assert callable(calls["backend_initializer"])
    assert "parameters" not in loaded
    assert parent == ("parent-head", "target")
    assert template.static_owner.include_modelout_diagnostics is False
    assert artifacts[0].schedule_entry["update"] == 11
    assert metadata["artifact_ids"] == (11,)
