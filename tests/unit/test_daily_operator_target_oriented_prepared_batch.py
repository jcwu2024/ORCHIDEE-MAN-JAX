from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.model import (
    bind_daily_operator_dynamic_retained_inputs,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    build_target_oriented_training_schedule,
    load_target_oriented_training_contract,
    prepare_target_oriented_training_batch,
)
from research.daily_coarse_graining.daily_operator.training_data import (
    MaskedProcessTarget,
    ProcessTrainingExample,
    ProcessTrainingTarget,
)
from research.daily_coarse_graining.daily_operator.types import (
    StateDefinedMasks,
    StateFeatureGroups,
)
from scripts.hpc.prepare_target_oriented_training_batches import (
    _retained_inputs_from_artifact,
)
from scripts.hpc.target_oriented_training_prepared import (
    PREPARED_ARTIFACT_SCHEMA,
    PREPARED_MANIFEST_SCHEMA,
    PreparedTargetOrientedUpdateArtifact,
    atomic_json,
    load_prepared_target_artifact,
    load_prepared_target_manifest,
    write_prepared_target_artifact,
)


@dataclass(frozen=True)
class ParentReference:
    landpoint_id: str
    year: int
    sha256: str
    path: str
    spatial_split: str = "train"
    temporal_split: str = "train"


@dataclass(frozen=True)
class Reference:
    parent: ParentReference
    sha256: str
    path: str
    day_count: int = 40


class FakeShard:
    def __init__(self, point: int):
        self.point = point
        self.day_index = np.arange(1, 41, dtype=np.int32)
        self.days = self.day_index.size

    def sample(self, row: int):
        return {"point": self.point, "row": row}


class FakeAdapter:
    contract = object()
    typed_contract = object()

    def assemble(self, sample, *, supplemental_values, supplemental_defined):
        point = int(sample["point"])
        row = int(sample["row"])
        token = float(point * 100 + row)
        state = StateFeatureGroups(
            **{name: np.asarray([token + index]) for index, name in enumerate(StateFeatureGroups._fields)}
        )
        defined = StateDefinedMasks(**{name: np.ones((1,), dtype=np.bool_) for name in StateFeatureGroups._fields})
        supervision = {
            "daily.fast_gpp": MaskedProcessTarget(
                np.asarray([token + float(supplemental_values["typed"]) / 1000.0]),
                np.asarray([bool(supplemental_defined["typed"])]),
            ),
            "auxiliary": MaskedProcessTarget(np.asarray([token]), np.ones((1,), dtype=np.bool_)),
        }
        return ProcessTrainingExample(
            inputs=np.asarray([point, row], dtype=np.int32),
            targets=ProcessTrainingTarget(
                supervision=supervision,
                retained_boundary=None,
                next_state=state,
                next_state_defined=defined,
                next_discrete_state={"flag": np.asarray(point % 2, dtype=np.int32)},
                diagnostics=np.asarray([0.0]),
            ),
        )


def _references():
    return tuple(
        Reference(
            parent=ParentReference(
                landpoint_id=f"point-{point:03d}",
                year=1979,
                sha256=f"parent-{point:03d}",
                path=f"parent-{point:03d}",
            ),
            sha256=f"typed-{point:03d}",
            path=f"typed-{point:03d}",
        )
        for point in range(28)
    )


def test_prepared_batch_keeps_landpoint_day_inputs_dynamic(monkeypatch):
    references = _references()
    contract = load_target_oriented_training_contract()
    entry = build_target_oriented_training_schedule(references, contract, updates=1)["entries"][0]
    parent_loads = []
    typed_loads = []
    target_builds = []
    schema = object()
    owner_contract = object()

    def load_parent(path, *, contract):
        del contract
        parent_loads.append(path)
        return FakeShard(int(str(path).rsplit("-", 1)[1]))

    def load_typed(path, *, contract, expected_days):
        del contract
        typed_loads.append(path)
        point = int(str(path).rsplit("-", 1)[1])
        return (
            {"typed": np.full((expected_days,), point, dtype=np.float64)},
            {"typed": np.ones((expected_days,), dtype=np.bool_)},
            np.arange(1, expected_days + 1, dtype=np.int32),
        )

    monkeypatch.setattr(
        "research.daily_coarse_graining.daily_operator.target_oriented_training.load_markov_shard",
        load_parent,
    )
    monkeypatch.setattr(
        "research.daily_coarse_graining.daily_operator.target_oriented_training.read_typed_sidecar_shard",
        load_typed,
    )

    def build_targets(value, *, schema, markov_contract, owner_contract):
        target_builds.append((value, schema, markov_contract, owner_contract))
        return value.targets.supervision

    monkeypatch.setattr(
        "research.daily_coarse_graining.daily_operator.target_oriented_training.build_successor_fast_owner_targets",
        build_targets,
    )
    day_inputs = {
        (
            reference.parent.landpoint_id,
            reference.parent.year,
            reference.parent.sha256,
            reference.sha256,
        ): np.asarray([point], dtype=np.float64)
        for point, reference in enumerate(references)
    }

    batch = prepare_target_oriented_training_batch(
        references,
        entry,
        adapter=FakeAdapter(),
        schema=schema,
        owner_contract=owner_contract,
        day_inputs_by_reference=day_inputs,
    )

    assert batch.one_day.inputs.shape == (32, 2)
    assert batch.seven_day.inputs.shape == (8, 7, 2)
    assert batch.thirty_day.inputs.shape == (4, 30, 2)
    np.testing.assert_array_equal(batch.one_day.inputs[:, 0], batch.one_day.day_inputs[:, 0])
    np.testing.assert_array_equal(batch.seven_day.inputs[:, 0, 0], batch.seven_day.day_inputs[:, 0])
    np.testing.assert_array_equal(batch.thirty_day.inputs[:, 0, 0], batch.thirty_day.day_inputs[:, 0])
    assert np.unique(batch.one_day.day_inputs).size == 16
    assert np.unique(batch.seven_day.day_inputs).size == 8
    assert np.unique(batch.thirty_day.day_inputs).size == 4
    assert len(parent_loads) == len(typed_loads) == 28
    assert len(target_builds) == 32
    assert all(values[1:] == (schema, FakeAdapter.contract, owner_contract) for values in target_builds)


@dataclass(frozen=True)
class FakeRetainedTail:
    day_inputs: object


def test_dynamic_retained_binding_maps_each_sample_day_inputs(monkeypatch):
    def fake_transition(parameters, inputs, *, input_assembler, process_head, retained_tail):
        del input_assembler, process_head
        return parameters + inputs + retained_tail.day_inputs

    monkeypatch.setattr(
        "research.daily_coarse_graining.daily_operator.model.daily_operator_transition",
        fake_transition,
    )
    transition = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=object(),
        process_head=object(),
        retained_tail_template=FakeRetainedTail(day_inputs=jnp.asarray(-1.0)),
    )
    result = jax.jit(jax.vmap(transition, in_axes=(None, 0, 0)))(
        jnp.asarray(2.0),
        jnp.asarray([10.0, 20.0, 30.0]),
        jnp.asarray([1.0, 3.0, 5.0]),
    )
    np.testing.assert_array_equal(result, np.asarray([13.0, 25.0, 37.0]))


def test_reuses_distinct_point_year_retained_inputs_from_prepared_artifact():
    def record(point: int, starts: tuple[int, ...]):
        return {
            "landpoint_id": f"point-{point:03d}",
            "year": 1979,
            "parent_shard_sha256": f"parent-{point:03d}",
            "typed_shard_sha256": f"typed-{point:03d}",
            "start_rows": starts,
        }

    one_records = tuple(record(point, (2 * point + 1, 2 * point + 2)) for point in range(16))
    seven_records = tuple(record(point, (point + 1,)) for point in range(16, 24))
    thirty_records = tuple(record(point, (point + 1,)) for point in range(24, 28))
    artifact = SimpleNamespace(
        batch=SimpleNamespace(
            one_day=SimpleNamespace(day_inputs=np.repeat(np.arange(16), 2)[:, None]),
            seven_day=SimpleNamespace(day_inputs=np.arange(16, 24)[:, None]),
            thirty_day=SimpleNamespace(day_inputs=np.arange(24, 28)[:, None]),
        )
    )
    entry = {
        "components": {
            "one_day": one_records,
            "seven_day": seven_records,
            "thirty_day": thirty_records,
        }
    }

    reused = _retained_inputs_from_artifact(jax, artifact, entry)

    assert len(reused) == 28
    for point in range(28):
        identity = (f"point-{point:03d}", 1979, f"parent-{point:03d}", f"typed-{point:03d}")
        np.testing.assert_array_equal(reused[identity], np.asarray([point]))


def test_prepared_target_artifact_roundtrip(tmp_path):
    entry = {"update": 1, "runtime_year": 1979, "components": {}}
    artifact = PreparedTargetOrientedUpdateArtifact(
        schema_version=PREPARED_ARTIFACT_SCHEMA,
        training_contract_sha256="contract",
        schedule_entry=entry,
        trace_signature="trace",
        batch={"values": np.arange(4)},
    )
    path = tmp_path / "update-0001.pkl.gz"
    record = {"trace_signature": "trace", **write_prepared_target_artifact(path, artifact)}
    loaded = load_prepared_target_artifact(
        tmp_path / "prepared_manifest.json",
        record,
        training_contract_sha256="contract",
        schedule_entry=entry,
    )
    np.testing.assert_array_equal(loaded.batch["values"], np.arange(4))


def test_prepared_target_manifest_requires_complete_producer_identity(tmp_path):
    path = tmp_path / "prepared_manifest.json"
    payload = {
        "schema_version": PREPARED_MANIFEST_SCHEMA,
        "status": "passed",
        "identity": {
            "training_contract_sha256": "contract",
            "schedule_sha256": "schedule",
            "source_git_head": "producer",
        },
        "updates": 2,
        "entries": [{"update": 1}, {"update": 2}],
    }
    atomic_json(path, payload)
    assert (
        load_prepared_target_manifest(
            path,
            training_contract_sha256="contract",
            schedule_sha256="schedule",
            updates=2,
            source_git_head="producer",
        )["status"]
        == "passed"
    )
    with pytest.raises(ValueError, match="identity or coverage drift"):
        load_prepared_target_manifest(
            path,
            training_contract_sha256="contract",
            schedule_sha256="schedule",
            updates=2,
            source_git_head="different",
        )
