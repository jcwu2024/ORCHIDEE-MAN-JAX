from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.expanded_one_day import (
    prepare_one_day_rows,
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


@dataclass(frozen=True)
class ParentReference:
    landpoint_id: str = "point-001"
    year: int = 1979
    sha256: str = "parent-sha"
    path: str = "parent-path"
    spatial_split: str = "train"
    temporal_split: str = "train"


@dataclass(frozen=True)
class Reference:
    parent: ParentReference
    sha256: str = "typed-sha"
    path: str = "typed-path"
    day_count: int = 4


class FakeShard:
    day_index = np.asarray([1, 2, 3, 4], dtype=np.int32)
    days = 4

    @staticmethod
    def sample(row):
        return {"row": row}


class FakeAdapter:
    contract = object()
    typed_contract = object()

    def __init__(self):
        self.assembled = []

    def assemble(self, sample, *, supplemental_values, supplemental_defined):
        row = int(sample["row"])
        self.assembled.append((row, supplemental_values, supplemental_defined))
        state = StateFeatureGroups(
            **{
                name: np.asarray([row + index], dtype=np.float32)
                for index, name in enumerate(StateFeatureGroups._fields)
            }
        )
        defined = StateDefinedMasks(
            **{
                name: np.asarray([index % 2 == 0], dtype=np.bool_)
                for index, name in enumerate(StateFeatureGroups._fields)
            }
        )
        return ProcessTrainingExample(
            inputs=np.asarray([row, supplemental_values["typed"]], dtype=np.int32),
            targets=ProcessTrainingTarget(
                supervision={},
                retained_boundary=None,
                next_state=state,
                next_state_defined=defined,
                next_discrete_state={"row_flag": np.asarray(row % 2, dtype=np.int16)},
                diagnostics=np.asarray([row], dtype=np.float32),
            ),
        )


def _reference(*, spatial="train", temporal="train"):
    return Reference(parent=ParentReference(spatial_split=spatial, temporal_split=temporal))


def _install_loaders(monkeypatch):
    parent_loads, typed_loads = [], []

    def load_parent(path, *, contract):
        parent_loads.append((path, contract))
        return FakeShard()

    def load_typed(path, *, contract, expected_days):
        typed_loads.append((path, contract, expected_days))
        return (
            {"typed": np.asarray([10, 20, 30, 40], dtype=np.int32)},
            {"typed": np.asarray([True, False, True, True], dtype=np.bool_)},
            FakeShard.day_index.copy(),
        )

    monkeypatch.setattr(
        "research.daily_coarse_graining.daily_operator.expanded_one_day.load_markov_shard",
        load_parent,
    )
    monkeypatch.setattr(
        "research.daily_coarse_graining.daily_operator.expanded_one_day.read_typed_sidecar_shard",
        load_typed,
    )
    return parent_loads, typed_loads


def test_materializes_rows_once_in_caller_order_with_targets_and_retained_inputs(monkeypatch):
    parent_loads, typed_loads = _install_loaders(monkeypatch)
    adapter = FakeAdapter()
    target_calls = []
    schema, owner_contract = object(), object()

    def build_targets(example, *, schema, markov_contract, owner_contract):
        target_calls.append((example, schema, markov_contract, owner_contract))
        row = int(example.inputs[0])
        return {
            "process.fast": MaskedProcessTarget(
                value=np.asarray([row + 0.5], dtype=np.float64),
                defined=np.asarray([row != 1], dtype=np.bool_),
            )
        }

    monkeypatch.setattr(
        "research.daily_coarse_graining.daily_operator.expanded_one_day.build_successor_fast_owner_targets",
        build_targets,
    )
    retained = np.asarray([7.25], dtype=np.float32)

    batch = prepare_one_day_rows(
        _reference(),
        [3, 1, 2],
        adapter=adapter,
        schema=schema,
        owner_contract=owner_contract,
        day_inputs=retained,
    )

    assert len(parent_loads) == len(typed_loads) == 1
    np.testing.assert_array_equal(batch.inputs[:, 0], [3, 1, 2])
    np.testing.assert_array_equal(batch.inputs[:, 1], [40, 20, 30])
    np.testing.assert_array_equal(batch.process_targets["process.fast"].value[:, 0], [3.5, 1.5, 2.5])
    np.testing.assert_array_equal(batch.process_targets["process.fast"].defined[:, 0], [True, False, True])
    np.testing.assert_array_equal(batch.expected_next_discrete_state["row_flag"], [1, 1, 0])
    np.testing.assert_array_equal(batch.day_inputs, [[7.25], [7.25], [7.25]])
    assert batch.inputs.dtype == np.int32
    assert batch.process_targets["process.fast"].value.dtype == np.float64
    assert batch.process_targets["process.fast"].defined.dtype == np.bool_
    assert batch.expected_next_state.canopy.dtype == np.float32
    assert batch.expected_next_state_defined.canopy.dtype == np.bool_
    assert batch.day_inputs.dtype == np.float32
    assert [row for row, _values, _defined in adapter.assembled] == [3, 1, 2]
    assert all(call[1:] == (schema, FakeAdapter.contract, owner_contract) for call in target_calls)


@pytest.mark.parametrize("spatial,temporal", [("validation", "train"), ("train", "test")])
def test_rejects_non_train_reference_before_reading(monkeypatch, spatial, temporal):
    parent_loads, typed_loads = _install_loaders(monkeypatch)

    with pytest.raises(ValueError, match="non-train shard"):
        prepare_one_day_rows(
            _reference(spatial=spatial, temporal=temporal),
            [1],
            adapter=FakeAdapter(),
            schema=object(),
            owner_contract=object(),
            day_inputs=np.asarray([1.0]),
        )

    assert not parent_loads and not typed_loads


@pytest.mark.parametrize(
    ("rows", "exception", "message"),
    [
        ([], ValueError, "empty"),
        ([True], TypeError, "integer"),
        ([-1], ValueError, "non-negative"),
        ([4], ValueError, "outside its shard"),
        ([0], ValueError, "first-day lifecycle"),
    ],
)
def test_rejects_invalid_or_lifecycle_rows(monkeypatch, rows, exception, message):
    _install_loaders(monkeypatch)

    with pytest.raises(exception, match=message):
        prepare_one_day_rows(
            _reference(),
            rows,
            adapter=FakeAdapter(),
            schema=object(),
            owner_contract=object(),
            day_inputs=np.asarray([1.0]),
        )
