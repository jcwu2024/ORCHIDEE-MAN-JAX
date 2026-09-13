"""Materialize explicit raw-v6 one-day rows for target-oriented training.

This host-only boundary deliberately owns no sampling, artifact serialization,
model initialization, or Teacher execution.  It preserves the one-day portion
of :func:`prepare_target_oriented_training_batch` while allowing a caller to
expand an accepted train/train typed shard beyond a prepared 32-row package.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import numpy as np

from research.daily_coarse_graining.daily_markov_contract import load_markov_shard
from research.daily_coarse_graining.typed_sidecar import read_typed_sidecar_shard

from .successor_fast_owner_training import build_successor_fast_owner_targets
from .target_oriented_training import PreparedTargetOrientedOneDayBatch


def _stack(values: Sequence[Any]) -> Any:
    """Use the same NumPy host stacking boundary as the prepared-v6 path."""

    if not values:
        raise ValueError("cannot stack an empty expanded one-day batch")
    return jax.tree_util.tree_map(lambda *items: np.stack(items), *values)


def _row_index(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("expanded one-day row must be an integer")
    try:
        row = int(operator.index(value))
    except TypeError as error:
        raise TypeError("expanded one-day row must be an integer") from error
    if row < 0:
        raise ValueError("expanded one-day row must be non-negative")
    return row


def _rows(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes, bytearray, Mapping)):
        raise TypeError("expanded one-day rows must be a sequence of integers")
    try:
        raw_rows = tuple(values)
    except TypeError as error:
        raise TypeError("expanded one-day rows must be a sequence of integers") from error
    rows = tuple(_row_index(value) for value in raw_rows)
    if not rows:
        raise ValueError("cannot materialize an empty expanded one-day batch")
    return rows


def prepare_one_day_rows(
    reference: Any,
    rows: Sequence[int],
    *,
    adapter: Any,
    schema: Any,
    owner_contract: Any,
    day_inputs: Any,
) -> PreparedTargetOrientedOneDayBatch:
    """Construct an ordered target-oriented one-day batch from one raw-v6 shard.

    ``reference`` is one typed-sidecar shard reference.  ``rows`` are explicit
    parent-shard offsets, retained exactly in caller order.  The same retained
    day inputs apply to every row because they are point-year owned.
    """

    selected_rows = _rows(rows)
    parent_ref = reference.parent
    if parent_ref.spatial_split != "train" or parent_ref.temporal_split != "train":
        raise ValueError("expanded one-day batch attempted to load a non-train shard")

    # Exactly one parent and typed read per point-year materialization call.
    parent = load_markov_shard(parent_ref.path, contract=adapter.contract)
    values, defined, days = read_typed_sidecar_shard(
        reference.path,
        contract=adapter.typed_contract,
        expected_days=reference.day_count,
    )
    if not np.array_equal(days, parent.day_index):
        raise ValueError("expanded one-day typed and parent day axes differ")

    examples = []
    for row in selected_rows:
        if row >= int(parent.days):
            raise ValueError("expanded one-day row lies outside its shard")
        if int(parent.day_index[row]) < 2:
            raise ValueError("expanded one-day batch entered a first-day lifecycle row")
        examples.append(
            adapter.assemble(
                parent.sample(row),
                supplemental_values={name: value[row] for name, value in values.items()},
                supplemental_defined={name: value[row] for name, value in defined.items()},
            )
        )

    return PreparedTargetOrientedOneDayBatch(
        inputs=_stack([value.inputs for value in examples]),
        process_targets=_stack(
            [
                build_successor_fast_owner_targets(
                    value,
                    schema=schema,
                    markov_contract=adapter.contract,
                    owner_contract=owner_contract,
                )
                for value in examples
            ]
        ),
        expected_next_state=_stack([value.targets.next_state for value in examples]),
        expected_next_state_defined=_stack([value.targets.next_state_defined for value in examples]),
        expected_next_discrete_state=_stack(
            [value.targets.next_discrete_state for value in examples]
        ),
        day_inputs=_stack([day_inputs for _value in examples]),
    )


__all__ = ["prepare_one_day_rows"]
