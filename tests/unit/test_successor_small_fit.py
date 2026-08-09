from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import research.daily_coarse_graining.daily_operator.successor_small_fit as small_fit
from research.daily_coarse_graining.daily_operator.successor_small_fit import (
    build_successor_small_fit_schedule,
    initialize_successor_adamw,
    load_successor_small_fit_contract,
    successor_adamw_update,
    successor_learning_rate,
    successor_small_fit_loss,
    successor_teacher_physical_outputs,
    verify_successor_small_fit_schedule,
)
from research.daily_coarse_graining.daily_operator.successor_training import (
    LONG_HORIZON_COMPONENTS,
)


@dataclass(frozen=True)
class _Parent:
    landpoint_id: str
    year: int
    spatial_split: str = "train"
    temporal_split: str = "train"
    sha256: str = "a" * 64


@dataclass(frozen=True)
class _Reference:
    parent: _Parent
    day_count: int = 365
    sha256: str = "b" * 64


def _references() -> tuple[_Reference, ...]:
    return tuple(
        _Reference(
            _Parent(f"{landpoint:03d}.0-071.0", year),
            sha256=f"{landpoint + year:064x}"[-64:],
        )
        for landpoint in range(1, 5)
        for year in range(1961, 1964)
    )


def test_small_fit_schedule_is_deterministic_matched_and_excludes_first_row():
    contract = load_successor_small_fit_contract()
    references = _references()
    left = build_successor_small_fit_schedule(references, contract, updates=8)
    right = build_successor_small_fit_schedule(tuple(reversed(references)), contract, updates=8)

    assert left == right
    assert len(left["entries"]) == 8
    assert len({entry["landpoint_id"] for entry in left["entries"][:4]}) == 4
    assert all(min(entry["anchor_starts"]) >= 1 for entry in left["entries"])
    assert all(min(entry["rollout_starts"]) >= 1 for entry in left["entries"])
    assert all(len(set(entry["anchor_starts"])) == 16 for entry in left["entries"])
    assert all(len(set(entry["rollout_starts"])) == 2 for entry in left["entries"])
    verify_successor_small_fit_schedule(left, references, contract)


def test_small_fit_schedule_rejects_non_train_reference():
    contract = load_successor_small_fit_contract()
    bad = _Reference(_Parent("001.0-071.0", 1961, temporal_split="validation"))
    with pytest.raises(ValueError, match="non-train"):
        build_successor_small_fit_schedule((bad,), contract, updates=1)


def test_successor_adamw_synthetic_step_is_finite_clipped_and_bounded():
    contract = load_successor_small_fit_contract()
    parameters = {"weight": jnp.asarray([1.0, -2.0], dtype=jnp.float64)}
    state = initialize_successor_adamw(parameters)

    def loss(values):
        return jnp.sum((values["weight"] - 4.0) ** 2)

    loss_value, gradients = jax.value_and_grad(loss)(parameters)
    updated, next_state, norm = successor_adamw_update(
        parameters,
        gradients,
        state,
        successor_learning_rate(contract, 1),
        contract,
    )

    assert np.isfinite(float(loss_value))
    assert float(norm) > contract.optimizer["global_gradient_norm_clip"]
    assert int(next_state.step) == 1
    assert np.all(np.isfinite(np.asarray(updated["weight"])))
    assert successor_learning_rate(contract, 50) == pytest.approx(3.0e-4)
    assert successor_learning_rate(contract, 500) == pytest.approx(3.0e-5)
    with pytest.raises(ValueError, match="outside"):
        successor_learning_rate(contract, 501)


def test_successor_combined_objective_supports_one_synthetic_update(monkeypatch):
    parameters = {"weight": jnp.asarray(2.0, dtype=jnp.float64)}
    coefficients = {
        name: value
        for name, value in zip(
            LONG_HORIZON_COMPONENTS,
            (0.1, 0.2, 0.3, 0.4),
            strict=True,
        )
    }
    monkeypatch.setattr(
        small_fit,
        "successor_anchor_batch_loss",
        lambda values, *_args, **_kwargs: values["weight"] ** 2,
    )
    monkeypatch.setattr(
        small_fit,
        "successor_rollout_component_losses",
        lambda values, *_args, **_kwargs: (
            values["weight"] * jnp.arange(1.0, 5.0),
            None,
        ),
    )

    def objective(values):
        return successor_small_fit_loss(
            values,
            SimpleNamespace(anchor_inputs=None, anchor_targets=None),
            head=None,
            schema=None,
            loss_scales={},
            transition=None,
            markov_contract=None,
            flux_scales={},
            stock_scales={},
            coefficients=coefficients,
        )

    (loss, auxiliary), gradient = jax.value_and_grad(objective, has_aux=True)(parameters)
    assert float(loss) == pytest.approx(10.0)
    assert float(auxiliary[0]) == pytest.approx(4.0)
    assert np.asarray(auxiliary[1]).tolist() == pytest.approx([2.0, 4.0, 6.0, 8.0])
    assert float(gradient["weight"]) == pytest.approx(7.0)


def test_teacher_replay_inverts_delta_and_defined_logit_representations():
    fields = (
        SimpleNamespace(
            name="direct",
            value_representation="direct",
            shape=(1,),
            day_start_state_start=None,
            day_start_state_stop=None,
        ),
        SimpleNamespace(
            name="delta",
            value_representation="day_start_delta",
            shape=(1,),
            day_start_state_start=1,
            day_start_state_stop=2,
        ),
        SimpleNamespace(
            name="status",
            value_representation="defined_logit",
            shape=(2,),
            day_start_state_start=None,
            day_start_state_stop=None,
        ),
    )
    schema = SimpleNamespace(fields=fields)
    targets = {
        "direct": SimpleNamespace(value=np.asarray([3.0])),
        "delta": SimpleNamespace(value=np.asarray([0.5])),
        "status": SimpleNamespace(value=np.asarray([1.0, 0.0])),
    }
    inputs = SimpleNamespace(continuous_state=np.asarray([1.0, 2.0]))
    outputs = successor_teacher_physical_outputs(targets, inputs, schema)

    assert np.asarray(outputs["direct"]).tolist() == [3.0]
    assert np.asarray(outputs["delta"]).tolist() == [2.5]
    assert np.asarray(outputs["status"]).tolist() == [20.0, -20.0]
