import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.mixed_training_state import (
    CHECKPOINT_SCHEMA,
    MAXIMUM_UPDATES,
    MILESTONE_UPDATES,
    load_mixed_training_checkpoint,
    save_mixed_training_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    initialize_target_oriented_adamw,
    target_oriented_adamw_update,
)

OPTIMIZER = {
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1.0e-8,
    "weight_decay": 0.0,
}
METHOD = {
    "schema_version": "fresh_mixed_learning_method_v1",
    "updates": 200,
    "daily_batch_size": 64,
    "daily_microbatch_size": 8,
}
SOURCES = {
    "daily": {"identity": "expanded-one-day-v1"},
    "recurrent": {"identity": "prepared-recurrent-v1"},
}
SCHEDULE = {
    "schema_version": "mixed_learning_recurrent_schedule_v1",
    "sha256": "schedule-hash",
}
FRESH = {"seed": 20260911, "parameter_count": 109_464_375}


def _loss(jnp, parameters, batch):
    features, targets = batch
    prediction = features @ parameters["active"]
    return jnp.mean((prediction - targets) ** 2)


def _assert_tree_exact(jax, left, right):
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    assert left_tree == right_tree
    for actual, expected in zip(left_leaves, right_leaves, strict=True):
        np.testing.assert_array_equal(actual, expected)


def _save(path, *, jax, parameters, optimizer_state, completed=1, **kwargs):
    return save_mixed_training_checkpoint(
        path,
        jax=jax,
        parameters=parameters,
        optimizer_state=optimizer_state,
        completed_updates=completed,
        daily_sampler_cursor={"epoch": 0, "batch_offset": completed},
        recurrent_schedule_cursor=completed,
        method=METHOD,
        source_identities=SOURCES,
        schedule_identity=SCHEDULE,
        fresh_initialization_identity=FRESH,
        **kwargs,
    )


def _load(path, **kwargs):
    return load_mixed_training_checkpoint(
        path,
        expected_method=kwargs.get("method", METHOD),
        expected_source_identities=kwargs.get("sources", SOURCES),
        expected_schedule_identity=kwargs.get("schedule", SCHEDULE),
        expected_fresh_initialization_identity=kwargs.get("fresh", FRESH),
    )


def test_full_state_resume_is_exact_and_preserves_none_placeholders(tmp_path):
    import jax
    import jax.numpy as jnp

    initial = {
        "active": jnp.asarray([0.2, -0.1], dtype=jnp.float32),
        "inactive_axis_embedding": None,
    }
    batches = (
        (
            jnp.asarray([[1.0, 3.0], [-2.0, 0.5]], dtype=jnp.float32),
            jnp.asarray([1.5, -0.5], dtype=jnp.float32),
        ),
        (
            jnp.asarray([[4.0, -1.0], [0.25, 2.0]], dtype=jnp.float32),
            jnp.asarray([-2.0, 0.75], dtype=jnp.float32),
        ),
    )
    contract = SimpleNamespace(optimizer=OPTIMIZER)
    gradient = jax.grad(lambda values, batch: _loss(jnp, values, batch))

    state0 = initialize_target_oriented_adamw(initial)
    parameters1, state1, _, _ = target_oriented_adamw_update(
        initial,
        gradient(initial, batches[0]),
        state0,
        learning_rate=5.0e-3,
        clip_norm=0.5,
        contract=contract,
    )
    uninterrupted_parameters, uninterrupted_state, _, _ = target_oriented_adamw_update(
        parameters1,
        gradient(parameters1, batches[1]),
        state1,
        learning_rate=5.0e-3,
        clip_norm=0.5,
        contract=contract,
    )

    path = tmp_path / "mixed-update-001.pkl"
    saved = _save(
        path,
        jax=jax,
        parameters=parameters1,
        optimizer_state=state1,
    )
    restored = _load(path)
    assert saved["schema_version"] == CHECKPOINT_SCHEMA
    assert restored["completed_updates"] == 1
    assert restored["daily_sampler_cursor"] == {"epoch": 0, "batch_offset": 1}
    assert restored["recurrent_schedule_cursor"] == 1
    assert restored["parameters"]["inactive_axis_embedding"] is None
    assert restored["optimizer_state"].first["inactive_axis_embedding"] is None
    _assert_tree_exact(jax, restored["parameters"], jax.device_get(parameters1))
    _assert_tree_exact(jax, restored["optimizer_state"], jax.device_get(state1))

    resumed_parameters1 = jax.device_put(restored["parameters"])
    resumed_state1 = jax.device_put(restored["optimizer_state"])
    resumed_parameters, resumed_state, _, _ = target_oriented_adamw_update(
        resumed_parameters1,
        gradient(resumed_parameters1, batches[1]),
        resumed_state1,
        learning_rate=5.0e-3,
        clip_norm=0.5,
        contract=contract,
    )
    _assert_tree_exact(jax, resumed_parameters, uninterrupted_parameters)
    _assert_tree_exact(jax, resumed_state, uninterrupted_state)
    assert not tuple(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize(
    ("daily", "recurrent", "message"),
    (
        ({"epoch": 0, "batch_offset": 0}, 1, "daily sampler cursor"),
        ({"epoch": 0, "batch_offset": 1}, 0, "recurrent schedule cursor"),
        ({"epoch": 1, "batch_offset": 1}, 1, "daily sampler cursor"),
    ),
)
def test_save_rejects_unpaired_progress(tmp_path, daily, recurrent, message):
    import jax
    import jax.numpy as jnp

    parameters = {"active": jnp.asarray([1.0])}
    with pytest.raises(ValueError, match=message):
        save_mixed_training_checkpoint(
            tmp_path / "invalid.pkl",
            jax=jax,
            parameters=parameters,
            optimizer_state=initialize_target_oriented_adamw(parameters)._replace(
                step=jnp.asarray(1, dtype=jnp.int32)
            ),
            completed_updates=1,
            daily_sampler_cursor=daily,
            recurrent_schedule_cursor=recurrent,
            method=METHOD,
            source_identities=SOURCES,
            schedule_identity=SCHEDULE,
            fresh_initialization_identity=FRESH,
        )


def test_load_rejects_every_identity_mismatch(tmp_path):
    import jax
    import jax.numpy as jnp

    parameters = {"active": jnp.asarray([1.0])}
    state = initialize_target_oriented_adamw(parameters)
    path = tmp_path / "mixed-update-000.pkl"
    _save(path, jax=jax, parameters=parameters, optimizer_state=state, completed=0)

    mismatches = (
        ({"method": {**METHOD, "updates": 201}}, "method mismatch"),
        ({"sources": {**SOURCES, "daily": {"identity": "other"}}}, "source_identities mismatch"),
        ({"schedule": {**SCHEDULE, "sha256": "other"}}, "schedule_identity mismatch"),
        ({"fresh": {**FRESH, "seed": 3}}, "fresh_initialization_identity mismatch"),
    )
    for arguments, message in mismatches:
        with pytest.raises(ValueError, match=message):
            _load(path, **arguments)


def test_load_rejects_tampered_paired_cursor_and_optimizer_step(tmp_path):
    import jax
    import jax.numpy as jnp

    parameters = {"active": jnp.asarray([1.0])}
    state0 = initialize_target_oriented_adamw(parameters)
    path = tmp_path / "mixed-update-000.pkl"
    _save(path, jax=jax, parameters=parameters, optimizer_state=state0, completed=0)
    with path.open("rb") as handle:
        payload = pickle.load(handle)

    payload["recurrent_schedule_cursor"] = 1
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with pytest.raises(ValueError, match="recurrent schedule cursor"):
        _load(path)

    payload["recurrent_schedule_cursor"] = 0
    payload["optimizer_state"] = state0._replace(step=jnp.asarray(1, dtype=jnp.int32))
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with pytest.raises(ValueError, match="AdamW step"):
        _load(path)


def test_milestone_contract_limit_and_immutable_default(tmp_path):
    import jax
    import jax.numpy as jnp

    assert MILESTONE_UPDATES == (0, 50, 100, 200)
    assert MAXIMUM_UPDATES == 200
    parameters = {"active": jnp.asarray([1.0])}
    state = initialize_target_oriented_adamw(parameters)
    path = tmp_path / "mixed-update-000.pkl"
    _save(path, jax=jax, parameters=parameters, optimizer_state=state, completed=0)
    with pytest.raises(FileExistsError):
        _save(path, jax=jax, parameters=parameters, optimizer_state=state, completed=0)
    _save(
        path,
        jax=jax,
        parameters=parameters,
        optimizer_state=state,
        completed=0,
        replace=True,
    )

    with pytest.raises(ValueError, match="exceeds"):
        _save(
            tmp_path / "too-far.pkl",
            jax=jax,
            parameters=parameters,
            optimizer_state=state._replace(step=jnp.asarray(201, dtype=jnp.int32)),
            completed=201,
        )
