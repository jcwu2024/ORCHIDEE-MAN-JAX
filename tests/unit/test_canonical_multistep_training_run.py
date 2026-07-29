from __future__ import annotations

import copy
import hashlib
import json
import pickle
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining import canonical_multistep_training_run
from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalModelConfig,
    initialize_canonical_model,
)
from research.daily_coarse_graining.canonical_multistep import (
    CanonicalMultistepSequence,
)
from research.daily_coarse_graining.canonical_multistep_training_run import (
    _compiled_forcing_batch,
    _load_initial_parameters,
    _load_plan,
    _make_train_step,
    build_parser,
    parse_curriculum,
    train_multistep,
)
from research.daily_coarse_graining.canonical_training import (
    FastDayTargetRepresentation,
)
from research.daily_coarse_graining.canonical_training_run import (
    CHECKPOINT_SCHEMA_VERSION,
    _adam_init,
)
from research.daily_coarse_graining.daily_model_architecture import (
    AXIS_PROCESS_COUPLED_V1,
    CANONICAL_FLAT_V1,
    STRUCTURED_PROCESS_FILM_V1,
    build_daily_model_definition,
)
from research.daily_coarse_graining.markov_dataset import (
    FiniteColumnStatistics,
    TrainingStatistics,
)


def _statistics():
    shapes = {
        "state": (2,),
        "forcing_native": (1,),
        "parameters": (1,),
        "landpoint_static": (1,),
        "annual_conditions": (1,),
        "fast_day_target": (2,),
        "state_delta": (2,),
    }
    arrays = {
        name: FiniteColumnStatistics(
            count=np.ones(shape, dtype=np.uint64),
            mean=np.zeros(shape),
            variance=np.ones(shape),
            scale=np.ones(shape),
        )
        for name, shape in shapes.items()
    }
    return TrainingStatistics("test", "test", "test", 1, ("test",), {}, arrays)


def _tail(state, discrete, fast_day, forcing, year, day_index):
    del forcing, year, day_index
    return state + 0.1 * fast_day, discrete


def test_curriculum_parser_requires_unique_increasing_supported_horizons():
    assert parse_curriculum("1:8,3:4,7:2") == (
        parse_curriculum("1:8")[0],
        parse_curriculum("3:4")[0],
        parse_curriculum("7:2")[0],
    )
    with pytest.raises(ValueError, match="unique and increasing"):
        parse_curriculum("3:1,1:1")
    with pytest.raises(ValueError, match="1, 3, or 7"):
        parse_curriculum("2:1")


def test_multistep_cli_preserves_v1_default_and_accepts_v2_objective():
    parser = build_parser()
    required = [
        "--dataset",
        "dataset.json",
        "--statistics",
        "statistics.json",
        "--acceptance",
        "acceptance.json",
        "--output-dir",
        "output",
    ]

    baseline = parser.parse_args(required)
    candidate = parser.parse_args(required + ["--objective", "process_increment_v2"])
    axis_process = parser.parse_args(
        required + ["--model-architecture", AXIS_PROCESS_COUPLED_V1]
    )

    assert baseline.objective == "canonical_multistep_v1"
    assert baseline.model_architecture == CANONICAL_FLAT_V1
    assert candidate.objective == "process_increment_v2"
    assert candidate.state_increment_loss_weight == 1.0
    assert candidate.state_delta_floor_ratio == 1.0e-3
    assert axis_process.model_architecture == AXIS_PROCESS_COUPLED_V1


def test_structured_candidate_requires_a_frozen_flat_initialization():
    with pytest.raises(ValueError, match="requires --initialize-checkpoint"):
        train_multistep(
            type(
                "Args",
                (),
                {
                    "model_architecture": STRUCTURED_PROCESS_FILM_V1,
                    "initialize_checkpoint": None,
                },
            )()
        )


def test_flat_checkpoint_upgrades_to_zero_residual_structured_parameters(tmp_path):
    config = CanonicalModelConfig(8, 1, 1, 1, 1, 2, 1, 2, 2, 2, 3)
    metadata = {
        "continuous_state_width": 8,
        "state_leaves": [
            {"component": component, "path": [field], "start": i, "stop": i + 1}
            for i, (component, field) in enumerate(
                (
                    ("slowproc_stomate_previous_step_state", "gpp_daily"),
                    ("slowproc_stomate_previous_step_state", "biomass"),
                    ("slowproc_stomate_previous_step_state", "litter"),
                    ("slowproc_stomate_previous_step_state", "DOC"),
                    ("slowproc_stomate_previous_step_state", "herbivores"),
                    ("hydrol_previous_step_state", "soil_moisture"),
                    ("thermosoil_previous_step_state", "soil_temperature"),
                    ("diffuco_previous_step_state", "rveget"),
                )
            )
        ],
    }
    definition = build_daily_model_definition(
        STRUCTURED_PROCESS_FILM_V1,
        config,
        metadata,
    )
    identity = {
        "dataset_id": "dataset",
        "teacher_git_head": "teacher",
        "markov_contract_sha256": "contract",
        "statistics_sha256": "statistics",
        "acceptance_sha256": "acceptance",
        "model_config": config._asdict(),
        "model_architecture": definition.identity(),
    }
    canonical = initialize_canonical_model(config, seed=17)
    checkpoint = tmp_path / "flat.pkl"
    checkpoint.write_bytes(
        pickle.dumps(
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "identity": {
                    key: value
                    for key, value in identity.items()
                    if key != "model_architecture"
                },
                "parameters": jax.device_get(canonical),
            }
        )
    )

    upgraded = _load_initial_parameters(
        checkpoint,
        identity=identity,
        config=config,
        seed=17,
        model_definition=definition,
    )

    assert all(
        np.array_equal(np.asarray(actual), np.asarray(expected))
        for actual, expected in zip(
            jax.tree_util.tree_leaves(upgraded.canonical),
            jax.tree_util.tree_leaves(canonical),
            strict=True,
        )
    )
    assert all(
        np.all(np.asarray(value.weight) == 0.0)
        for value in upgraded.state_group_outputs
    )


def test_load_plan_uses_canonical_json_hash_not_file_formatting(tmp_path):
    plan = {
        "entries": [
            {
                "landpoint_id": "001.0-071.0",
                "run_def": "run.def",
                "reference_run_dir": "reference",
            }
        ]
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    canonical = json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()

    loaded, entries = _load_plan(path, hashlib.sha256(canonical).hexdigest())

    assert loaded == plan
    assert entries["001.0-071.0"] == plan["entries"][0]


def test_load_plan_rejects_wrong_canonical_hash(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text('{"entries": []}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="generation plan hash"):
        _load_plan(path, "0" * 64)


def test_compiled_forcing_batch_deduplicates_overlapping_calendar_days(
    monkeypatch,
):
    calls = []

    def reconstruct(native, _spec, _context):
        calls.append(float(native[0, 0]))
        return {"value": jnp.asarray([float(native[0, 0])])}

    monkeypatch.setattr(
        canonical_multistep_training_run,
        "reconstruct_retained_tail_forcing_day",
        reconstruct,
    )
    batch = {
        "forcing_native": np.asarray(
            [
                [[[10.0]], [[11.0]], [[12.0]]],
                [[[11.0]], [[12.0]], [[13.0]]],
            ]
        ),
        "year": np.full((2, 3), 1962),
        "day_index": np.asarray([[10, 11, 12], [11, 12, 13]]),
    }

    result = _compiled_forcing_batch(
        batch,
        SimpleNamespace(native_forcing=object()),
        object(),
    )

    assert calls == [10.0, 11.0, 12.0, 13.0]
    assert result["value"].shape == (2, 3, 1)
    assert np.array_equal(
        np.asarray(result["value"][:, :, 0]),
        np.asarray([[10.0, 11.0, 12.0], [11.0, 12.0, 13.0]]),
    )

    changed = copy.deepcopy(batch)
    changed["forcing_native"][1, 0, 0, 0] = 99.0
    with pytest.raises(ValueError, match="different native data"):
        _compiled_forcing_batch(
            changed,
            SimpleNamespace(native_forcing=object()),
            object(),
        )


def test_compiled_multistep_train_step_updates_parameters_with_finite_gradient():
    config = CanonicalModelConfig(2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 3)
    parameters = initialize_canonical_model(config, seed=2)
    optimizer = _adam_init(parameters)
    representation = FastDayTargetRepresentation(
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        np.asarray([1.0e20]),
    )
    batch = 2
    horizon = 3
    sequence = CanonicalMultistepSequence(
        forcing_native=jnp.ones((batch, horizon, 5, 1)),
        parameters=jnp.ones((batch, horizon, 1)),
        landpoint_static=jnp.ones((batch, horizon, 1)),
        annual_conditions=jnp.ones((batch, horizon, 1)),
        year=jnp.full((batch, horizon), 1962),
        day_index=jnp.tile(jnp.arange(2, 5), (batch, 1)),
        teacher_state=jnp.zeros((batch, horizon, 2)),
        teacher_fast_day_target=jnp.full((batch, horizon, 2), 0.2),
        teacher_next_state=jnp.full((batch, horizon, 2), 0.4),
        retained_tail_inputs=jnp.ones((batch, horizon, 1)),
    )
    step = _make_train_step(
        statistics=_statistics(),
        representation=representation,
        weights=jnp.asarray([0.5, 0.5]),
        transition=_tail,
        state_loss_weight=1.0,
        undefined_loss_weight=0.1,
    )
    updated, _, loss, gradient_norm = step(
        parameters,
        optimizer,
        jnp.asarray([[0.1, 0.2], [0.2, 0.3]]),
        {"flag": jnp.asarray([[True], [False]])},
        sequence,
        jnp.asarray(1.0e-3),
    )
    assert np.isfinite(np.asarray(loss))
    assert float(gradient_norm) > 0.0
    assert any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree_util.tree_leaves(parameters),
            jax.tree_util.tree_leaves(updated),
            strict=True,
        )
    )


def test_compiled_process_increment_train_step_has_finite_gradient():
    config = CanonicalModelConfig(2, 1, 1, 1, 1, 2, 1, 2, 2, 2, 3)
    parameters = initialize_canonical_model(config, seed=7)
    optimizer = _adam_init(parameters)
    representation = FastDayTargetRepresentation(
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        np.asarray([1.0e20]),
    )
    batch = 2
    horizon = 3
    sequence = CanonicalMultistepSequence(
        forcing_native=jnp.ones((batch, horizon, 5, 1)),
        parameters=jnp.ones((batch, horizon, 1)),
        landpoint_static=jnp.ones((batch, horizon, 1)),
        annual_conditions=jnp.ones((batch, horizon, 1)),
        year=jnp.full((batch, horizon), 1962),
        day_index=jnp.tile(jnp.arange(2, 5), (batch, 1)),
        teacher_state=jnp.zeros((batch, horizon, 2)),
        teacher_fast_day_target=jnp.full((batch, horizon, 2), 0.2),
        teacher_next_state=jnp.full((batch, horizon, 2), 0.4),
        retained_tail_inputs=jnp.ones((batch, horizon, 1)),
    )
    step = _make_train_step(
        statistics=_statistics(),
        representation=representation,
        weights=jnp.asarray([0.5, 0.5]),
        transition=_tail,
        state_loss_weight=1.0,
        undefined_loss_weight=0.1,
        state_weights=jnp.asarray([0.75, 0.25]),
        state_delta_scale=jnp.asarray([0.1, 0.2]),
        state_increment_loss_weight=1.0,
    )

    _, _, loss, gradient_norm = step(
        parameters,
        optimizer,
        jnp.asarray([[0.1, 0.2], [0.2, 0.3]]),
        {"flag": jnp.asarray([[True], [False]])},
        sequence,
        jnp.asarray(1.0e-3),
    )

    assert np.isfinite(np.asarray(loss))
    assert np.isfinite(np.asarray(gradient_norm))
    assert float(gradient_norm) > 0.0
