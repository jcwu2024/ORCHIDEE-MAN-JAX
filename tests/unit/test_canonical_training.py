from __future__ import annotations

import numpy as np

from research.daily_coarse_graining.canonical_training import (
    FastDayTargetRepresentation,
    _calendar_features,
    loss_weights_from_contract,
    model_config_from_batch,
    prepare_canonical_batch,
    prepare_canonical_inference_batch,
    restore_fast_day_inference_prediction,
    restore_fast_day_prediction,
)
from research.daily_coarse_graining.markov_dataset import (
    FiniteColumnStatistics,
    TrainingStatistics,
)


def _statistics(shapes):
    arrays = {
        name: FiniteColumnStatistics(
            count=np.full(shape, 4, dtype=np.uint64),
            mean=np.zeros(shape),
            variance=np.ones(shape),
            scale=np.ones(shape),
        )
        for name, shape in shapes.items()
    }
    return TrainingStatistics(
        dataset_id="dataset",
        teacher_git_head="teacher",
        contract_sha256="contract",
        sample_count=4,
        source_shards=("point:1961:hash",),
        observations_per_column={name: 4 for name in arrays},
        arrays=arrays,
    )


def test_prepare_canonical_batch_uses_native_forcing_and_explicit_masks():
    batch_size = 2
    batch = {
        "state": np.ones((batch_size, 4)),
        "forcing_native": np.ones((batch_size, 5, 3)),
        "parameters": np.ones((batch_size, 2)),
        "landpoint_static": np.ones((batch_size, 5)),
        "annual_conditions": np.ones((batch_size, 1)),
        "fast_day_target": np.ones((batch_size, 6)),
        "year": np.asarray([1961, 1964]),
        "day_index": np.asarray([2, 60]),
    }
    batch["state"][0, 1] = np.nan
    statistics = _statistics(
        {
            "state": (4,),
            "forcing_native": (3,),
            "parameters": (2,),
            "landpoint_static": (5,),
            "annual_conditions": (1,),
            "fast_day_target": (6,),
        }
    )

    representation = FastDayTargetRepresentation(
        state_indices=np.full(6, -1, dtype=np.int32),
        dynamic_undefined_indices=np.asarray([0], dtype=np.int32),
        dynamic_undefined_fill_values=np.asarray([1.0e20]),
    )
    prepared = prepare_canonical_batch(batch, statistics, representation)

    assert prepared.model_input.forcing_native.shape == (2, 5, 3)
    assert prepared.model_input.calendar.shape == (2, 4)
    assert not prepared.model_input.state_finite[0, 1]
    assert prepared.model_input.state[0, 1] == 0.0
    config = model_config_from_batch(prepared, hidden_width=32)
    assert config.state_width == 4
    assert config.forcing_width == 3
    assert config.fast_day_target_width == 6
    assert config.hidden_width == 32


def test_prepare_canonical_batch_persists_source_sentinel_and_learns_delta():
    batch = {
        "state": np.asarray([[10.0, 1.0e20]]),
        "forcing_native": np.ones((1, 5, 1)),
        "parameters": np.ones((1, 1)),
        "landpoint_static": np.ones((1, 1)),
        "annual_conditions": np.ones((1, 1)),
        "fast_day_target": np.asarray([[12.0, 1.0e20]]),
        "year": np.asarray([1961]),
        "day_index": np.asarray([2]),
    }
    statistics = _statistics(
        {
            "state": (2,),
            "forcing_native": (1,),
            "parameters": (1,),
            "landpoint_static": (1,),
            "annual_conditions": (1,),
            "fast_day_target": (2,),
        }
    )
    representation = FastDayTargetRepresentation(
        state_indices=np.asarray([0, 1], dtype=np.int32),
        dynamic_undefined_indices=np.asarray([1], dtype=np.int32),
        dynamic_undefined_fill_values=np.asarray([1.0e20]),
    )

    prepared = prepare_canonical_batch(batch, statistics, representation)

    np.testing.assert_allclose(
        prepared.model_input.normalized_fast_day_baseline, [[10.0, 0.0]]
    )
    np.testing.assert_array_equal(
        prepared.fast_day_target_finite, [[True, False]]
    )
    np.testing.assert_array_equal(
        prepared.fast_day_target_undefined, [[False, True]]
    )
    assert prepared.fast_day_target_undefined_values[0, 1] == 1.0e20
    np.testing.assert_array_equal(
        prepared.persistent_dynamic_undefined,
        [[True]],
    )
    np.testing.assert_array_equal(
        prepared.dynamic_undefined_flip_target,
        [[False]],
    )


def test_dynamic_undefined_classifier_learns_flips_from_persistence():
    batch = {
        "state": np.asarray([[10.0, 1.0e20], [10.0, 1.0e20]]),
        "forcing_native": np.ones((2, 5, 1)),
        "parameters": np.ones((2, 1)),
        "landpoint_static": np.ones((2, 1)),
        "annual_conditions": np.ones((2, 1)),
        "fast_day_target": np.asarray([[12.0, 1.0e20], [12.0, 5.0]]),
        "year": np.asarray([1961, 1961]),
        "day_index": np.asarray([2, 3]),
    }
    statistics = _statistics(
        {
            "state": (2,),
            "forcing_native": (1,),
            "parameters": (1,),
            "landpoint_static": (1,),
            "annual_conditions": (1,),
            "fast_day_target": (2,),
        }
    )
    representation = FastDayTargetRepresentation(
        state_indices=np.asarray([0, 1], dtype=np.int32),
        dynamic_undefined_indices=np.asarray([1], dtype=np.int32),
        dynamic_undefined_fill_values=np.asarray([1.0e20]),
    )
    prepared = prepare_canonical_batch(batch, statistics, representation)

    np.testing.assert_array_equal(
        prepared.persistent_dynamic_undefined,
        [[True], [True]],
    )
    np.testing.assert_array_equal(
        prepared.dynamic_undefined_flip_target,
        [[False], [True]],
    )
    restored = restore_fast_day_prediction(
        prepared.normalized_fast_day_target,
        np.asarray([[-4.0], [4.0]]),
        prepared,
        statistics,
        representation,
    )
    assert restored[0, 1] == 1.0e20
    assert restored[1, 1] == 5.0


def test_inference_batch_never_requires_or_reads_teacher_target():
    batch = {
        "state": np.asarray([[10.0, 1.0e20]]),
        "forcing_native": np.ones((1, 5, 1)),
        "parameters": np.ones((1, 1)),
        "landpoint_static": np.ones((1, 1)),
        "annual_conditions": np.ones((1, 1)),
        "year": np.asarray([1961]),
        "day_index": np.asarray([2]),
    }
    statistics = _statistics(
        {
            "state": (2,),
            "forcing_native": (1,),
            "parameters": (1,),
            "landpoint_static": (1,),
            "annual_conditions": (1,),
            "fast_day_target": (2,),
        }
    )
    representation = FastDayTargetRepresentation(
        state_indices=np.asarray([0, 1], dtype=np.int32),
        dynamic_undefined_indices=np.asarray([1], dtype=np.int32),
        dynamic_undefined_fill_values=np.asarray([1.0e20]),
    )

    prepared = prepare_canonical_inference_batch(
        batch,
        statistics,
        representation,
    )
    np.testing.assert_allclose(
        prepared.model_input.normalized_fast_day_baseline,
        [[10.0, 0.0]],
    )
    restored = restore_fast_day_inference_prediction(
        np.asarray([[12.0, 5.0]]),
        np.asarray([[-4.0]]),
        prepared,
        statistics,
        representation,
    )
    np.testing.assert_array_equal(restored, [[12.0, 1.0e20]])


def test_prepare_canonical_batch_rejects_nonpersistent_sentinel():
    batch = {
        "state": np.asarray([[1.0]]),
        "forcing_native": np.ones((1, 5, 1)),
        "parameters": np.ones((1, 1)),
        "landpoint_static": np.ones((1, 1)),
        "annual_conditions": np.ones((1, 1)),
        "fast_day_target": np.asarray([[1.0e20]]),
        "year": np.asarray([1961]),
        "day_index": np.asarray([2]),
    }
    statistics = _statistics(
        {
            "state": (1,),
            "forcing_native": (1,),
            "parameters": (1,),
            "landpoint_static": (1,),
            "annual_conditions": (1,),
            "fast_day_target": (1,),
        }
    )
    representation = FastDayTargetRepresentation(
        state_indices=np.asarray([0], dtype=np.int32),
        dynamic_undefined_indices=np.asarray([], dtype=np.int32),
        dynamic_undefined_fill_values=np.asarray([], dtype=np.float64),
    )

    with np.testing.assert_raises_regex(ValueError, "not persistent"):
        prepare_canonical_batch(batch, statistics, representation)


def test_calendar_features_follow_paper_noleap_calendar():
    features = _calendar_features(
        np.asarray([1963, 1964]),
        np.asarray([365, 365]),
    )
    np.testing.assert_allclose(features[0], features[1], rtol=0.0, atol=0.0)
    assert features[1, 2] == 1.0
    assert features[1, 3] == 0.0


def test_contract_weights_balance_components_and_diagnostic_leaves():
    metadata = {
        "fast_day_target_width": 6,
        "fast_day_target_leaves": [
            {"family": "hydrol", "start": 0, "stop": 2},
            {"family": "hydrol", "start": 2, "stop": 4},
            {"family": "ok_leak", "start": 4, "stop": 6},
        ],
    }

    target = loss_weights_from_contract(metadata)

    np.testing.assert_allclose(target[:4].sum(), 0.5)
    np.testing.assert_allclose(target[4:].sum(), 0.5)
