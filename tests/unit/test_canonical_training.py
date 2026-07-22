from __future__ import annotations

import numpy as np

from research.daily_coarse_graining.canonical_training import (
    loss_weights_from_contract,
    model_config_from_batch,
    prepare_canonical_batch,
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
        "state_delta": np.ones((batch_size, 4)),
        "diagnostics": np.ones((batch_size, 2)),
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
            "state_delta": (4,),
            "diagnostics": (2,),
        }
    )

    prepared = prepare_canonical_batch(batch, statistics)

    assert prepared.model_input.forcing_native.shape == (2, 5, 3)
    assert prepared.model_input.calendar.shape == (2, 4)
    assert not prepared.model_input.state_finite[0, 1]
    assert prepared.model_input.state[0, 1] == 0.0
    config = model_config_from_batch(prepared, hidden_width=32)
    assert config.state_width == 4
    assert config.forcing_width == 3
    assert config.hidden_width == 32


def test_contract_weights_balance_components_and_diagnostic_leaves():
    metadata = {
        "continuous_state_width": 6,
        "diagnostic_width": 3,
        "state_leaves": [
            {"component": "hydrol", "start": 0, "stop": 2, "classification": "prognostic"},
            {"component": "hydrol", "start": 2, "stop": 4, "classification": "prognostic"},
            {"component": "stomate", "start": 4, "stop": 6, "classification": "prognostic"},
            {"component": "driver", "start": None, "stop": None, "classification": "discrete"},
        ],
        "diagnostic_leaves": [
            {"name": "gpp", "start": 0, "stop": 1},
            {"name": "temperature", "start": 1, "stop": 3},
        ],
    }

    state, diagnostic = loss_weights_from_contract(metadata)

    np.testing.assert_allclose(state[:4].sum(), 0.5)
    np.testing.assert_allclose(state[4:].sum(), 0.5)
    np.testing.assert_allclose(diagnostic[:1].sum(), 0.5)
    np.testing.assert_allclose(diagnostic[1:].sum(), 0.5)
