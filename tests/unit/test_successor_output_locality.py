from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.neural_architecture import (
    NeuralOutputSchema,
    OutputFieldSpec,
)
from research.daily_coarse_graining.daily_operator.successor_fast_owner_decoder import (
    DAY_START_DELTA,
    physical_outputs_from_training_representation,
    training_representation_outputs,
)

ROOT = Path(__file__).resolve().parents[2]




def _projection_record(
    correction_norm: float,
    *,
    safe: bool,
    maintenance_rms: float,
    maintenance_maximum: float,
    displacement_fraction: float = 0.5,
) -> dict:
    return {
        "correction_norm": correction_norm,
        "loss_summary": {
            "all_group_components_nonincreasing": safe,
            "component_median_relative_changes": {
                "one_day_anchor": -2.0e-5,
                "npp": -2.0e-5,
                "aboveground_biomass_tendency": -2.0e-5,
                "belowground_biomass_tendency": -2.0e-5,
            },
        },
        "maintenance_change": {
            "normalized_exact_rms": maintenance_rms,
            "maximum_absolute_exact_change": maintenance_maximum,
        },
        "maintenance_cap": {
            "normalized_exact_rms": 0.1,
            "maximum_absolute_exact_change": 1.0,
        },
        "displacement_fraction": displacement_fraction,
        "hard_failure_count": 0,
        "nonfinite_value_count": 0,
    }






def test_fast_owner_training_representation_round_trips_physical_outputs():
    schema = NeuralOutputSchema(
        fields=(
            OutputFieldSpec(
                name="delta",
                shape=(2,),
                label_index=0,
                constraint="signed",
                value_representation=DAY_START_DELTA,
                day_start_owner="state",
                day_start_state_start=0,
                day_start_state_stop=2,
            ),
            OutputFieldSpec(
                name="direct",
                shape=(1,),
                label_index=0,
                constraint="signed",
            ),
        ),
        labels=("label",),
        active_pft_indices=(0,),
    )
    inputs = SimpleNamespace(
        canonical_input=SimpleNamespace(
            continuous_state=jnp.asarray([10.0, 20.0], dtype=jnp.float64)
        )
    )
    physical = {
        "delta": jnp.asarray([13.0, 24.0], dtype=jnp.float64),
        "direct": jnp.asarray([5.0], dtype=jnp.float64),
    }

    represented = training_representation_outputs(physical, inputs, schema)
    rebuilt = physical_outputs_from_training_representation(
        represented,
        inputs,
        schema,
    )

    np.testing.assert_array_equal(represented["delta"], [3.0, 4.0])
    np.testing.assert_array_equal(represented["direct"], [5.0])
    np.testing.assert_array_equal(rebuilt["delta"], physical["delta"])
    np.testing.assert_array_equal(rebuilt["direct"], physical["direct"])
