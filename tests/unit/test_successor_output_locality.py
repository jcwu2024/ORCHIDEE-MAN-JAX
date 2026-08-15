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
from research.daily_coarse_graining.daily_operator.output_locality import (
    aggregate_output_locality,
    classify_output_locality,
    summarize_output_locality_group,
)
from research.daily_coarse_graining.daily_operator.successor_fast_owner_decoder import (
    DAY_START_DELTA,
    physical_outputs_from_training_representation,
    training_representation_outputs,
)

ROOT = Path(__file__).resolve().parents[2]


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


def _group_record() -> dict:
    fields = (
        OutputFieldSpec("a", (1,), 0, "signed", process_domain="carbon"),
        OutputFieldSpec("b", (1,), 1, "signed", process_domain="carbon"),
    )
    base = {"a": np.zeros((2, 1)), "b": np.zeros((2, 1))}
    tangent = {"a": np.full((2, 1), 2.0), "b": np.full((2, 1), 2.0)}
    proposed = {"a": np.full((2, 1), 1.5), "b": np.full((2, 1), 0.5)}
    gradients = {
        "a": np.asarray([[[1.0], [1.0]], [[0.0], [0.0]]]),
        "b": np.asarray([[[0.0], [0.0]], [[2.0], [2.0]]]),
    }
    record = summarize_output_locality_group(
        fields=fields,
        field_labels={"a": "alpha", "b": "beta"},
        scales={"a": np.asarray([2.0]), "b": np.asarray([1.0])},
        base_outputs=base,
        output_tangents=tangent,
        proposed_outputs=proposed,
        output_gradients=gradients,
        loss_labels=("npp", "bgb"),
        base_losses=(10.0, 20.0),
        proposed_losses=(13.5, 23.0),
        end_to_end_base_losses=(10.0, 20.0),
        end_to_end_proposed_losses=(13.5, 23.0),
        end_to_end_first_order_changes=(2.0, 4.0),
        step_norm=0.5,
    )
    record["hard_failure_count"] = 0
    return record


def test_output_locality_decomposition_closes_and_ranks_fields():
    group = _group_record()
    npp = group["loss_decomposition"]["npp"]
    bgb = group["loss_decomposition"]["bgb"]

    assert npp["output_nonlinearity_component"] == 1.0
    assert npp["downstream_finite_response_component"] == 0.5
    assert bgb["output_nonlinearity_component"] == -2.0
    assert bgb["downstream_finite_response_component"] == 1.0
    assert npp["chain_rule_relative_error"] == 0.0
    assert bgb["chain_rule_relative_error"] == 0.0

    aggregate = aggregate_output_locality(
        (group, group),
        loss_labels=("npp", "bgb"),
        top_k=1,
    )
    assert aggregate["rankings"]["npp"]["fields"][0]["name"] == "a"
    assert aggregate["rankings"]["bgb"]["fields"][0]["name"] == "b"
    assert aggregate["rankings"]["bgb"]["top_k_absolute_coverage"] == 1.0


def test_output_locality_classification_selects_metric_scope():
    group = _group_record()
    aggregate = aggregate_output_locality(
        (group, group),
        loss_labels=("npp", "bgb"),
        top_k=1,
    )
    gates = {
        "expected_group_count": 2,
        "expected_field_count": 2,
        "key_loss_labels": ["bgb"],
        "minimum_regressing_groups_per_key": 1,
        "maximum_replay_relative_error": 1e-12,
        "maximum_chain_rule_relative_error": 1e-12,
        "minimum_output_nonlinearity_share": 0.6,
        "minimum_top_k_absolute_coverage": 0.8,
        "hard_failure_count": 0,
        "nonfinite_value_count": 0,
    }
    localized = classify_output_locality(
        aggregate,
        (group, group),
        gates=gates,
        localized_action="field",
        distributed_action="all",
        downstream_action="trajectory",
    )
    assert localized["status"] == "decoded_output_nonlinearity_localized"
    assert localized["next_action"] == "field"

    distributed_aggregate = json.loads(json.dumps(aggregate))
    distributed_aggregate["rankings"]["bgb"]["top_k_absolute_coverage"] = 0.5
    distributed = classify_output_locality(
        distributed_aggregate,
        (group, group),
        gates=gates,
        localized_action="field",
        distributed_action="all",
        downstream_action="trajectory",
    )
    assert distributed["status"] == "decoded_output_nonlinearity_distributed"
    assert distributed["next_action"] == "all"

    downstream_aggregate = json.loads(json.dumps(aggregate))
    downstream_aggregate["nonlinearity_decomposition"]["bgb"][
        "output_nonlinearity_share"
    ] = 0.2
    downstream = classify_output_locality(
        downstream_aggregate,
        (group, group),
        gates=gates,
        localized_action="field",
        distributed_action="all",
        downstream_action="trajectory",
    )
    assert downstream["status"] == "downstream_finite_response_dominant"
    assert downstream["next_action"] == "trajectory"


def test_output_locality_contract_and_launcher_are_bounded():
    contract_path = (
        ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_output_locality_audit_v1.json"
    )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    canonical = dict(payload)
    expected_hash = canonical.pop("contract_sha256")
    observed_hash = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert expected_hash == observed_hash
    assert payload["matched_reference"]["total_daily_transitions"] == 1200
    assert payload["matched_reference"]["field_intervention_runs"] == 0
    assert payload["execution"]["optimizer_updates"] == 0
    assert payload["execution"]["checkpoint_written"] is False
    assert payload["execution"]["validation_values_evaluated"] is False

    launcher = (
        ROOT / "scripts" / "hpc" / "run_successor_output_locality_audit.sh"
    ).read_text(encoding="utf-8")
    assert "output-locality audit requires free gln01" in launcher
    assert "--output-locality" in launcher
    assert "SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false" in launcher
    assert "#SBATCH" not in launcher
    assert "successor_adamw_update" not in launcher
