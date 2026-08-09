import copy
import json
from pathlib import Path

import pytest

from scripts.dev import (
    build_successor_fast_owner_reconstruction_contract as builder,
)

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "manifests" / "coarse_graining" / "successor_fast_owner_reconstruction_v1.json"
DOCUMENT_PATH = (
    ROOT / "docs" / "research" / "daily_coarse_graining" / "successor_fast_owner_reconstruction_contract_v1.md"
)


def _contract():
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _rehash(payload):
    payload["contract_sha256"] = builder._payload_sha256(payload)
    return payload


def test_frozen_fast_owner_contract_is_complete_and_self_validating():
    payload = _contract()

    summary = builder.validate_contract(payload)

    assert summary == {
        "contract_sha256": "4db5f828442212c58903096c36015f9d769c19aeaaa4eb9a4326fbea2fdfc4c9",
        "dynamic_defined_output_count": 1,
        "dynamic_defined_output_width": 2,
        "leaf_count": 160,
        "neural_value_output_count": 79,
        "neural_value_output_width": 839,
        "role_leaf_counts": builder.EXPECTED_ROLE_LEAF_COUNTS,
        "target_width": 2855,
        "typed_collision_count": 14,
    }
    assert payload["role_column_counts"] == builder.EXPECTED_ROLE_COLUMN_COUNTS
    assert payload["decoder_extension"]["total_added_scalar_width"] == 841


def test_fast_owner_slices_outputs_and_provenance_have_unique_explicit_ownership():
    payload = _contract()
    leaves = payload["leaves"]
    keys = [record["key"] for record in leaves]
    outputs = [record["output_key"] for record in leaves if record["role"] == builder.ROLE_NEURAL]
    defined_outputs = [record["defined_output_key"] for record in leaves if "defined_output_key" in record]

    assert len(keys) == len(set(keys)) == 160
    assert len(outputs) == len(set(outputs)) == 79
    assert len(defined_outputs) == len(set(defined_outputs)) == 1
    assert set(outputs).isdisjoint(defined_outputs)
    assert all("anonymous" not in record["canonical_owner"] for record in leaves)
    assert all(record["provenance"]["source_ref"] for record in leaves)
    assert [(record["start"], record["stop"]) for record in leaves] == [
        (start, stop)
        for start, stop in zip(
            [0, *(record["stop"] for record in leaves[:-1])],
            (record["stop"] for record in leaves),
            strict=True,
        )
    ]


def test_mirrors_are_direct_and_a_mirror_chain_is_rejected():
    payload = _contract()
    by_key = {record["key"]: record for record in payload["leaves"]}
    mirrors = [record for record in payload["leaves"] if record["role"] == builder.ROLE_MIRROR]

    assert len(mirrors) == 30
    assert all(record["canonical_owner"] in by_key for record in mirrors)
    assert all(by_key[record["canonical_owner"]]["role"] != builder.ROLE_MIRROR for record in mirrors)

    tampered = copy.deepcopy(payload)
    tampered_by_key = {record["key"]: record for record in tampered["leaves"]}
    tampered_by_key["diffuco_previous_step_state.snow"]["canonical_owner"] = "diffuco_previous_step_state.snow_age"
    with pytest.raises(ValueError, match="mirror chain or cycle"):
        builder.validate_contract(_rehash(tampered))


def test_typed_integrals_never_own_instantaneous_fast_state():
    payload = _contract()
    collisions = {record["key"]: record for record in payload["leaves"] if "typed_integral_collision" in record}

    assert set(collisions) == builder._SEMANTIC_COLLISION_KEYS
    assert len(collisions) == 14
    assert all(not record["canonical_owner"].startswith("typed.") for record in collisions.values())
    assert all(
        record["typed_integral_collision"]["required_source"] == record["canonical_owner"]
        for record in collisions.values()
    )


def test_only_rveget_has_a_dynamic_defined_head_and_pft_axis():
    payload = _contract()
    dynamic = [
        record for record in payload["leaves"] if record["defined_policy"] == "neural_dynamic_value_and_defined_logit"
    ]

    assert [record["key"] for record in dynamic] == ["diffuco_previous_step_state.rveget"]
    assert dynamic[0]["shape"] == [1, 2]
    assert dynamic[0]["pft_axis"] == 1
    assert dynamic[0]["pft_axis_policy"] == "compact_active_pft_axis"
    assert dynamic[0]["loss_mask_policy"].endswith("eligible_slot_mask_bce")


def test_v6_and_source_contract_identities_are_locked():
    payload = _contract()

    assert payload["dataset"] == {
        "dataset_id": builder.DATASET_ID,
        "dataset_manifest_sha256": builder.DATASET_MANIFEST_SHA256,
        "historical_v5_allowed": False,
        "markov_contract_sha256": builder.MARKOV_CONTRACT_SHA256,
        "release_id": "pft14-daily-teacher-v6",
        "source_contract_sha256": builder.SOURCE_CONTRACT_SHA256,
        "teacher_regeneration_allowed": False,
    }
    assert builder._sha256_file(builder.DEFAULT_SOURCE_CONTRACT) == (builder.SOURCE_CONTRACT_SHA256)

    tampered = copy.deepcopy(payload)
    tampered["dataset"]["markov_contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="markov_contract_sha256 identity drift"):
        builder.validate_contract(_rehash(tampered))


def test_promoted_nonexact_owner_leaves_are_independent_neural_endpoints():
    payload = _contract()
    by_key = {record["key"]: record for record in payload["leaves"]}
    expected_domains = {
        "daily_interface.wspeed_daily": "energy",
        "daily_interface.resp_maint_radia": "carbon",
        "final_diagnostics.temp_sol": "energy",
        "diffuco_previous_step_state.control_inudate": "carbon",
    }

    for key, domain in expected_domains.items():
        assert by_key[key]["role"] == builder.ROLE_NEURAL
        assert by_key[key]["proposed_process_domain"] == domain


def test_review_document_is_generated_from_all_79_named_value_heads():
    payload = _contract()
    expected = builder.render_markdown(payload)

    assert DOCUMENT_PATH.read_text(encoding="utf-8") == expected
    neural = [record for record in payload["leaves"] if record["role"] == builder.ROLE_NEURAL]
    assert len(neural) == 79
    assert all(expected.count(f"| `{record['key']}` | `{record['output_key']}` |") == 1 for record in neural)
    assert "Total added scalar output width: 841" in expected
    assert "schema implemented in the all-Teacher materializer" in expected
