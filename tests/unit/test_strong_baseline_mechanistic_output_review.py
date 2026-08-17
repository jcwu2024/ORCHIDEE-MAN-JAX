from __future__ import annotations

import json
import math
from pathlib import Path

from tests.unit.test_daily_operator_successor_process_head import (
    extended_successor_setup,
    successor_setup,
)

REVIEW = (
    Path(__file__).resolve().parents[2]
    / "manifests"
    / "coarse_graining"
    / "strong_baseline_mechanistic_output_review_v1.json"
)


def _schema():
    base = successor_setup.__wrapped__()
    return extended_successor_setup.__wrapped__(base)["schema"]


def test_mechanistic_output_review_classifies_complete_strong_schema_once() -> None:
    payload = json.loads(REVIEW.read_text(encoding="utf-8"))
    assert payload["status"] == "proposal_only_user_review_required"
    assert payload["implementation_authorized"] is False

    schema = _schema()
    categories = payload["classification"]
    explicit_names = {
        key: set(record.get("fields", ()))
        for key, record in categories.items()
        if key != "learned_primary_or_endpoint"
    }
    flattened = [name for names in explicit_names.values() for name in names]
    assert len(flattened) == len(set(flattened))

    fields = {field.name: field for field in schema.fields}
    assert set(flattened) <= set(fields)
    learned = set(fields) - set(flattened)
    classified = {"learned_primary_or_endpoint": learned} | explicit_names

    assert sum(len(names) for names in classified.values()) == 181
    assert sum(
        sum(math.prod(fields[name].shape) for name in names)
        for names in classified.values()
    ) == 39_371
    for name, names in classified.items():
        declared = categories[name]
        assert len(names) == declared["field_count"]
        assert sum(math.prod(fields[field].shape) for field in names) == declared["scalar_count"]


def test_doc_interface_review_rejects_signed_or_once_daily_exact_claims() -> None:
    payload = json.loads(REVIEW.read_text(encoding="utf-8"))
    semantics = payload["doc_interface_semantics"]
    advective = semantics["typed.ok_leak.doc_advective_interface"]
    diffusive = semantics["typed.ok_leak.doc_diffusive_interface"]

    assert "nonnegative" in advective["stored_value"]
    assert "wat_flux" in advective["direction_owner"]
    assert "nonnegative" in diffusive["stored_value"]
    assert "concentration" in diffusive["direction_owner"]
    assert "exact daily" in diffusive["not_valid"]
