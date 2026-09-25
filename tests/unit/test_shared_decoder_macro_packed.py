"""Focused checks for the diagnostic non-GPP macro-row decoder."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.shared_decoder_macro_packed import (
    packed_decode_fields,
    reference_decode_fields,
)
from research.daily_coarse_graining.daily_operator.shared_transformer import SharedTransformerConfig
from scripts.hpc import audit_shared_decoder_cost
from scripts.hpc import probe_shared_decoder_macro_packed as probe
from scripts.hpc.probe_shared_decoder_macro_packed import FIELD_SPECS, fixture, prepare_fields


@pytest.fixture(scope="module")
def case():
    shared, payload, masks = fixture(8)
    fields, experts = prepare_fields(shared, payload, masks)
    config = SharedTransformerConfig(width=8, ffn_width=32, coordinate_blocks=2, readout_width=8)
    identities = tuple(spec[0] for spec in FIELD_SPECS)
    return shared, payload, masks, fields, experts, config, identities


def test_routing_padding_and_restored_shapes(case):
    shared, _payload, _masks, fields, experts, config, identities = case
    reference = reference_decode_fields(shared, fields, config)
    for chunk_size in (3, 16, 64):
        packed = packed_decode_fields(shared, fields, identities, experts, config, chunk_size=chunk_size)
        assert tuple(value.shape for value in packed) == tuple((spec[1],) for spec in FIELD_SPECS)
        for actual, expected, field in zip(packed, reference, fields, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            np.testing.assert_array_equal(np.asarray(actual)[~np.asarray(field.output_active)], 0.0)


def test_expert_selection_is_row_local(case):
    shared, payload, masks, fields, experts, config, identities = case
    baseline = packed_decode_fields(shared, fields, identities, experts, config, chunk_size=16)
    changed_payload = dict(payload)
    changed_payload["family"] = payload["family"].at[5].add(0.5)
    changed_fields, changed_experts = prepare_fields(shared, changed_payload, masks)
    changed = packed_decode_fields(shared, changed_fields, identities, changed_experts, config, chunk_size=16)
    for index, (before, after) in enumerate(zip(baseline, changed, strict=True)):
        difference = np.abs(np.asarray(after - before))
        if identities[index] == 5:
            assert np.max(difference) > 1e-6
        else:
            np.testing.assert_array_equal(after, before)


def test_large_local_bank_gradient_and_alias(case):
    shared, payload, masks, _fields, _experts, config, identities = case

    def objective(local_bank, *, include_alias):
        changed_payload = dict(payload)
        changed_fields = list(payload["fields"])
        changed_fields[5] = dict(changed_fields[5], local=local_bank)
        changed_payload["fields"] = tuple(changed_fields)
        fields, experts = prepare_fields(shared, changed_payload, masks)
        outputs = packed_decode_fields(shared, fields, identities, experts, config, chunk_size=16)
        return jnp.sum(outputs[5] ** 2) + (jnp.sum(outputs[7] ** 2) if include_alias else 0.0)

    local_bank = payload["fields"][5]["local"]
    gradient = jax.grad(lambda value: objective(value, include_alias=False))(local_bank)
    with_alias = jax.grad(lambda value: objective(value, include_alias=True))(local_bank)
    assert gradient.shape == (896, 8)
    assert np.all(np.isfinite(gradient))
    assert np.count_nonzero(np.asarray(gradient)) > 0
    np.testing.assert_allclose(with_alias, gradient, rtol=1e-5, atol=1e-6)


def test_carbon_bank_is_rejected(case):
    shared, _payload, _masks, fields, experts, config, identities = case
    carbon_field = fields[0]._replace(carbon_mask=jnp.ones((1, 1, 1), dtype=bool))
    with pytest.raises(ValueError, match="carbon bank"):
        packed_decode_fields(shared, (carbon_field,) + fields[1:], identities, experts, config, chunk_size=16)


def test_actual_layout_json_only_runtime(tmp_path, monkeypatch):
    path = tmp_path / "macro-layout.json"
    probe.export_layout_metadata(path)
    monkeypatch.setattr(
        audit_shared_decoder_cost, "_load_static_layout",
        lambda: pytest.fail("runtime loader must not rebuild the static layout"),
    )
    layout = probe.load_layout_metadata(path)
    assert layout["field_count"] == 160
    assert layout["eligible_rows"] == 8548
    assert len({row["expert_index"] for row in layout["fields"]}) == 7
    assert all(row["field_name"] != "daily.fast_gpp" for row in layout["fields"])
    assert any(not all(row["local_bank_mapped"]) for row in layout["fields"])
    _shared, payload, masks = probe.fixture(8, layout_metadata=layout)
    assert len(payload["fields"]) == len(masks) == 160
    for row, mask in zip(layout["fields"], masks, strict=True):
        expected_local = np.asarray(row["local_bank_mapped"])[row["local_indices"]]
        observed_local = np.asarray(mask["dynamic"])[:, 0, -1]
        assert not np.any(observed_local & ~expected_local)
        assert mask["axis_indices"].shape == (row["eligible_rows"], row["axis_slots"])
    modified = dict(layout)
    modified["fields"] = [dict(row) for row in layout["fields"]]
    modified["fields"][0]["field_name"] = "daily.fast_gpp"
    with pytest.raises(ValueError, match="identity"):
        probe.validate_layout_metadata(modified)
