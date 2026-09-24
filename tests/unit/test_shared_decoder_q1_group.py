"""Checks for the diagnostic within-expert Q1 decoder grouping."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from research.daily_coarse_graining.daily_operator.shared_decoder_dense import reference_decode_fields
from research.daily_coarse_graining.daily_operator.shared_decoder_q1_group import grouped_decode_q1_fields
from research.daily_coarse_graining.daily_operator.shared_transformer import SharedTransformerConfig
from scripts.hpc.audit_shared_decoder_cost import _load_static_layout
from scripts.hpc.audit_shared_decoder_packing import static_grouping_audit
from scripts.hpc.probe_shared_decoder_q1_group import (
    AUDITED_FIELD_INDICES,
    AUDITED_SIGNATURE,
    FAMILY_COUNT,
    compare,
    fixture,
    prepare_fields,
    run,
)


def test_audited_q1_scope_and_full_native_gradient_parity():
    assert len(AUDITED_FIELD_INDICES) == 22
    groups = static_grouping_audit(width=384)["groups"]
    audited = next(group for group in groups if group["signature"] == AUDITED_SIGNATURE)
    assert tuple(member["field_index"] for member in audited["members"] if member["expert"] == "hydrology_snow") == AUDITED_FIELD_INDICES
    assert len(_load_static_layout()["layout"].source_family_names) == FAMILY_COUNT
    shared, payload, masks = fixture(16)
    config = SharedTransformerConfig(width=16, ffn_width=64, coordinate_blocks=2, readout_width=16)

    def output(decode, parameters, inputs):
        return jnp.concatenate(decode(parameters, prepare_fields(parameters, inputs, masks), config))

    reference = output(reference_decode_fields, shared, payload)
    grouped = output(grouped_decode_q1_fields, shared, payload)
    assert compare(grouped, reference)["passed"]
    assert jnp.all(grouped[jnp.asarray([0, 11])] == 0.0)

    def loss(decode, parameters, inputs):
        values = output(decode, parameters, inputs)
        return jnp.mean((values - jnp.arange(values.size, dtype=jnp.float32) * 0.1) ** 2)

    expected = jax.grad(lambda p, x: loss(reference_decode_fields, p, x), argnums=(0, 1))(shared, payload)
    actual = jax.grad(lambda p, x: loss(grouped_decode_q1_fields, p, x), argnums=(0, 1))(shared, payload)
    assert compare(actual, expected)["passed"]


def test_micro4_mean_loss_preserves_full_native_gradients():
    report = run(16, 1, grouped_decode_q1_fields, require_gpu=False, microbatch=4)
    assert report["microbatch"] == 4
    assert report["numerical_parity_passed"]
    assert report["comparison"]["loss_output"]["coordinates"] == 89
    assert report["comparison"]["parameter_payload_gradients"]["coordinates"] == 11009
