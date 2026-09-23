from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.shared_decoder_dense import (
    DenseDecoderField,
    packed_decode_fields,
    reference_decode_fields,
)
from research.daily_coarse_graining.daily_operator.shared_transformer import (
    SharedTransformerConfig,
    _DecoderBlockProjection,
    _field_dynamic_projection_cache,
    _project_key_value,
)
from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
    _linear_parameters,
    _residual_block_parameters,
)


class _Payload(NamedTuple):
    query_first: Any
    axis_first: Any
    local_first: Any
    family_first: Any
    process_first: Any
    carbon: Any
    query_second: Any
    axis_second: Any
    local_second: Any
    family_second: Any
    process_second: Any


@pytest.fixture(scope="module")
def dense_case():
    config = SharedTransformerConfig(
        width=16,
        attention_heads=4,
        ffn_width=64,
        backbone_blocks=1,
        coordinate_blocks=2,
        readout_width=8,
        output_chunk_size=4,
    )
    keys = iter(jax.random.split(jax.random.PRNGKey(20260923), 15))
    blocks = []
    for block_index in range(config.coordinate_blocks):
        block = _residual_block_parameters(next(keys), config.width, config.ffn_width)
        attention = dict(block["attention"])
        for projection_index, name in enumerate(("key", "value"), start=1):
            projection = dict(attention[name])
            projection["bias"] = jnp.linspace(
                0.01 * (block_index + 1) * projection_index,
                0.02 * (block_index + 1) * projection_index,
                config.width,
                dtype=jnp.float32,
            )
            attention[name] = projection
        block = dict(block)
        block["attention"] = attention
        blocks.append(block)
    parameters = {
        "coordinate_blocks": tuple(blocks),
        "readout_first": _linear_parameters(next(keys), config.width, config.readout_width),
        "readout_second": _linear_parameters(next(keys), config.readout_width, 1),
    }

    def normal(shape):
        return jax.random.normal(next(keys), shape, dtype=jnp.float32)

    payload = _Payload(
        query_first=normal((3, 1, config.width)),
        axis_first=normal((4, config.width)),
        local_first=normal((3, config.width)),
        family_first=normal((3, config.width)),
        process_first=normal((1, config.width)),
        carbon=normal((2, config.width)),
        query_second=normal((2, 1, config.width)),
        axis_second=normal((2, config.width)),
        local_second=normal((2, config.width)),
        family_second=normal((2, config.width)),
        process_second=normal((1, config.width)),
    )
    return parameters, payload, config


def _base_projections(parameters, family, process, carbon):
    projections = []
    for block in parameters["coordinate_blocks"]:
        attention = block["attention"]
        family_key, family_value = _project_key_value(attention, family)
        process_key, process_value = _project_key_value(attention, process)
        carbon_key, carbon_value = _project_key_value(attention, carbon)
        projections.append(
            _DecoderBlockProjection(
                family_key=family_key,
                family_value=family_value,
                process_key=process_key,
                process_value=process_value,
                carbon_key=carbon_key,
                carbon_value=carbon_value,
                axis_key=None,
                axis_value=None,
                local_key=None,
                local_value=None,
            )
        )
    return tuple(projections)


def _fields(parameters, payload):
    axis_first = jnp.asarray([[0, 1], [2, 3], [1, 2]], dtype=jnp.int32)
    local_first = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    active_first = jnp.asarray([True, True, False])
    dynamic_first = jnp.concatenate(
        (
            payload.axis_first[axis_first],
            payload.local_first[local_first, None, :],
        ),
        axis=1,
    )
    projections_first = _field_dynamic_projection_cache(
        {"shared_transformer": parameters},
        _base_projections(
            parameters,
            payload.family_first,
            payload.process_first,
            payload.carbon,
        ),
        (payload.axis_first, payload.local_first),
    )

    axis_second = jnp.asarray([[0], [1]], dtype=jnp.int32)
    local_second = jnp.asarray([0, 1], dtype=jnp.int32)
    active_second = jnp.asarray([True, False])
    dynamic_second = jnp.concatenate(
        (
            payload.axis_second[axis_second],
            payload.local_second[local_second, None, :],
        ),
        axis=1,
    )
    projections_second = _field_dynamic_projection_cache(
        {"shared_transformer": parameters},
        _base_projections(
            parameters,
            payload.family_second,
            payload.process_second,
            payload.carbon,
        ),
        (payload.axis_second, payload.local_second),
    )
    return (
        DenseDecoderField(
            query=payload.query_first,
            dynamic_tokens=dynamic_first,
            dynamic_mask=jnp.asarray(
                [[[True, True, True]], [[True, False, True]], [[False, False, False]]]
            ),
            family_mask=jnp.asarray(
                [[[True, False, True]], [[False, True, True]], [[False, False, False]]]
            ),
            process_mask=active_first[:, None, None],
            carbon_mask=jnp.asarray(
                [[[True, False]], [[False, True]], [[False, False]]]
            ),
            projections=projections_first,
            dynamic_projection_indices=(axis_first, local_first, active_first),
            output_active=active_first,
        ),
        DenseDecoderField(
            query=payload.query_second,
            dynamic_tokens=dynamic_second,
            dynamic_mask=jnp.asarray([[[False, False]], [[False, False]]]),
            family_mask=jnp.asarray([[[False, False]], [[False, False]]]),
            process_mask=jnp.asarray([[[False]], [[False]]]),
            carbon_mask=None,
            projections=projections_second,
            dynamic_projection_indices=(axis_second, local_second, active_second),
            output_active=jnp.asarray([True, False]),
        ),
    )


def _assert_tree_close(expected, actual):
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    assert actual_tree == expected_tree
    for expected_leaf, actual_leaf in zip(expected_leaves, actual_leaves, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
            rtol=5.0e-4,
            atol=2.0e-5,
        )


@pytest.mark.parametrize("dense_chunk_size", [0, 2, 3])
def test_packed_dense_stages_match_production_field_execution(dense_case, dense_chunk_size):
    parameters, payload, config = dense_case
    fields = _fields(parameters, payload)
    reference = reference_decode_fields(parameters, fields, config)
    candidate = packed_decode_fields(
        parameters,
        fields,
        config,
        dense_chunk_size=dense_chunk_size,
    )
    for expected, actual in zip(reference, candidate, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual),
            np.asarray(expected),
            rtol=2.0e-5,
            atol=2.0e-6,
        )
        assert np.all(np.isfinite(np.asarray(actual)))
    assert float(candidate[0][-1]) == 0.0
    assert float(candidate[1][-1]) == 0.0
    assert np.isfinite(float(candidate[1][0]))
    for block in parameters["coordinate_blocks"]:
        assert np.all(np.asarray(block["attention"]["key"]["bias"]) != 0.0)
        assert np.all(np.asarray(block["attention"]["value"]["bias"]) != 0.0)


def test_packed_dense_stages_match_complete_parameter_and_payload_vjp(dense_case):
    parameters, payload, config = dense_case

    def objective(decoder, current_parameters, current_payload, *, dense_chunk_size=0):
        fields = _fields(current_parameters, current_payload)
        if decoder is packed_decode_fields:
            outputs = decoder(
                current_parameters,
                fields,
                config,
                dense_chunk_size=dense_chunk_size,
            )
        else:
            outputs = decoder(current_parameters, fields, config)
        terms = []
        for output in outputs:
            weights = jnp.linspace(0.5, 1.5, output.size, dtype=output.dtype)
            terms.append(jnp.vdot(output, weights))
        return sum(terms)

    reference_value, reference_pullback = jax.vjp(
        lambda current_parameters, current_payload: objective(
            reference_decode_fields,
            current_parameters,
            current_payload,
        ),
        parameters,
        payload,
    )
    candidate_value, candidate_pullback = jax.vjp(
        lambda current_parameters, current_payload: objective(
            packed_decode_fields,
            current_parameters,
            current_payload,
            dense_chunk_size=2,
        ),
        parameters,
        payload,
    )
    np.testing.assert_allclose(
        np.asarray(candidate_value),
        np.asarray(reference_value),
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    reference_vjp = reference_pullback(jnp.ones_like(reference_value))
    candidate_vjp = candidate_pullback(jnp.ones_like(candidate_value))
    _assert_tree_close(reference_vjp, candidate_vjp)
    for leaf in jax.tree_util.tree_leaves(candidate_vjp):
        assert np.all(np.isfinite(np.asarray(leaf)))
    assert np.linalg.norm(
        np.asarray(candidate_vjp[0]["coordinate_blocks"][0]["attention"]["key"]["kernel"])
    ) > 0.0
    assert np.linalg.norm(np.asarray(candidate_vjp[1].query_first)) > 0.0
    assert np.linalg.norm(np.asarray(candidate_vjp[1].carbon)) > 0.0
