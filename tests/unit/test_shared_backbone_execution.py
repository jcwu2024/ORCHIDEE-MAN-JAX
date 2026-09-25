"""Canonical context and native-gradient parity for diagnostic source reuse."""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.shared_backbone_execution import (
    assemble_shared_transformer_context_with_source_reuse,
)
from research.daily_coarse_graining.daily_operator.shared_backbone_projection_reuse import (
    run_backbone_with_shared_source_projection,
)
from research.daily_coarse_graining.daily_operator.shared_transformer import (
    _run_blocks,
    assemble_shared_transformer_context,
)

pytest_plugins = ("tests.unit.test_shared_transformer",)


def _context(assembler, setup, parameters, inputs):
    return assembler(
        parameters,
        inputs,
        normalization=setup["input_normalization"],
        layout=setup["layout"],
        config=setup["config"],
    )


def test_source_reuse_preserves_complete_context_and_inactive_masks(shared_transformer_setup):
    setup = shared_transformer_setup
    inputs = setup["assembled"]
    inactive_pft = jnp.asarray(inputs.pft.active_pft_mask).at[1].set(False)
    inputs = inputs._replace(pft=inputs.pft._replace(active_pft_mask=inactive_pft))
    reference = _context(assemble_shared_transformer_context, setup, setup["parameters"], inputs)
    candidate = _context(
        assemble_shared_transformer_context_with_source_reuse,
        setup,
        setup["parameters"],
        inputs,
    )
    assert reference._fields == candidate._fields
    for name in reference._fields:
        expected = np.asarray(getattr(reference, name))
        actual = np.asarray(getattr(candidate, name))
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        if expected.dtype.kind in "biu":
            np.testing.assert_array_equal(actual, expected, err_msg=name)
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6, err_msg=name)
    assert np.any(~np.asarray(reference.pft_mask))


def diagnose_strict_full_context_gradients(setup):
    """Return every FP32 parameter-gradient failure at the fixed real-loss gate."""

    inputs = setup["assembled"]
    expected = _context(assemble_shared_transformer_context, setup, setup["parameters"], inputs)
    weights = {
        name: jnp.sin(jnp.arange(getattr(expected, name).size, dtype=jnp.float32)).reshape(
            getattr(expected, name).shape
        )
        for name in ("source_tokens", "expert_axis_tokens", "pft_tokens", "process_tokens")
    }

    def loss(assembler, parameters, other):
        changed_inputs = inputs._replace(state=inputs.state._replace(other=other))
        context = _context(assembler, setup, parameters, changed_inputs)
        return sum(jnp.sum(getattr(context, name) * weight) for name, weight in weights.items())

    source_state = inputs.state.other
    reference_grads = jax.grad(loss, argnums=(1, 2))(
        assemble_shared_transformer_context, setup["parameters"], source_state
    )
    candidate_grads = jax.grad(loss, argnums=(1, 2))(
        assemble_shared_transformer_context_with_source_reuse, setup["parameters"], source_state
    )
    expected_paths, expected_tree = jax.tree_util.tree_flatten_with_path(reference_grads[0])
    actual_paths, actual_tree = jax.tree_util.tree_flatten_with_path(candidate_grads[0])
    assert actual_tree == expected_tree
    failures = []
    coordinates = 0
    for (path, actual), (_, expected) in zip(actual_paths, expected_paths, strict=True):
        actual_array, expected_array = np.asarray(actual), np.asarray(expected)
        failed = ~np.isfinite(actual_array) | ~np.isfinite(expected_array)
        failed |= ~np.isclose(actual_array, expected_array, rtol=1e-5, atol=1e-6)
        coordinates += actual_array.size
        if np.any(failed):
            index = tuple(np.argwhere(failed)[0])
            failures.append(
                {
                    "path": jax.tree_util.keystr(path),
                    "failed": int(np.count_nonzero(failed)),
                    "first_index": [int(value) for value in index],
                    "candidate": float(actual_array[index]),
                    "canonical": float(expected_array[index]),
                }
            )
    state_failed = ~np.isclose(np.asarray(candidate_grads[1]), np.asarray(reference_grads[1]), rtol=1e-5, atol=1e-6)
    return {
        "parameter_coordinates": coordinates,
        "parameter_failed": sum(row["failed"] for row in failures),
        "failed_leaves": failures,
        "state_failed": int(np.count_nonzero(state_failed)),
    }


def test_source_reuse_preserves_assembled_state_gradient(shared_transformer_setup):
    setup = shared_transformer_setup
    inputs = setup["assembled"]
    context = _context(assemble_shared_transformer_context, setup, setup["parameters"], inputs)
    weights = jnp.sin(jnp.arange(context.process_tokens.size, dtype=jnp.float32)).reshape(context.process_tokens.shape)

    def loss(assembler, other):
        changed_inputs = inputs._replace(state=inputs.state._replace(other=other))
        assembled = _context(assembler, setup, setup["parameters"], changed_inputs)
        return jnp.sum(assembled.process_tokens * weights)

    source_state = inputs.state.other
    reference = jax.grad(loss, argnums=1)(assemble_shared_transformer_context, source_state)
    candidate = jax.grad(loss, argnums=1)(assemble_shared_transformer_context_with_source_reuse, source_state)
    np.testing.assert_allclose(candidate, reference, rtol=1e-5, atol=1e-6)
    assert np.linalg.norm(np.asarray(candidate)) > 0


def _true64_backbone(blocks, query, source, axis, mask, heads, epsilon, *, reuse_source=False):
    """Independent dtype-preserving two-block attention/FFN calculation."""

    def linear(parameters, value):
        return value @ parameters["kernel"] + parameters["bias"]

    def layer_norm(value, scale, bias):
        mean = jnp.mean(value, axis=-1, keepdims=True)
        variance = jnp.mean((value - mean) ** 2, axis=-1, keepdims=True)
        return (value - mean) * jax.lax.rsqrt(variance + 1e-6) * scale + bias

    def unit(value):
        maximum = jnp.max(jnp.abs(value), axis=-1, keepdims=True)
        scaled = value / jnp.where(maximum > 0, maximum, 1)
        norm = jnp.sqrt(jnp.sum(scaled * scaled, axis=-1, keepdims=True) + epsilon**2)
        return jnp.where(maximum * norm > epsilon, scaled / norm, 0)

    width = query.shape[-1]
    head_width = width // heads
    bank = jnp.concatenate((jnp.broadcast_to(source[None, :, :], (axis.shape[0],) + source.shape), axis), axis=1)

    def per_head(value):
        return value.reshape(value.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)

    for block in blocks:
        attention = block["attention"]
        head_query = unit(per_head(linear(attention["query"], query)))
        if reuse_source:
            source_key = linear(attention["key"], source)
            source_value = linear(attention["value"], source)
            key = jnp.concatenate(
                (
                    jnp.broadcast_to(source_key[None, :, :], (axis.shape[0],) + source_key.shape),
                    linear(attention["key"], axis),
                ),
                axis=1,
            )
            value = jnp.concatenate(
                (
                    jnp.broadcast_to(source_value[None, :, :], (axis.shape[0],) + source_value.shape),
                    linear(attention["value"], axis),
                ),
                axis=1,
            )
        else:
            key = linear(attention["key"], bank)
            value = linear(attention["value"], bank)
        head_key = unit(per_head(key))
        head_value = per_head(value)
        logits = jnp.clip(jnp.einsum("...hqd,...hkd->...hqk", head_query, head_key), -1, 1)
        logits *= jnp.sqrt(jnp.asarray(head_width, dtype=jnp.float64))
        head_mask = mask[:, None, :, :]
        row_valid = jnp.any(head_mask, axis=-1, keepdims=True)
        weights = jax.nn.softmax(jnp.where(row_valid, jnp.where(head_mask, logits, -jnp.inf), 0), axis=-1)
        weights = jnp.where(head_mask, weights, 0)
        attended = jnp.einsum("...hqk,...hkd->...hqd", weights, head_value)
        attended = attended.swapaxes(-3, -2).reshape(query.shape)
        query = layer_norm(
            query + linear(attention["output"], attended),
            block["attention_norm_scale"],
            block["attention_norm_bias"],
        )
        hidden = jax.nn.gelu(linear(block["ffn"]["first"], query))
        query = layer_norm(
            query + linear(block["ffn"]["second"], hidden),
            block["ffn_norm_scale"],
            block["ffn_norm_bias"],
        )
    return query


def test_two_block_key_gradient_against_true_fp64_oracle(shared_transformer_setup):
    setup = shared_transformer_setup
    context = _context(assemble_shared_transformer_context, setup, setup["parameters"], setup["assembled"])
    shared = setup["parameters"]["shared_transformer"]
    blocks = shared["backbone_blocks"]
    query = (shared["process_query"] + shared["process_embedding"])[:, None, :]
    source = context.source_tokens
    axis = context.expert_axis_tokens
    mask = jnp.concatenate((context.process_source_mask, context.expert_axis_mask), axis=1)[:, None, :]
    weights = jnp.sin(jnp.arange(context.process_tokens.size, dtype=jnp.float32)).reshape(context.process_tokens.shape)

    def canonical(trainable):
        bank = jnp.concatenate((jnp.broadcast_to(source[None, :, :], (axis.shape[0],) + source.shape), axis), axis=1)
        return jnp.sum(_run_blocks(query, bank, mask, trainable, setup["config"])[:, 0, :] * weights)

    def reused(trainable):
        values = run_backbone_with_shared_source_projection(query, source, axis, mask, trainable, setup["config"])
        return jnp.sum(values[:, 0, :] * weights)

    canonical_gradient = jax.grad(canonical)(blocks)
    reused_gradient = jax.grad(reused)(blocks)
    with jax.enable_x64():
        blocks64 = jax.tree_util.tree_map(lambda value: jnp.asarray(value, dtype=jnp.float64), blocks)
        query64 = jnp.asarray(query, dtype=jnp.float64)
        source64 = jnp.asarray(source, dtype=jnp.float64)
        axis64 = jnp.asarray(axis, dtype=jnp.float64)
        weights64 = jnp.asarray(weights, dtype=jnp.float64)

        def oracle(trainable, *, reuse_source=False):
            values = _true64_backbone(
                trainable,
                query64,
                source64,
                axis64,
                mask,
                setup["config"].attention_heads,
                setup["config"].normalization_epsilon,
                reuse_source=reuse_source,
            )
            return jnp.sum(values[:, 0, :] * weights64)

        oracle_value_full = oracle(blocks64)
        oracle_value_reused = oracle(blocks64, reuse_source=True)
        oracle_gradient = jax.grad(oracle)(blocks64)
        oracle_reused_gradient = jax.grad(lambda trainable: oracle(trainable, reuse_source=True))(blocks64)

    np.testing.assert_allclose(oracle_value_reused, oracle_value_full, rtol=1e-11, atol=1e-11)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(oracle_reused_gradient),
        jax.tree_util.tree_leaves(oracle_gradient),
        strict=True,
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)

    index = (27, 15)
    canonical_value = float(canonical_gradient[0]["attention"]["key"]["kernel"][index])
    reused_value = float(reused_gradient[0]["attention"]["key"]["kernel"][index])
    oracle_value = float(oracle_gradient[0]["attention"]["key"]["kernel"][index])
    assert np.isfinite([canonical_value, reused_value, oracle_value]).all()
    print(
        "two-block key[27,15] canonical/reused/true64="
        f"{canonical_value:.12g}/{reused_value:.12g}/{oracle_value:.12g}; "
        f"absolute errors={abs(canonical_value - oracle_value):.12g}/"
        f"{abs(reused_value - oracle_value):.12g}"
    )


if __name__ == "__main__":
    from tests.unit.test_shared_transformer import shared_transformer_setup

    print(json.dumps(diagnose_strict_full_context_gradients(shared_transformer_setup.__wrapped__()), indent=2))
