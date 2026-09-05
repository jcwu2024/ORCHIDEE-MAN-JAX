"""Pure helpers for coupled-GPP potential-block internal attribution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .target_oriented_coupled_gpp_structured_precision_attribution import (
    COMPARABLE_ENDPOINT_RATIO_MAXIMUM,
    COMPARABLE_ENDPOINT_RATIO_MINIMUM,
    INDIVIDUALLY_MATERIAL_FRACTION_MINIMUM,
    INTERPOLATION_FRACTIONS,
    SHARP_ADJACENT_FRACTION_MINIMUM,
    SMOOTHED_ADJACENT_FRACTION_MAXIMUM,
    classify_subtree_controls,
    summarize_gpp_path,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_potential_internal_attribution_result_v2"
PARENT_REPORT_SHA256 = "40c617ba096f5b9b50482316473d416b17dafb90b7162cbf0d59c9fffaf5b693"
LOWER_FRACTION = 11.0 / 16.0
UPPER_FRACTION = 12.0 / 16.0
EXPECTED_MAXIMUM_INTERVAL = (LOWER_FRACTION, UPPER_FRACTION)
POTENTIAL_BLOCK_COUNT = 2
POTENTIAL_BLOCK_LEAF_COUNT = 32

ParameterPath = tuple[int | str, ...]
ParameterGroupRegistry = Mapping[str, tuple[ParameterPath, ...]]

BLOCK_PARAMETER_GROUPS: dict[str, tuple[ParameterPath, ...]] = {
    "block_0": ((0,),),
    "block_1": ((1,),),
}

_COMPONENT_PATHS: tuple[tuple[str, tuple[ParameterPath, ...]], ...] = (
    ("attention.query", (("attention", "query"),)),
    ("attention.key", (("attention", "key"),)),
    ("attention.value", (("attention", "value"),)),
    ("attention.output", (("attention", "output"),)),
    ("attention_norm", (("attention_norm_scale",), ("attention_norm_bias",))),
    ("ffn.first", (("ffn", "first"),)),
    ("ffn.second", (("ffn", "second"),)),
    ("ffn_norm", (("ffn_norm_scale",), ("ffn_norm_bias",))),
)

COMPONENT_PARAMETER_GROUPS: dict[str, tuple[ParameterPath, ...]] = {
    f"block_{block}.{name}": tuple((block, *path) for path in paths)
    for block in range(POTENTIAL_BLOCK_COUNT)
    for name, paths in _COMPONENT_PATHS
}


def parameter_leaf_paths(tree: Any, prefix: ParameterPath = ()) -> tuple[ParameterPath, ...]:
    """Return structural array-leaf paths without relying on mapping order."""

    if isinstance(tree, Mapping):
        return tuple(
            path for key in sorted(tree, key=str) for path in parameter_leaf_paths(tree[key], (*prefix, str(key)))
        )
    if isinstance(tree, (tuple, list)):
        return tuple(path for index, value in enumerate(tree) for path in parameter_leaf_paths(value, (*prefix, index)))
    value = np.asarray(tree)
    if value.dtype == np.dtype("O"):
        raise TypeError(f"potential-block leaf is not an array: {prefix}")
    return (prefix,)


def _value_at_path(tree: Any, path: ParameterPath) -> Any:
    value = tree
    for key in path:
        value = value[key]
    return value


def _replace_at_path(tree: Any, path: ParameterPath, value: Any) -> Any:
    if not path:
        return value
    key, *tail = path
    remaining = tuple(tail)
    if isinstance(tree, Mapping):
        if not isinstance(key, str) or key not in tree:
            raise KeyError(path)
        result = dict(tree)
        result[key] = _replace_at_path(tree[key], remaining, value)
        return result
    if isinstance(tree, tuple):
        if not isinstance(key, int) or key < 0 or key >= len(tree):
            raise KeyError(path)
        result = list(tree)
        result[key] = _replace_at_path(tree[key], remaining, value)
        return tuple(result)
    if isinstance(tree, list):
        if not isinstance(key, int) or key < 0 or key >= len(tree):
            raise KeyError(path)
        result = list(tree)
        result[key] = _replace_at_path(tree[key], remaining, value)
        return result
    raise KeyError(path)


def replace_parameter_paths(
    base: Any,
    source: Any,
    paths: Sequence[ParameterPath],
) -> Any:
    """Copy selected subtrees from source into a structurally matched base."""

    base_leaves = frozenset(parameter_leaf_paths(base))
    source_leaves = frozenset(parameter_leaf_paths(source))
    if base_leaves != source_leaves:
        raise ValueError("potential-block parameter tree structure drift")
    result = base
    for path in paths:
        prefix = tuple(path)
        if not any(leaf[: len(prefix)] == prefix for leaf in base_leaves):
            raise ValueError(f"potential-block parameter path is not a subtree: {prefix}")
        result = _replace_at_path(result, prefix, _value_at_path(source, prefix))
    return result


def validate_parameter_group_coverage(potential_blocks: Any) -> Mapping[str, Any]:
    """Prove that the frozen component groups partition all block leaves."""

    if not isinstance(potential_blocks, (tuple, list)) or len(potential_blocks) != POTENTIAL_BLOCK_COUNT:
        raise ValueError("potential-block count drift")
    leaves = frozenset(parameter_leaf_paths(potential_blocks))
    covered: dict[ParameterPath, str] = {}
    group_counts = {}
    for group, prefixes in COMPONENT_PARAMETER_GROUPS.items():
        selected = {leaf for prefix in prefixes for leaf in leaves if leaf[: len(prefix)] == prefix}
        if not selected:
            raise ValueError(f"potential-block component group is empty: {group}")
        overlap = frozenset(selected) & frozenset(covered)
        if overlap:
            raise ValueError(f"potential-block component groups overlap: {group}")
        covered.update({leaf: group for leaf in selected})
        group_counts[group] = len(selected)
    if frozenset(covered) != leaves or len(leaves) != POTENTIAL_BLOCK_LEAF_COUNT:
        raise ValueError("potential-block component groups do not exactly cover the parameter leaves")
    return {
        "block_count": len(potential_blocks),
        "leaf_count": len(leaves),
        "component_group_count": len(COMPONENT_PARAMETER_GROUPS),
        "group_leaf_counts": group_counts,
        "exact_nonoverlapping_coverage": True,
    }


def classify_parameter_localization(
    *,
    lower_gpp: float,
    upper_gpp: float,
    block_upper_only_gpp: Mapping[str, float],
    block_upper_without_gpp: Mapping[str, float],
    component_upper_only_gpp: Mapping[str, float],
    component_upper_without_gpp: Mapping[str, float],
) -> Mapping[str, Any]:
    """Classify matched whole-block and component causal controls."""

    blocks = classify_subtree_controls(
        control_gpp=lower_gpp,
        treatment_gpp=upper_gpp,
        treatment_group_only_gpp=block_upper_only_gpp,
        treatment_without_group_gpp=block_upper_without_gpp,
        groups=tuple(BLOCK_PARAMETER_GROUPS),
    )
    components = classify_subtree_controls(
        control_gpp=lower_gpp,
        treatment_gpp=upper_gpp,
        treatment_group_only_gpp=component_upper_only_gpp,
        treatment_without_group_gpp=component_upper_without_gpp,
        groups=tuple(COMPONENT_PARAMETER_GROUPS),
    )
    selected_blocks = tuple(blocks["selected_groups"])
    selected_components = tuple(components["selected_groups"])
    if selected_components:
        classification = "potential_jump_localized_to_internal_parameter_components"
    elif selected_blocks:
        classification = "potential_jump_requires_distributed_within_block_interaction"
    else:
        classification = "potential_jump_not_localized_by_registered_internal_groups"
    return {
        "classification": classification,
        "selected_blocks": list(selected_blocks),
        "selected_components": list(selected_components),
        "blocks": blocks,
        "components": components,
        "material_fraction_minimum": INDIVIDUALLY_MATERIAL_FRACTION_MINIMUM,
    }


def classify_activation_precision_paths(
    *,
    fractions: Sequence[float],
    production_gpp: Sequence[float],
    full_float64_gpp: Sequence[float],
) -> Mapping[str, Any]:
    """Separate intermediate float32 effects from intrinsic block conditioning."""

    production = summarize_gpp_path(fractions, production_gpp)
    diagnostic = summarize_gpp_path(fractions, full_float64_gpp)
    first = float(production["endpoint_gpp_jump"])
    second = float(diagnostic["endpoint_gpp_jump"])
    ratio = abs(second) / abs(first)
    same_sign = bool(np.signbit(first) == np.signbit(second))
    comparable = bool(same_sign and COMPARABLE_ENDPOINT_RATIO_MINIMUM <= ratio <= COMPARABLE_ENDPOINT_RATIO_MAXIMUM)
    production_sharp = bool(
        float(production["maximum_adjacent_fraction_of_endpoint_jump"]) >= SHARP_ADJACENT_FRACTION_MINIMUM
    )
    float64_sharp = bool(
        float(diagnostic["maximum_adjacent_fraction_of_endpoint_jump"]) >= SHARP_ADJACENT_FRACTION_MINIMUM
    )
    float64_smoothed = bool(
        float(diagnostic["maximum_adjacent_fraction_of_endpoint_jump"]) <= SMOOTHED_ADJACENT_FRACTION_MAXIMUM
    )
    if comparable and production_sharp and float64_smoothed:
        classification = "float32_internal_activation_precision_dominant_for_sampled_jump"
    elif comparable and production_sharp and float64_sharp:
        classification = "potential_block_conditioning_persists_in_full_float64"
    else:
        classification = "mixed_or_inconclusive_internal_activation_precision_effect"
    return {
        "classification": classification,
        "production_float32": production,
        "full_float64_potential_branch": diagnostic,
        "endpoint_jump_magnitude_ratio": ratio,
        "endpoint_jump_same_sign": same_sign,
        "endpoint_jumps_comparable": comparable,
        "production_path_sharp": production_sharp,
        "full_float64_path_sharp": float64_sharp,
        "full_float64_path_smoothed": float64_smoothed,
        "thresholds": {
            "sharp_adjacent_fraction_minimum": SHARP_ADJACENT_FRACTION_MINIMUM,
            "smoothed_adjacent_fraction_maximum": SMOOTHED_ADJACENT_FRACTION_MAXIMUM,
            "comparable_endpoint_ratio_minimum": COMPARABLE_ENDPOINT_RATIO_MINIMUM,
            "comparable_endpoint_ratio_maximum": COMPARABLE_ENDPOINT_RATIO_MAXIMUM,
        },
    }


def traced_potential_branch(
    jax: Any,
    jnp: Any,
    potential_blocks: Any,
    potential_output: Mapping[str, Any],
    query: Any,
    bank: Any,
    bank_mask: Any,
    active: Any,
    *,
    heads: int,
    dtype: Any,
    attention_head_mask: Any | None = None,
) -> Mapping[str, Any]:
    """Run the existing potential branch operation order while exposing activations."""

    dtype = jnp.dtype(dtype)

    def linear(parameters: Mapping[str, Any], value: Any) -> Any:
        return jnp.asarray(value, dtype=dtype) @ jnp.asarray(parameters["kernel"], dtype=dtype) + jnp.asarray(
            parameters["bias"], dtype=dtype
        )

    def layer_norm(value: Any, scale: Any, bias: Any) -> tuple[Any, Any]:
        value = jnp.asarray(value, dtype=dtype)
        mean = jnp.mean(value, axis=-1, keepdims=True)
        variance = jnp.mean((value - mean) ** 2, axis=-1, keepdims=True)
        normalized = (value - mean) * jax.lax.rsqrt(variance + jnp.asarray(1.0e-6, dtype=dtype)) * jnp.asarray(
            scale, dtype=dtype
        ) + jnp.asarray(bias, dtype=dtype)
        return normalized, variance

    encoded = jnp.asarray(query, dtype=dtype)[:, None, :]
    bank = jnp.asarray(bank, dtype=dtype)
    mask = jnp.asarray(bank_mask, dtype=jnp.bool_)[:, None, :]
    records = []
    effective_attention_masks = []
    if attention_head_mask is not None:
        attention_head_mask = jnp.asarray(attention_head_mask, dtype=jnp.bool_)
        expected_shape = (len(potential_blocks), int(query.shape[0]), int(heads), int(bank.shape[-2]))
        if tuple(attention_head_mask.shape) != expected_shape:
            raise ValueError(
                "potential diagnostic head mask shape drift: "
                f"{tuple(attention_head_mask.shape)} versus {expected_shape}"
            )
    for block_index, block in enumerate(potential_blocks):
        block_input = encoded
        attention = block["attention"]
        projected_query = linear(attention["query"], encoded)
        projected_key = linear(attention["key"], bank)
        projected_value = linear(attention["value"], bank)
        width = int(projected_query.shape[-1])
        head_width = width // int(heads)
        head_query = projected_query.reshape(projected_query.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)
        head_key = projected_key.reshape(projected_key.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)
        head_value = projected_value.reshape(projected_value.shape[:-1] + (heads, head_width)).swapaxes(-3, -2)
        logits = jnp.einsum("...hqd,...hkd->...hqk", head_query, head_key) / jnp.sqrt(
            jnp.asarray(head_width, dtype=dtype)
        )
        effective_mask = jnp.broadcast_to(mask[..., None, :, :], logits.shape)
        if attention_head_mask is not None:
            effective_mask &= attention_head_mask[block_index, :, :, None, :]
        masked_logits = jnp.where(effective_mask, logits, -jnp.inf)
        row_valid = jnp.any(effective_mask, axis=-1, keepdims=True)
        safe_logits = jnp.where(row_valid, masked_logits, 0.0)
        weights = jax.nn.softmax(safe_logits, axis=-1)
        weights = jnp.where(effective_mask, weights, 0.0)
        attention_context = jnp.einsum("...hqk,...hkd->...hqd", weights, head_value)
        attention_context = attention_context.swapaxes(-3, -2).reshape(
            attention_context.shape[:-3] + (attention_context.shape[-2], width)
        )
        attention_output = linear(attention["output"], attention_context)
        attention_residual = encoded + attention_output
        attention_norm, attention_variance = layer_norm(
            attention_residual,
            block["attention_norm_scale"],
            block["attention_norm_bias"],
        )
        ffn_first = linear(block["ffn"]["first"], attention_norm)
        ffn_gelu = jax.nn.gelu(ffn_first)
        ffn_second = linear(block["ffn"]["second"], ffn_gelu)
        ffn_residual = attention_norm + ffn_second
        encoded, ffn_variance = layer_norm(
            ffn_residual,
            block["ffn_norm_scale"],
            block["ffn_norm_bias"],
        )
        records.append(
            {
                "input": block_input,
                "attention_query": projected_query,
                "attention_key": projected_key,
                "attention_value": projected_value,
                "attention_logits": logits,
                "attention_weights": weights,
                "attention_context": attention_context,
                "attention_output": attention_output,
                "attention_residual": attention_residual,
                "attention_norm_variance": attention_variance,
                "attention_norm": attention_norm,
                "ffn_first": ffn_first,
                "ffn_gelu": ffn_gelu,
                "ffn_second": ffn_second,
                "ffn_residual": ffn_residual,
                "ffn_norm_variance": ffn_variance,
                "ffn_norm": encoded,
            }
        )
        effective_attention_masks.append(effective_mask)
    hidden = jnp.where(jnp.asarray(active, dtype=jnp.bool_)[:, None], encoded[:, 0, :], 0.0)
    potential_logit = linear(potential_output, hidden)[:, 0]
    return {
        "blocks": tuple(records),
        "attention_mask": mask,
        "effective_attention_masks": tuple(effective_attention_masks),
        "hidden": hidden,
        "potential_logit": potential_logit,
        "potential": jax.nn.softplus(potential_logit),
    }


def _numeric_summary(value: Any) -> Mapping[str, Any]:
    array = np.asarray(value)
    finite = np.isfinite(array)
    selected = array[finite].astype(np.float64)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nonfinite_count": int(array.size - np.sum(finite, dtype=np.int64)),
        "rms": float(np.sqrt(np.mean(selected * selected))) if selected.size else None,
        "maximum_absolute": float(np.max(np.abs(selected))) if selected.size else None,
    }


def _selected_stage_arrays(trace: Mapping[str, Any], pft_index: int) -> Mapping[str, np.ndarray]:
    stages: dict[str, np.ndarray] = {}
    for block_index, block in enumerate(trace["blocks"]):
        for name, value in block.items():
            stages[f"block_{block_index}.{name}"] = np.asarray(value)[pft_index]
    stages["hidden"] = np.asarray(trace["hidden"])[pft_index]
    stages["potential_logit"] = np.asarray(trace["potential_logit"])[pft_index]
    stages["potential"] = np.asarray(trace["potential"])[pft_index]
    return stages


def _attention_head_summaries(trace: Mapping[str, Any], pft_index: int) -> list[Mapping[str, Any]]:
    mask = np.asarray(trace["attention_mask"], dtype=np.bool_)[pft_index, 0]
    records = []
    for block_index, block in enumerate(trace["blocks"]):
        logits = np.asarray(block["attention_logits"])[pft_index, :, 0]
        weights = np.asarray(block["attention_weights"])[pft_index, :, 0]
        for head in range(weights.shape[0]):
            selected_logits = logits[head, mask].astype(np.float64)
            selected_weights = weights[head, mask].astype(np.float64)
            positive = selected_weights > 0.0
            entropy = float(-np.sum(selected_weights[positive] * np.log(selected_weights[positive])))
            top_local = int(np.argmax(selected_weights))
            bank_indices = np.flatnonzero(mask)
            records.append(
                {
                    "block": block_index,
                    "head": head,
                    "valid_bank_count": int(bank_indices.size),
                    "logit_minimum": float(np.min(selected_logits)),
                    "logit_maximum": float(np.max(selected_logits)),
                    "logit_range": float(np.ptp(selected_logits)),
                    "entropy": entropy,
                    "effective_support": float(np.exp(entropy)),
                    "maximum_weight": float(selected_weights[top_local]),
                    "maximum_weight_bank_index": int(bank_indices[top_local]),
                }
            )
    return records


def summarize_trace_pair(
    lower: Mapping[str, Any],
    upper: Mapping[str, Any],
    *,
    pft_index: int,
) -> Mapping[str, Any]:
    """Summarize adjacent lower/upper traces without retaining full activations."""

    lower_stages = _selected_stage_arrays(lower, pft_index)
    upper_stages = _selected_stage_arrays(upper, pft_index)
    if frozenset(lower_stages) != frozenset(upper_stages):
        raise ValueError("potential activation stage registry drift")
    stages = {}
    for name in lower_stages:
        first = lower_stages[name]
        second = upper_stages[name]
        if first.shape != second.shape:
            raise ValueError(f"potential activation shape drift: {name}")
        delta = second.astype(np.float64) - first.astype(np.float64)
        stages[name] = {
            "lower": _numeric_summary(first),
            "upper": _numeric_summary(second),
            "delta_rms": float(np.sqrt(np.mean(delta * delta))),
            "delta_maximum_absolute": float(np.max(np.abs(delta))),
        }
    lower_attention = _attention_head_summaries(lower, pft_index)
    upper_attention = _attention_head_summaries(upper, pft_index)
    attention = []
    for first, second in zip(lower_attention, upper_attention, strict=True):
        if (first["block"], first["head"]) != (second["block"], second["head"]):
            raise ValueError("potential attention-head registry drift")
        attention.append(
            {
                "block": first["block"],
                "head": first["head"],
                "lower": first,
                "upper": second,
                "maximum_bank_index_changed": (
                    first["maximum_weight_bank_index"] != second["maximum_weight_bank_index"]
                ),
                "entropy_change": second["entropy"] - first["entropy"],
                "maximum_weight_change": second["maximum_weight"] - first["maximum_weight"],
            }
        )
    return {
        "pft_index": int(pft_index),
        "stages": stages,
        "attention_heads": attention,
        "all_stages_finite": all(
            record[side]["nonfinite_count"] == 0 for record in stages.values() for side in ("lower", "upper")
        ),
    }


__all__ = [
    "BLOCK_PARAMETER_GROUPS",
    "COMPONENT_PARAMETER_GROUPS",
    "EXPECTED_MAXIMUM_INTERVAL",
    "INTERPOLATION_FRACTIONS",
    "LOWER_FRACTION",
    "PARENT_REPORT_SHA256",
    "POTENTIAL_BLOCK_LEAF_COUNT",
    "SCHEMA_VERSION",
    "UPPER_FRACTION",
    "classify_activation_precision_paths",
    "classify_parameter_localization",
    "parameter_leaf_paths",
    "replace_parameter_paths",
    "summarize_trace_pair",
    "traced_potential_branch",
    "validate_parameter_group_coverage",
]
