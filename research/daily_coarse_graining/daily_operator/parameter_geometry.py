"""Owned parameter blocks for bounded successor update-geometry audits."""

from __future__ import annotations

from typing import Any, Mapping

import jax
import jax.numpy as jnp
import numpy as np

SUCCESSOR_PARAMETER_BLOCKS: Mapping[str, tuple[str, ...]] = {
    "input_projection": (
        "forcing_input",
        "summary_input",
        "pft_input",
        "calendar_input",
    ),
    "input_embedding": (
        "family_embedding",
        "forcing_field_embedding",
        "forcing_position_embedding",
        "semantic_field_embedding",
        "semantic_role_embedding",
        "semantic_position_embedding",
        "state_field_embedding",
        "static_field_embedding",
        "annual_field_embedding",
        "process_static_field_embedding",
    ),
    "forcing_interaction": ("forcing_blocks",),
    "process_query": ("field_queries", "label_embedding"),
    "process_interaction": ("process_blocks",),
    "output_embedding": (
        "output_field_embedding",
        "axis_ordinal_embedding",
        "output_position_embedding",
    ),
    "output_decoder": ("decoder_first", "decoder_second"),
}

SUCCESSOR_DECODER_LEAVES: Mapping[str, tuple[str, str]] = {
    "decoder_first.kernel": ("decoder_first", "kernel"),
    "decoder_first.bias": ("decoder_first", "bias"),
    "decoder_second.kernel": ("decoder_second", "kernel"),
    "decoder_second.bias": ("decoder_second", "bias"),
}


def successor_parameter_block_by_key() -> Mapping[str, str]:
    result = {}
    for block, keys in SUCCESSOR_PARAMETER_BLOCKS.items():
        for key in keys:
            if key in result:
                raise ValueError(f"successor parameter key has two blocks: {key}")
            result[key] = block
    return result


def validate_successor_parameter_blocks(parameters: Mapping[str, Any]) -> None:
    expected = set(successor_parameter_block_by_key())
    observed = set(parameters)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            "successor parameter block coverage drift: "
            f"missing={missing}, unexpected={unexpected}"
        )


def exclude_successor_parameter_block(
    direction: Mapping[str, Any],
    block: str,
) -> Mapping[str, Any]:
    """Set one owned block to zero while preserving the parameter pytree."""

    validate_successor_parameter_blocks(direction)
    if block not in SUCCESSOR_PARAMETER_BLOCKS:
        raise ValueError(f"unknown successor parameter block {block!r}")
    excluded = set(SUCCESSOR_PARAMETER_BLOCKS[block])
    return {
        key: (
            jax.tree_util.tree_map(jnp.zeros_like, value)
            if key in excluded
            else value
        )
        for key, value in direction.items()
    }


def exclude_successor_decoder_leaf(
    direction: Mapping[str, Any],
    leaf_name: str,
) -> Mapping[str, Any]:
    """Set one decoder kernel or bias to zero while preserving the pytree."""

    validate_successor_parameter_blocks(direction)
    if leaf_name not in SUCCESSOR_DECODER_LEAVES:
        raise ValueError(f"unknown successor decoder leaf {leaf_name!r}")
    top_level, child = SUCCESSOR_DECODER_LEAVES[leaf_name]
    decoder = dict(direction[top_level])
    if set(decoder) != {"kernel", "bias"}:
        raise ValueError(f"successor decoder leaf layout drift for {top_level}")
    decoder[child] = jnp.zeros_like(decoder[child])
    return {**direction, top_level: decoder}


def successor_parameter_geometry_report(
    parameters: Mapping[str, Any],
    direction: Mapping[str, Any],
    *,
    step_norm: float,
) -> Mapping[str, Any]:
    """Measure Euclidean direction and relative-update concentration by block."""

    validate_successor_parameter_blocks(parameters)
    validate_successor_parameter_blocks(direction)
    if not np.isfinite(step_norm) or step_norm <= 0.0:
        raise ValueError("successor geometry step norm must be positive and finite")
    records = {}
    total_direction_squared = 0.0
    total_parameter_count = 0
    for block, keys in SUCCESSOR_PARAMETER_BLOCKS.items():
        parameter_leaves = [
            np.asarray(leaf, dtype=np.float64)
            for key in keys
            for leaf in jax.tree_util.tree_leaves(parameters[key])
        ]
        direction_leaves = [
            np.asarray(leaf, dtype=np.float64)
            for key in keys
            for leaf in jax.tree_util.tree_leaves(direction[key])
        ]
        if len(parameter_leaves) != len(direction_leaves):
            raise ValueError(f"successor parameter/direction leaf drift for {block}")
        element_count = sum(int(value.size) for value in parameter_leaves)
        parameter_squared = sum(float(np.vdot(value, value)) for value in parameter_leaves)
        direction_squared = sum(float(np.vdot(value, value)) for value in direction_leaves)
        nonfinite_count = sum(
            int(np.count_nonzero(~np.isfinite(value)))
            for value in (*parameter_leaves, *direction_leaves)
        )
        records[block] = {
            "top_level_keys": list(keys),
            "element_count": element_count,
            "parameter_norm": float(np.sqrt(parameter_squared)),
            "direction_norm": float(np.sqrt(direction_squared)),
            "nonfinite_value_count": nonfinite_count,
        }
        total_direction_squared += direction_squared
        total_parameter_count += element_count
    total_direction_norm = float(np.sqrt(total_direction_squared))
    if not np.isfinite(total_direction_norm) or total_direction_norm <= 0.0:
        raise ValueError("successor geometry direction norm is invalid")
    for record in records.values():
        direction_norm = float(record["direction_norm"])
        parameter_norm = float(record["parameter_norm"])
        delta_norm = step_norm * direction_norm / total_direction_norm
        record["direction_squared_fraction"] = float(
            direction_norm * direction_norm / total_direction_squared
        )
        record["step_delta_norm"] = delta_norm
        record["relative_step_to_parameter_norm"] = float(
            delta_norm / max(parameter_norm, 1.0e-30)
        )
    return {
        "blocks": records,
        "block_count": len(records),
        "top_level_key_count": len(successor_parameter_block_by_key()),
        "parameter_count": total_parameter_count,
        "direction_norm": total_direction_norm,
        "direction_squared_fraction_sum": float(
            sum(record["direction_squared_fraction"] for record in records.values())
        ),
        "nonfinite_value_count": sum(
            int(record["nonfinite_value_count"]) for record in records.values()
        ),
        "step_norm": float(step_norm),
    }


def successor_decoder_leaf_geometry_report(
    parameters: Mapping[str, Any],
    direction: Mapping[str, Any],
    *,
    step_norm: float,
) -> Mapping[str, Any]:
    """Measure the four decoder leaves against the full direction norm."""

    validate_successor_parameter_blocks(parameters)
    validate_successor_parameter_blocks(direction)
    if not np.isfinite(step_norm) or step_norm <= 0.0:
        raise ValueError("successor decoder geometry step norm must be positive")
    full_direction_squared = sum(
        float(np.vdot(np.asarray(leaf), np.asarray(leaf)))
        for leaf in jax.tree_util.tree_leaves(direction)
    )
    full_direction_norm = float(np.sqrt(full_direction_squared))
    if not np.isfinite(full_direction_norm) or full_direction_norm <= 0.0:
        raise ValueError("successor decoder geometry direction norm is invalid")
    records = {}
    parameter_count = 0
    for name, (top_level, child) in SUCCESSOR_DECODER_LEAVES.items():
        parameter = np.asarray(parameters[top_level][child], dtype=np.float64)
        value = np.asarray(direction[top_level][child], dtype=np.float64)
        parameter_norm = float(np.linalg.norm(parameter.reshape(-1)))
        direction_norm = float(np.linalg.norm(value.reshape(-1)))
        delta_norm = step_norm * direction_norm / full_direction_norm
        records[name] = {
            "top_level_key": top_level,
            "child_key": child,
            "shape": list(parameter.shape),
            "element_count": int(parameter.size),
            "parameter_norm": parameter_norm,
            "direction_norm": direction_norm,
            "direction_squared_fraction": float(
                direction_norm * direction_norm / full_direction_squared
            ),
            "step_delta_norm": delta_norm,
            "relative_step_to_parameter_norm": float(
                delta_norm / max(parameter_norm, 1.0e-30)
            ),
            "nonfinite_value_count": int(
                np.count_nonzero(~np.isfinite(parameter))
                + np.count_nonzero(~np.isfinite(value))
            ),
        }
        parameter_count += int(parameter.size)
    return {
        "blocks": records,
        "block_count": len(records),
        "top_level_key_count": 2,
        "parameter_count": parameter_count,
        "direction_norm": full_direction_norm,
        "direction_squared_fraction_sum": float(
            sum(record["direction_squared_fraction"] for record in records.values())
        ),
        "nonfinite_value_count": sum(
            int(record["nonfinite_value_count"]) for record in records.values()
        ),
        "step_norm": float(step_norm),
    }
