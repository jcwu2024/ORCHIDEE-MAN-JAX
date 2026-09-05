"""Pure helpers for coupled-GPP attention-head and source-token attribution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .target_oriented_coupled_gpp_structured_precision_attribution import (
    INDIVIDUALLY_MATERIAL_FRACTION_MINIMUM,
    classify_subtree_controls,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_attention_head_token_attribution_result_v1"
PARENT_REPORT_SHA256 = "f781e4a92eeac1dd9bc0f986a55fa83478d18d8c20025eb12a2d74067207e0a3"
PARENT_SCHEMA_VERSION = "target_oriented_coupled_gpp_potential_internal_attribution_result_v2"
PARENT_CLASSIFICATION = "potential_block_conditioning_persists_in_full_float64"
MODEL_WIDTH = 384
ATTENTION_HEADS = 8
HEAD_WIDTH = 48
TARGET_BLOCK_INDEX = 0
TARGET_HEAD_INDEX = 7
DOC_BANK_INDEX = 153
LITTERPART_BANK_INDEX = 205
TARGET_TOKEN_KEYS = {
    "doc": "slowproc_stomate_previous_step_state.DOC",
    "litterpart": "slowproc_stomate_previous_step_state.litterpart",
}


def _projection_shape(projection: Mapping[str, Any]) -> tuple[int, int]:
    if frozenset(projection) != frozenset(("kernel", "bias")):
        raise ValueError("attention projection parameter registry drift")
    kernel = projection["kernel"]
    bias = projection["bias"]
    if len(kernel.shape) != 2 or len(bias.shape) != 1 or int(kernel.shape[-1]) != int(bias.shape[0]):
        raise ValueError("attention projection parameter shape drift")
    return int(kernel.shape[0]), int(kernel.shape[1])


def head_output_bounds(width: int, heads: int, head: int) -> tuple[int, int]:
    """Return the output-coordinate interval owned by one attention head."""

    width = int(width)
    heads = int(heads)
    head = int(head)
    if width <= 0 or heads <= 0 or width % heads or head < 0 or head >= heads:
        raise ValueError("invalid attention head geometry")
    head_width = width // heads
    return head * head_width, (head + 1) * head_width


def _replace_last_axis_slice(base: Any, source: Any, start: int, stop: int) -> Any:
    if tuple(base.shape) != tuple(source.shape):
        raise ValueError("attention projection source shape drift")
    if hasattr(base, "at"):
        return base.at[..., start:stop].set(source[..., start:stop])
    result = np.array(base, copy=True)
    result[..., start:stop] = np.asarray(source)[..., start:stop]
    return result


def replace_projection_head(
    base: Mapping[str, Any],
    source: Mapping[str, Any],
    *,
    head: int,
    heads: int,
) -> Mapping[str, Any]:
    """Copy one projection head's kernel columns and matching bias slice."""

    base_shape = _projection_shape(base)
    source_shape = _projection_shape(source)
    if base_shape != source_shape:
        raise ValueError("attention projection pair shape drift")
    start, stop = head_output_bounds(base_shape[1], heads, head)
    return {
        "kernel": _replace_last_axis_slice(base["kernel"], source["kernel"], start, stop),
        "bias": _replace_last_axis_slice(base["bias"], source["bias"], start, stop),
    }


def replace_block_attention_head(
    base_blocks: Sequence[Mapping[str, Any]],
    source_blocks: Sequence[Mapping[str, Any]],
    *,
    block_index: int,
    head: int,
    heads: int,
    projections: Sequence[str] = ("query", "key"),
) -> tuple[Mapping[str, Any], ...]:
    """Copy selected Q/K output slices for one head into a block tree."""

    if len(base_blocks) != len(source_blocks) or block_index < 0 or block_index >= len(base_blocks):
        raise ValueError("potential block registry drift")
    projections = tuple(str(value) for value in projections)
    if not projections or len(projections) != len(frozenset(projections)) or not set(projections) <= {"query", "key"}:
        raise ValueError("head attribution supports unique query/key projections only")
    result = list(base_blocks)
    block = dict(result[block_index])
    source_block = source_blocks[block_index]
    attention = dict(block["attention"])
    source_attention = source_block["attention"]
    for name in projections:
        attention[name] = replace_projection_head(
            attention[name],
            source_attention[name],
            head=head,
            heads=heads,
        )
    block["attention"] = attention
    result[block_index] = block
    return tuple(result)


def projection_head_delta_coverage(
    lower_blocks: Sequence[Mapping[str, Any]],
    upper_blocks: Sequence[Mapping[str, Any]],
    *,
    block_index: int,
    heads: int,
) -> Mapping[str, Any]:
    """Measure Q/K head deltas and prove their slices exactly partition outputs."""

    if len(lower_blocks) != len(upper_blocks) or block_index < 0 or block_index >= len(lower_blocks):
        raise ValueError("potential block registry drift")
    records: dict[str, Any] = {}
    projection_checks = {}
    widths = set()
    for projection_name in ("query", "key"):
        lower = lower_blocks[block_index]["attention"][projection_name]
        upper = upper_blocks[block_index]["attention"][projection_name]
        lower_shape = _projection_shape(lower)
        if lower_shape != _projection_shape(upper):
            raise ValueError("attention projection pair shape drift")
        width = lower_shape[1]
        widths.add(width)
        kernel_delta = np.asarray(upper["kernel"], dtype=np.float64) - np.asarray(lower["kernel"], dtype=np.float64)
        bias_delta = np.asarray(upper["bias"], dtype=np.float64) - np.asarray(lower["bias"], dtype=np.float64)
        reconstructed_squared = 0.0
        for head in range(int(heads)):
            start, stop = head_output_bounds(width, heads, head)
            kernel_squared = float(np.sum(kernel_delta[..., start:stop] ** 2))
            bias_squared = float(np.sum(bias_delta[start:stop] ** 2))
            reconstructed_squared += kernel_squared + bias_squared
            records.setdefault(f"head_{head}", {})[projection_name] = {
                "output_start": start,
                "output_stop": stop,
                "kernel_delta_l2": float(np.sqrt(kernel_squared)),
                "bias_delta_l2": float(np.sqrt(bias_squared)),
                "combined_delta_l2": float(np.sqrt(kernel_squared + bias_squared)),
            }
        full_squared = float(np.sum(kernel_delta**2) + np.sum(bias_delta**2))
        projection_checks[projection_name] = {
            "full_delta_l2": float(np.sqrt(full_squared)),
            "reconstructed_delta_l2": float(np.sqrt(reconstructed_squared)),
            "squared_delta_reproduced": bool(
                np.isclose(reconstructed_squared, full_squared, rtol=1.0e-12, atol=1.0e-24)
            ),
        }
    width = widths.pop() if len(widths) == 1 else -1
    exact = bool(
        width > 0
        and width % int(heads) == 0
        and all(value["squared_delta_reproduced"] for value in projection_checks.values())
        and tuple(
            (records[f"head_{head}"]["query"]["output_start"], records[f"head_{head}"]["query"]["output_stop"])
            for head in range(int(heads))
        )
        == tuple((head * (width // int(heads)), (head + 1) * (width // int(heads))) for head in range(int(heads)))
    )
    return {
        "block_index": int(block_index),
        "width": width,
        "head_count": int(heads),
        "head_width": width // int(heads) if width > 0 and width % int(heads) == 0 else None,
        "exact_nonoverlapping_coverage": exact,
        "projections": projection_checks,
        "heads": records,
    }


def classify_head_controls(
    *,
    lower_gpp: float,
    upper_gpp: float,
    upper_head_only_gpp: Mapping[str, float],
    upper_without_head_gpp: Mapping[str, float],
    target_head: int = TARGET_HEAD_INDEX,
) -> Mapping[str, Any]:
    """Classify paired Q/K head-slice sufficiency and necessity controls."""

    groups = tuple(f"head_{head}" for head in range(ATTENTION_HEADS))
    result = dict(
        classify_subtree_controls(
            control_gpp=lower_gpp,
            treatment_gpp=upper_gpp,
            treatment_group_only_gpp=upper_head_only_gpp,
            treatment_without_group_gpp=upper_without_head_gpp,
            groups=groups,
        )
    )
    target = result["groups"][f"head_{int(target_head)}"]
    sufficient = abs(float(target["signed_sufficiency_fraction"])) >= INDIVIDUALLY_MATERIAL_FRACTION_MINIMUM
    necessary = abs(float(target["necessity_capture_fraction"])) >= INDIVIDUALLY_MATERIAL_FRACTION_MINIMUM
    if sufficient and necessary:
        classification = "target_head_qk_pair_sufficient_and_necessary"
    elif sufficient:
        classification = "target_head_qk_pair_sufficient_not_necessary"
    elif necessary:
        classification = "target_head_qk_pair_necessary_not_sufficient"
    elif result["selected_groups"]:
        classification = "other_or_distributed_heads_material"
    else:
        classification = "no_individual_qk_head_pair_material"
    result.update(
        {
            "classification": classification,
            "target_head": int(target_head),
            "target_head_sufficient": sufficient,
            "target_head_necessary": necessary,
        }
    )
    return result


def state_summary_bank_indices(
    state_leaves: Sequence[Any],
    *,
    forcing_token_count: int,
    keys: Sequence[str],
) -> Mapping[str, int]:
    """Resolve structured-bank indices for continuous state-summary tokens."""

    continuous = tuple(str(leaf.key) for leaf in state_leaves if not bool(leaf.discrete))
    if len(continuous) != len(frozenset(continuous)):
        raise ValueError("continuous state-summary key registry overlaps")
    result = {}
    for key in keys:
        key = str(key)
        if key not in continuous:
            raise ValueError(f"continuous state-summary key is absent: {key}")
        result[key] = int(forcing_token_count) + continuous.index(key)
    return result


def attention_token_summary(
    trace: Mapping[str, Any],
    *,
    pft_index: int,
    block_index: int,
    head_index: int,
    tokens: Mapping[str, int],
) -> Mapping[str, Any]:
    """Report exact logits, weights, validity, ranks, and winner for named tokens."""

    block = trace["blocks"][block_index]
    logits = np.asarray(block["attention_logits"])[pft_index, head_index, 0].astype(np.float64)
    weights = np.asarray(block["attention_weights"])[pft_index, head_index, 0].astype(np.float64)
    effective = np.asarray(trace["effective_attention_masks"], dtype=np.bool_)[block_index, pft_index, head_index, 0]
    valid_indices = np.flatnonzero(effective)
    if not valid_indices.size:
        raise ValueError("selected attention row has no valid source token")
    valid_weights = weights[valid_indices]
    winner_local = int(np.argmax(valid_weights))
    winner_index = int(valid_indices[winner_local])
    positive = valid_weights > 0.0
    entropy = float(-np.sum(valid_weights[positive] * np.log(valid_weights[positive])))
    token_records = {}
    for name, raw_index in tokens.items():
        index = int(raw_index)
        if index < 0 or index >= logits.size:
            raise ValueError(f"attention token index is outside bank: {name}={index}")
        valid = bool(effective[index])
        token_records[str(name)] = {
            "bank_index": index,
            "valid": valid,
            "logit": float(logits[index]),
            "weight": float(weights[index]),
            "rank_among_valid": int(1 + np.sum(logits[valid_indices] > logits[index])) if valid else None,
        }
    return {
        "block": int(block_index),
        "head": int(head_index),
        "valid_bank_count": int(valid_indices.size),
        "winner_bank_index": winner_index,
        "winner_logit": float(logits[winner_index]),
        "winner_weight": float(weights[winner_index]),
        "entropy": entropy,
        "tokens": token_records,
    }


def summarize_token_mask_controls(
    *,
    lower_gpp: float,
    upper_gpp: float,
    scenarios: Mapping[str, Mapping[str, float]],
) -> Mapping[str, Any]:
    """Normalize matched lower/upper token-removal interventions."""

    full_effect = float(upper_gpp) - float(lower_gpp)
    if not np.isfinite(full_effect) or abs(full_effect) <= 1.0e-12:
        raise ValueError("token-mask controls require a finite nonzero endpoint effect")
    records = {}
    for name, values in scenarios.items():
        if frozenset(values) != frozenset(("lower_gpp", "upper_gpp")):
            raise ValueError(f"token-mask scenario endpoint registry drift: {name}")
        lower = float(values["lower_gpp"])
        upper = float(values["upper_gpp"])
        if not np.all(np.isfinite((lower, upper))):
            raise ValueError(f"token-mask scenario is nonfinite: {name}")
        residual = upper - lower
        capture = (float(upper_gpp) - upper) / full_effect
        records[str(name)] = {
            "lower_gpp": lower,
            "upper_gpp": upper,
            "lower_shift_fraction": (lower - float(lower_gpp)) / full_effect,
            "upper_removal_capture_fraction": capture,
            "residual_effect": residual,
            "residual_effect_fraction": residual / full_effect,
            "upper_token_descriptively_necessary": bool(abs(capture) >= INDIVIDUALLY_MATERIAL_FRACTION_MINIMUM),
        }
    return {
        "full_parameter_effect": full_effect,
        "material_fraction_minimum": INDIVIDUALLY_MATERIAL_FRACTION_MINIMUM,
        "scenarios": records,
    }


__all__ = [
    "ATTENTION_HEADS",
    "DOC_BANK_INDEX",
    "HEAD_WIDTH",
    "LITTERPART_BANK_INDEX",
    "MODEL_WIDTH",
    "PARENT_CLASSIFICATION",
    "PARENT_REPORT_SHA256",
    "PARENT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "TARGET_BLOCK_INDEX",
    "TARGET_HEAD_INDEX",
    "TARGET_TOKEN_KEYS",
    "attention_token_summary",
    "classify_head_controls",
    "head_output_bounds",
    "projection_head_delta_coverage",
    "replace_block_attention_head",
    "replace_projection_head",
    "state_summary_bank_indices",
    "summarize_token_mask_controls",
]
