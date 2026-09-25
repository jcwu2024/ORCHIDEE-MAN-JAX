"""Independent FP64 single-coordinate oracle for non-GPP decoder diagnostics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _array64(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def _linear64(parameters: Mapping[str, Any], value: np.ndarray) -> np.ndarray:
    return value @ _array64(parameters["kernel"]) + _array64(parameters["bias"])


def _normalize_head64(value: np.ndarray, epsilon: float) -> tuple[np.ndarray, float, float]:
    maximum = float(np.max(np.abs(value)))
    scaled = value / (maximum if maximum > 0.0 else 1.0)
    scaled_norm = float(np.sqrt(np.sum(scaled * scaled) + epsilon**2))
    raw_norm = maximum * scaled_norm
    normalized = scaled / scaled_norm if raw_norm > epsilon else np.zeros_like(value)
    return normalized, maximum, raw_norm


def _gelu_tanh64(value: np.ndarray) -> np.ndarray:
    scale = np.sqrt(2.0 / np.pi)
    return 0.5 * value * (1.0 + np.tanh(scale * (value + 0.044715 * value**3)))


def _json_stage(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        item = array.item()
        return item if isinstance(item, (bool, int, float)) else float(item)
    if array.dtype == np.dtype("bool") or np.issubdtype(array.dtype, np.integer):
        if array.size <= 8:
            return array.tolist()
    numeric = array.astype(np.float64, copy=False)
    return {
        "shape": list(array.shape),
        "finite": bool(np.all(np.isfinite(numeric))),
        "min": float(np.min(numeric)) if numeric.size else 0.0,
        "max": float(np.max(numeric)) if numeric.size else 0.0,
        "l2": float(np.linalg.norm(numeric.ravel())),
        "preview": numeric.ravel()[:8].tolist(),
    }


def _layer_norm64(value: np.ndarray, scale: Any, bias: Any) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(value))
    variance = float(np.mean((value - mean) ** 2))
    result = (value - mean) / np.sqrt(variance + 1.0e-6)
    return result * _array64(scale) + _array64(bias), mean, variance


def coordinate_oracle(
    shared: Mapping[str, Any],
    sample_payload: Mapping[str, Any],
    field_mask: Mapping[str, Any],
    field_index: int,
    row_index: int,
    expert_index: int,
    *,
    heads: int = 8,
    epsilon: float = 1.0e-6,
) -> dict[str, Any]:
    """Evaluate one non-GPP output coordinate using explicit NumPy float64 ops.

    Stored fixture/model arrays are typically FP32; they are promoted before
    arithmetic. The return value contains stable intermediate arrays for
    inspecting discrepant coordinates. Carbon-bank attention is intentionally
    outside this oracle's scope.
    """

    if heads <= 0 or epsilon <= 0.0:
        raise ValueError("heads and normalization epsilon must be positive")
    fields = sample_payload["fields"]
    if not 0 <= field_index < len(fields):
        raise IndexError("field index outside payload")
    data = fields[field_index]
    query_bank = _array64(data["query"])
    rows, query_count, width = query_bank.shape
    if not 0 <= row_index < rows or query_count != 1:
        raise IndexError("row index outside one-query field")
    if width % heads:
        raise ValueError("decoder width must be divisible by attention heads")
    if expert_index < 0 or expert_index >= np.asarray(sample_payload["family"]).shape[0]:
        raise IndexError("expert index outside family/process payload")
    blocks = shared["coordinate_blocks"]
    if len(blocks) != 2:
        raise ValueError("oracle currently requires the two-block coordinate decoder")
    family_bank = _array64(sample_payload["family"])[expert_index]
    process_bank = _array64(sample_payload["process"])[expert_index]
    axis_bank = _array64(data["axis"])
    local_bank = _array64(data["local"])
    axis_indices = np.asarray(field_mask["axis_indices"][row_index], dtype=np.int64)
    local_index = int(field_mask["local_indices"][row_index])
    active = bool(field_mask["active"][row_index])
    dynamic_indices = np.concatenate((axis_indices, np.asarray([local_index], dtype=np.int64)))
    dynamic_tokens = np.concatenate((axis_bank[axis_indices], local_bank[local_index][None, :]), axis=0)
    family_mask = np.asarray(field_mask["family"][row_index, 0], dtype=bool)
    dynamic_mask = np.asarray(field_mask["dynamic"][row_index, 0], dtype=bool)
    process_mask = np.asarray(field_mask["process"][row_index, 0], dtype=bool)
    if family_mask.shape != (family_bank.shape[0],):
        raise ValueError("family mask does not match selected expert bank")
    if dynamic_mask.shape != (dynamic_tokens.shape[0],):
        raise ValueError("dynamic mask does not match gathered axis/local tokens")
    if process_mask.shape != (process_bank.shape[0],):
        raise ValueError("process mask does not match selected expert bank")

    stages: dict[str, Any] = {
        "input_query": query_bank[row_index, 0].copy(),
        "dynamic_indices": dynamic_indices.copy(),
        "dynamic_tokens": dynamic_tokens.copy(),
        "family_mask": family_mask.copy(),
        "dynamic_mask": dynamic_mask.copy(),
        "process_mask": process_mask.copy(),
        "output_active": active,
    }
    query = query_bank[row_index].copy()
    head_width = width // heads
    temperature = np.sqrt(float(head_width))

    for block_index, block in enumerate(blocks):
        attention = block["attention"]
        query_projected = _linear64(attention["query"], query)
        family_key = _linear64(attention["key"], family_bank)
        family_value = _linear64(attention["value"], family_bank)
        process_key = _linear64(attention["key"], process_bank)
        process_value = _linear64(attention["value"], process_bank)
        axis_key = _linear64(attention["key"], axis_bank)[axis_indices]
        axis_value = _linear64(attention["value"], axis_bank)[axis_indices]
        local_key = _linear64(attention["key"], local_bank)[local_index][None, :]
        local_value = _linear64(attention["value"], local_bank)[local_index][None, :]
        dynamic_key = np.concatenate((axis_key, local_key), axis=0)
        dynamic_value = np.concatenate((axis_value, local_value), axis=0)
        if not active:
            dynamic_key = np.broadcast_to(_array64(attention["key"]["bias"]), dynamic_key.shape).copy()
            dynamic_value = np.broadcast_to(_array64(attention["value"]["bias"]), dynamic_value.shape).copy()

        query_heads = query_projected.reshape(1, heads, head_width)[0]
        family_key_heads = family_key.reshape(-1, heads, head_width).transpose(1, 0, 2)
        family_value_heads = family_value.reshape(-1, heads, head_width).transpose(1, 0, 2)
        dynamic_key_heads = dynamic_key.reshape(-1, heads, head_width).transpose(1, 0, 2)
        dynamic_value_heads = dynamic_value.reshape(-1, heads, head_width).transpose(1, 0, 2)
        process_key_heads = process_key.reshape(-1, heads, head_width).transpose(1, 0, 2)
        process_value_heads = process_value.reshape(-1, heads, head_width).transpose(1, 0, 2)
        query_norm = np.zeros_like(query_heads)
        family_key_norm = np.zeros_like(family_key_heads)
        dynamic_key_norm = np.zeros_like(dynamic_key_heads)
        process_key_norm = np.zeros_like(process_key_heads)
        query_norm_stats = []
        family_norm_stats = []
        dynamic_norm_stats = []
        process_norm_stats = []
        for head in range(heads):
            query_norm[head], maximum, raw_norm = _normalize_head64(query_heads[head], epsilon)
            query_norm_stats.append((maximum, raw_norm))
            for position in range(family_key_heads.shape[1]):
                family_key_norm[head, position], maximum, raw_norm = _normalize_head64(
                    family_key_heads[head, position], epsilon,
                )
                family_norm_stats.append((head, position, maximum, raw_norm))
            for position in range(dynamic_key_heads.shape[1]):
                dynamic_key_norm[head, position], maximum, raw_norm = _normalize_head64(
                    dynamic_key_heads[head, position], epsilon,
                )
                dynamic_norm_stats.append((head, position, maximum, raw_norm))
            for position in range(process_key_heads.shape[1]):
                process_key_norm[head, position], maximum, raw_norm = _normalize_head64(
                    process_key_heads[head, position], epsilon,
                )
                process_norm_stats.append((head, position, maximum, raw_norm))

        family_logits = np.clip(np.einsum("hd,hkd->hk", query_norm, family_key_norm), -1.0, 1.0) * temperature
        dynamic_logits = np.clip(np.einsum("hd,hkd->hk", query_norm, dynamic_key_norm), -1.0, 1.0) * temperature
        process_logits = np.clip(np.einsum("hd,hkd->hk", query_norm, process_key_norm), -1.0, 1.0) * temperature
        combined_mask = np.concatenate((family_mask, dynamic_mask, process_mask))
        combined_logits = np.concatenate((family_logits, dynamic_logits, process_logits), axis=1)
        masked_logits = np.where(combined_mask[None, :], combined_logits, -np.inf)
        row_valid = bool(np.any(combined_mask))
        if row_valid:
            shifted = masked_logits - np.max(masked_logits, axis=-1, keepdims=True)
            exponentials = np.where(combined_mask[None, :], np.exp(shifted), 0.0)
            weights = exponentials / np.sum(exponentials, axis=-1, keepdims=True)
        else:
            weights = np.zeros_like(combined_logits)

        family_end = family_mask.size
        dynamic_end = family_end + dynamic_mask.size
        family_weights = weights[:, :family_end]
        dynamic_weights = weights[:, family_end:dynamic_end]
        process_weights = weights[:, dynamic_end:]
        message_heads = np.einsum("hk,hkd->hd", family_weights, family_value_heads)
        message_heads += np.einsum("hk,hkd->hd", dynamic_weights, dynamic_value_heads)
        message_heads += np.einsum("hk,hkd->hd", process_weights, process_value_heads)
        attention_message = _linear64(attention["output"], message_heads.reshape(1, width))
        attention_residual = query + attention_message
        query, attention_mean, attention_variance = _layer_norm64(
            attention_residual,
            block["attention_norm_scale"],
            block["attention_norm_bias"],
        )
        attention_norm = query.copy()
        ffn_first = _linear64(block["ffn"]["first"], query)
        ffn_activation = _gelu_tanh64(ffn_first)
        ffn_second = _linear64(block["ffn"]["second"], ffn_activation)
        ffn_residual = query + ffn_second
        query, ffn_mean, ffn_variance = _layer_norm64(
            ffn_residual, block["ffn_norm_scale"], block["ffn_norm_bias"],
        )
        prefix = f"block_{block_index}"
        stages.update({
            f"{prefix}_query_projected": query_projected.copy(),
            f"{prefix}_family_key_projected": family_key.copy(),
            f"{prefix}_family_value_projected": family_value.copy(),
            f"{prefix}_dynamic_key_projected": dynamic_key.copy(),
            f"{prefix}_dynamic_value_projected": dynamic_value.copy(),
            f"{prefix}_process_key_projected": process_key.copy(),
            f"{prefix}_process_value_projected": process_value.copy(),
            f"{prefix}_query_heads_normalized": query_norm.copy(),
            f"{prefix}_family_keys_normalized": family_key_norm.copy(),
            f"{prefix}_dynamic_keys_normalized": dynamic_key_norm.copy(),
            f"{prefix}_process_keys_normalized": process_key_norm.copy(),
            f"{prefix}_query_norm_stats": np.asarray(query_norm_stats, dtype=np.float64),
            f"{prefix}_family_key_norm_stats": np.asarray(family_norm_stats, dtype=np.float64),
            f"{prefix}_dynamic_key_norm_stats": np.asarray(dynamic_norm_stats, dtype=np.float64),
            f"{prefix}_process_key_norm_stats": np.asarray(process_norm_stats, dtype=np.float64),
            f"{prefix}_family_logits": family_logits.copy(),
            f"{prefix}_dynamic_logits": dynamic_logits.copy(),
            f"{prefix}_process_logits": process_logits.copy(),
            f"{prefix}_combined_mask": combined_mask.copy(),
            f"{prefix}_combined_logits": combined_logits.copy(),
            f"{prefix}_weights": weights.copy(),
            f"{prefix}_attention_message": attention_message.copy(),
            f"{prefix}_attention_residual": attention_residual.copy(),
            f"{prefix}_attention_norm": attention_norm,
            f"{prefix}_attention_norm_stats": np.asarray([attention_mean, attention_variance]),
            f"{prefix}_ffn_first": ffn_first.copy(),
            f"{prefix}_ffn_activation": ffn_activation.copy(),
            f"{prefix}_ffn_second": ffn_second.copy(),
            f"{prefix}_ffn_residual": ffn_residual.copy(),
            f"{prefix}_output_norm": query.copy(),
            f"{prefix}_output_norm_stats": np.asarray([ffn_mean, ffn_variance]),
        })

    readout_first = _linear64(shared["readout_first"], query)
    readout_activation = _gelu_tanh64(readout_first)
    raw_output = float(_linear64(shared["readout_second"], readout_activation)[0, 0])
    output = raw_output if active else 0.0
    stages["readout_first"] = readout_first.copy()
    stages["readout_activation"] = readout_activation.copy()
    stages["raw_output"] = raw_output
    stages["output"] = output
    return {
        "output": float(output),
        "stages": {name: _json_stage(value) for name, value in stages.items()},
        "scientific_scope": (
            "single non-GPP coordinate; source-equivalent two-block decoder; "
            "NumPy float64 arithmetic over stored fixture/model values; no carbon bank"
        ),
        "field_index": int(field_index),
        "row_index": int(row_index),
        "expert_index": int(expert_index),
        "output_active": active,
    }
