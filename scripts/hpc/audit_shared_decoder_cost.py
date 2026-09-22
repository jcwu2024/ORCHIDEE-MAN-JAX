"""Static shape/MAC audit for the admitted shared coordinate decoder.

This script reads contracts, fixture-shaped metadata, and an optional accepted
admission report.  It does not initialize neural parameters, read Teacher data,
compile JAX programs, or claim that static MAC counts predict wall time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_PROTOCOL = ROOT / "manifests/coarse_graining/shared_transformer_admission_v3.json"
DEFAULT_ACCEPTED_REPORT = ROOT / "outputs/shared-decoder-094dfd2-cpu.json"
DEFAULT_OUTPUT = ROOT / "outputs/shared-decoder-cost-audit-20260923.json"


def padded_work(size: int, maximum_chunk: int) -> tuple[int, int]:
    """Match ``_decode_field`` field-local chunk and padding arithmetic."""

    if size < 0 or maximum_chunk <= 0:
        raise ValueError("coordinate size must be nonnegative and chunk size positive")
    if size == 0:
        return 0, 0
    chunk = min(maximum_chunk, size)
    chunks = math.ceil(size / chunk)
    return chunks * chunk, chunks


def dense_mac_breakdown(
    fields: Iterable[Mapping[str, Any]],
    *,
    width: int,
    ffn_width: int,
    decoder_blocks: int,
    readout_width: int,
    family_tokens: int,
    experts: int,
    carbon_tokens: int,
    unique_axis_variants: int,
    unique_local_variants: int,
) -> dict[str, dict[str, int]]:
    """Count forward dense/dot MACs at source-visible tensor shapes.

    Bias, normalization, activation, masking, gather/scatter, and retained
    physical-process arithmetic are deliberately excluded.  A multiply-add is
    counted as one MAC, not two FLOPs.
    """

    rows = tuple(fields)
    valid = sum(int(row["eligible_coordinates"]) for row in rows)
    padded = sum(int(row["padded_coordinates"]) for row in rows)
    padded_dynamic = sum(int(row["padded_coordinates"]) * int(row["dynamic_tokens_per_query"]) for row in rows)
    valid_attention_keys = sum(
        int(row["eligible_coordinates"])
        * (
            family_tokens
            + int(row["dynamic_tokens_per_query"])
            + 1
            + (carbon_tokens if bool(row["uses_carbon_bank"]) else 0)
        )
        for row in rows
    )
    padded_attention_keys = sum(
        int(row["padded_coordinates"])
        * (
            family_tokens
            + int(row["dynamic_tokens_per_query"])
            + 1
            + (carbon_tokens if bool(row["uses_carbon_bank"]) else 0)
        )
        for row in rows
    )

    projection_cache = decoder_blocks * 2 * (carbon_tokens + experts * (family_tokens + 1)) * width * width
    current = {
        "shared_family_process_carbon_kv": projection_cache,
        "query_and_output_projections": decoder_blocks * 2 * padded * width * width,
        "dynamic_axis_local_kv": decoder_blocks * 2 * padded_dynamic * width * width,
        "ffn": decoder_blocks * 2 * padded * width * ffn_width,
        "attention_score_and_value_dots": decoder_blocks * 2 * padded_attention_keys * width,
        "readout": padded * (width * readout_width + readout_width),
        "local_value_input": valid * 2 * width,
        "carbon_value_input": carbon_tokens * 2 * width,
    }
    packed_cached = {
        "shared_family_process_carbon_kv": projection_cache,
        "query_and_output_projections": decoder_blocks * 2 * valid * width * width,
        "dynamic_axis_local_kv": decoder_blocks * 2 * (unique_axis_variants + unique_local_variants) * width * width,
        "ffn": decoder_blocks * 2 * valid * width * ffn_width,
        "attention_score_and_value_dots": decoder_blocks * 2 * valid_attention_keys * width,
        "readout": valid * (width * readout_width + readout_width),
        "local_value_input": unique_local_variants * 2 * width,
        "carbon_value_input": carbon_tokens * 2 * width,
    }
    current["total"] = sum(current.values())
    packed_cached["total"] = sum(packed_cached.values())
    return {"current_gather_reuse": current, "packed_cached_ceiling": packed_cached}


def backbone_mac_breakdown(
    *,
    width: int,
    input_width: int,
    ffn_width: int,
    backbone_blocks: int,
    experts: int,
    source_tokens: int,
    axis_tokens: int,
    pft_tokens: int,
) -> dict[str, dict[str, int]]:
    bank = source_tokens + axis_tokens
    common = {
        "input_projection": (source_tokens + experts * axis_tokens + pft_tokens) * input_width * width,
        "ffn": backbone_blocks * 2 * experts * width * ffn_width,
        "attention_score_and_value_dots": backbone_blocks * 2 * experts * bank * width,
    }
    current = {
        **common,
        "query_key_value_output_projections": backbone_blocks * (2 * experts + 2 * experts * bank) * width * width,
    }
    source_cached = {
        **common,
        "query_key_value_output_projections": backbone_blocks
        * (2 * experts + 2 * source_tokens + 2 * experts * axis_tokens)
        * width
        * width,
    }
    current["total"] = sum(current.values())
    source_cached["total"] = sum(source_cached.values())
    return {
        "current_broadcast_source_projection": current,
        "shared_source_kv_cached_ceiling": source_cached,
    }


def _load_static_layout() -> Mapping[str, Any]:
    """Build the canonical schema/layout while replacing both neural initializers."""

    from research.daily_coarse_graining.daily_operator.output_transform import field_coordinates
    from research.daily_coarse_graining.daily_operator.shared_transformer import (
        shared_transformer_static_coordinate_indices,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
        load_target_oriented_contract,
    )
    from tests.unit.test_daily_operator_mechanistic_output import mechanistic_setup

    with (
        patch(
            "tests.unit.test_daily_operator_successor_process_head.initialize_successor_parameters",
            return_value={},
        ),
        patch(
            "tests.unit.test_daily_operator_mechanistic_output.initialize_strong_baseline_parameters",
            return_value={},
        ),
    ):
        setup = mechanistic_setup.__wrapped__()

    schema = setup["learned_schema"]
    layout = setup["layout"]
    selected = shared_transformer_static_coordinate_indices(schema=schema, layout=layout)
    categories = load_target_oriented_contract().labels_by_category
    category_by_label = {label: category for category, labels in categories.items() for label in labels}
    axis_variants: set[tuple[Any, ...]] = set()
    local_variants: set[int] = set()
    fields = []
    for field_index, (field, ownership) in enumerate(zip(schema.fields, layout.ledger, strict=True)):
        coordinates = np.asarray(field_coordinates(field.shape), dtype=np.int32)
        indices = np.asarray(selected[field.name], dtype=np.int32)
        retained = coordinates[indices]
        local = np.asarray(layout.local_state_indices[field_index], dtype=np.int32).reshape(-1)
        if local.size == 0:
            retained_local = np.full((indices.size,), -1, dtype=np.int32)
        else:
            if local.size != coordinates.shape[0]:
                raise ValueError(f"local-state index layout differs for {field.name}")
            retained_local = local[indices]
        local_variants.update(int(value) for value in retained_local)
        expert_index = int(layout.field_expert_index[field_index])
        axis_count = 0
        for slot, axis_name in enumerate(ownership.axes):
            if axis_name == "landpoint":
                continue
            axis_count += 1
            for coordinate in retained:
                value = int(coordinate[slot])
                if axis_name == "pft" and field.full_pft_axis:
                    value = layout.active_pft_indices.index(value)
                axis_variants.add((expert_index, slot, axis_name, value))
        eligible = int(indices.size)
        padded, chunks = padded_work(eligible, 64)
        label = schema.labels[field.label_index]
        fields.append(
            {
                "field": field.name,
                "category": category_by_label[label],
                "expert": layout.expert_names[expert_index],
                "expert_index": expert_index,
                "axes": list(ownership.axes),
                "full_coordinates": int(coordinates.shape[0]),
                "eligible_coordinates": eligible,
                "padded_coordinates": padded,
                "padding_coordinates": padded - eligible,
                "chunks": chunks,
                "dynamic_tokens_per_query": axis_count + 1,
                "uses_carbon_bank": field.name == "daily.fast_gpp",
                "unique_local_state_indices": len(set(int(value) for value in retained_local)),
            }
        )
    return {
        "schema": schema,
        "layout": layout,
        "fields": fields,
        "unique_axis_variants": len(axis_variants),
        "unique_local_variants": len(local_variants),
    }


def _aggregate(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        values = grouped[str(row[key])]
        values["fields"] += 1
        for name in (
            "full_coordinates",
            "eligible_coordinates",
            "padded_coordinates",
            "padding_coordinates",
            "chunks",
        ):
            values[name] += int(row[name])
    return {name: dict(values) for name, values in sorted(grouped.items())}


def _validate_accepted_report(path: Path, fields: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(fields)
    if not path.is_file():
        return {"path": str(path), "available": False, "matched": None}
    report = json.loads(path.read_text(encoding="utf-8"))
    accepted = report["decoder_execution"]
    observed = {row["field"]: int(row["eligible_coordinates"]) for row in rows}
    expected = {name: int(value) for name, value in accepted["eligible_by_field"].items()}
    matched = (
        expected == observed
        and int(accepted["eligible_coordinates"]) == sum(observed.values())
        and int(report["learned_fields"]) == len(rows)
        and int(report["learned_scalars"]) == sum(int(row["full_coordinates"]) for row in rows)
    )
    if not matched:
        raise ValueError("source-derived static layout differs from accepted decoder report")
    return {
        "path": str(path),
        "available": True,
        "matched": True,
        "mode": accepted["mode"],
        "rematerialization": bool(accepted["rematerialization"]),
    }


def _run_self_test() -> None:
    assert padded_work(0, 64) == (0, 0)
    assert padded_work(17, 64) == (17, 1)
    assert padded_work(65, 64) == (128, 2)
    fields = (
        {
            "eligible_coordinates": 65,
            "padded_coordinates": 128,
            "dynamic_tokens_per_query": 3,
            "uses_carbon_bank": False,
        },
        {
            "eligible_coordinates": 2,
            "padded_coordinates": 2,
            "dynamic_tokens_per_query": 2,
            "uses_carbon_bank": True,
        },
    )
    costs = dense_mac_breakdown(
        fields,
        width=8,
        ffn_width=32,
        decoder_blocks=2,
        readout_width=4,
        family_tokens=3,
        experts=2,
        carbon_tokens=5,
        unique_axis_variants=7,
        unique_local_variants=4,
    )
    assert costs["current_gather_reuse"]["total"] > costs["packed_cached_ceiling"]["total"]


def audit(args: argparse.Namespace) -> dict[str, Any]:
    _run_self_test()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    architecture = protocol["architecture"]
    current = _load_static_layout()
    layout = current["layout"]
    fields = current["fields"]
    width = int(architecture["width"])
    ffn_width = int(architecture["ffn_width"])
    decoder_blocks = int(architecture["decoder_blocks"])
    chunk_size = int(architecture["query_chunk_size"])
    if chunk_size != 64:
        raise ValueError("audit field padding was derived for the admitted chunk size 64")
    from research.daily_coarse_graining.daily_operator.shared_transformer import SharedTransformerConfig

    defaults = SharedTransformerConfig()
    if (
        width != defaults.width
        or ffn_width != defaults.ffn_width
        or decoder_blocks != defaults.coordinate_blocks
        or architecture["decoder_execution"] != defaults.decoder_execution
    ):
        raise ValueError("protocol and executable shared Transformer configuration differ")

    accepted = _validate_accepted_report(args.accepted_report, fields)
    decoder = dense_mac_breakdown(
        fields,
        width=width,
        ffn_width=ffn_width,
        decoder_blocks=decoder_blocks,
        readout_width=defaults.readout_width,
        family_tokens=len(layout.source_family_names),
        experts=len(layout.expert_names),
        carbon_tokens=args.carbon_tokens,
        unique_axis_variants=int(current["unique_axis_variants"]),
        unique_local_variants=int(current["unique_local_variants"]),
    )
    backbone = backbone_mac_breakdown(
        width=width,
        input_width=defaults.input_width,
        ffn_width=ffn_width,
        backbone_blocks=int(architecture["backbone_blocks"]),
        experts=len(layout.expert_names),
        source_tokens=args.source_tokens,
        axis_tokens=layout.axis_token_count,
        pft_tokens=len(layout.active_pft_indices),
    )
    current_backbone = backbone["current_broadcast_source_projection"]
    cached_backbone = backbone["shared_source_kv_cached_ceiling"]
    total_current = current_backbone["total"] + decoder["current_gather_reuse"]["total"]
    total_packed = cached_backbone["total"] + decoder["packed_cached_ceiling"]["total"]
    return {
        "schema_version": "shared_decoder_static_cost_audit_v1",
        "scope": "metadata_only_forward_shape_mac_estimate_no_runtime_or_compiler_claim",
        "source": {
            "protocol": str(args.protocol),
            "accepted_report": accepted,
            "implementation": "research/daily_coarse_graining/daily_operator/shared_transformer.py",
            "layout_fixture": "tests.unit.test_daily_operator_mechanistic_output.mechanistic_setup",
            "neural_initialization": False,
            "teacher_data_read": False,
        },
        "configuration": {
            "width": width,
            "ffn_width": ffn_width,
            "backbone_blocks": int(architecture["backbone_blocks"]),
            "decoder_blocks": decoder_blocks,
            "readout_width": defaults.readout_width,
            "chunk_size": chunk_size,
            "decoder_execution": architecture["decoder_execution"],
            "decoder_rematerialization": bool(architecture["decoder_rematerialization"]),
            "experts": len(layout.expert_names),
            "expert_names": list(layout.expert_names),
            "source_families": len(layout.source_family_names),
            "source_tokens": args.source_tokens,
            "axis_tokens_per_expert": layout.axis_token_count,
            "raw_carbon_tokens": args.carbon_tokens,
            "active_pft_indices": list(layout.active_pft_indices),
        },
        "coordinate_work": {
            "fields": len(fields),
            "full_coordinates": sum(int(row["full_coordinates"]) for row in fields),
            "eligible_coordinates": sum(int(row["eligible_coordinates"]) for row in fields),
            "field_local_padded_coordinates": sum(int(row["padded_coordinates"]) for row in fields),
            "padding_coordinates": sum(int(row["padding_coordinates"]) for row in fields),
            "field_local_chunks": sum(int(row["chunks"]) for row in fields),
            "unique_axis_token_variants": int(current["unique_axis_variants"]),
            "unique_local_state_variants": int(current["unique_local_variants"]),
            "by_category": _aggregate(fields, "category"),
            "by_expert": _aggregate(fields, "expert"),
            "largest_padded_fields": sorted(
                fields,
                key=lambda row: (
                    int(row["padded_coordinates"]),
                    int(row["eligible_coordinates"]),
                ),
                reverse=True,
            )[:20],
        },
        "forward_mac_estimate": {
            "definition": "one multiply-add is one MAC; forward pass only",
            "excluded": [
                "tokenizer_before_shared_input_projection",
                "bias_layernorm_activation_mask_gather_scatter",
                "output_physical_transforms_and_retained_processes",
                "loss_reverse_ad_optimizer_and_compiler_effects",
            ],
            "shared_input_and_backbone": backbone,
            "coordinate_decoder": decoder,
            "current_backbone_plus_decoder": total_current,
            "source_and_decoder_cached_combined_ceiling": total_packed,
            "backbone_static_mac_reduction_ceiling_fraction": 1.0
            - cached_backbone["total"] / current_backbone["total"],
            "decoder_static_mac_reduction_ceiling_fraction": 1.0
            - decoder["packed_cached_ceiling"]["total"] / decoder["current_gather_reuse"]["total"],
            "combined_static_mac_reduction_ceiling_fraction": 1.0 - total_packed / total_current,
        },
        "interpretation_limits": [
            "Counts follow source-visible tensor shapes before XLA fusion, common-subexpression elimination, or dead-code elimination.",
            "The shared-source backbone ceiling may overlap work that XLA already removes from the broadcasted source-token projection.",
            "The packed/cache estimate is an optimistic function-preserving execution ceiling, not an implemented or timed result.",
            "Forward MACs do not predict reverse-mode memory, rematerialization work, kernel occupancy, launch overhead, or wall time.",
            "The 393 source-token and 2386 carbon-token counts are accepted fixture shape evidence supplied explicitly to this audit.",
        ],
        "fields": fields,
        "status": "passed",
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    result.add_argument("--accepted-report", type=Path, default=DEFAULT_ACCEPTED_REPORT)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--source-tokens", type=int, default=393)
    result.add_argument("--carbon-tokens", type=int, default=2386)
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if args.source_tokens <= 0 or args.carbon_tokens <= 0:
        raise ValueError("accepted token counts must be positive")
    if args.self_test:
        _run_self_test()
        print(json.dumps({"status": "passed", "scope": "pure_arithmetic_self_test"}))
        return 0
    payload = audit(args)
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output),
                "coordinate_work": payload["coordinate_work"],
                "forward_mac_estimate": payload["forward_mac_estimate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
