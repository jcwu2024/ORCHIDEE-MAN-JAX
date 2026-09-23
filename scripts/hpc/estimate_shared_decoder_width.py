"""Estimate one proposed split-width decoder from existing audited metadata.

This is arithmetic only: no model construction, Teacher reads, compilation,
architecture implementation, or wall-time/memory prediction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def estimate(cost: dict, packing: dict) -> dict:
    cfg = cost["configuration"]
    shapes = packing["shape_evidence"]
    if cost["coordinate_work"]["eligible_coordinates"] != shapes["eligible_coordinates"]:
        raise ValueError("saved audits describe different coordinate layouts")
    width = int(cfg["width"])
    proposed_width = 384
    if width != 768:
        raise ValueError("proposal assumes the audited width-768 backbone")
    blocks = int(cfg["decoder_blocks"])
    padded = int(shapes["existing_field_local_padded_coordinates"])
    dynamic_rows = int(shapes["field_local_axis_bank_entries"]) + int(shapes["field_local_state_bank_entries"])
    original = cost["forward_mac_estimate"]["coordinate_decoder"]["current_gather_reuse"].copy()
    original.pop("total")
    # Match the already-measured field-local reuse, NOT the older repeated-K/V path.
    original["dynamic_axis_local_kv"] = blocks * 2 * dynamic_rows * width**2
    original["local_value_input"] = int(shapes["field_local_state_bank_entries"]) * 2 * width
    assert original["dynamic_axis_local_kv"] == shapes["field_local_dynamic_projection_macs"]
    square_terms = {
        "shared_family_process_carbon_kv",
        "query_and_output_projections",
        "dynamic_axis_local_kv",
        "ffn",
    }
    proposed = {}
    for name, value in original.items():
        if name in square_terms:
            proposed[name] = value * proposed_width**2 // width**2
        elif name == "readout":
            proposed[name] = padded * (proposed_width * int(cfg["readout_width"]) + int(cfg["readout_width"]))
        else:
            proposed[name] = value * proposed_width // width
    # One shared token-local affine bridge after lawful family pooling and
    # before per-output coordinate expansion. Raw carbon/local values and
    # output coordinate embeddings are constructed directly at decoder width.
    bridge_rows = int(cfg["experts"]) * (int(cfg["source_families"]) + int(cfg["axis_tokens_per_expert"]) + 1) + len(
        cfg["active_pft_indices"]
    )
    proposed["context_bridge"] = bridge_rows * width * proposed_width
    backbone = cost["forward_mac_estimate"]["shared_input_and_backbone"]["current_broadcast_source_projection"]["total"]
    old_total = backbone + sum(original.values())
    new_total = backbone + sum(proposed.values())
    # Existing block has 12*w*w matrix entries and 13*w bias/norm entries.
    # Decoder embedding tables also shrink, so omitting that saving is an upper bound.
    parameter_upper_bound = (
        75_578_497
        - blocks * (12 * width**2 + 13 * width)
        + blocks * (12 * proposed_width**2 + 13 * proposed_width)
        + width * proposed_width
        + proposed_width
        - (width - proposed_width) * int(cfg["readout_width"])
    )
    # Hypothetical concatenation of all FFN rows; actual code chunks fields.
    # This is a logical activation-volume comparison, not an AD live-memory estimate.
    hidden = {str(w): 4 * padded * (4 * w) * 4 for w in (width, proposed_width)}
    return {
        "status": "proposal_only",
        "scope": "source_shape_forward_MACs_not_wall_time_or_peak_memory",
        "architecture": {
            "backbone_width": width,
            "backbone_blocks": cfg["backbone_blocks"],
            "decoder_width": proposed_width,
            "decoder_blocks": blocks,
            "decoder_heads": 8,
            "decoder_ffn_width": 4 * proposed_width,
            "readout_width": cfg["readout_width"],
            "uniform_for_all_fields": True,
        },
        "coordinate_count": shapes["eligible_coordinates"],
        "padded_coordinate_count": padded,
        "field_local_reference_decoder_macs": original,
        "proposed_decoder_macs": proposed,
        "context_bridge_rows": bridge_rows,
        "unchanged_backbone_macs": backbone,
        "field_local_reference_total_macs": old_total,
        "proposed_total_macs": new_total,
        "counted_mac_reduction_fraction": 1 - new_total / old_total,
        "parameter_upper_bound_excluding_additional_embedding_savings": parameter_upper_bound,
        "logical_all_query_FFN_activation_bytes_micro4": hidden,
        "limitations": [
            "Source-shape arithmetic excludes tokenizer, physics, AD, loss, optimizer and compiler effects.",
            "Backbone broadcast projections may already be optimized by XLA; this is not measured attribution.",
            "Local-token input is counted once per field-local bank; features may also cause compiler-dependent duplicates.",
            "Chunk padding and field boundaries remain; smaller width does not eliminate launch overhead.",
            "No parameter/state gradient equality to the wider model is expected: architecture changes.",
            "Width reduction is a capacity tradeoff, not evidence that any output is scientifically unimportant.",
        ],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cost", type=Path, default=root / "outputs/shared-decoder-cost-audit-20260923.json")
    parser.add_argument("--packing", type=Path, default=root / "outputs/shared-decoder-packing-audit-20260923.json")
    parser.add_argument("--output", type=Path, default=root / "outputs/shared-decoder-width-proposal-20260923.json")
    args = parser.parse_args()
    result = estimate(json.loads(args.cost.read_text()), json.loads(args.packing.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
