"""Offline checks for conservative split-profile attribution."""

import gzip
import json

import pytest

from scripts.dev.summarize_shared_split_profile import (
    _hlo_instruction_sources,
    source_category,
    summarize_assets,
    summarize_hlo,
    summarize_trace_sources,
)


def test_hlo_counts_and_shared_helper_stays_ambiguous():
    hlo = """HloModule test
ENTRY %main (x: f32[]) -> f32[] {
  %a = f32[] add(%x, %x), metadata={op_name="a" source_file="/repo/shared_transformer.py" source_line=537}
  ROOT %b = f32[] multiply(%a, %a), metadata={op_name="b" source_file="/repo/strong_baseline_architecture.py" source_line=887}
}
"""
    summary = summarize_hlo(hlo)
    assert summary["computation_count"] == 1
    assert summary["instruction_count"] == 2
    assert summary["operation_counts"] == {"add": 1, "multiply": 1}
    assert summary["source_category_instruction_counts"] == {
        "shared_transformer": 1,
        "ambiguous_or_unmapped": 1,
    }


def test_tuple_hlo_shape_and_colliding_op_names_remain_unmapped():
    hlo = """%forward (x: f32[]) -> (f32[], f32[]) {
  ROOT %pair = (f32[], f32[]) tuple(%x, %x), metadata={source_file="/repo/shared_transformer.py" source_line=537}
}
%reverse (x: f32[]) -> (f32[], f32[]) {
  ROOT %pair = (f32[], f32[]) tuple(%x, %x), metadata={source_file="/repo/jax_orchidee/tail.py" source_line=20}
}
"""
    summary = summarize_hlo(hlo)
    assert summary["instruction_count"] == 2
    assert summary["operation_counts"] == {"tuple": 2}
    assert summary["ambiguous_instruction_names"] == 1
    assert "pair" not in _hlo_instruction_sources(hlo)


def test_trace_uses_only_gpu_events_and_preserves_unmapped():
    trace = {
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:GPU:0"}},
            {"ph": "X", "pid": 1, "name": "kernel", "ts": 0, "dur": 10, "args": {"hlo_op": "a"}},
            {"ph": "X", "pid": 1, "name": "kernel", "ts": 10, "dur": 5, "args": {"hlo_op": "unknown"}},
            {"ph": "X", "pid": 2, "name": "host", "ts": 0, "dur": 100},
        ]
    }
    result = summarize_trace_sources(trace, {"a": ("/repo/shared_transformer.py", 537)})
    assert result["shared_transformer"]["events"] == 1
    assert result["shared_transformer"]["summed_seconds"] == pytest.approx(10e-6)
    assert result["ambiguous_or_unmapped"]["events"] == 1
    assert result["ambiguous_or_unmapped"]["summed_seconds"] == pytest.approx(5e-6)
    assert source_category(("/repo/strong_baseline_architecture.py", 622)) == "tokenizer"
    assert source_category(("/repo/strong_baseline_architecture.py", 887)) == "ambiguous_or_unmapped"


def test_saved_asset_layout_can_be_summarized_without_jax(tmp_path):
    hlo = "ENTRY %main (x: f32[]) -> f32[] {\n  ROOT %a = f32[] add(%x, %x)\n}\n"
    with gzip.open(tmp_path / "daily.optimized-hlo.txt.gz", "wt", encoding="utf-8") as handle:
        handle.write(hlo)
    trace_dir = tmp_path / "trace-daily_micro4" / "plugins" / "profile" / "run"
    trace_dir.mkdir(parents=True)
    trace = {
        "traceEvents": [
            {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:GPU:0"}},
            {"ph": "X", "pid": 1, "name": "fusion", "ts": 0, "dur": 5, "args": {"hlo_op": "a"}},
        ]
    }
    with gzip.open(trace_dir / "host.trace.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(trace, handle)
    report = summarize_assets(tmp_path)
    assert report["components"]["daily"]["hlo"]["instruction_count"] == 1
    assert report["components"]["daily"]["trace"]["event_count"] == 1
    assert report["components"]["T7"]["status"] == "missing_hlo"
