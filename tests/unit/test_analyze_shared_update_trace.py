from __future__ import annotations

from scripts.hpc.analyze_shared_update_trace import summarize


def _event(name, start, duration, *, pid=2, args=None):
    return {"name": name, "ts": start, "dur": duration, "ph": "X", "pid": pid,
            "args": args or {}}


def test_trace_summary_clips_gpu_intervals_to_blocking_root_and_keeps_calls_nonexclusive():
    trace = {"traceEvents": [
        {"name": "process_name", "ph": "M", "pid": 2,
         "args": {"name": "/device:GPU:0"}},
        _event("full_update", 100, 1000, pid=1),
        _event("call_T7", 300, 400, pid=1),
        _event("ampere_sgemm_128x64", 250, 500, args={"hlo_op": "dot"}),
        _event("fusion", 650, 400),
        _event("outside", 1110, 100),
    ]}
    result = summarize(trace, {"compile_or_get_cached_calls": 0, "backend_compile_calls": 0})
    assert result["status"] == "completed"
    assert result["gpu_event_count_clipped_to_update"] == 2
    assert abs(result["gpu_active_union_seconds"] - 0.0008) < 1e-12
    assert abs(result["gpu_events_summed_seconds"] - 0.0009) < 1e-12
    call = result["call_annotations"][0]
    assert call["name"] == "call_T7"
    assert abs(call["gpu_active_overlap_union_seconds"] - 0.0004) < 1e-12
    assert call["call_interval_is_full_blocking_branch_wall"] is False
    assert result["compile_free_confirmed"] is True
    assert result["gpu_event_hlo_op_metadata_coverage"] == {
        "events_with_hlo_op": 1, "events_total": 2, "fraction": 0.5
    }
    assert result["gpu_kernel_categories_by_name_only"]["matrix_multiply"]["event_count"] == 1
    assert abs(result["gpu_first_last_event_span_fraction_of_root"] - 0.8) < 1e-12
    assert abs(result["conditional_gpu_active_union_fraction"] - 0.8) < 1e-12
    assert "not a hardware bound" in result["gpu_union_limitation"]


def test_trace_summary_keeps_amdahl_separate_and_surfaces_profiler_warnings():
    trace = {"traceEvents": [
        {"name": "process_name", "ph": "M", "pid": 2,
         "args": {"name": "/device:GPU:0"}},
        _event("full_update", 0, 1000, pid=1),
        _event("other_kernel", 100, 100, pid=2),
    ]}
    run_log = "WARNING CUPTI buffer unavailable; events dropped\nnormal message"
    update = {"amdahl_style_optimistic_ceilings": {"source": "untraced"}}
    result = summarize(trace, {
        "compile_or_get_cached_calls": 0, "backend_compile_calls": 0
    }, run_log, update)
    assert result["status"] == "partial_profiler_warning"
    assert result["amdahl_style_optimistic_ceilings_from_untraced_runner"] == {"source": "untraced"}
    assert len(result["profiler_warnings_from_run_log"]) == 1
    assert result["compile_free_confirmed"] is True
    assert abs(result["conditional_wall_over_gpu_active_union"] - 10.0) < 1e-12
    assert "not a hardware bound" in result["gpu_union_limitation"]
    assert "Amdahl speedup ceiling" in result["gpu_union_limitation"]


def test_trace_summary_rejects_missing_or_ambiguous_root_annotation():
    assert summarize({"traceEvents": []})["status"] == "invalid_update_annotation_count"
    trace = {"traceEvents": [_event("full_update", 0, 1, pid=1),
                              _event("full_update", 2, 1, pid=1)]}
    assert summarize(trace)["status"] == "invalid_update_annotation_count"


def test_trace_summary_does_not_claim_compile_free_without_both_counters():
    trace = {"traceEvents": [_event("full_update", 0, 100, pid=1)]}
    assert not summarize(trace, {"compile_or_get_cached_calls": 0})["compile_free_confirmed"]


def test_trace_summary_reports_call_annotation_overlap_in_seconds():
    trace = {"traceEvents": [
        _event("full_update", 0, 1000, pid=1),
        _event("call_T7", 100, 400, pid=1),
        _event("call_T30", 300, 400, pid=1),
    ]}
    result = summarize(trace)
    assert abs(result["call_annotation_overlap_seconds"] - 0.0002) < 1e-12


def test_trace_summary_marks_missing_gpu_activity_as_incomplete_not_host_time():
    trace = {"traceEvents": [_event("full_update", 0, 1000, pid=1)]}
    result = summarize(trace)
    assert result["status"] == "no_gpu_events"
    assert result["gpu_trace_coverage_status"] == "missing_gpu_process_metadata"
    assert result["host_gap_seconds_relative_to_gpu_active_union"] is None
    assert result["conditional_wall_over_gpu_active_union"] is None

    trace["traceEvents"].extend([
        {"name": "process_name", "ph": "M", "pid": 2,
         "args": {"name": "/device:GPU:0"}},
        _event("before_update", 1001, 20),
    ])
    result = summarize(trace)
    assert result["status"] == "no_gpu_events"
    assert result["gpu_trace_coverage_status"] == "no_gpu_intervals_clipped_to_update"
    assert result["host_gap_seconds_relative_to_gpu_active_union"] is None
