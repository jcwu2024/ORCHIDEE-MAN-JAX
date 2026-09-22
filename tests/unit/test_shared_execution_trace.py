import json

from scripts.hpc.analyze_shared_execution_trace import hlo_gemm_shapes, hlo_sources, summarize


def test_trace_excludes_host_nesting_and_unions_overlapping_streams():
    trace = {"traceEvents": [
        {"ph": "M", "pid": 1, "name": "process_name", "args": {"name": "/device:GPU:0"}},
        {"ph": "X", "pid": 700, "ts": 0, "dur": 10000, "name": "host"},
        {"ph": "X", "pid": 1, "ts": 0, "dur": 10, "name": "volta_sgemm"},
        {"ph": "X", "pid": 1, "ts": 5, "dur": 10, "name": "fusion"},
        {"ph": "X", "pid": 1, "ts": 20, "dur": 5, "name": "MemcpyD2D"},
    ]}
    result = summarize(trace)
    assert result["event_count"] == 3
    assert round(result["summed_device_event_seconds"]*1e6) == 25
    assert round(result["device_active_union_seconds"]*1e6) == 20
    assert round(result["device_event_span_seconds"]*1e6) == 25
    assert round(result["gaps_within_device_span_seconds"]*1e6) == 5
    assert result["gemm_precision_by_explicit_kernel_name"][0]["name"] == "explicit_fp32"


def test_ambiguous_hlo_names_are_not_assigned_to_arbitrary_source():
    text = '''
      %fusion.1 = f32[] fusion(), metadata={source_file="one.py" source_line=3}
      %fusion.1 = f32[] fusion(), metadata={source_file="two.py" source_line=4}
      ROOT %fusion.2 = f32[] fusion(), metadata={source_file="two.py" source_line=8}
    '''
    assert hlo_sources(text) == {"fusion.2": ("two.py", 8)}


def test_missing_gpu_trace_is_not_reported_as_zero_cost():
    assert summarize({"traceEvents": []})["status"] == "no_gpu_events"


def test_hlo_gemm_dimensions_include_batch_and_contraction_axes():
    config = {"gemm_backend_config": {"dot_dimension_numbers": {
        "lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"],
        "lhs_batch_dimensions": ["0"], "rhs_batch_dimensions": ["0"]}}}
    text = (' %gemm = f32[2,3,5] custom-call(f32[2,3,4] %a, f32[2,4,5] %b), '
            'custom_call_target="__cublas$gemm", backend_config='+json.dumps(config))
    assert hlo_gemm_shapes(text)["gemm"] == {
        "operand_dtype": "f32", "batch": 2, "m": 3, "n": 5, "k": 4, "mac_per_hlo_call": 120}
