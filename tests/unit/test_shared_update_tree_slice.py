from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from scripts.hpc import probe_shared_update_tree_slice as probe
from scripts.hpc.probe_shared_update_tree_slice import interleaved_orders


def test_full_proposal_schedule_has_three_interleaved_matched_pairs():
    assert interleaved_orders() == (
        ("leaf", "tree"),
        ("tree", "leaf"),
        ("leaf", "tree"),
    )


def test_full_proposal_schedule_rejects_underpowered_comparison():
    with pytest.raises(ValueError, match="three matched full-proposal pairs"):
        interleaved_orders(2)


def test_profiled_update_requires_parity_and_zero_compilation_boundaries():
    assert probe.profiled_update_admitted(True, {
        "compile_or_get_cached_calls": 0, "backend_compile_calls": 0
    })
    assert not probe.profiled_update_admitted(False, {
        "compile_or_get_cached_calls": 0, "backend_compile_calls": 0
    })
    assert not probe.profiled_update_admitted(True, {
        "compile_or_get_cached_calls": 1, "backend_compile_calls": 0
    })
    assert not probe.profiled_update_admitted(True, {
        "compile_or_get_cached_calls": 0, "backend_compile_calls": 1
    })


def test_compile_guard_rejects_before_calling_either_compiler_entry():
    calls = []
    compiler = SimpleNamespace(
        compile_or_get_cached=lambda *args: calls.append("lookup"),
        backend_compile=lambda *args: calls.append("backend"),
    )
    original_lookup = compiler.compile_or_get_cached
    original_backend = compiler.backend_compile
    counters = {"compile_or_get_cached_calls": 0, "backend_compile_calls": 0}
    restore = probe.reject_compilation_before_entry(compiler, counters)
    try:
        with pytest.raises(probe._CompilationAttempt, match="compile_or_get_cached"):
            compiler.compile_or_get_cached(None)
        with pytest.raises(probe._CompilationAttempt, match="backend_compile"):
            compiler.backend_compile(None)
    finally:
        restore()
    assert calls == []
    assert counters == {"compile_or_get_cached_calls": 1, "backend_compile_calls": 1}
    assert compiler.compile_or_get_cached is original_lookup
    assert compiler.backend_compile is original_backend


def test_adamw_annotation_wraps_the_separate_update_callable_not_operations_mapping():
    events = []

    @contextmanager
    def annotation(name):
        events.append(("begin", name))
        try:
            yield
        finally:
            events.append(("end", name))

    operations = {"finite": object(), "norm": object()}
    update_calls = []

    def update(parameters, gradient, optimizer):
        update_calls.append((parameters, gradient, optimizer))
        return "new-parameters", "new-optimizer", 2.0, 0.5

    wrapped = probe._annotated_call(update, annotation, "call_adamw")
    result = wrapped("parameters", "gradient", "optimizer")
    assert result == ("new-parameters", "new-optimizer", 2.0, 0.5)
    assert update_calls == [("parameters", "gradient", "optimizer")]
    assert "update" not in operations
    assert events == [("begin", "call_adamw"), ("end", "call_adamw")]


def test_synchronized_evaluator_summary_requires_expected_call_counts_and_reports_residual():
    values = {
        "daily32_micro4": [0.1] * 8,
        "T7": [0.2],
        "T30": [0.3],
    }
    report = probe.synchronized_evaluator_summary(2.0, values)
    assert report["synchronized_evaluator_sum_seconds"] == 1.3
    assert report["outside_evaluator_wall_seconds"] == 0.7
    with pytest.raises(ValueError, match="exactly eight daily micro4 calls"):
        probe.synchronized_evaluator_summary(2.0, {**values, "daily32_micro4": [0.1]})


def test_overlay_loader_registers_module_before_dataclass_definition(tmp_path, monkeypatch):
    name = "scripts.hpc._test_update_slice_overlay"
    monkeypatch.setitem(sys.modules, name, None)
    source = tmp_path / "overlay.py"
    source.write_text(
        "from dataclasses import dataclass\n@dataclass\nclass Payload:\n    value: int\n",
        encoding="ascii",
    )
    module = probe._load_overlay_module(name, source)
    assert module.Payload(7).value == 7
    assert sys.modules[name] is module


def test_overlay_loader_restores_previous_module_on_failure(tmp_path, monkeypatch):
    name = "scripts.hpc._test_update_slice_overlay"
    previous = SimpleNamespace(__file__=str(tmp_path / "old.py"))
    monkeypatch.setitem(sys.modules, name, previous)
    source = tmp_path / "broken.py"
    source.write_text("raise RuntimeError('failed overlay')\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="failed overlay"):
        probe._load_overlay_module(name, source)
    assert sys.modules[name] is previous


def test_gpu_initialization_precedes_overlay_imports(monkeypatch):
    events = []
    monkeypatch.setitem(sys.modules, "scripts.hpc.orcjax_gpu_runtime", SimpleNamespace(
        initialize_gpu_backend=lambda: events.append("gpu_initialized")
    ))
    monkeypatch.setattr(probe, "_load_overlay_module", lambda name, path: events.append(name))
    args = SimpleNamespace(inspect_only=False, slice_helper_file="slice.py", support_owner_file="owner.py")
    probe._prepare_runtime(args)
    assert events == [
        "gpu_initialized", "scripts.hpc.shared_daily_microbatch",
        "scripts.hpc.probe_shared_serialized_dual",
    ]


def test_cpu_inspection_does_not_initialize_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "scripts.hpc.orcjax_gpu_runtime", SimpleNamespace(
        initialize_gpu_backend=lambda: pytest.fail("CPU inspection initialized GPU")
    ))
    loaded = []
    monkeypatch.setattr(probe, "_load_overlay_module", lambda name, path: loaded.append(name))
    probe._prepare_runtime(SimpleNamespace(
        inspect_only=True, slice_helper_file="slice.py", support_owner_file="owner.py"
    ))
    assert loaded == [
        "scripts.hpc.shared_daily_microbatch", "scripts.hpc.probe_shared_serialized_dual"
    ]


def test_profile_trace_is_explicit_opt_in():
    args = probe.parser().parse_args([
        "--source-profile", "profile.json", "--input-bundle", "inputs.pkl",
        "--slice-helper-file", "slice.py", "--support-owner-file", "owner.py",
        "--daily-admission-report", "daily.json", "--seven-admission-report", "t7.json",
        "--thirty-admission-report", "t30.json", "--output", "result.json",
    ])
    assert args.profile_trace_dir is None
    assert args.synchronize_evaluators is False
    args = probe.parser().parse_args([
        "--source-profile", "profile.json", "--input-bundle", "inputs.pkl",
        "--slice-helper-file", "slice.py", "--support-owner-file", "owner.py",
        "--daily-admission-report", "daily.json", "--seven-admission-report", "t7.json",
        "--thirty-admission-report", "t30.json", "--output", "result.json",
        "--profile-trace-dir", "trace",
    ])
    assert str(args.profile_trace_dir) == "trace"
