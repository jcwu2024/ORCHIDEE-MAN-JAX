from types import SimpleNamespace

import pytest

from scripts.hpc.probe_shared_split_timing_cache import observe_compiler


@pytest.mark.parametrize("component,hit,allowed", [("daily", True, True), ("T7", False, False), ("T30", False, True)])
def test_cache_recovery_preserves_result_and_restores_compiler(component, hit, allowed):
    calls = []
    compiler = SimpleNamespace()

    def backend(*_args):
        calls.append("backend")
        return "executable"

    def lookup(*args):
        return "executable" if hit else compiler.backend_compile(*args)

    compiler.backend_compile = backend
    compiler.compile_or_get_cached = lookup
    module = SimpleNamespace(operation=SimpleNamespace(attributes={"sym_name": '"jit_checkpointed_loss_with_aux"'}))
    events = []
    if allowed:
        with observe_compiler(compiler, lambda: component, events.append):
            assert compiler.compile_or_get_cached(None, module) == "executable"
        assert events[-1]["status"] == ("cache_hit" if hit else "backend_compiled")
    else:
        with pytest.raises(RuntimeError, match="stopped before backend"):
            with observe_compiler(compiler, lambda: component, events.append):
                compiler.compile_or_get_cached(None, module)
        assert events[-1]["status"] == "completed_graph_cache_miss"
    assert calls == ([] if hit or not allowed else ["backend"])
    assert compiler.backend_compile is backend
    assert compiler.compile_or_get_cached is lookup


def test_completed_t30_can_be_required_cached_before_profiling():
    compiler = SimpleNamespace()
    backend_calls = []
    compiler.backend_compile = lambda *_args: backend_calls.append(True)
    compiler.compile_or_get_cached = lambda *args: compiler.backend_compile(*args)
    module = SimpleNamespace(operation=SimpleNamespace(attributes={"sym_name": '"jit_checkpointed_loss_with_aux"'}))
    events = []
    with pytest.raises(RuntimeError, match="stopped before backend"):
        with observe_compiler(compiler, lambda: "T30", events.append, required_cached=("daily", "T7", "T30")):
            compiler.compile_or_get_cached(None, module)
    assert not backend_calls
    assert events[-1]["component"] == "T30"
