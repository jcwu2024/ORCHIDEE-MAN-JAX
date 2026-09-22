from types import SimpleNamespace

import pytest

from scripts.hpc import probe_shared_execution as probe


class _Jax:
    @staticmethod
    def device_get(value):
        return value


def _identity(value=1):
    return {
        "schema_version": "shared_execution_state_identity_v1",
        "source_checkpoint": "checkpoint.pkl",
        "source_update": 100,
        "artifact_id": 742,
        "configuration": {"value": value},
        "jax_version": "0.4.38",
        "method": {"name": "shared"},
    }


def test_partial_reference_state_resumes_at_first_missing_component(tmp_path):
    identity = _identity()
    probe.initialize_state_store(tmp_path, _Jax(), identity)
    probe._save_stage_record(tmp_path, "daily", identity, {"value": 1.0}, _Jax())

    assert probe.pending_reference_components(tmp_path, identity) == ("T7", "T30")
    assert probe._load_stage_record(tmp_path, "daily", identity, required=True) == {
        "value": 1.0
    }


def test_state_store_refuses_mismatched_reference_identity(tmp_path):
    probe.initialize_state_store(tmp_path, _Jax(), _identity(1))

    with pytest.raises(ValueError, match="identity differs"):
        probe.initialize_state_store(tmp_path, _Jax(), _identity(2))


def test_cache_only_miss_rejects_backend_compile_and_restores_hooks():
    backend_calls = []

    def backend_compile(*args, **kwargs):
        backend_calls.append((args, kwargs))
        return "compiled"

    compiler = SimpleNamespace(backend_compile=backend_compile)

    def compile_or_get_cached():
        return compiler.backend_compile()

    compiler.compile_or_get_cached = compile_or_get_cached
    original_boundary = compiler.compile_or_get_cached
    original_backend = compiler.backend_compile

    class Lowered:
        @staticmethod
        def compile():
            return compiler.compile_or_get_cached()

    executable, evidence = probe.compile_cached_lowered(Lowered(), compiler)

    assert executable is None
    assert evidence["backend_compile_calls"] == 1
    assert not backend_calls
    assert compiler.compile_or_get_cached is original_boundary
    assert compiler.backend_compile is original_backend


def test_cache_only_hit_is_distinguished_and_restores_hooks():
    compiler = SimpleNamespace(
        backend_compile=lambda: pytest.fail("cache hit called backend compile")
    )
    compiler.compile_or_get_cached = lambda: "cached"
    original_boundary = compiler.compile_or_get_cached
    original_backend = compiler.backend_compile

    class Lowered:
        @staticmethod
        def compile():
            return compiler.compile_or_get_cached()

    executable, evidence = probe.compile_cached_lowered(Lowered(), compiler)

    assert executable == "cached"
    assert evidence == {
        "compile_or_get_cached_calls": 1,
        "backend_compile_calls": 0,
        "persistent_cache_hit": True,
        "in_process_hit": False,
    }
    assert compiler.compile_or_get_cached is original_boundary
    assert compiler.backend_compile is original_backend


@pytest.mark.parametrize(
    ("stage", "expected"),
    (("reference", ()), ("micro2", (2,)), ("micro4", (4,)), ("all", (2, 4))),
)
def test_selected_size_dispatch(stage, expected):
    assert probe.selected_microbatch_sizes(stage, {"microbatch_sizes": [2, 4]}) == expected


def test_stage_record_refuses_mismatched_embedded_identity(tmp_path):
    probe.initialize_state_store(tmp_path, _Jax(), _identity())
    probe._save_stage_record(tmp_path, "daily", _identity(2), {}, _Jax())

    with pytest.raises(ValueError, match="identity or schema differs"):
        probe._load_stage_record(tmp_path, "daily", _identity(), required=True)
