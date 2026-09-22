from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.hpc import probe_shared_compile_cache as probe


class _Value:
    def __init__(self, value):
        self.value = value


class _IgnoreCallbacks:
    NO = SimpleNamespace(name="NO")
    CUSTOM_PARTITIONING = SimpleNamespace(name="CUSTOM_PARTITIONING")


def _fixture(tmp_path: Path, *, cache_hit: bool = False):
    backend_calls = []
    backend = SimpleNamespace(platform="cpu", platform_version="test")
    devices = np.asarray([SimpleNamespace(__str__=lambda self: "cpu:0")], dtype=object)

    def backend_compile(*args, **kwargs):
        backend_calls.append((args, kwargs))
        return "compiled"

    compiler = SimpleNamespace(backend_compile=backend_compile)

    def compile_or_get_cached(*args, **kwargs):
        if cache_hit:
            return "cached"
        return compiler.backend_compile(args[0], args[1], args[3], args[4])

    compiler.compile_or_get_cached = compile_or_get_cached
    cache_file = tmp_path / "toy-key-cache"
    if cache_hit:
        cache_file.write_bytes(b"cached")
    compilation_cache = SimpleNamespace(
        is_cache_used=lambda unused: True,
        get_cache_key=lambda *args, **kwargs: "toy-key",
    )
    cache_key = SimpleNamespace(
        __name__="fake.cache_key",
        IgnoreCallbacks=_IgnoreCallbacks,
        custom_hook=lambda: "",
    )
    config = SimpleNamespace(
        remove_custom_partitioning_ptr_from_cache_key=_Value(False),
        compilation_cache_dir=_Value(str(tmp_path)),
        compilation_cache_include_metadata_in_key=_Value(False),
    )
    internals = SimpleNamespace(
        compiler=compiler,
        compilation_cache=compilation_cache,
        cache_key=cache_key,
        config=config,
    )

    class Lowered:
        def compile(self):
            return compiler.compile_or_get_cached(
                backend, object(), devices, object(), (), None
            )

    return internals, Lowered(), backend_calls


def _report():
    return {
        "status": "running",
        "counters": {
            "compile_or_get_cached_calls": 0,
            "backend_compile_calls": 0,
            "persistent_cache_hits": 0,
        },
        "timing_seconds": {},
    }


def test_key_only_never_calls_backend_and_restores_hooks(tmp_path):
    internals, lowered, backend_calls = _fixture(tmp_path)
    original_compile = internals.compiler.compile_or_get_cached
    original_backend = internals.compiler.backend_compile
    report = _report()

    with pytest.raises(probe.DiagnosticComplete):
        probe.diagnose_lowered(
            lowered=lowered,
            mode="key-only",
            report=report,
            emit=lambda stage: None,
            internals=internals,
        )

    assert not backend_calls
    assert report["status"] == "key_observed"
    assert report["cache"]["key"] == "toy-key"
    assert report["cache"]["file_exists"] is False
    assert internals.compiler.compile_or_get_cached is original_compile
    assert internals.compiler.backend_compile is original_backend


def test_cached_only_cold_cache_rejects_backend_and_restores_hooks(tmp_path):
    internals, lowered, backend_calls = _fixture(tmp_path)
    original_compile = internals.compiler.compile_or_get_cached
    original_backend = internals.compiler.backend_compile
    report = _report()

    with pytest.raises(probe.DiagnosticComplete):
        probe.diagnose_lowered(
            lowered=lowered,
            mode="cached-only",
            report=report,
            emit=lambda stage: None,
            internals=internals,
        )

    assert not backend_calls
    assert report["status"] == "cache_miss_backend_compile_rejected"
    assert report["counters"] == {
        "compile_or_get_cached_calls": 1,
        "backend_compile_calls": 1,
        "persistent_cache_hits": 0,
    }
    assert internals.compiler.compile_or_get_cached is original_compile
    assert internals.compiler.backend_compile is original_backend


def test_cached_only_hit_is_reported_without_backend_compile(tmp_path):
    internals, lowered, backend_calls = _fixture(tmp_path, cache_hit=True)
    report = _report()

    with pytest.raises(probe.DiagnosticComplete):
        probe.diagnose_lowered(
            lowered=lowered,
            mode="cached-only",
            report=report,
            emit=lambda stage: None,
            internals=internals,
        )

    assert not backend_calls
    assert report["status"] == "cached_executable_loaded"
    assert report["cache"]["file_readable"] is True
    assert report["counters"]["persistent_cache_hits"] == 1


def test_version_guard_is_explicit():
    with pytest.raises(RuntimeError, match="requires JAX 0.4.38"):
        probe._require_supported_jax(SimpleNamespace(__version__="0.4.39"))


def test_self_test_parser_needs_only_output(tmp_path):
    argument_parser = probe.parser()
    args = argument_parser.parse_args(["--self-test", "--output", str(tmp_path / "out.json")])
    probe._validate_args(args, argument_parser)


def test_real_parser_rejects_missing_probe_inputs(tmp_path):
    argument_parser = probe.parser()
    args = argument_parser.parse_args(["--output", str(tmp_path / "out.json")])
    with pytest.raises(SystemExit):
        probe._validate_args(args, argument_parser)
