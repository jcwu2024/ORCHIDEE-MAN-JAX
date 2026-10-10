"""Native observations remain read-only and preserve fail-closed interfaces."""
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.hpc import probe_shared_union_cache_preparation_cpu as preparation
from scripts.hpc import shared_native_cache_evidence as evidence


def test_legacy_signature_is_explicitly_incomplete_and_ordered_change_fails():
    old = {'argument_leaves': 2, 'argument_nominal_bytes': 8,
           'result_leaves': 1, 'result_nominal_bytes': 4}
    current = old | {'argument_types': ['tensor<f32>', 'tensor<i32>'], 'result_types': ['tensor<f32>']}
    result = evidence.compare_recorded_signatures(old, current)
    assert result['ordered_prior_types'] == 'unavailable_in_legacy_report'
    assert result['cache_compatibility'] == 'requires_actual_native_cache_hit'
    with pytest.raises(ValueError, match='ordered interface'):
        evidence.compare_recorded_signatures(current, current | {'argument_types': current['argument_types'][::-1]})
    with pytest.raises(ValueError, match='signature changed'):
        evidence.compare_recorded_signatures(old, current | {'result_nominal_bytes': 8})


def test_native_observer_records_native_logs_without_calling_hook(monkeypatch):
    from jax._src import cache_key
    hook_calls = []
    logger = cache_key.logger
    def custom_hook():
        hook_calls.append(1)
        return 'actual-hook'
    def native_get(module, devices, compile_options, backend, compression_algorithm='zlib', ignore_callbacks=None):
        # This stands in for the two native DEBUG hash calls, not observer calls.
        custom_hook()
        custom_hook()
        logger.debug('get_cache_key hash of serialized %s: %s', 'custom_hook', 'native-component')
        logger.debug('get_cache_key hash after serializing %s: %s', 'custom_hook', 'native-cumulative')
        return 'native-key'
    monkeypatch.setattr(cache_key, 'custom_hook', custom_hook)
    monkeypatch.setattr(cache_key, 'get', native_get)
    old_level, old_propagate = logger.level, logger.propagate
    event = {}
    backend = SimpleNamespace(platform='cpu', platform_version='test', runtime_type='test')
    options = SimpleNamespace(SerializeAsString=lambda: b'actual-options')
    with evidence.observe_native_cache_keys(lambda: event):
        assert cache_key.get(None, np.array([], dtype=object), options, backend) == 'native-key'
    assert cache_key.get is native_get
    assert (logger.level, logger.propagate) == (old_level, old_propagate)
    assert hook_calls == [1, 1]
    entry = event['native_cache_key_calls'][0]
    assert entry['component_sha256'] == {'custom_hook': 'native-component'}
    assert entry['cumulative_sha256'] == {'custom_hook': 'native-cumulative'}
    serialized = entry['inputs']['raw_serialized_compile_options']
    assert serialized['sha256'] == hashlib.sha256(b'actual-options').hexdigest()
    assert serialized['scope'] == 'raw_actual_options_not_native_normalized_component'


def test_model_preparation_guard_forbids_lookup_and_backend(monkeypatch):
    from jax._src import compiler
    real_calls = []
    monkeypatch.setattr(compiler, 'compile_or_get_cached', lambda *a, **kw: real_calls.append('lookup'))
    monkeypatch.setattr(compiler, 'backend_compile', lambda *a, **kw: real_calls.append('backend'), raising=False)
    report = {}
    with preparation.forbid_model_compilation(report):
        with pytest.raises(RuntimeError, match='lookup is forbidden'):
            compiler.compile_or_get_cached()
        with pytest.raises(RuntimeError, match='backend compilation is forbidden'):
            compiler.backend_compile()
    assert real_calls == []
    assert report == {'model_compile_or_get_cached_calls': 1, 'model_backend_compile_calls': 1}


def test_native_get_failure_is_recorded_and_observer_restored(monkeypatch):
    from jax._src import cache_key
    def failed_get(module, devices, compile_options, backend, compression_algorithm='zlib', ignore_callbacks=None):
        raise RuntimeError('actual native get failed')
    monkeypatch.setattr(cache_key, 'get', failed_get)
    event = {}
    with evidence.observe_native_cache_keys(lambda: event):
        with pytest.raises(RuntimeError, match='actual native get failed'):
            cache_key.get(None, np.array([], dtype=object),
                          SimpleNamespace(SerializeAsString=lambda: b'options'), SimpleNamespace(platform='cpu'))
    assert cache_key.get is failed_get
    call = event['native_cache_key_calls'][0]
    assert call['status'] == 'native_get_failed' and call['error_type'] == 'RuntimeError'
    assert 'key' not in call


def test_six_stage_extraction_requires_exact_order():
    names = ('neural_stage', 'transition_stage', 'loss_stage', 'loss_stage', 'transition_stage', 'neural_stage')
    equations = [SimpleNamespace(primitive=SimpleNamespace(name='pjit'), params={'name': name}) for name in names]
    closed = SimpleNamespace(jaxpr=SimpleNamespace(eqns=equations))
    assert preparation.stage_equations(closed) == equations
    closed.jaxpr.eqns = [equations[1], equations[0], *equations[2:]]
    with pytest.raises(ValueError, match='expected six ordered'):
        preparation.stage_equations(closed)
