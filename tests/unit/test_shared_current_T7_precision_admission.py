"""Bounded runner behavior; original scientific assets are never mutated."""
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from scripts.hpc import probe_shared_current_T7_precision_admission as probe


def raw(*, nan_ad=False, hard=0, width=3854):
    gradient = np.ones(width, dtype=np.float64)
    if nan_ad:
        gradient[10] = np.nan
    return ((np.float64(3), {'hard': {'physical': np.int32(hard)}, 'flux': np.float64(2)}), gradient, np.float64(1))


def test_forward_finite_ad_nan_is_separate():
    gates = probe.result_gates(jax, raw(nan_ad=True))
    assert gates['forward_passed'] and not gates['passed']
    assert gates['nonfinite']


def test_physics_failure_and_state_abi_fail_independently():
    assert not probe.result_gates(jax, raw(hard=1))['forward_passed']
    gates = probe.result_gates(jax, raw(width=3))
    assert gates['forward_passed'] and not gates['state_vjp_abi'] and not gates['passed']


def test_strict_nonfinite_json_raw_preserved(tmp_path):
    output = tmp_path / 'evidence.json'
    source = raw(nan_ad=True)
    receipt = probe.save_raw(tmp_path / 'raw.pkl', source)
    probe.save_json(output, {'raw': receipt, 'value': np.nan, 'samples': np.array([1, np.inf])})
    parsed = json.loads(output.read_text(), parse_constant=lambda token: pytest.fail(token))
    assert parsed['value']['finite_value'] is None
    import pickle
    with (tmp_path / 'raw.pkl').open('rb') as f:
        assert np.isnan(pickle.load(f)[1][10])
    with pytest.raises(FileExistsError):
        probe.save_raw(tmp_path / 'raw.pkl', source)


def test_unit_direction_native_scale():
    for coordinate in probe.COORDINATES:
        direction = probe.unit_direction(coordinate)
        assert direction.shape == (3854,) and direction.dtype == np.float64
        assert direction[coordinate] == 1 and np.count_nonzero(direction) == 1
    with pytest.raises(ValueError):
        probe.unit_direction(3854)


def test_save_before_gate_exception(tmp_path, monkeypatch):
    class Journal:
        report = {'results': []}
        def emit(self, *args, **kwargs):
            pass
    def bad_gate(*args):
        raise RuntimeError('gate failure fixture')
    monkeypatch.setattr(probe, 'result_gates', bad_gate)
    journal = Journal()
    with pytest.raises(RuntimeError):
        probe.run_calls(jax, lambda *args: raw(),
            (None, np.zeros(3854), None, None, None, None), {}, tmp_path/'result.json', journal)
    assert (tmp_path/'result.native.0.434.raw.pkl').is_file()
    assert journal.report['full_model_calls'] == 1 and journal.report['full_model_call_attempts'] == 1


def test_fixed_tolerance_full_state_failure():
    reference = np.ones(3854, dtype=np.float64)
    actual = reference.copy()
    actual[2621] += 0.1
    result = probe.compare_arrays(actual, reference)
    assert not result['passed'] and result['failed_indices'] == [[2621]]
    assert result['rtol'] == 1e-5 and result['atol'] == 1e-6


def test_complete_public_raw_repeat_includes_aux_and_state():
    first = raw()
    second = raw()
    second[0][1]['flux'] = np.float64(3)
    assert not probe.compare_result_tree(jax, first[:2], second[:2])['passed']


def test_masks_and_large_integers_compare_exact():
    assert probe.compare_arrays(np.array([True]), np.array([True]))['passed']
    assert not probe.compare_arrays(np.array([True]), np.array([False]))['passed']
    assert not probe.compare_arrays(np.array([1000001]), np.array([1000000]))['passed']
    first = raw()
    first[0][1]['mask'] = np.array([True, False])
    second = raw()
    second[0][1]['mask'] = np.array([True, True])
    assert not probe.compare_result_tree(jax, first[:2], second[:2])['passed']


def test_memory_estimate_is_gate_and_not_sampled_peak():
    from types import SimpleNamespace
    class Executable:
        def memory_analysis(self):
            return SimpleNamespace(argument_size_in_bytes=1, output_size_in_bytes=1,
                temp_size_in_bytes=25*2**30, alias_size_in_bytes=0)
    gate = probe.compiled_memory_gate(Executable())
    assert not gate['under_24_GiB'] and gate['estimate_is_not_sampled_runtime_peak']
    with pytest.raises(RuntimeError):
        probe.compiled_memory_gate(SimpleNamespace(memory_analysis=lambda: None))


def test_direct_execute_rejected_before_gpu_bootstrap(tmp_path, monkeypatch):
    arguments = ['entry', '--stage', 'execute']
    for name in ('parameters', 'prepared-context', 'training-prepared-manifest', 'dataset-manifest',
                 'saved-identity-receipt', 'state-input-metadata', 'turnover-overlay', 'output'):
        arguments += ['--'+name, str(tmp_path/name)]
    monkeypatch.setattr(probe.sys, 'argv', arguments)
    monkeypatch.setattr(probe, 'check_approval', lambda args: {'approved': True})
    monkeypatch.delenv('ORCJAX_CURRENT_T7_SUPERVISED', raising=False)
    with pytest.raises(PermissionError, match='before GPU bootstrap'):
        probe.main()
