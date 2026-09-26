"""Behavioral checks for reusable helpers detached from experiment runners."""
from __future__ import annotations

import ast
import base64
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.hpc import canonical_training_runtime as canonical
from scripts.hpc import retained_training_runtime as retained
from scripts.hpc.runtime_diagnostics import _tree_numeric_report
from scripts.hpc.training_artifact_io import _atomic_json, _json_value, _sha256, _verify


def test_atomic_json_preserves_numpy_values_and_replaces_existing_file(tmp_path):
    path = tmp_path / 'nested' / 'report.json'
    payload = {'scalar': np.int64(4), 'arrays': (np.array([1.0, 2.0]),), 7: np.bool_(True)}
    _atomic_json(path, {'old': True})
    _atomic_json(path, payload)
    assert json.loads(path.read_text()) == {'scalar': 4, 'arrays': [[1.0, 2.0]], '7': True}
    assert path.read_text().endswith('\n')
    assert tuple(path.parent.iterdir()) == (path,)
    assert _json_value(None) is None


def test_artifact_hash_verification(tmp_path):
    path = tmp_path / 'artifact.bin'
    path.write_bytes(b'canonical artifact')
    expected = hashlib.sha256(b'canonical artifact').hexdigest()
    assert _sha256(path) == expected
    _verify(path, expected, 'artifact')
    with pytest.raises(ValueError, match='artifact hash drift'):
        _verify(path, 'wrong', 'artifact')


def _plan_manifest(tmp_path):
    for name in ('config', 'run_def', 'reference'):
        (tmp_path / name).touch()
    plan = {
        'teacher_config': str(tmp_path / 'config'),
        'entries': [{'landpoint_id': canonical.FROZEN_LANDPOINT, 'year': 1961,
                     'run_def': str(tmp_path / 'run_def'),
                     'reference_run_dir': str(tmp_path / 'reference')}],
    }
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    return {'plan': str(path), 'plan_sha256': retained._canonical_sha256(plan)}


def test_plan_context_preserves_frozen_and_general_resolution(tmp_path):
    manifest = _plan_manifest(tmp_path)
    general = retained._plan_context(manifest, landpoint_id=canonical.FROZEN_LANDPOINT, year=1961)
    assert general == canonical._plan_context(manifest, canonical.FROZEN_LANDPOINT)
    assert general['config'] == (tmp_path / 'config').resolve()
    with pytest.raises(ValueError, match='hash drift'):
        retained._plan_context(manifest | {'plan_sha256': 'wrong'}, landpoint_id=canonical.FROZEN_LANDPOINT, year=1961)
    with pytest.raises(ValueError, match='unique generation-plan entry'):
        retained._plan_context(manifest, landpoint_id=canonical.FROZEN_LANDPOINT, year=1962)


def test_numeric_tree_report_preserves_empty_and_nonfinite_checks():
    jax = SimpleNamespace(tree_util=SimpleNamespace(tree_leaves=lambda value: value))
    report = _tree_numeric_report(jax, [np.array([3.0, 4.0]), np.zeros(2), np.array([])])
    assert report == {'all_finite': True, 'element_count': 4, 'leaf_count': 3,
                      'nonzero_leaf_count': 1, 'l2_norm': 5.0, 'maximum_absolute_value': 4.0}
    assert not _tree_numeric_report(jax, [np.array([np.nan])])['all_finite']


def test_prepared_artifact_pickle_keeps_original_class_modules():
    from scripts.hpc.target_oriented_training_prepared import PreparedTargetOrientedUpdateArtifact
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        PreparedTargetOrientedOneDayBatch, PreparedTargetOrientedWindowBatch,
        PreparedTargetOrientedTrainingBatch,
    )

    one_day = PreparedTargetOrientedOneDayBatch(*(None for _ in range(6)))
    window = PreparedTargetOrientedWindowBatch(*(None for _ in range(7)))
    batch = PreparedTargetOrientedTrainingBatch(one_day, window, window)
    artifact = PreparedTargetOrientedUpdateArtifact('prepared', 'contract', {}, 'trace', batch)
    restored = pickle.loads(pickle.dumps(artifact))
    assert type(restored) is PreparedTargetOrientedUpdateArtifact
    assert type(restored.batch) is PreparedTargetOrientedTrainingBatch
    assert type(restored.batch.one_day) is PreparedTargetOrientedOneDayBatch
    assert type(restored.batch.seven_day) is PreparedTargetOrientedWindowBatch
    assert PreparedTargetOrientedUpdateArtifact.__module__ == 'scripts.hpc.target_oriented_training_prepared'


def test_baseline_created_prepared_pickle_loads_with_stored_signature():
    from scripts.hpc.target_oriented_training_prepared import PreparedTargetOrientedUpdateArtifact
    from research.daily_coarse_graining.daily_operator.target_oriented_training import PreparedTargetOrientedTrainingBatch

    # Protocol-4 fixture generated from recovery-tag class definitions, not current classes.
    fixture = (
        "gASVcAEAAAAAAACMLXNjcmlwdHMuaHBjLnRhcmdldF9vcmllbnRlZF90cmFpbmluZ19wcmVwYXJlZJSMJFByZXBh"
        "cmVkVGFyZ2V0T3JpZW50ZWRVcGRhdGVBcnRpZmFjdJSTlCiMCHByZXBhcmVklIwIY29udHJhY3SUfZSMBnVwZGF0"
        "ZZRLC3OMDHN0b3JlZC10cmFjZZSMRnJlc2VhcmNoLmRhaWx5X2NvYXJzZV9ncmFpbmluZy5kYWlseV9vcGVyYXRv"
        "ci50YXJnZXRfb3JpZW50ZWRfdHJhaW5pbmeUjCNQcmVwYXJlZFRhcmdldE9yaWVudGVkVHJhaW5pbmdCYXRjaJST"
        "lGgIjCFQcmVwYXJlZFRhcmdldE9yaWVudGVkT25lRGF5QmF0Y2iUk5QoTk5OTk5OdJSBlGgIjCFQcmVwYXJlZFRh"
        "cmdldE9yaWVudGVkV2luZG93QmF0Y2iUk5QoTk5OTk5OTnSUgZRoEoeUgZR0lIGULg=="
    )
    artifact = pickle.loads(base64.b64decode(fixture))
    assert type(artifact) is PreparedTargetOrientedUpdateArtifact
    assert type(artifact.batch) is PreparedTargetOrientedTrainingBatch
    assert artifact.schedule_entry == {'update': 11}
    assert artifact.trace_signature == 'stored-trace'


def test_packet_keeps_state_reconstruction_and_provenance():
    seen = {}

    def reconstruct(state, discrete, contract):
        seen.update(state=state, discrete=discrete, contract=contract)
        return {'driver': {'value': 4}}

    teacher = SimpleNamespace(DriverPreviousStepStatePacket=lambda **kwargs: kwargs)
    packet = canonical._packet(teacher, reconstruct, {'state': [2], 'discrete_state': 'discrete', 'contract': 'contract'}, 47)
    assert seen['state'].dtype == np.float64
    assert packet == {'tstep': 47, 'fields_by_component': {'driver': {'value': 4}},
                      'provenance_by_component': {'driver': ('successor real-v6 canonical gate',)}}


@pytest.mark.parametrize('initialize', [False, True])
@pytest.mark.parametrize('backend', [None, 'explicit-backend'])
def test_extended_bootstrap_forwards_backend_and_initialization_without_work(monkeypatch, initialize, backend):
    calls = {}

    class StopBeforeDataRead(Exception):
        pass

    def load(args, **kwargs):
        calls.update(args=args, kwargs=kwargs)
        raise StopBeforeDataRead

    monkeypatch.setattr(canonical, '_load_runtime', load)
    args = object()
    with pytest.raises(StopBeforeDataRead):
        canonical._load_extended_runtime(args, backend_initializer=backend,
                                         initialize_probe_parameters=initialize)
    expected = {} if initialize else {'initialize_probe_parameters': False}
    if backend is not None:
        expected['backend_initializer'] = backend
    assert calls == {'args': args, 'kwargs': expected}


def test_prepare_template_keeps_source_identity_and_lifecycle(monkeypatch):
    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining import daily_markov_contract

    first = {'landpoint_id': 'point', 'year': 1963, 'parent_shard_sha256': 'parent',
             'typed_shard_sha256': 'typed', 'start_rows': [7]}
    reference = SimpleNamespace(parent=SimpleNamespace(landpoint_id='point', year=1963,
                                                       sha256='parent', path='parent-path'), sha256='typed')
    runtime = {'index': SimpleNamespace(select=lambda **kwargs: (reference,)),
               'manifest': 'manifest', 'contract': 'contract', 'materializer': 'materializer'}
    paths = {'config': 'config', 'run_def': 'run-def', 'reference_run_dir': 'reference'}
    calls = {}
    monkeypatch.setattr(retained, '_plan_context', lambda manifest, **kwargs: paths)
    monkeypatch.setattr(teacher, 'prepare_paper_1961_driver_context', lambda *args, **kwargs: 'context')
    monkeypatch.setattr(daily_markov_contract, 'load_markov_shard',
                        lambda path, **kwargs: SimpleNamespace(sample=lambda row: {'row': row}))

    def adapter(loaded, context, sample, **kwargs):
        calls.update(loaded=loaded, context=context, sample=sample, flags=kwargs)
        return 'template'

    monkeypatch.setattr(retained, '_retained_adapter', adapter)
    assert retained._prepare_template(runtime, {'components': {'one_day': [first]}}) == 'template'
    assert calls['loaded']['context_paths'] is paths
    assert calls['context'] == 'context'
    assert calls['sample'] == {'row': 7}
    assert calls['flags'] == {'lifecycle_start_tstep': 48, 'result_day_index': 2,
                              'fast_owner_materializer': 'materializer'}
    runtime['index'] = SimpleNamespace(select=lambda **kwargs: (reference, reference))
    with pytest.raises(ValueError, match='ambiguous'):
        retained._prepare_template(runtime, {'components': {'one_day': [first]}})


@pytest.mark.parametrize('module', ['canonical_training_runtime', 'retained_training_runtime',
                                   'runtime_diagnostics', 'training_artifact_io', 'shared_transformer_runtime'])
def test_current_runtime_helpers_do_not_import_experiment_runners(module):
    path = Path(__file__).resolve().parents[2] / 'scripts' / 'hpc' / (module + '.py')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or '').startswith('scripts.hpc.run_')
