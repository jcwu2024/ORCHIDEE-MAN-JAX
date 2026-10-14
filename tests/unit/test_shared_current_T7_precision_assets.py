"""Small metadata/temp-copy failure tests; original scientific assets unused."""
import ast
from copy import deepcopy
import gzip
import hashlib
import os
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.hpc import shared_current_T7_precision_assets as assets


def receipt():
    math = dict(source_git_head=assets.SOURCE_HEAD,
        architecture='shared_latent32x384_reader64x2', parameter_count=66123713,
        parameter_leaves=355, training_updates=0, horizon=7,
        prepared_context_sha256=assets.CONTEXT_SHA,
        parameter_provenance=dict(sha256=assets.PARAMETERS_SHA, identity=dict(source_head=assets.SOURCE_HEAD)))
    selection = dict(artifact_id=299, window_index=0, horizon=7, spatial_split='train',
        temporal_split='train', case_sha256=assets.CASE_SHA, prepared_artifact_sha256=assets.ARTIFACT_SHA)
    return dict(route='B1', identity=dict(asset_identity=dict(saved_B1_mathematical_identity=math,
        sampling=[{}, {}, {}, selection])))


def manifests():
    identity = dict(training_contract_sha256='c'*64, dataset_manifest_sha256='d'*64)
    prepared = dict(schema_version='target_oriented_prepared_manifest_v1', status='passed', identity=identity,
                    entries=[dict(update=299, sha256=assets.ARTIFACT_SHA)])
    dataset = dict(schema_version='daily_teacher_dataset_manifest_v6', status='complete')
    method = dict(configuration=dict(recurrent_window_index=0), training_plan=dict(
        prepared_manifest_identity=identity, training_artifact_ids=[742, 275, 482, 299]))
    return prepared, dataset, method


def test_exact_metadata_selection_positive():
    math, selection = assets.validate_receipt(receipt())
    assert math['parameter_count'] == 66123713 and selection['artifact_id'] == 299
    prepared, dataset, method = manifests()
    assert assets.validate_manifests(prepared, dataset, method)['update'] == 299


@pytest.mark.parametrize('key,value', [('source_git_head', 'x'*40), ('architecture', 'original'),
    ('parameter_count', 64884481), ('parameter_leaves', 300), ('training_updates', 1),
    ('horizon', 30), ('prepared_context_sha256', 'x'*64)])
def test_receipt_rejects_old_or_changed_math(key, value):
    data = receipt()
    data['identity']['asset_identity']['saved_B1_mathematical_identity'][key] = value
    with pytest.raises(ValueError):
        assets.validate_receipt(data)


@pytest.mark.parametrize('key,value', [('artifact_id', 742), ('window_index', 1), ('window_index', False),
    ('horizon', 30), ('spatial_split', 'test'), ('temporal_split', 'validation'),
    ('case_sha256', 'x'*64), ('prepared_artifact_sha256', 'x'*64)])
def test_receipt_rejects_wrong_window_or_held_out(key, value):
    data = receipt()
    data['identity']['asset_identity']['sampling'][3][key] = value
    with pytest.raises(ValueError):
        assets.validate_receipt(data)


def test_receipt_rejects_non_b1_and_wrong_parameters():
    data = receipt()
    data['route'] = 'B4'
    with pytest.raises(ValueError):
        assets.validate_receipt(data)
    data = receipt()
    data['identity']['asset_identity']['saved_B1_mathematical_identity']['parameter_provenance']['sha256'] = 'b'*64
    with pytest.raises(ValueError):
        assets.validate_receipt(data)


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'wrong_sha', 'wrong_order', 'heldout_status', 'wrong_identity'])
def test_training_manifest_failures(change):
    prepared, dataset, method = deepcopy(manifests())
    if change == 'duplicate':
        prepared['entries'] *= 2
    elif change == 'missing':
        prepared['entries'] = []
    elif change == 'wrong_sha':
        prepared['entries'][0]['sha256'] = 'b'*64
    elif change == 'wrong_order':
        method['training_plan']['training_artifact_ids'] = [299, 275, 482, 742]
    elif change == 'heldout_status':
        dataset['status'] = 'incomplete'
    elif change == 'wrong_identity':
        prepared['identity']['training_contract_sha256'] = 'b'*64
        # deepcopy preserves shared identity within fixture; independently break it.
        method['training_plan']['prepared_manifest_identity'] = {'training_contract_sha256': 'c'*64}
    with pytest.raises(ValueError):
        assets.validate_manifests(prepared, dataset, method)


@pytest.mark.parametrize('compressed', [False, True])
def test_verified_temp_pickle_and_corruption_rejected_before_unpickle(tmp_path, monkeypatch, compressed):
    path = tmp_path/('fixture.pkl.gz' if compressed else 'fixture.pkl')
    opener = gzip.open if compressed else open
    with opener(path, 'wb') as stream:
        pickle.dump({'tiny': np.array([1, 2], dtype=np.int32)}, stream)
    sha = assets.sha256(path)
    value, identity = assets.read_verified_pickle(path, sha)
    np.testing.assert_array_equal(value['tiny'], [1, 2])
    assert identity['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(path.read_bytes()+b'changed-fixture')
    monkeypatch.setattr(assets.HistoricalAssetUnpickler, 'load', lambda self: pytest.fail('Unverified pickle loaded'))
    with pytest.raises(ValueError, match='hash differs'):
        assets.read_verified_pickle(path, sha)


def test_windows_lexical_historical_path_fixture(tmp_path):
    path = tmp_path/'posix.pkl'
    # Protocol0 GLOBAL+REDUCE constructs a tiny path metadata object only.
    path.write_bytes(b'cpathlib\nPosixPath\n(V/WORK/saved\ntR.')
    value, _ = assets.read_verified_pickle(path, assets.sha256(path))
    assert str(value) == '/WORK/saved'
    if os.name == 'nt':
        assert type(value).__name__ == 'PurePosixPath'


def test_loader_rejects_non_x64_before_any_file_access():
    fake = SimpleNamespace(config=SimpleNamespace(x64_enabled=False))
    with pytest.raises(ValueError, match='x64'):
        assets.load_assets(fake, parameters='absent', prepared_context='absent',
            training_prepared_manifest='absent', dataset_manifest='absent', saved_identity_receipt='absent')


def test_small_wrong_parameter_scope_rejected():
    fake = SimpleNamespace(tree_util=SimpleNamespace(tree_leaves=lambda value: [value]))
    with pytest.raises(ValueError, match='355-leaf'):
        assets.parameter_gate(fake, np.zeros(1, dtype=np.float32))


def test_loader_source_has_no_capture_import_or_model_compilation_api():
    source = Path(assets.__file__).read_text()
    tree = ast.parse(source)
    imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any('capture_' in module for module in imports)
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert not any(isinstance(call, ast.Attribute) and call.attr in ('jit', 'eval_shape', 'lower', 'compile', 'trace') for call in calls)
    assert not any(isinstance(call, ast.Name) and call.id in ('make_shared_recurrent_objective', 'make_shared_daily_objective') for call in calls)
