"""Operation-scoped admission through the ordinary public Teacher indices."""
import json
from pathlib import Path

import pytest

from research.daily_coarse_graining import teacher_assets as assets
from research.daily_coarse_graining.markov_dataset import load_dataset_index
from research.daily_coarse_graining.typed_sidecar import load_coherent_typed_sidecar_index
from test_teacher_asset_readers import teacher_fixture, _publish


def test_coherent_base_and_typed_share_one_admission_per_operation(teacher_fixture, monkeypatch):
    f = teacher_fixture
    _publish(f.root, f.entry.landpoint_id, f.paths)
    opened, scanned, resolved = [], [], []
    original_init = assets.PointContainer.__init__
    original_scandir, original_resolve = assets.os.scandir, Path.resolve

    def init(self, *args, **kwargs):
        opened.append(kwargs.get('metadata_only'))
        return original_init(self, *args, **kwargs)

    def scandir(path):
        scanned.append(Path(path))
        return original_scandir(path)

    def resolve(path, *args, **kwargs):
        if path in (f.base, f.typed):
            resolved.append(path)
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(assets.PointContainer, '__init__', init)
    monkeypatch.setattr(assets.os, 'scandir', scandir)
    monkeypatch.setattr(Path, 'resolve', resolve)
    for _ in range(2):
        before = len(scanned)
        index = load_coherent_typed_sidecar_index(f.manifest, verify_files=True, verify_hashes=False)
        assert index.shards[0].path == f.typed
        assert len(scanned[before:]) == len(set(scanned[before:]))
    assert opened == [True, True]  # no trust survives the first constructor
    assert not resolved  # retired annual paths never need per-record realpath


def test_retired_members_do_not_require_original_directories(teacher_fixture):
    f = teacher_fixture
    _publish(f.root, f.entry.landpoint_id, f.paths)
    for directory in sorted({p.parent for p in f.paths}, key=lambda p: len(p.parts), reverse=True):
        directory.rmdir()
    assert load_coherent_typed_sidecar_index(f.manifest, verify_hashes=False).shards[0].path == f.typed


@pytest.mark.parametrize('target', ['map', 'manifest', 'map_appears'])
def test_load_cannot_bind_old_content_to_replacement_identity(teacher_fixture, monkeypatch, target):
    f = teacher_fixture
    if target != 'map_appears':
        _publish(f.root, f.entry.landpoint_id, f.paths)
    original = assets.load_storage_map

    def changed(root):
        loaded = original(root)
        path = f.manifest if target == 'manifest' else f.root / assets.MAP_NAME
        path.write_bytes((path.read_bytes() if path.exists() else b'{}') + b'\n')
        return loaded

    monkeypatch.setattr(assets, 'load_storage_map', changed)
    with pytest.raises(ValueError, match='changed during admission|changed during index admission'):
        load_dataset_index(f.manifest, verify_hashes=False)


@pytest.mark.parametrize('target', ['map', 'manifest', 'package', 'directory'])
def test_scope_exit_rechecks_all_admitted_identities(teacher_fixture, target):
    f = teacher_fixture
    _publish(f.root, f.entry.landpoint_id, f.paths)
    with pytest.raises(ValueError, match='changed during admission'):
        with assets.asset_index_admission(f.root, f.manifest) as admission:
            admission.resolve(f.base.relative_to(f.root).as_posix(), verify_files=True)
            path = {'map': f.root / assets.MAP_NAME, 'manifest': f.manifest,
                    'package': f.root / 'points' / (f.entry.landpoint_id + '.sqlite'),
                    'directory': f.base.parent}[target]
            if target == 'directory':
                (path / 'unexpected').write_bytes(b'x')
            else:
                path.write_bytes(path.read_bytes() + b'\n')


def test_changed_map_binding_invalidates_cached_connection(teacher_fixture):
    f = teacher_fixture
    _publish(f.root, f.entry.landpoint_id, f.paths)
    assert assets.asset_is_file(f.base)
    mapping_path = f.root / assets.MAP_NAME
    mapping = json.loads(mapping_path.read_bytes())
    mapping['points'][f.entry.landpoint_id]['bytes'] += 1
    mapping_path.write_text(json.dumps(mapping), encoding='utf-8')
    with pytest.raises(ValueError, match='container/map binding mismatch'):
        assets.asset_is_file(f.base)
