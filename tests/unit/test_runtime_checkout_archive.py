import json
import shutil
import subprocess

import pytest

from scripts.hpc import archive_runtime_checkouts as archive


def test_rejects_broad_or_symlink_targets(tmp_path):
    with pytest.raises(ValueError):
        archive.validate_targets(tmp_path, ['checkouts'])
    with pytest.raises(ValueError):
        archive.validate_targets(tmp_path, ['checkouts/..'])


@pytest.mark.parametrize('changed', [False, True])
def test_verified_archive_restores_and_detects_new_files(tmp_path, monkeypatch, changed):
    if not shutil.which('tar') or 'GNU tar' not in subprocess.run(
            ['tar', '--version'], capture_output=True, text=True).stdout:
        pytest.skip('production archiver requires GNU tar')
    root = tmp_path / 'runtime'
    checkout = root / 'checkouts' / 'old'
    (checkout / '.git').mkdir(parents=True)
    (root / 'archives').mkdir()
    (checkout / '.git' / 'HEAD').write_text('preserve dependency metadata\n')
    (checkout / 'source.py').write_text('original source\n')
    (checkout / 'untracked').mkdir()
    (checkout / 'untracked' / 'notes.txt').write_text('uncommitted notes\n')
    manifest = tmp_path / 'plan.json'
    manifest.write_text(json.dumps(dict(targets=['checkouts/old'], nodes=[], python='python')))
    target = root / 'archives' / 'old.tar.gz'
    monkeypatch.setattr(archive, 'ROOT', root)
    calls = []

    def activity(*args):
        calls.append(1)
        if len(calls) == 2 and changed:
            (checkout / 'new-after-packing').write_text('must not be deleted\n')
        return {}

    monkeypatch.setattr(archive, 'check_activity', activity)
    args = type('Args', (), dict(processes_only=False, manifest=manifest,
                               archive=target, apply=True))()
    if changed:
        with pytest.raises(RuntimeError, match='changed while archiving'):
            archive.run(args)
        assert (checkout / 'new-after-packing').exists()
        assert (checkout / 'source.py').exists()
        return
    archive.run(args)
    assert (checkout / '.git' / 'HEAD').read_text() == 'preserve dependency metadata\n'
    assert not (checkout / 'source.py').exists()
    assert (checkout / 'ARCHIVED_WORKING_FILES.md').exists()
    subprocess.run(['tar', '-xzf', str(target), '-C', str(root)], check=True)
    assert (checkout / 'source.py').read_text() == 'original source\n'
    assert (checkout / 'untracked' / 'notes.txt').read_text() == 'uncommitted notes\n'


def test_nested_git_is_not_silently_excluded(tmp_path):
    (tmp_path / '.git').mkdir()
    (tmp_path / 'nested' / '.git').mkdir(parents=True)
    with pytest.raises(ValueError, match='nested Git'):
        archive.working_snapshot(tmp_path)
