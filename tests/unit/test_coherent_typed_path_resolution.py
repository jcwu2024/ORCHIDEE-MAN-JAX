from __future__ import annotations

import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.daily_coarse_graining.typed_sidecar import _resolve_coherent_typed_shard


def _symlink(link: Path, target: Path, *, directory=False):
    try:
        link.symlink_to(target, target_is_directory=directory)
    except (NotImplementedError, OSError) as error:
        pytest.skip(f"filesystem cannot create symlinks: {error}")


def test_regular_shards_reuse_directory_resolution_but_check_each_leaf(
    tmp_path, monkeypatch
):
    root = tmp_path.resolve()
    directory = root / "typed" / "train"
    directory.mkdir(parents=True)
    for index in range(3):
        (directory / f"{index}.npz").write_bytes(bytes((index,)))

    original_resolve = Path.resolve
    original_lstat = os.lstat
    directory_resolves = []
    leaf_lstats = []

    def counted_resolve(path, *args, **kwargs):
        if path == directory:
            directory_resolves.append(path)
        return original_resolve(path, *args, **kwargs)

    def counted_lstat(path, *args, **kwargs):
        if Path(path).parent == directory:
            leaf_lstats.append(Path(path))
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counted_resolve)
    monkeypatch.setattr(os, "lstat", counted_lstat)
    cached = {}
    paths = [
        _resolve_coherent_typed_shard(root, f"typed/train/{index}.npz", cached)
        for index in range(3)
    ]
    assert paths == [directory / f"{index}.npz" for index in range(3)]
    assert directory_resolves == [directory]
    assert leaf_lstats == paths


def test_file_and_directory_symlinks_resolve_inside_root(tmp_path):
    root = tmp_path.resolve()
    target_dir = root / "actual"
    target_dir.mkdir()
    target = target_dir / "typed.npz"
    target.write_bytes(b"typed")
    _symlink(root / "linked_file.npz", target)
    _symlink(root / "linked_dir", target_dir, directory=True)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    _symlink(outside_dir / "return.npz", target)
    _symlink(root / "outward_dir", outside_dir, directory=True)

    cached = {}
    assert _resolve_coherent_typed_shard(root, "linked_file.npz", cached) == target
    assert _resolve_coherent_typed_shard(root, "linked_dir/typed.npz", cached) == target
    assert _resolve_coherent_typed_shard(root, "outward_dir/return.npz", cached) == target


def test_escape_missing_and_nonfile_paths_remain_rejected(tmp_path):
    root = (tmp_path / "release").resolve()
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_file = outside / "shard.npz"
    outside_file.write_bytes(b"outside")
    _symlink(root / "escape_file.npz", outside_file)
    _symlink(root / "escape_dir", outside, directory=True)
    (root / "not_a_file").mkdir()

    cached = {}
    with pytest.raises(ValueError, match="escapes dataset root"):
        _resolve_coherent_typed_shard(root, "escape_file.npz", cached)
    with pytest.raises(ValueError, match="escapes dataset root"):
        _resolve_coherent_typed_shard(root, "escape_dir/shard.npz", cached)
    with pytest.raises(ValueError, match="escapes dataset root"):
        _resolve_coherent_typed_shard(root, "../outside/shard.npz", cached)
    with pytest.raises(FileNotFoundError):
        _resolve_coherent_typed_shard(root, "missing.npz", cached)
    with pytest.raises(FileNotFoundError):
        _resolve_coherent_typed_shard(root, "not_a_file", cached)
    with pytest.raises(FileNotFoundError):
        _resolve_coherent_typed_shard(root, "", cached)
    with pytest.raises(FileNotFoundError) as parent_path:
        _resolve_coherent_typed_shard(root, "not_a_file/..", cached)
    assert str(root) in str(parent_path.value)


def test_symlink_leaf_resolution_and_escape_logic_without_platform_symlinks(
    tmp_path, monkeypatch
):
    root = (tmp_path / "release").resolve()
    root.mkdir()
    inside = root / "inside.npz"
    inside.write_bytes(b"inside")
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"outside")
    link = root / "link.npz"
    original_resolve = Path.resolve
    original_lstat = os.lstat

    def fake_lstat(path, *args, **kwargs):
        if Path(path) == link:
            return SimpleNamespace(st_mode=stat.S_IFLNK)
        return original_lstat(path, *args, **kwargs)

    def fake_resolve(path, *args, **kwargs):
        if path == link:
            return destination
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(os, "lstat", fake_lstat)
    monkeypatch.setattr(Path, "resolve", fake_resolve)
    destination = inside
    assert _resolve_coherent_typed_shard(root, "link.npz", {}) == inside
    destination = outside
    with pytest.raises(ValueError, match="escapes dataset root"):
        _resolve_coherent_typed_shard(root, "link.npz", {})


def test_outward_directory_and_inward_leaf_link_is_allowed(tmp_path, monkeypatch):
    root = (tmp_path / "release").resolve()
    root.mkdir()
    target = root / "target.npz"
    target.write_bytes(b"inside")
    outward = root / "outward"
    outside = tmp_path / "outside"
    outside.mkdir()
    leaf = outside / "return.npz"
    original_resolve = Path.resolve
    original_lstat = os.lstat

    def fake_resolve(path, *args, **kwargs):
        if path == outward:
            return outside
        if path == leaf:
            return target
        return original_resolve(path, *args, **kwargs)

    def fake_lstat(path, *args, **kwargs):
        if Path(path) == leaf:
            return SimpleNamespace(st_mode=stat.S_IFLNK)
        return original_lstat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    monkeypatch.setattr(os, "lstat", fake_lstat)
    assert _resolve_coherent_typed_shard(root, "outward/return.npz", {}) == target
