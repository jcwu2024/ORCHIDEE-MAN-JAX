"""Read original Teacher assets from loose files or published per-point stores.

Logical Paths and serialized scientific classes do not change. A storage map is
authoritative for its members even while loose originals remain. No global IO
patching, extracted per-record cache or writable SQLite connection is used.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import threading
from contextlib import contextmanager
from functools import wraps

MAP_NAME = "teacher-storage.json"
SCHEMA = "teacher_point_storage_v1"
_lock = threading.RLock()
_maps: dict[Path, tuple[tuple, dict, dict]] = {}
_manifest_hashes: dict[Path, tuple[tuple, str]] = {}
_local = threading.local()
MAX_CONNECTIONS_PER_THREAD = 8
CHUNK = 1024 * 1024


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def relative_key(value: str) -> str:
    p = PurePosixPath(value)
    if (not value or value == "." or p.is_absolute() or ".." in p.parts
            or str(p) != value or "\\" in value or ":" in value or "\x00" in value):
        raise ValueError(f"unsafe Teacher member {value!r}")
    return value


def point_key(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,80}", value) or value in {".", ".."}:
        raise ValueError(f"unsafe landpoint ID {value!r}")
    return value


def safe_path(path: str | Path) -> Path:
    p = Path(path).absolute()
    if ".." in p.parts:
        raise ValueError(f"parent traversal in Teacher path {p}")
    for ancestor in (p, *p.parents):
        if ancestor.is_symlink():
            raise ValueError(f"symlink in stored Teacher path {ancestor}")
    if os.path.normcase(os.path.realpath(p)) != os.path.normcase(str(p)):
        raise ValueError(f"Teacher realpath mismatch {p}")
    return p


def fingerprint(path: Path) -> tuple:
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def loose_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _bound_manifest_hash(root: Path) -> str:
    p = safe_path(root / "dataset_manifest.json")
    identity = fingerprint(p)
    cached = _manifest_hashes.get(p)
    if cached is None or cached[0] != identity:
        h = loose_sha256(p)
        if fingerprint(p) != identity:
            raise ValueError("Teacher manifest changed while hashing")
        _manifest_hashes[p] = identity, h
    return _manifest_hashes[p][1]


def load_storage_map(root: str | Path) -> tuple[dict, dict[str, str]] | None:
    """Return validated map and exact logical-member→point lookup, if published."""
    root = safe_path(root)
    p = root / MAP_NAME
    if not p.exists():
        return None
    safe_path(p)
    identity = fingerprint(p)
    with _lock:
        cached = _maps.get(root)
        if cached is not None and cached[0] == identity:
            # A changed scientific manifest invalidates the map even without a map edit.
            if cached[1]["dataset_manifest_sha256"] != _bound_manifest_hash(root):
                raise ValueError("Teacher storage map release hash mismatch")
            return cached[1], cached[2]
        raw = json.loads(p.read_bytes())
        if fingerprint(p) != identity:
            raise ValueError("Teacher storage map changed while reading")
        if raw.get("schema_version") != SCHEMA:
            raise ValueError("unsupported Teacher storage map")
        if raw.get("dataset_manifest_sha256") != _bound_manifest_hash(root):
            raise ValueError("Teacher storage map release hash mismatch")
        members: dict[str, str] = {}
        for point, record in raw["points"].items():
            point_key(point)
            if record["container"] != "points/" + point + ".sqlite":
                raise ValueError("Teacher point container path mismatch")
            relative_key(record["container"])
            if len(record["members"]) != record["member_count"]:
                raise ValueError("Teacher point member count mismatch")
            for name in record["members"]:
                relative_key(name)
                if not name.startswith("workers/") or name in members:
                    raise ValueError("duplicate or unsupported mapped Teacher member")
                members[name] = point
        _maps[root] = identity, raw, members
        return raw, members


class AssetIndexAdmission:
    """One bounded index operation; mapped paths never probe missing leaves.

    Each distinct directory and package is admitted once, then all identities
    are checked again before returning the index. No trust survives this scope.
    Loose references retain normal resolve/containment and existence semantics.
    """
    def __init__(self, root: Path, manifest_path: Path | None = None):
        self.root = safe_path(root)
        self.identities: dict[Path, tuple] = {}
        self.directories: dict[Path, set[str]] = {}
        self.packages: dict[str, dict] = {}
        self._track(self.root / 'dataset_manifest.json' if manifest_path is None else manifest_path)
        map_path = self.root / MAP_NAME
        map_before = fingerprint(map_path) if map_path.exists() else None
        if map_before is not None:
            self.identities[map_path] = map_before
            # The map is always bound to this canonical manifest, including
            # callers opening a differently named index in the same directory.
            self._track(self.root / 'dataset_manifest.json')
        self.loaded = load_storage_map(self.root)
        map_after = fingerprint(map_path) if map_path.exists() else None
        if map_before != map_after:
            raise ValueError('Teacher storage map changed during index admission')
        for path in tuple(self.identities):
            self._track(path)

    def _track(self, path: Path) -> None:
        identity = fingerprint(path)
        old = self.identities.setdefault(path, identity)
        if old != identity:
            raise ValueError(f'Teacher index input changed during admission: {path}')

    def _directory(self, directory: Path) -> set[str]:
        cached = self.directories.get(directory)
        if cached is not None:
            return cached
        if directory != self.root:
            self._directory(directory.parent)
            if directory.name in self.directories[directory.parent]:
                raise ValueError(f'symlink in stored Teacher path {directory}')
        if not directory.exists():
            # Retired logical members need not retain their original empty
            # directories. The nearest existing parent's identity protects
            # this absence until the operation ends.
            self.directories[directory] = set()
            return self.directories[directory]
        self._track(directory)
        with os.scandir(directory) as entries:
            links = {entry.name for entry in entries if entry.is_symlink()}
        self._track(directory)
        self.directories[directory] = links
        return links

    def mapped(self, path: Path):
        if self.loaded is None:
            return None
        try:
            name = path.relative_to(self.root).as_posix()
        except ValueError:
            return None
        relative_key(name)
        mapping, lookup = self.loaded
        point = lookup.get(name)
        if point is None:
            return None
        if path.name in self._directory(path.parent):
            raise ValueError(f'symlink in stored Teacher path {path}')
        return self.root, mapping, name, point

    def resolve(self, name: str, *, verify_files: bool, loose_resolver=None) -> Path:
        candidate = self.root / name
        mapped = self.mapped(candidate)
        if mapped is not None:
            if verify_files:
                self.admit(mapped)
            return candidate
        if loose_resolver is not None:
            return loose_resolver(name)
        path = candidate.resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f'shard path escapes dataset root: {path}') from error
        if verify_files and not path.is_file():
            raise FileNotFoundError(path)
        return path

    def admit(self, mapped) -> None:
        root, mapping, name, point = mapped
        if point in self.packages:
            return
        record = mapping['points'][point]
        path = root / record['container']
        if path.name in self._directory(path.parent):
            raise ValueError(f'symlink in stored Teacher path {path}')
        self._track(path)
        reader = PointContainer(path, record, metadata_only=True)
        try:
            if reader.point != point or reader.manifest_sha256 != mapping['dataset_manifest_sha256']:
                raise ValueError('Teacher point/release binding mismatch')
        finally:
            reader.close()
        self._track(path)
        self.packages[point] = record

    def finish(self) -> None:
        if self.loaded is None and (self.root / MAP_NAME).exists():
            raise ValueError('Teacher storage map appeared during index admission')
        for ancestor in (self.root, *self.root.parents):
            if ancestor.is_symlink():
                raise ValueError(f'symlink in stored Teacher path {ancestor}')
        for path, identity in self.identities.items():
            if path.is_symlink() or fingerprint(path) != identity:
                raise ValueError(f'Teacher index input changed during admission: {path}')


@contextmanager
def asset_index_admission(root: Path, manifest_path: Path | None = None):
    """Share a scope through coherent typed→base construction, never globally."""
    active = getattr(_local, 'index_admission', None)
    if active is not None and active.root == root:
        if manifest_path is not None:
            active._track(manifest_path)
        yield active
        return
    admission = AssetIndexAdmission(root, manifest_path)
    _local.index_admission = admission
    try:
        yield admission
        admission.finish()
    finally:
        _local.index_admission = active


def with_asset_index_admission(function):
    @wraps(function)
    def load(manifest_path, **kwargs):
        path = Path(manifest_path).resolve()
        root = path.parent
        with asset_index_admission(root, path):
            return function(manifest_path, **kwargs)
    return load


def resolve_index_asset(root: Path, name: str, *, verify_files: bool, loose_resolver=None) -> Path:
    admission = getattr(_local, 'index_admission', None)
    if admission is None or admission.root != root:
        raise RuntimeError('index asset resolution requires a bounded admission scope')
    return admission.resolve(name, verify_files=verify_files, loose_resolver=loose_resolver)


class PointContainer:
    """Readonly point store bound by the central map and immutable file identity.

    Maintenance defaults to validating every table metadata row. Index admission
    may use only the bound snapshot and covering path index; ``read`` always
    checks the selected table row and original bytes in either mode.
    """
    def __init__(self, path: Path, expected: dict | None = None, *, metadata_only: bool = False):
        self.path = safe_path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.identity = fingerprint(self.path)
        self.connection = sqlite3.connect(self.path.as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            self.connection.execute("PRAGMA query_only=ON")
            metadata = dict(self.connection.execute("SELECT key,value FROM metadata"))
            if metadata["schema_version"] != SCHEMA:
                raise ValueError("unsupported Teacher container")
            self.point = point_key(metadata["landpoint_id"])
            self.manifest_sha256 = metadata["dataset_manifest_sha256"]
            if not re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256):
                raise ValueError("invalid Teacher container release hash")
            self.rows = json.loads(metadata["source_rows"])
            if not isinstance(self.rows, dict):
                raise ValueError("invalid Teacher source row metadata")
            self.snapshot_sha256 = hashlib.sha256(canonical(self.rows)).hexdigest()
            if self.snapshot_sha256 != metadata["snapshot_sha256"] or metadata.get("complete") != "true":
                raise ValueError("incomplete/tampered Teacher snapshot")
            for name, row in self.rows.items():
                relative_key(name)
                if (not isinstance(row, dict)
                        or any(type(row.get(k)) is not int for k in ("size", "mode", "mtime_ns"))
                        or row["size"] < 0 or row["mode"] < 0
                        or not isinstance(row.get("sha256"), str)
                        or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])):
                    raise ValueError("invalid Teacher source row metadata " + name)
            if metadata_only:
                # The path primary-key index covers this inventory query. Reading
                # other columns here would visit every BLOB-adjacent table page.
                names = [r[0] for r in self.connection.execute("SELECT path FROM assets")]
                if len(names) != len(self.rows) or set(names) != self.rows.keys():
                    raise ValueError("Teacher container member inventory drift")
            else:
                actual = {r[0]: {"size":r[1], "mode":r[2], "mtime_ns":r[3], "sha256":r[4]}
                          for r in self.connection.execute("SELECT path,size,mode,mtime_ns,sha256 FROM assets")}
                if actual.keys() != self.rows.keys():
                    raise ValueError("Teacher container member inventory drift")
                for name, row in self.rows.items():
                    if any(actual[name][k] != row[k] for k in actual[name]):
                        raise ValueError("Teacher container metadata drift " + name)
            if expected is not None:
                if (self.snapshot_sha256 != expected["snapshot_sha256"]
                        or set(self.rows) != set(expected["members"])
                        or self.path.stat().st_size != expected["bytes"]):
                    raise ValueError("Teacher container/map binding mismatch")
            self._assert_unchanged()
        except BaseException:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()

    def _assert_unchanged(self):
        if fingerprint(safe_path(self.path)) != self.identity:
            raise ValueError("published Teacher container changed")

    def read(self, name: str) -> tuple[bytes, str]:
        relative_key(name)
        self._assert_unchanged()
        row = self.rows[name]
        value = self.connection.execute(
            "SELECT size,mode,mtime_ns,sha256,data FROM assets WHERE path=?", (name,)).fetchone()
        if value is None:
            raise ValueError("Teacher BLOB size/inventory mismatch")
        if any(value[i] != row[k] for i, k in enumerate(("size", "mode", "mtime_ns", "sha256"))):
            raise ValueError("Teacher container metadata drift " + name)
        payload = value[4]
        if not isinstance(payload, bytes) or len(payload) != row["size"]:
            raise ValueError("Teacher BLOB size/inventory mismatch")
        actual = hashlib.sha256(payload).hexdigest()
        if actual != row["sha256"]:
            raise ValueError("Teacher BLOB hash mismatch " + name)
        self._assert_unchanged()
        return payload, actual


def _mapped(path: str | Path):
    p = Path(path).absolute()
    if ".." in p.parts:
        raise ValueError("Teacher path contains parent traversal")
    admission = getattr(_local, 'index_admission', None)
    if admission is not None and p.is_relative_to(admission.root):
        return admission.mapped(p)
    for root in p.parents:
        if (root / MAP_NAME).is_file():
            loaded = load_storage_map(root)
            if loaded is None:
                continue
            mapping, lookup = loaded
            name = p.relative_to(root).as_posix()
            point = lookup.get(name)
            if point is None:
                return None
            safe_path(p)
            return root, mapping, name, point
    return None


def _container(mapped) -> PointContainer:
    root, mapping, name, point = mapped
    record = mapping["points"][point]
    path = safe_path(root / record["container"])
    identity = fingerprint(path)
    cache = getattr(_local, "connections", None)
    if cache is None:
        _local.connections = cache = OrderedDict()
    key = str(path)
    cached = cache.pop(key, None)
    binding = canonical(record), point, mapping["dataset_manifest_sha256"]
    if cached is not None and (cached.identity != identity or cached.map_binding != binding):
        cached.close()
        cached = None
    if cached is None:
        cached = PointContainer(path, record, metadata_only=True)
        if cached.point != point or cached.manifest_sha256 != mapping["dataset_manifest_sha256"]:
            cached.close()
            raise ValueError("Teacher point/release binding mismatch")
        cached.map_binding = binding
    cache[key] = cached
    while len(cache) > MAX_CONNECTIONS_PER_THREAD:
        cache.popitem(last=False)[1].close()
    return cached


def close_asset_readers() -> None:
    """Close the calling thread's bounded readonly connection cache."""
    cache = getattr(_local, "connections", {})
    for reader in cache.values():
        reader.close()
    cache.clear()


def asset_is_mapped(path: str | Path) -> bool:
    return _mapped(path) is not None


def reject_asset_write(path: str | Path) -> None:
    if asset_is_mapped(path):
        raise ValueError(f"published Teacher asset is immutable; restore/unmap before regeneration: {path}")


def reject_release_write(root: str | Path) -> None:
    if load_storage_map(root) is not None:
        raise ValueError(f"published Teacher release is immutable; restore to a separate loose root for generation: {root}")


def reject_scientific_manifest_write(path: str | Path) -> None:
    p = Path(path).absolute()
    for root in p.parents:
        if (root / MAP_NAME).is_file():
            relative = p.relative_to(root).as_posix()
            if (relative == "dataset_manifest.json" or
                    (relative.startswith("workers/") and relative.endswith("/manifest.json"))):
                reject_release_write(root)
            return


def open_asset(path: str | Path):
    mapped = _mapped(path)
    if mapped is None:
        return Path(path).open("rb")
    return io.BytesIO(_container(mapped).read(mapped[2])[0])


def asset_is_file(path: str | Path) -> bool:
    mapped = _mapped(path)
    if mapped is None:
        return Path(path).is_file()
    return mapped[2] in _container(mapped).rows


def asset_read_text(path: str | Path, encoding: str = "utf-8", errors: str = "strict") -> str:
    with open_asset(path) as handle:
        return handle.read().decode(encoding, errors)


def asset_sha256(path: str | Path) -> str:
    mapped = _mapped(path)
    if mapped is None:
        return loose_sha256(Path(path))
    return _container(mapped).read(mapped[2])[1]
