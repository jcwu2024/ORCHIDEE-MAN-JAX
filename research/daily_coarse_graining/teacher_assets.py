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


class PointContainer:
    """One readonly per-thread connection; metadata bound by the central map."""
    def __init__(self, path: Path, expected: dict | None = None):
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
            self.rows = json.loads(metadata["source_rows"])
            self.snapshot_sha256 = hashlib.sha256(canonical(self.rows)).hexdigest()
            if self.snapshot_sha256 != metadata["snapshot_sha256"] or metadata.get("complete") != "true":
                raise ValueError("incomplete/tampered Teacher snapshot")
            actual = {r[0]: {"size":r[1], "mode":r[2], "mtime_ns":r[3], "sha256":r[4]}
                      for r in self.connection.execute("SELECT path,size,mode,mtime_ns,sha256 FROM assets")}
            if actual.keys() != self.rows.keys():
                raise ValueError("Teacher container member inventory drift")
            for name, row in self.rows.items():
                relative_key(name)
                if any(actual[name][k] != row[k] for k in actual[name]):
                    raise ValueError("Teacher container metadata drift " + name)
            if expected is not None:
                if (self.snapshot_sha256 != expected["snapshot_sha256"]
                        or set(self.rows) != set(expected["members"])
                        or self.path.stat().st_size != expected["bytes"]):
                    raise ValueError("Teacher container/map binding mismatch")
        except BaseException:
            self.connection.close()
            raise

    def close(self):
        self.connection.close()

    def read(self, name: str) -> tuple[bytes, str]:
        relative_key(name)
        if fingerprint(self.path) != self.identity:
            raise ValueError("published Teacher container changed")
        row = self.rows[name]
        value = self.connection.execute("SELECT data FROM assets WHERE path=?", (name,)).fetchone()
        if value is None or not isinstance(value[0], bytes) or len(value[0]) != row["size"]:
            raise ValueError("Teacher BLOB size/inventory mismatch")
        payload = value[0]
        actual = hashlib.sha256(payload).hexdigest()
        if actual != row["sha256"]:
            raise ValueError("Teacher BLOB hash mismatch " + name)
        return payload, actual


def _mapped(path: str | Path):
    p = Path(path).absolute()
    if ".." in p.parts:
        raise ValueError("Teacher path contains parent traversal")
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
    if cached is not None and (cached.identity != identity or cached.snapshot_sha256 != record["snapshot_sha256"]):
        cached.close()
        cached = None
    if cached is None:
        cached = PointContainer(path, record)
        if cached.point != point or cached.manifest_sha256 != mapping["dataset_manifest_sha256"]:
            cached.close()
            raise ValueError("Teacher point/release binding mismatch")
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
