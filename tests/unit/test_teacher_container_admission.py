"""Bounded container admission and exact validation at the BLOB read boundary."""
import hashlib
import os
import sqlite3

import pytest

from research.daily_coarse_graining import teacher_assets as assets


@pytest.fixture
def package(tmp_path):
    path = tmp_path / "point.sqlite"
    payloads = {"workers/a/one.npz": b"first" * 20000,
                "workers/a/two.pkl": b"second" * 20000}
    rows = {name: {"size": len(data), "mode": 0o644, "mtime_ns": 123456789,
                   "sha256": hashlib.sha256(data).hexdigest()}
            for name, data in payloads.items()}
    snapshot = hashlib.sha256(assets.canonical(rows)).hexdigest()
    metadata = {"schema_version": assets.SCHEMA, "landpoint_id": "087.0-109.0",
                "dataset_manifest_sha256": "a" * 64, "source_rows": assets.canonical(rows).decode(),
                "snapshot_sha256": snapshot, "complete": "true"}
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        connection.execute("CREATE TABLE assets(path TEXT PRIMARY KEY,size INTEGER NOT NULL,"
                           "mode INTEGER NOT NULL,mtime_ns INTEGER NOT NULL,sha256 TEXT NOT NULL,data BLOB NOT NULL)")
        connection.executemany("INSERT INTO metadata VALUES (?,?)", metadata.items())
        connection.executemany("INSERT INTO assets VALUES (?,?,?,?,?,?)",
                               [(name, row["size"], row["mode"], row["mtime_ns"], row["sha256"], payloads[name])
                                for name, row in rows.items()])
    connection.close()
    expected = {"snapshot_sha256": snapshot, "members": list(rows), "bytes": path.stat().st_size}
    return path, rows, payloads, expected


def test_admission_reads_only_metadata_and_covering_member_index(package, monkeypatch):
    path, rows, payloads, expected = package
    statements = []
    original_connect = sqlite3.connect

    def traced_connect(*args, **kwargs):
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(assets.sqlite3, "connect", traced_connect)
    reader = assets.PointContainer(path, expected, metadata_only=True)
    try:
        assert reader.rows == rows
        assert "SELECT path FROM assets" in statements
        assert not any("data" in sql.lower().split("from assets")[0]
                       for sql in statements if "from assets" in sql.lower())
        plan = reader.connection.execute("EXPLAIN QUERY PLAN SELECT path FROM assets").fetchall()
        assert any("COVERING INDEX" in row[3] for row in plan)
        for name, payload in payloads.items():
            assert reader.read(name) == (payload, rows[name]["sha256"])
        assert "SELECT path,size,mode,mtime_ns,sha256 FROM assets" not in statements
    finally:
        reader.close()


@pytest.mark.parametrize("column,value", [("size", 100001), ("mode", 0o600),
                                         ("mtime_ns", 123456790), ("sha256", "b" * 64)])
def test_row_metadata_validated_lazily_and_strict_default_preserved(package, column, value):
    path, rows, _, expected = package
    name = next(iter(rows))
    with sqlite3.connect(path) as connection:
        connection.execute(f"UPDATE assets SET {column}=? WHERE path=?", (value, name))
    # Only table metadata changed; snapshot/map and path inventory remain bound.
    reader = assets.PointContainer(path, expected, metadata_only=True)
    try:
        with pytest.raises(ValueError, match="metadata drift"):
            reader.read(name)
    finally:
        reader.close()
    with pytest.raises(ValueError, match="metadata drift"):
        assets.PointContainer(path, expected)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_exact_member_inventory_admitted(package, mutation):
    path, rows, _, expected = package
    name = next(iter(rows))
    with sqlite3.connect(path) as connection:
        if mutation == "missing":
            connection.execute("DELETE FROM assets WHERE path=?", (name,))
        else:
            connection.execute("INSERT INTO assets SELECT 'workers/extra',size,mode,mtime_ns,sha256,data "
                               "FROM assets WHERE path=?", (name,))
    expected["bytes"] = path.stat().st_size
    with pytest.raises(ValueError, match="member inventory drift"):
        assets.PointContainer(path, expected, metadata_only=True)


@pytest.mark.parametrize("field,value", [("snapshot_sha256", "b" * 64),
                                       ("members", ["workers/other"]), ("bytes", 1)])
def test_map_binding_mismatch_rejected(package, field, value):
    path, _, _, expected = package
    expected[field] = value
    with pytest.raises(ValueError, match="map binding mismatch"):
        assets.PointContainer(path, expected, metadata_only=True)


@pytest.mark.parametrize("field,value,message", [
    ("schema_version", "unknown", "unsupported"),
    ("complete", "false", "incomplete/tampered"),
    ("landpoint_id", "../unsafe", "unsafe landpoint"),
    ("dataset_manifest_sha256", "invalid", "release hash"),
    ("source_rows", "{}", "incomplete/tampered"),
    ("snapshot_sha256", "b" * 64, "incomplete/tampered"),
])
def test_metadata_contract_rejected(package, field, value, message):
    path, _, _, expected = package
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE metadata SET value=? WHERE key=?", (value, field))
    expected["bytes"] = path.stat().st_size
    with pytest.raises(ValueError, match=message):
        assets.PointContainer(path, expected, metadata_only=True)


def test_same_size_corrupt_blob_rejected_on_read(package):
    path, rows, _, expected = package
    name, row = next(iter(rows.items()))
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE assets SET data=? WHERE path=?", (b"x" * row["size"], name))
    reader = assets.PointContainer(path, expected, metadata_only=True)
    try:
        with pytest.raises(ValueError, match="BLOB hash mismatch"):
            reader.read(name)
    finally:
        reader.close()


@pytest.mark.parametrize("mutation", ["replace", "append"])
def test_published_package_identity_change_rejected(package, mutation):
    path, rows, _, expected = package
    reader = assets.PointContainer(path, expected, metadata_only=True)
    try:
        if mutation == "replace":
            replacement = path.with_suffix(".replacement")
            replacement.write_bytes(path.read_bytes())
            if os.name == "nt":
                # Windows SQLite denies file replacement while its handle is
                # open. Retain the admitted identity and release that handle;
                # the read must reject replacement before querying SQLite.
                reader.connection.close()
            os.replace(replacement, path)
        else:
            with path.open("ab") as handle:
                handle.write(b"changed")
        with pytest.raises(ValueError, match="published Teacher container changed"):
            reader.read(next(iter(rows)))
    finally:
        reader.close()


def test_package_change_during_payload_hash_fails_before_return(package, monkeypatch):
    path, rows, _, expected = package
    reader = assets.PointContainer(path, expected, metadata_only=True)
    original_sha256 = hashlib.sha256

    def mutate_while_hashing(payload):
        with path.open("ab") as handle:
            handle.write(b"changed-during-read")
        return original_sha256(payload)

    monkeypatch.setattr(assets.hashlib, "sha256", mutate_while_hashing)
    try:
        with pytest.raises(ValueError, match="published Teacher container changed"):
            reader.read(next(iter(rows)))
    finally:
        reader.close()
