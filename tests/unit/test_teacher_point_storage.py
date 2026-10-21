"""Byte-preserving point migration, failure boundaries and restart evidence."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import pytest

from research.daily_coarse_graining.teacher_assets import (
    MAP_NAME, PointContainer, asset_is_file, asset_sha256, close_asset_readers,
    open_asset, relative_key,
)
from research.daily_coarse_graining.teacher_point_storage import Migration, file_stat


def sha(data): return hashlib.sha256(data).hexdigest()


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "release"; root.mkdir()
    work = tmp_path / "maintenance"
    point = "087.0-109.0"; worker = "workers/worker-000-of-001"
    records = []; previous = None
    for year in range(1961,2011):
        record = {"landpoint_id":point,"year":year,"spatial_split":"train","temporal_split":"train",
                  "preceding_checkpoint_sha256":previous}
        for kind, suffix in (("shard","npz"),("typed_shard","npz"),("checkpoint","pkl")):
            name = f"{worker}/{kind}/{point}_{year}.{suffix}"
            raw = (kind+str(year)).encode()*31
            path = root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
            record[kind] = name; record[kind+"_sha256"] = sha(raw)
        name = f"{worker}/metadata/{point}_{year}.json"
        raw = json.dumps(record,sort_keys=True).encode()
        path = root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
        record["metadata"] = name; record["metadata_sha256"] = sha(raw)
        previous = record["checkpoint_sha256"]
        records.append(record)
    keys = [point+":"+str(y) for y in range(1961,2011)]
    worker_records = []
    for record in records:
        r = dict(record)
        for field in ("shard","metadata","typed_shard","checkpoint"):
            r[field] = r[field].removeprefix(worker+"/")
        worker_records.append(r)
    raw = json.dumps({"complete":True,"assigned_entries":keys,"completed_entries":keys,"shards":worker_records}).encode()
    (root/worker/"manifest.json").write_bytes(raw)
    manifest = {"schema_version":"daily_teacher_dataset_manifest_v6","status":"complete","provisional_teacher":False,
                "worker_manifests":[{"manifest":worker+"/manifest.json","manifest_sha256":sha(raw)}],"shards":records}
    (root/"dataset_manifest.json").write_text(json.dumps(manifest))
    yield root,work,point
    close_asset_readers()


def proof(migration, point, tmp_path):
    record = migration.record(point)
    p = tmp_path/"reader-proof.json"
    p.write_text(json.dumps({"status":"PRODUCTION_POINT_READERS_AND_ABSENT_LOOSE_PASS","landpoint_id":point,
                           "container_sha256":record["container"]["sha256"],"snapshot_sha256":record["snapshot_sha256"]}))
    migration.accept_reader_proof(point,p)
    d = tmp_path/"dependencies.json"
    d.write_text(json.dumps({"dataset_manifest_sha256":migration.manifest_sha256,
                            "active_queued_audit_complete":True,"expires_unix":time.time()+300,"preserve_paths":[]}))
    return d


def test_complete_original_bytes_restore_and_idempotent_publication(release,tmp_path):
    root,work,point = release
    before = {p.relative_to(root).as_posix():(file_stat(p),sha(p.read_bytes())) for p in root.rglob("*") if p.is_file()}
    with Migration(root,work) as m:
        record = m.pack(point)
        assert len(record["rows"]) == 200
        db = root/"points"/(point+".sqlite"); identity = db.stat().st_ino
        assert m.pack(point)["container"] == record["container"]
        assert db.stat().st_ino == identity
        for name,row in record["rows"].items():
            assert asset_is_file(root/name)
            assert asset_sha256(root/name) == row["sha256"]
            with open_asset(root/name) as h: assert sha(h.read()) == row["sha256"]
        result = m.restore(point,tmp_path/"restore")
        assert len(result["written"]) == 200 and result["content_mode_nsmtime_equal"]
        with pytest.raises(FileExistsError):m.restore(point,tmp_path/"restore")
    for name,(identity,digest) in before.items():
        assert file_stat(root/name) == identity and sha((root/name).read_bytes()) == digest
    assert not list(root.glob("points/*-journal"))


def test_refuse_source_drift_hardlinks_and_unsafe_paths(release,tmp_path):
    root,work,point = release
    with Migration(root,work) as m:
        row = m.inventory(point); name = next(iter(row["rows"]))
        m.update(point,row,"planned")
        path = root/name; path.write_bytes(b"changed")
        with pytest.raises(ValueError,match="source stat drift"):m.pack(point)
    for name in ("../x","/x","a//b","a/./b","a\\b","C:/x"):
        with pytest.raises(ValueError):relative_key(name)


def test_single_link_source_requirement(release,tmp_path):
    root,work,point = release
    source = next((root/"workers").rglob("*.npz"))
    os.link(source,tmp_path/"duplicate")
    with Migration(root,work) as m:
        with pytest.raises(ValueError,match="single-link"):m.pack(point)


def test_retire_requires_proof_dependencies_and_preserves_pins(release,tmp_path):
    root,work,point = release
    with Migration(root,work) as m:
        record = m.pack(point)
        with pytest.raises(ValueError,match="reviewed real-reader"):m.retire(point,review_token="",dependencies=tmp_path/"none")
        m.restore(point,tmp_path/"restore")
        deps = proof(m,point,tmp_path)
        pinned = next(iter(record["rows"]))
        d = json.loads(deps.read_bytes()); d["preserve_paths"] = [pinned];deps.write_text(json.dumps(d))
        result = m.retire(point,review_token="fixture-review",dependencies=deps)
        assert result["removed_files"] == 199 and result["retained_pinned_files"] == 1
        assert (root/pinned).is_file()
        assert sum((root/name).is_file() for name in record["rows"]) == 1
        assert m.retire(point,review_token="fixture-review",dependencies=deps) == result
        # Reader remains usable with 199 exact original objects physically absent.
        for name,row in record["rows"].items():assert asset_sha256(root/name) == row["sha256"]
        restored = m.restore(point,root,original_paths=True)
        assert len(restored["written"]) == 199 and len(restored["retained_identical"]) == 1


def test_interrupted_unlink_reconciles_only_with_durable_intent(release,tmp_path):
    root,work,point = release
    with Migration(root,work) as m:
        record = m.pack(point);m.restore(point,tmp_path/"restore")
        deps = proof(m,point,tmp_path)
        name = next(iter(record["rows"]))
        m.connection.execute("INSERT INTO retirement VALUES (?,?,?)",(point,name,"intent"));m.connection.commit()
        (root/name).unlink()  # interruption after unlink, before completion commit
        assert m.retire(point,review_token="fixture-review",dependencies=deps)["removed_files"] == 200
        assert m.retire(point,review_token="fixture-review",dependencies=deps)["removed_files"] == 200


def test_tampered_payload_or_release_rejected(release):
    root,work,point = release
    with Migration(root,work) as m:
        record = m.pack(point)
    close_asset_readers()
    db = root/"points"/(point+".sqlite");os.chmod(db,0o600)
    name,row = next(iter(record["rows"].items()))
    with sqlite3.connect(db) as c:c.execute("UPDATE assets SET data=? WHERE path=?",(b"x"*row["size"],name))
    with pytest.raises(ValueError,match="hash mismatch"):asset_sha256(root/name)
    close_asset_readers()
    manifest = root/"dataset_manifest.json";manifest.write_bytes(manifest.read_bytes()+b" ")
    with pytest.raises(ValueError,match="release hash mismatch"):asset_is_file(root/name)


def test_dataset_lock_rejects_another_work_directory(release,tmp_path):
    root,work,point = release
    with Migration(root,work):
        owner = (root/"teacher-storage.lock").read_bytes()
        with pytest.raises(FileExistsError):
            with Migration(root,tmp_path/"different-maintenance"):pass
        assert (root/"teacher-storage.lock").read_bytes() == owner
    assert not (root/"teacher-storage.lock").exists()


@pytest.mark.parametrize("mutation",["remove-pack","replace-pack","replace-map"])
def test_changed_pack_or_map_after_verification_preserves_sources(release,tmp_path,monkeypatch,mutation):
    root,work,point = release
    with Migration(root,work) as m:
        record = m.pack(point);m.restore(point,tmp_path/"restore")
        deps = proof(m,point,tmp_path)
        original_update = m.update
        def update(p,r,stage):
            original_update(p,r,stage)
            if stage != "retiring":return
            db = root/"points"/(point+".sqlite")
            if mutation == "remove-pack":
                os.chmod(db,0o600);db.unlink()
            elif mutation == "replace-pack":
                replacement = db.with_suffix(".replacement");replacement.write_bytes(db.read_bytes())
                os.chmod(db,0o600);db.unlink();replacement.rename(db)
            else:
                path = root/MAP_NAME
                raw = json.loads(path.read_bytes());raw["points"] = {}
                path.write_text(json.dumps(raw))
        monkeypatch.setattr(m,"update",update)
        with pytest.raises((ValueError,FileNotFoundError)):
            m.retire(point,review_token="fixture-review",dependencies=deps)
        assert all((root/name).is_file() for name in record["rows"])
        assert m.connection.execute("SELECT COUNT(*) FROM retirement").fetchone()[0] == 0


def test_actual_verified_pack_hash_must_match_ledger_before_retirement(release,tmp_path):
    root,work,point = release
    with Migration(root,work) as m:
        record = m.pack(point);m.restore(point,tmp_path/"restore")
        deps = proof(m,point,tmp_path)
        db = root/"points"/(point+".sqlite");os.chmod(db,0o600)
        with db.open("ab") as f:f.write(b"unbound-trailing-bytes")
        with pytest.raises(ValueError,match="SHA differs"):
            m.retire(point,review_token="fixture-review",dependencies=deps)
        assert all((root/name).is_file() for name in record["rows"])


def test_discard_partial_stage_requires_hash_and_intact_originals(release):
    root,work,point = release
    with Migration(root,work) as m:
        record = m.inventory(point);m.update(point,record,"planned")
        stage = work/(point+".building.sqlite");stage.write_bytes(b"interrupted SQLite build")
        with pytest.raises(ValueError,match="identity/hash"):
            m.discard_incomplete_stage(point,"0"*64)
        name = next(iter(record["rows"]));source = root/name
        data = source.read_bytes();original = source.stat();source.write_bytes(b"drift")
        with pytest.raises(ValueError,match="original differs"):
            m.discard_incomplete_stage(point,sha(stage.read_bytes()))
        assert stage.exists()
        source.write_bytes(data);os.utime(source,ns=(original.st_atime_ns,original.st_mtime_ns))
        result = m.discard_incomplete_stage(point,sha(stage.read_bytes()))
        assert not stage.exists() and result["source_files_removed"] == 0
        assert all((root/name).is_file() for name in record["rows"])


def test_complete_stage_is_never_discarded_as_partial(release):
    root,work,point = release
    with Migration(root,work) as m:
        record = m.pack(point)
        stage = work/(point+".building.sqlite")
        stage.write_bytes((root/"points"/(point+".sqlite")).read_bytes())
        del record["container"];m.update(point,record,"planned")
        with pytest.raises(ValueError,match="complete valid stage"):
            m.discard_incomplete_stage(point,sha(stage.read_bytes()))
        assert stage.exists()
