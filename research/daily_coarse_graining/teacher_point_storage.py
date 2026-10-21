"""Original-byte point packing, publication, exact restoration and retirement.

No model execution. The original scientific manifests stay unchanged. All
per-member provenance/retirement progress lives inside one durable SQLite ledger.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import time
import socket

from .teacher_assets import (CHUNK, MAP_NAME, SCHEMA, PointContainer, canonical,
    close_asset_readers, fingerprint, load_storage_map, loose_sha256, point_key,
    relative_key, safe_path)

FIELDS = ("shard", "metadata", "typed_shard", "checkpoint")
YEARS = tuple(range(1961, 2011))


def file_stat(path: Path) -> dict:
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
        raise ValueError("source must be a regular single-link file: " + str(path))
    return {"dev":s.st_dev,"ino":s.st_ino,"size":s.st_size,
            "mode":stat.S_IMODE(s.st_mode),"mtime_ns":s.st_mtime_ns,"nlink":s.st_nlink}


def fsync_dir(path: Path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)


def atomic_json(path: Path, value):
    temporary = path.with_name(path.name + ".publishing")
    if temporary.exists():
        raise ValueError("unresolved publication temporary: " + str(temporary))
    with temporary.open("xb") as handle:
        handle.write(canonical(value) + b"\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)
    fsync_dir(path.parent)


class Migration:
    """One serialized maintenance owner, exact manifest-bound per-point state."""
    def __init__(self, root: str | Path, work: str | Path):
        self.root = safe_path(root)
        self.work = safe_path(work)
        if not self.root.is_dir() or self.work == self.root or self.root in self.work.parents:
            raise ValueError("maintenance directory must be outside the dataset")
        self.work.mkdir(parents=True, exist_ok=True)
        safe_path(self.work)
        self.lock = self.work / "migration.lock"
        self.dataset_lock = self.root / "teacher-storage.lock"
        self.connection = None

    def __enter__(self):
        self.dataset_lock_fd = os.open(self.dataset_lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        owner = canonical({"pid":os.getpid(),"host":socket.gethostname(),"root":str(self.root),"work":str(self.work),"time":time.time()})
        os.write(self.dataset_lock_fd, owner); os.fsync(self.dataset_lock_fd); fsync_dir(self.root)
        try:
            self.lock_fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(self.lock_fd, owner); os.fsync(self.lock_fd)
            self.connection = sqlite3.connect(self.work / "migration.sqlite")
            self.connection.execute("PRAGMA journal_mode=DELETE")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
            self.connection.execute("CREATE TABLE IF NOT EXISTS points(point TEXT PRIMARY KEY,record TEXT NOT NULL)")
            self.connection.execute("CREATE TABLE IF NOT EXISTS retirement(point TEXT,path TEXT,status TEXT NOT NULL,PRIMARY KEY(point,path))")
            self.connection.execute("CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,utc TEXT,point TEXT,stage TEXT,detail TEXT)")
            self.manifest_identity = fingerprint(self.root / "dataset_manifest.json")
            self.manifest_sha256 = loose_sha256(self.root / "dataset_manifest.json")
            self.manifest = json.loads((self.root / "dataset_manifest.json").read_bytes())
            if fingerprint(self.root / "dataset_manifest.json") != self.manifest_identity:
                raise ValueError("scientific manifest changed during admission")
            if self.manifest.get("status") != "complete" or self.manifest.get("provisional_teacher") is not False:
                raise ValueError("only a complete nonprovisional release may migrate")
            binding = {"root":str(self.root),"dataset_manifest_sha256":self.manifest_sha256,"schema_version":SCHEMA}
            old = dict(self.connection.execute("SELECT key,value FROM metadata"))
            if old and old != binding:
                raise ValueError("migration ledger belongs to another root/release")
            if not old:
                self.connection.executemany("INSERT INTO metadata VALUES (?,?)", binding.items())
            self.connection.commit()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        if self.connection is not None:
            self.connection.close()
        if hasattr(self, "lock_fd"):
            os.close(self.lock_fd); self.lock.unlink(); fsync_dir(self.work)
        os.close(self.dataset_lock_fd); self.dataset_lock.unlink(); fsync_dir(self.root)

    def event(self, point: str, stage: str, detail):
        self.connection.execute("INSERT INTO events(utc,point,stage,detail) VALUES (?,?,?,?)",
            (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),point,stage,canonical(detail).decode()))
        self.connection.commit()

    def record(self, point: str) -> dict | None:
        row = self.connection.execute("SELECT record FROM points WHERE point=?", (point,)).fetchone()
        return None if row is None else json.loads(row[0])

    def update(self, point: str, record: dict, stage: str):
        record["stage"] = stage
        self.connection.execute("INSERT OR REPLACE INTO points VALUES (?,?)", (point, canonical(record).decode()))
        self.connection.commit()
        self.event(point, stage, {"snapshot_sha256":record.get("snapshot_sha256"),"source_files":len(record.get("rows", {}))})

    def assert_release(self):
        if fingerprint(self.root / "dataset_manifest.json") != self.manifest_identity:
            raise ValueError("scientific manifest identity changed during maintenance")

    def inventory(self, point: str) -> dict:
        point_key(point)
        records = sorted((r for r in self.manifest["shards"] if r["landpoint_id"] == point), key=lambda r:r["year"])
        if [r["year"] for r in records] != list(YEARS):
            raise ValueError("point must contain exactly every year 1961–2010")
        rows, directories, workers, expected = {}, {}, {}, {}
        for record in records:
            for field in FIELDS:
                name = relative_key(record[field])
                if not name.startswith("workers/") or name in expected:
                    raise ValueError("duplicate or unsupported source member")
                expected[name] = record[field + "_sha256"]
                worker_root = "/".join(name.split("/")[:2])
                workers.setdefault(worker_root, None)
        if len(expected) != 200:
            raise ValueError("point needs 50 years × four original objects")
        manifest_refs = {r["manifest"]:r["manifest_sha256"] for r in self.manifest["worker_manifests"]}
        for worker in workers:
            name = worker + "/manifest.json"
            path = safe_path(self.root / name)
            h = loose_sha256(path)
            if manifest_refs.get(name) != h:
                raise ValueError("worker manifest/release hash mismatch")
            raw = json.loads(path.read_bytes())
            if not raw["complete"] or len(raw["assigned_entries"]) != len(raw["completed_entries"]) or set(raw["assigned_entries"]) != set(raw["completed_entries"]):
                raise ValueError("worker completion keys differ")
            by_key = {(r["landpoint_id"],r["year"]):r for r in raw["shards"]}
            for record in records:
                if not record["shard"].startswith(worker + "/"):
                    continue
                wr = by_key[(point, record["year"])]
                for field in FIELDS:
                    if worker + "/" + wr[field] != record[field] or wr[field + "_sha256"] != record[field + "_sha256"]:
                        raise ValueError("worker/root member binding drift")
                for field in ("spatial_split","temporal_split","preceding_checkpoint_sha256","input_hashes","data_release_id","data_release_sha256","teacher_source_sha256","typed_contract_sha256"):
                    if field in record and wr.get(field) != record[field]:
                        raise ValueError("worker/root scientific identity drift " + field)
            workers[worker] = {"manifest":name,"sha256":h,"stat":file_stat(path)}
        for name, h in sorted(expected.items()):
            path = safe_path(self.root / name)
            row = file_stat(path)
            row["sha256"] = h  # actual bytes checked while building; no redundant source hash pass
            rows[name] = row
            for parent in path.parents:
                if parent == self.root:
                    break
                rel = parent.relative_to(self.root).as_posix()
                s = parent.lstat()
                if not stat.S_ISDIR(s.st_mode):
                    raise ValueError("non-directory source parent")
                directories[rel] = {"mode":stat.S_IMODE(s.st_mode),"mtime_ns":s.st_mtime_ns}
        self.assert_release()
        return {"landpoint_id":point,"dataset_manifest_sha256":self.manifest_sha256,
                "rows":rows,"directories":directories,"worker_manifests":workers,
                "snapshot_sha256":hashlib.sha256(canonical(rows)).hexdigest(),
                "records":records,"source_bytes":sum(r["size"] for r in rows.values())}

    def _verify_pack(self, path: Path, record: dict, full=True) -> dict:
        initial_identity = fingerprint(path)
        reader = PointContainer(path)
        try:
            if (reader.point != record["landpoint_id"] or reader.manifest_sha256 != self.manifest_sha256
                    or reader.rows != record["rows"] or reader.snapshot_sha256 != record["snapshot_sha256"]):
                raise ValueError("pack/source snapshot mismatch")
            if reader.connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise ValueError("point SQLite integrity failure")
            if full:
                for name in reader.rows:
                    reader.read(name)
                chain = None
                for r in record["records"]:
                    metadata = json.loads(reader.read(r["metadata"])[0])
                    for field in ("landpoint_id","year","spatial_split","temporal_split","input_hashes","data_release_id","data_release_sha256","teacher_source_sha256","typed_contract_sha256"):
                        if field in r and metadata.get(field) != r[field]:
                            raise ValueError("annual metadata scientific drift " + field)
                    for field in ("shard","typed_shard","checkpoint"):
                        if metadata[field + "_sha256"] != r[field + "_sha256"]:
                            raise ValueError("annual metadata payload hash drift")
                    if metadata.get("preceding_checkpoint_sha256") != chain or r.get("preceding_checkpoint_sha256") != chain:
                        raise ValueError("checkpoint chain drift")
                    chain = r["checkpoint_sha256"]
        finally:
            reader.close()
        actual_hash = loose_sha256(path)
        if fingerprint(path) != initial_identity:
            raise ValueError("container identity changed during verification")
        return {"bytes":path.stat().st_size,"sha256":actual_hash,
                "all_200_member_hashes_verified":full,"sqlite_integrity":"ok",
                "verified_fingerprint":initial_identity}

    def pack(self, point: str) -> dict:
        point_key(point)
        record = self.record(point)
        destination = safe_path(self.root / "points" / (point + ".sqlite"))
        stage = safe_path(self.work / (point + ".building.sqlite"))
        if record is None:
            record = self.inventory(point)
            if stage.exists() or destination.exists():
                raise ValueError("unowned preexisting stage/container")
            self.update(point, record, "planned")
        if record.get("container") is not None:
            path = destination if destination.exists() else stage
            verified = self._verify_pack(path, record)
            if verified["sha256"] != record["container"]["sha256"]:
                raise ValueError("existing pack differs from durable verified record")
        else:
            if destination.exists():
                raise ValueError("published pack without a verified ledger binding")
            if stage.exists():
                # Crash after completed build but before ledger update: fully verify and reuse.
                verified = self._verify_pack(stage, record)
            else:
                fd = os.open(stage, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600); os.close(fd)
                c = sqlite3.connect(stage)
                try:
                    c.execute("PRAGMA journal_mode=DELETE"); c.execute("PRAGMA synchronous=FULL")
                    c.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
                    c.execute("CREATE TABLE assets(path TEXT PRIMARY KEY,size INTEGER NOT NULL,mode INTEGER NOT NULL,mtime_ns INTEGER NOT NULL,sha256 TEXT NOT NULL,data BLOB NOT NULL)")
                    meta = {"schema_version":SCHEMA,"landpoint_id":point,"dataset_manifest_sha256":self.manifest_sha256,
                            "source_rows":canonical(record["rows"]).decode(),"snapshot_sha256":record["snapshot_sha256"],
                            "source_directories":canonical(record["directories"]).decode(),"complete":"false"}
                    c.executemany("INSERT INTO metadata VALUES (?,?)", meta.items())
                    for name, row in record["rows"].items():
                        path = safe_path(self.root / name)
                        if file_stat(path) != {k:row[k] for k in file_stat(path)}:
                            raise ValueError("source stat drift before pack " + name)
                        cur = c.execute("INSERT INTO assets VALUES (?,?,?,?,?,zeroblob(?))",
                            (name,row["size"],row["mode"],row["mtime_ns"],row["sha256"],row["size"]))
                        h = hashlib.sha256()
                        with c.blobopen("assets","data",cur.lastrowid,readonly=False) as blob, path.open("rb") as src:
                            for chunk in iter(lambda:src.read(CHUNK), b""):
                                h.update(chunk); blob.write(chunk)
                        if h.hexdigest() != row["sha256"] or file_stat(path) != {k:row[k] for k in file_stat(path)}:
                            raise ValueError("source bytes/identity drift while packing " + name)
                        c.commit()
                    c.execute("UPDATE metadata SET value='true' WHERE key='complete'"); c.commit()
                finally:
                    c.close()
                # Windows fsync requires a writable descriptor; no content changes.
                with stage.open("r+b") as handle: os.fsync(handle.fileno())
                os.chmod(stage, 0o444)
                fsync_dir(stage.parent)
                verified = self._verify_pack(stage, record)
            record["container"] = {k:v for k,v in verified.items() if k != "verified_fingerprint"}
            self.update(point, record, "verified")
        self.assert_release()
        points_dir = destination.parent
        if not points_dir.exists():
            points_dir.mkdir(); fsync_dir(points_dir.parent)
        safe_path(points_dir)
        if not destination.exists():
            # Windows rename refuses an existing destination. POSIX rename can
            # overwrite, so use exclusive link publication there instead.
            if os.name == "nt":
                os.rename(stage, destination)
            else:
                os.link(stage, destination); fsync_dir(points_dir)
                stage.unlink()
            fsync_dir(points_dir); fsync_dir(self.work)
        elif stage.exists():
            if not os.path.samefile(stage, destination):
                raise ValueError("stage/published identity conflict")
            stage.unlink(); fsync_dir(self.work)
        published = load_storage_map(self.root)
        mapping = {"schema_version":SCHEMA,"dataset_manifest_sha256":self.manifest_sha256,"points":{}} if published is None else published[0]
        entry = {**record["container"],"container":"points/" + point + ".sqlite",
                 "snapshot_sha256":record["snapshot_sha256"],"member_count":len(record["rows"]),"members":sorted(record["rows"])}
        old = mapping["points"].get(point)
        if old is not None and old != entry:
            raise ValueError("published map contains different point content")
        if old is None:
            mapping = json.loads(canonical(mapping)); mapping["points"][point] = entry
            atomic_json(self.root / MAP_NAME, mapping)
        # Preserve completed proof/retirement state across an idempotent pack replay.
        self.update(point, record, record["stage"] if record["stage"] in {"reader_verified","retiring","retired","pinned"} else "published")
        return record

    def restore(self, point: str, destination: str | Path, *, original_paths=False) -> dict:
        record = self.record(point)
        if record is None or "container" not in record:
            raise ValueError("no verified point to restore")
        target = safe_path(destination)
        if original_paths:
            if target != self.root:
                raise ValueError("original restoration root mismatch")
        else:
            if target.exists():
                raise FileExistsError(target)
            target.mkdir(parents=True)
        reader = PointContainer(self.root / "points" / (point + ".sqlite"))
        written, retained = [], []
        try:
            for name, row in record["rows"].items():
                path = safe_path(target / name)
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    if not original_paths or loose_sha256(path) != row["sha256"] or any(file_stat(path)[k] != row[k] for k in ("size","mode","mtime_ns")):
                        raise ValueError("restore destination conflict " + name)
                    retained.append(name); continue
                payload, _ = reader.read(name)
                with path.open("xb") as handle:
                    handle.write(payload); handle.flush(); os.fsync(handle.fileno())
                os.chmod(path,row["mode"]); os.utime(path,ns=(row["mtime_ns"],row["mtime_ns"]))
                written.append(name)
            if not original_paths:
                for name, row in sorted(record["directories"].items(), key=lambda x:x[0].count("/"), reverse=True):
                    path = safe_path(target / name)
                    os.chmod(path,row["mode"]); os.utime(path,ns=(row["mtime_ns"],row["mtime_ns"]))
            for name, row in record["rows"].items():
                path = safe_path(target / name)
                if loose_sha256(path) != row["sha256"] or any(file_stat(path)[k] != row[k] for k in ("size","mode","mtime_ns")):
                    raise ValueError("restored byte/metadata mismatch " + name)
        finally:
            reader.close()
        proof = {"landpoint_id":point,"destination":str(target),"members":len(record["rows"]),
                 "written":written,"retained_identical":retained,"content_mode_nsmtime_equal":True,
                 "container_sha256":record["container"]["sha256"],"snapshot_sha256":record["snapshot_sha256"]}
        record["restore_proof"] = proof
        self.update(point, record, record["stage"])
        return proof

    def accept_reader_proof(self, point: str, proof_path: str | Path) -> dict:
        record = self.record(point)
        if record is None or record["stage"] not in {"published","reader_verified"}:
            raise ValueError("point not published")
        proof_path = safe_path(proof_path)
        proof = json.loads(proof_path.read_bytes())
        if (proof.get("status") != "PRODUCTION_POINT_READERS_AND_ABSENT_LOOSE_PASS"
                or proof.get("landpoint_id") != point or proof.get("container_sha256") != record["container"]["sha256"]
                or proof.get("snapshot_sha256") != record["snapshot_sha256"] or not record.get("restore_proof",{}).get("content_mode_nsmtime_equal")):
            raise ValueError("missing/mismatched real reader or restoration proof")
        record["reader_proof"] = {"path":str(proof_path),"sha256":loose_sha256(proof_path),"result":proof}
        self.update(point, record, "reader_verified")
        return record

    def retire(self, point: str, *, review_token: str, dependencies: str | Path) -> dict:
        """Only exact previously verified objects; durable per-file intent before unlink."""
        record = self.record(point)
        if not review_token or record is None or record["stage"] not in {"reader_verified","retiring","retired","pinned"}:
            raise ValueError("reviewed real-reader proof required before source retirement")
        dependency_path = safe_path(dependencies)
        dependency = json.loads(dependency_path.read_bytes())
        if (dependency.get("dataset_manifest_sha256") != self.manifest_sha256
                or dependency.get("active_queued_audit_complete") is not True
                or dependency.get("expires_unix",0) < time.time()):
            raise ValueError("current manifest-bound active/queued dependency audit required")
        pins = set(dependency["preserve_paths"])
        for name in pins: relative_key(name)
        record["review_token"] = review_token
        record["dependency_audit"] = {"path":str(dependency_path),"sha256":loose_sha256(dependency_path)}
        pack_path = self.root / "points" / (point + ".sqlite")
        verified = self._verify_pack(pack_path, record)
        if verified["sha256"] != record["container"]["sha256"]:
            raise ValueError("actual verified container SHA differs from ledger")
        pack_identity = verified["verified_fingerprint"]
        map_path = self.root / MAP_NAME
        map_identity = fingerprint(map_path)
        published = load_storage_map(self.root)
        if (published is None or point not in published[0]["points"]
                or published[0]["points"][point]["snapshot_sha256"] != record["snapshot_sha256"]
                or published[0]["points"][point]["sha256"] != verified["sha256"]
                or set(published[0]["points"][point]["members"]) != set(record["rows"])
                or fingerprint(map_path) != map_identity):
            raise ValueError("point not durably published")
        def assert_storage_live():
            try:
                stable = fingerprint(pack_path) == pack_identity and fingerprint(map_path) == map_identity
            except FileNotFoundError:
                stable = False
            if not stable:
                raise ValueError("verified container/storage map changed before unlink")
        self.assert_release()
        self.update(point, record, "retiring")
        reader = PointContainer(self.root / "points" / (point + ".sqlite"))
        try:
            for name, row in record["rows"].items():
                if name in pins: continue
                assert_storage_live()
                path = safe_path(self.root / name)
                status = self.connection.execute("SELECT status FROM retirement WHERE point=? AND path=?",(point,name)).fetchone()
                if not path.exists():
                    if status is None or status[0] not in {"intent","removed"}:
                        raise ValueError("source missing without durable retirement intent " + name)
                    reader.read(name)
                else:
                    if status is not None and status[0] == "removed":
                        raise ValueError("retired source was recreated; inspect before proceeding " + name)
                    if file_stat(path) != {k:row[k] for k in file_stat(path)} or loose_sha256(path) != row["sha256"]:
                        raise ValueError("source identity/hash changed before retirement " + name)
                    self.connection.execute("INSERT OR REPLACE INTO retirement VALUES (?,?,?)",(point,name,"intent")); self.connection.commit()
                    # Recheck immediately; exact leaf unlink only, no recursive deletion.
                    if file_stat(path) != {k:row[k] for k in file_stat(path)}:
                        raise ValueError("source identity changed at unlink " + name)
                    assert_storage_live()
                    path.unlink(); fsync_dir(path.parent)
                self.connection.execute("UPDATE retirement SET status='removed' WHERE point=? AND path=?",(point,name)); self.connection.commit()
        finally:
            reader.close()
        removed = [r[0] for r in self.connection.execute("SELECT path FROM retirement WHERE point=? AND status='removed' ORDER BY path",(point,))]
        assert_storage_live()
        record["retirement_pack_identity"] = list(pack_identity)
        record["retirement_map_identity"] = list(map_identity)
        record["retired_paths"] = removed; record["pinned_paths"] = sorted(pins.intersection(record["rows"]))
        self.update(point, record, "pinned" if record["pinned_paths"] else "retired")
        return {"landpoint_id":point,"removed_files":len(removed),"retained_pinned_files":len(record["pinned_paths"]),
                "source_bytes_retired":sum(record["rows"][n]["size"] for n in removed),
                "point_pack_files_added":1,"point_only_net_file_reduction":len(removed)-1,
                "shared_map_ledger_code_receipts_not_deducted_here":True}

    def discard_incomplete_stage(self, point: str, expected_stage_sha256: str) -> dict:
        """Explicit repair only after every original object is proved intact."""
        record = self.record(point)
        if record is None or record["stage"] != "planned" or "container" in record:
            raise ValueError("only an unverified planned stage may be discarded")
        stage = safe_path(self.work / (point_key(point) + ".building.sqlite"))
        identity = fingerprint(stage)
        if file_stat(stage)["nlink"] != 1 or loose_sha256(stage) != expected_stage_sha256:
            raise ValueError("partial stage identity/hash mismatch")
        # This bounded point rehash is required only for explicit crash repair.
        for name,row in record["rows"].items():
            path = safe_path(self.root/name)
            actual = file_stat(path)
            if actual != {k:row[k] for k in actual} or loose_sha256(path) != row["sha256"]:
                raise ValueError("cannot discard stage while an original differs/missing")
        try:
            self._verify_pack(stage,record)
        except (sqlite3.Error,ValueError,KeyError):
            pass
        else:
            raise ValueError("complete valid stage must be reused, not discarded")
        sidecars = []
        for suffix in ("-journal","-wal","-shm"):
            p = safe_path(str(stage)+suffix)
            if p.exists():sidecars.append((p,fingerprint(p)))
        if fingerprint(stage) != identity:raise ValueError("stage changed during repair")
        for p,owned in sidecars:
            if fingerprint(p) != owned:raise ValueError("stage sidecar changed during repair")
            p.unlink()
        if os.name == "nt":os.chmod(stage,0o600)
        stage.unlink();fsync_dir(self.work)
        self.event(point,"discarded_incomplete_stage",{"stage_sha256":expected_stage_sha256,"all_200_originals_intact":True})
        return {"stage_removed":str(stage),"sidecars_removed":[str(p) for p,_ in sidecars],"source_files_removed":0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root",type=Path,required=True)
    parser.add_argument("--work-dir",type=Path,required=True)
    sub = parser.add_subparsers(dest="action",required=True)
    sub.add_parser("plan")
    for name in ("pack","verify","restore","prove","accept-reader-proof","retire","status","discard-incomplete-stage"):
        p = sub.add_parser(name); p.add_argument("--point",required=True)
        if name == "restore":
            p.add_argument("--destination",type=Path,required=True);p.add_argument("--original-paths",action="store_true")
        if name == "accept-reader-proof":p.add_argument("--proof",type=Path,required=True)
        if name == "prove":
            p.add_argument("--proof",type=Path,required=True);p.add_argument("--scan-full-index",action="store_true")
        if name == "discard-incomplete-stage":p.add_argument("--stage-sha256",required=True)
        if name == "retire":
            p.add_argument("--review-token",required=True);p.add_argument("--dependencies",type=Path,required=True)
    args = parser.parse_args(argv)
    with Migration(args.dataset_root,args.work_dir) as migration:
        if args.action == "plan":
            points = sorted({r["landpoint_id"] for r in migration.manifest["shards"]})
            result = {"points":points,"point_count":len(points),"annual_records":len(migration.manifest["shards"]),
                      "source_objects":len(migration.manifest["shards"])*4,"dataset_manifest_sha256":migration.manifest_sha256}
        elif args.action == "pack":result = migration.pack(args.point)
        elif args.action == "verify":
            record = migration.record(args.point)
            result = migration._verify_pack(migration.root / "points" / (point_key(args.point) + ".sqlite"),record)
        elif args.action == "restore":result = migration.restore(args.point,args.destination,original_paths=args.original_paths)
        elif args.action == "prove":
            from .teacher_point_verification import prove_point
            if args.proof.exists():raise FileExistsError(args.proof)
            result = prove_point(migration,args.point,scan_full_index=args.scan_full_index)
            atomic_json(safe_path(args.proof),result)
            migration.accept_reader_proof(args.point,args.proof)
        elif args.action == "discard-incomplete-stage":result = migration.discard_incomplete_stage(args.point,args.stage_sha256)
        elif args.action == "accept-reader-proof":result = migration.accept_reader_proof(args.point,args.proof)
        elif args.action == "retire":result = migration.retire(args.point,review_token=args.review_token,dependencies=args.dependencies)
        else:result = migration.record(args.point)
    print(json.dumps(result,sort_keys=True,separators=(",",":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
