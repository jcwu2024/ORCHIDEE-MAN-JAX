"""Bounded diagnostic checkpoints, with no model or executable serialization."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path

from scripts.dev.probe_global_batch_cache import atomic_bytes, atomic_json


class StageStore:
    """Reuse only complete atomic snapshots with the declared experiment identity."""

    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "stage_manifest.json"
        self.identity = identity
        if self.path.exists():
            self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
            if self.manifest["identity"] != identity:
                raise ValueError("stage identity changed: input/science/baseline/runtime/device/compiler contract")
        else:
            self.manifest = {"version": 1, "identity": identity, "stages": {}}
        for record in self.manifest["stages"].values():
            if record["status"] == "running":
                record.update(status="stopped", reason="previous process ended without a complete checkpoint")
        self.flush()

    def flush(self):
        atomic_json(self.path, self.manifest)

    def checkpoint_path(self, name):
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in name):
            raise ValueError("invalid stage name")
        return self.directory / ("stage_" + name + ".pkl")

    def import_completed(self, directory, names):
        """Read only sealed numerical prerequisites; never open a writable source store."""
        source = Path(directory).resolve()
        if source == self.directory.resolve():
            raise ValueError("stage source must differ from destination")
        source_manifest_path = source / "stage_manifest.json"
        manifest_data = source_manifest_path.read_bytes()
        manifest = json.loads(manifest_data)
        if manifest["identity"] != self.identity:
            raise ValueError("source stage identity changed: input/science/baseline/runtime/device/compiler contract")
        # Validate the entire bounded import before publishing any destination copy.
        copies = []
        for name in names:
            target = self.checkpoint_path(name)
            record = manifest["stages"].get(name, {})
            if record.get("status") != "complete":
                raise ValueError(f"source prerequisite is not complete: {name}")
            path = source / target.name
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if record.get("sha256") != digest:
                raise ValueError(f"source stage checksum changed: {name}")
            envelope = pickle.loads(data)
            if envelope["identity"] != self.identity or envelope["name"] != name or envelope["status"] != "complete":
                raise ValueError(f"source stage checkpoint identity/status changed: {name}")
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f"destination prerequisite differs from source: {name}")
            copies.append((name, target, data, record, path))
        for name, target, data, record, path in copies:
            if not target.exists():
                atomic_bytes(target, data)
            self.manifest["stages"][name] = {
                **record, "path": target.name,
                "execution_origin": record.get("execution_origin", {
                    "available": False, "reason": "not recorded by originating harness",
                    "external_evidence": [str(source / "report_model1_origin.json"), str(source.parent / "windows")],
                }),
                "imported_from": {"path": str(path), "sha256": record["sha256"],
                                  "manifest": str(source_manifest_path),
                                  "manifest_sha256": hashlib.sha256(manifest_data).hexdigest(),
                                  "policy": "read-only complete numerical snapshot; no executable or old measurement import"},
            }
        self.flush()

    def load(self, name):
        path = self.checkpoint_path(name)
        record = self.manifest["stages"].get(name, {})
        if not path.exists():
            if record.get("status") == "complete":
                raise ValueError(f"complete stage checkpoint missing: {name}")
            return None
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if record.get("sha256") and record["sha256"] != digest:
            raise ValueError(f"stage checkpoint checksum changed: {name}")
        envelope = pickle.loads(data)
        if envelope["identity"] != self.identity or envelope["name"] != name or envelope["status"] != "complete":
            raise ValueError(f"stage checkpoint identity/status changed: {name}")
        # Recover a completed atomic snapshot if termination happened before its
        # manifest update. Incomplete temporary files are never considered.
        published_record = {**envelope["record"], "checkpoint_io": record.get("checkpoint_io", {
            "available": False, "reason": "not recorded at origin or manifest update interrupted"
        }), "execution_origin": envelope["record"].get("execution_origin", record.get("execution_origin", {
            "available": False, "reason": "not recorded by originating harness",
            "external_evidence": [str(self.directory / "report_model1_origin.json"),
                                  str(self.directory.parent / "windows"), str(self.directory / "windows")],
        }))}
        if "imported_from" in record:
            published_record["imported_from"] = record["imported_from"]
        self.manifest["stages"][name] = {
            "status": "complete", "path": path.name, "sha256": digest, **published_record
        }
        self.flush()
        return envelope["value"], published_record

    def save(self, name, value, record):
        path = self.checkpoint_path(name)
        envelope = {"identity": self.identity, "name": name, "status": "complete", "value": value, "record": record}
        started = time.perf_counter()
        data = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
        serialization_seconds = time.perf_counter() - started
        started = time.perf_counter()
        atomic_bytes(path, data)
        published_record = {**record, "checkpoint_io": {
            "available": True, "bytes": len(data), "serialization_seconds": serialization_seconds,
            "atomic_write_seconds": time.perf_counter() - started,
            "scope": "host snapshot pickle serialization and atomic write/fsync/replace; excludes manifest/report/event writes",
        }}
        self.manifest["stages"][name] = {
            "status": "complete", "path": path.name, "sha256": hashlib.sha256(data).hexdigest(), **published_record
        }
        self.flush()
        return published_record

    def run(self, name, operation):
        complete = self.load(name)
        if complete is not None:
            return complete, True
        old = self.manifest["stages"].get(name, {})
        self.manifest["stages"][name] = {"status": "running", "attempt": old.get("attempt", 0) + 1}
        self.flush()
        try:
            value, record = operation()
            published_record = self.save(name, value, record)
            return (value, published_record), False
        except (InterruptedError, KeyboardInterrupt) as exc:
            self.manifest["stages"][name].update(status="stopped", reason=str(exc))
            self.flush()
            raise
        except Exception as exc:
            self.manifest["stages"][name].update(status="failed", exception_type=type(exc).__name__, reason=str(exc))
            self.flush()
            raise
