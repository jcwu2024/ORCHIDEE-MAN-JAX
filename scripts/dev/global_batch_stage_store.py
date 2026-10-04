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
        })}
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
