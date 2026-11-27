"""File and runtime identity for the paper acceptance runner."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from functools import cache, lru_cache
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile

RESULT_PROTOCOL = 2


def read_selection_ids(path: Path) -> tuple[str, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ids = payload["selected_landpoint_ids"]
    if not isinstance(ids, list) or any(not isinstance(site, str) or not site for site in ids):
        raise ValueError("selected_landpoint_ids must be a list of nonempty site IDs")
    return tuple(ids)


def assigned_site_ids(selection: tuple[str, ...], shard_index: int, shard_count: int, max_points: int | None) -> tuple[str, ...]:
    return selection[shard_index::shard_count][:max_points]


def digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@lru_cache(maxsize=4096)
def _small_file_digest(path: Path, stamp: tuple[int, ...]) -> str:
    with path.open("rb") as handle:
        value = hashlib.file_digest(handle, "sha256").hexdigest()
    if _stamp(path) != stamp:
        raise OSError(f"input changed during identity read: {path}")
    return value


def _stamp(path: Path) -> tuple[int, ...]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def file_identity(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    path = Path(path).resolve()
    try:
        stamp = _stamp(path)
    except FileNotFoundError:
        return {"path": str(path), "missing": True}
    if not path.is_file():
        raise ValueError(f"run input is not a file: {path}")
    if stamp[0] <= 8 * 1024 * 1024:
        return {"path": str(path), "sha256": _small_file_digest(path, stamp)}
    # Large shared input assets are treated as immutable filesystem versions.
    # Avoid reading the entire 42 GB weather archive per worker or per point.
    return {"path": str(path), "size_mtime_ctime_inode": list(stamp)}


@cache
def _source_digest() -> str:
    package = Path(__file__).resolve().parents[1]
    return digest({
        path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package.rglob("*.py"))
    })


def runtime_identity() -> dict:
    import jax

    return {
        "source_sha256": _source_digest(),
        "python": platform.python_version(),
        "platform": [platform.system(), platform.machine()],
        "packages": {name: version(name) for name in ("jax", "jaxlib", "numpy", "netCDF4", "xarray", "PyYAML")},
        "backend": jax.default_backend(),
        "x64": bool(jax.config.jax_enable_x64),
        "disable_jit": bool(jax.config.jax_disable_jit),
        "matmul_precision": jax.config.jax_default_matmul_precision,
        "xla_flags": os.environ.get("XLA_FLAGS"),
    }


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        # Keep the same directory for replacement, with a short name distinct
        # from the target even when that target is already near MAX_PATH.
        prefix = "_" if path.name.startswith(".") else "."
        with NamedTemporaryFile(dir=path.parent, prefix=prefix, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, payload) -> None:
    write_atomic(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))
