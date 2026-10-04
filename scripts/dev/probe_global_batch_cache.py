"""Small, separate-process persistent-cache/device probe; no physical model data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import sys
import time
import uuid
from pathlib import Path

PROCESS_STARTED = time.perf_counter()


def atomic_bytes(path, data):
    """Publish a complete file in the same filesystem, preserving its predecessor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2) + "\n").encode("utf-8"))


class CacheEvents(logging.Handler):
    """Observe JAX's actual cache-key diagnostics; elapsed time is not hit evidence."""

    def __init__(self):
        super().__init__()
        self.events = []
        self.previous = []
        for name in ("jax._src.compiler", "jax._src.compilation_cache"):
            logger = logging.getLogger(name)
            self.previous.append((logger, logger.level, logger.propagate))
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            logger.addHandler(self)

    def emit(self, record):
        message = record.getMessage()
        if "cache" in message.lower():
            self.events.append({"logger": record.name, "message": message})

    def close(self):
        for logger, level, propagate in self.previous:
            logger.removeHandler(self)
            logger.setLevel(level)
            logger.propagate = propagate
        super().close()


def observed_cache_hits(events):
    return [event for event in events if "persistent compilation cache hit" in event["message"].lower()]


def configure_cache(mode, root):
    import jax

    if mode == "off":
        os.environ["ORCHJAX_DISABLE_COMPILATION_CACHE"] = "1"
        jax.config.update("jax_enable_compilation_cache", False)
        return {"mode": "off", "root": None}
    if root is None:
        raise ValueError("persistent mode requires an explicit --cache-root")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ["ORCHJAX_DISABLE_COMPILATION_CACHE"] = "0"
    jax.config.update("jax_enable_compilation_cache", True)
    jax.config.update("jax_compilation_cache_dir", str(root))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return {"mode": "persistent", "root": str(root), "min_compile_time_seconds": 0, "min_entry_bytes": 0}


def probe_transition(x):
    from jax import lax
    from jax import numpy as jnp

    return lax.fori_loop(0, 8, lambda i, value: jnp.tanh(value @ x * 0.01 + i * 0.001), x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--cache-mode", choices=["off", "persistent"], default="persistent")
    parser.add_argument("--cache-phase", choices=["write", "read"], required=True)
    args = parser.parse_args()
    import jax
    import jaxlib
    import numpy as np

    jax.config.update("jax_enable_x64", True)
    policy = configure_cache(args.cache_mode, args.cache_root)
    observer = CacheEvents()
    result = {
        "status": "running", "cache_phase": args.cache_phase, "cache_policy": policy,
        "python": sys.version, "jax": jax.__version__, "jaxlib": jaxlib.__version__,
        "numpy": np.__version__, "hostname": platform.node(),
        "devices": [{"platform": d.platform, "kind": d.device_kind, "id": d.id} for d in jax.devices()],
        "x64_enabled": bool(jax.config.jax_enable_x64),
        "backend_platform_version": str(jax.devices()[0].client.platform_version),
        "import_setup_seconds": time.perf_counter() - PROCESS_STARTED,
        "pid": os.getpid(), "xla_flags": os.environ.get("XLA_FLAGS"),
    }
    try:
        x = jax.device_put(np.arange(4096, dtype=np.float64).reshape(64, 64) / 4096)
        x.block_until_ready()
        start = time.perf_counter()
        lowered = jax.jit(probe_transition).lower(x)
        result["lower_seconds"] = time.perf_counter() - start
        result["stablehlo_sha256"] = hashlib.sha256(str(lowered.compiler_ir(dialect="stablehlo")).encode()).hexdigest()
        event_start = len(observer.events)
        start = time.perf_counter()
        executable = lowered.compile()
        result["compile_seconds"] = time.perf_counter() - start
        result["compile_cache_events"] = observer.events[event_start:]
        actual = np.asarray(executable(x).block_until_ready())
        result["output"] = {"shape": list(actual.shape), "dtype": str(actual.dtype),
                            "finite": bool(np.isfinite(actual).all()),
                            "sha256": hashlib.sha256(actual.tobytes()).hexdigest()}
        result["actual_disk_hits"] = observed_cache_hits(result["compile_cache_events"])
        if not result["output"]["finite"]:
            raise ValueError("nonfinite cache probe output")
        if args.cache_phase == "read" and args.cache_mode == "persistent" and not result["actual_disk_hits"]:
            raise ValueError("fresh-process read had no observed persistent compilation cache hit")
        if args.cache_mode == "persistent":
            identity = {k: result[k] for k in ("python", "jax", "jaxlib", "numpy", "devices", "x64_enabled",
                                               "backend_platform_version", "xla_flags", "stablehlo_sha256", "output")}
            write_record = args.cache_root.resolve() / "probe-write.json"
            if args.cache_phase == "write":
                if result["actual_disk_hits"]:
                    raise ValueError("write probe was not cold; use a new cache root")
                atomic_json(write_record, {"pid": os.getpid(), "identity": identity})
            else:
                origin = json.loads(write_record.read_text(encoding="utf-8"))
                if origin["pid"] == os.getpid() or origin["identity"] != identity:
                    raise ValueError("read must be a separate process with identical runtime/device/IR/output")
                result["matched_write_identity"] = True
        result["status"] = "passed_cache_probe"
        return 0
    except Exception as exc:
        result.update(status="failed", exception_type=type(exc).__name__, message=str(exc))
        return 1
    finally:
        result["all_cache_events"] = observer.events
        result["elapsed_seconds"] = time.perf_counter() - PROCESS_STARTED
        atomic_json(args.output_dir / "report.json", result)
        observer.close()
        print(json.dumps({k: v for k, v in result.items() if k != "all_cache_events"}), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
