"""Runtime configuration helpers for local ORCHIDEE-JAX development scripts."""

from __future__ import annotations

import os
from pathlib import Path


def configure_jax_compilation_cache(root: str | Path) -> Path | None:
    """Enable JAX's persistent cache under ``ORCHIDEE_OUTPUT_ROOT``.

    The cache stores XLA compilation artifacts only. It does not alter model
    inputs, process ordering, or numerical kernels.
    """

    if os.environ.get("ORCHJAX_DISABLE_COMPILATION_CACHE", "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        return None

    from jax import config

    cache_dir = Path(
        os.environ.get("ORCHIDEE_OUTPUT_ROOT", Path(root) / "outputs")
    ).expanduser() / "xla_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    config.update("jax_compilation_cache_dir", str(cache_dir))
    config.update("jax_persistent_cache_min_compile_time_secs", 0.5)
    config.update("jax_persistent_cache_min_entry_size_bytes", 0)
    return cache_dir
