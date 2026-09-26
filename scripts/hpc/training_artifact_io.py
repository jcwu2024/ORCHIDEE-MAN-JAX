"""Shared artifact serialization and content-identity checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str, name: str) -> None:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"{name} hash drift: {observed} != {expected}")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(_json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _json_value(child) for name, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()
