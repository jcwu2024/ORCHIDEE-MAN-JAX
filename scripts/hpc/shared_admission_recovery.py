"""Reuse a completed zero-update gradient from an interrupted admission."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


def load_resume_report(path, output, protocol):
    path = Path(path).resolve()
    if path == Path(output).resolve():
        raise ValueError("recovery must preserve the original report")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["protocol"] != protocol:
        raise ValueError("cannot reuse a gradient for a changed admission protocol")
    if report["completed_updates"] != 0:
        raise ValueError("saved-gradient recovery requires the unchanged initial parameters")
    if not report.get("gradient_artifact") or not report.get("gradient", {}).get("all_finite"):
        raise ValueError("report has no completed finite saved gradient")
    if report.get("reason") or report.get("status") in {"rejected", "error"}:
        raise ValueError("saved-gradient recovery must not bypass a rejected admission")
    if not all(all(int(value) == 0 for value in aux["hard"].values())
               for _loss, aux in report["initial"]):
        raise ValueError("source initial forwards failed physical admission")
    return report


def restore_gradient(jax, source, parameter_shapes):
    """Check small metadata and array structure, without scanning external data."""
    with Path(source["gradient_artifact"]).open("rb") as handle:
        saved = pickle.load(handle)
    if saved["protocol"] != source["protocol"] or saved["source_git_head"] != source["source_git_head"]:
        raise ValueError("saved gradient and source report disagree")
    reference, definition = jax.tree_util.tree_flatten(parameter_shapes)
    for name in ("parameters", "gradient"):
        arrays, observed = jax.tree_util.tree_flatten(saved[name])
        if definition != observed:
            raise ValueError(f"recovered {name} tree differs from current architecture")
        for value, shape in zip(arrays, reference, strict=True):
            if np.shape(value) != shape.shape or np.asarray(value).dtype != shape.dtype:
                raise ValueError(f"recovered {name} array shape/dtype differs")
    # Separate forward/AD compilations may reorder FP32 neural operations.
    # Use the already declared execution-equivalence tolerance, not bitwise equality.
    if not np.isclose(float(saved["value"]), float(source["initial"][0][0]), rtol=2e-5, atol=2e-6):
        raise ValueError("recovered gradient scalar differs from source initial forward")
    return jax.device_put(saved["parameters"]), jax.device_put(saved["gradient"]), saved["value"]
