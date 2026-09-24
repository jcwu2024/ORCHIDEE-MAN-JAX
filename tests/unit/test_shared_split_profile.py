import gzip
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.hpc.probe_shared_split_profile import (
    benchmark_replays,
    concatenate_daily,
    save_executable,
)


def test_replays_synchronize_warmup_and_every_measurement():
    events = []
    def invoke():
        events.append("invoke")
        return len(events)
    def synchronize(value):
        assert value == len(events)
        events.append("ready")
    result = benchmark_replays(invoke, synchronize, 3)
    assert events == ["invoke", "ready"] * 4
    assert len(result["seconds"]) == 3
    assert result["median_seconds"] >= 0
    with pytest.raises(ValueError):
        benchmark_replays(invoke, synchronize, 0)


def test_concatenation_preserves_both_artifacts_without_repeating_first():
    import jax
    left = {"x": np.arange(32)[:, None], "mask": np.ones((32, 2), dtype=bool)}
    right = {"x": np.arange(32, 64)[:, None], "mask": np.zeros((32, 2), dtype=bool)}
    batch = concatenate_daily(jax, left, right)
    np.testing.assert_array_equal(batch["x"].ravel(), np.arange(64))
    assert batch["mask"].dtype == np.bool_
    assert batch["mask"][:32].all() and not batch["mask"][32:].any()


def test_executable_and_hlo_are_persisted_together(tmp_path):
    executable = SimpleNamespace(as_text=lambda: "HloModule test\nROOT x = f32[] constant(1)\n")
    payload = (b"binary", "input-tree", "output-tree")
    result = save_executable(tmp_path, "T7", executable, lambda _: payload)
    with open(result["executable"], "rb") as handle:
        assert pickle.load(handle) == payload
    with gzip.open(result["optimized_hlo"], "rt") as handle:
        assert handle.read() == executable.as_text()
