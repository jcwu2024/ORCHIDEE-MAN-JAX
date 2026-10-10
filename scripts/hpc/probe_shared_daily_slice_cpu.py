"""Lightweight CPU check of exact all-leaf daily slices for 32/64 rows."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--helper", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("shared_daily_slice_cpu_helper", args.helper)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load slice helper: {args.helper}")
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    report = {"backend": jax.default_backend(), "leaf_count": 590, "batches": {}}
    if report["backend"] != "cpu":
        raise ValueError("CPU slice preparation check must not initialize a GPU backend")
    for count in (32, 64):
        rows = np.arange(count, dtype=np.int32)
        batch = {}
        for index in range(590):
            values = rows[:, None] + index
            value = (
                values.astype(np.float32)
                if index % 3 == 0
                else values.astype(np.int16)
                if index % 3 == 1
                else (values % 2) == 0
            )
            batch[f"leaf_{index:03d}"] = jnp.asarray(value)
        slicer = helper.make_daily_batch_slicer(jax, 4)
        for start in range(0, count, 4):
            selected = slicer(batch, jnp.asarray(start, dtype=jnp.int32))
            for name, value in batch.items():
                expected = value[start : start + 4]
                actual = selected[name]
                np.testing.assert_array_equal(actual, expected)
                if actual.dtype != value.dtype or actual.shape != expected.shape:
                    raise AssertionError(f"shape/dtype changed at {name}, row {start}")
        report["batches"][str(count)] = {"slices": count // 4, "exact_all_leaf_equivalence": True}
    print(report)


if __name__ == "__main__":
    main()
